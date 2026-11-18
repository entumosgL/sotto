"""Собеседования: план, привязка звонков, напоминания и разбор отказов.

Собеседование — один этап у одной компании (HR, техническое, финальное…). Звонок, начавшийся в окне
собеседования, привязывается к нему сам: суфлёр получает бриф, а придуманный опыт (легенда) — компанию.
Так легенда держится в пределах одной компании и удаляется, когда процесс с ней закрыт.

Отказ разбирает CLI-агент по расшифровкам всех звонков с компанией и подсказкам суфлёра; слабые темы
уходят в тренер. Общий разбор ищет повторяющиеся причины по всем отказам.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any

from .agents import AgentError
from .copilot import context, prompts
from .db import DB, norm_company

if TYPE_CHECKING:
    from .daemon import Daemon

log = logging.getLogger(__name__)

STATUSES = {
    "planned": "запланировано",
    "done": "прошло, жду ответа",
    "passed": "позвали дальше",
    "offer": "оффер",
    "rejected": "отказ",
    "declined": "я отказался",
    "cancelled": "отменено",
}
CLOSED = ("rejected", "declined", "cancelled")

BEFORE = 30 * 60      # звонок за полчаса до собеседования — уже он
AFTER = 90 * 60       # и до полутора часов после начала
REMIND = 15 * 60      # напоминание за 15 минут


def find_for_call(db: DB, at: float) -> dict[str, Any] | None:
    """Запланированное собеседование, в окно которого попадает начало звонка."""
    rows = db.all("""SELECT * FROM interviews WHERE status = 'planned' AND scheduled_at IS NOT NULL
                     AND scheduled_at BETWEEN ? AND ?""", [at - AFTER, at + BEFORE])
    return min(rows, key=lambda r: abs(r["scheduled_at"] - at)) if rows else None


def attach_call(db: DB, call_id: int, iv: dict[str, Any]) -> None:
    db.set_call_company(call_id, iv["company"], iv["id"])
    call = db.one("SELECT brief FROM calls WHERE id = ?", [call_id])
    if call and not call["brief"]:
        brief = iv["brief"] or (iv["vacancy"] or "")[:3000]
        head = f"Компания: {iv['company']}" + (f", роль: {iv['role']}" if iv["role"] else "") + \
               (f", этап: {iv['stage']}" if iv["stage"] else "")
        db.update("calls", call_id, brief=head + ("\n\n" + brief if brief else ""))


def link_after_call(db: DB, call_id: int, company: str | None) -> None:
    """После звонка: компания из итогов. Если звонок ещё не привязан — ищем собеседование с этой компанией."""
    call = db.one("SELECT * FROM calls WHERE id = ?", [call_id])
    if not call or call["app"] == "training":
        return
    if call["interview_id"]:
        iv = db.one("SELECT * FROM interviews WHERE id = ?", [call["interview_id"]])
        if iv and iv["status"] == "planned":
            db.update("interviews", iv["id"], status="done", updated_at=time.time())
        return
    name = norm_company(company)
    if not name:
        return
    candidates = [r for r in db.all("SELECT * FROM interviews WHERE status IN ('planned', 'done', 'passed')")
                  if norm_company(r["company"]) == name or name in norm_company(r["company"])
                  or norm_company(r["company"]) in name]
    if candidates:
        iv = min(candidates, key=lambda r: abs((r["scheduled_at"] or r["created_at"]) - call["started_at"]))
        db.set_call_company(call_id, iv["company"], iv["id"])
        if iv["status"] == "planned":
            db.update("interviews", iv["id"], status="done", updated_at=time.time())
    else:
        db.set_call_company(call_id, company)


def parse_when(value: str) -> float | None:
    """Поле datetime-local («2026-11-20T15:00») → время. Пусто — без даты."""
    value = (value or "").strip()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def company_calls(db: DB, iv: dict[str, Any]) -> list[dict[str, Any]]:
    return db.all("""SELECT * FROM calls WHERE app IS NOT 'training'
                     AND (interview_id = ? OR lower(trim(company)) = ?) ORDER BY started_at""",
                  [iv["id"], norm_company(iv["company"])])


def _transcript_with_hints(db: DB, call_id: int) -> str:
    """Расшифровка, где после вопроса видно, что подсказал суфлёр, — чтобы разбор видел и подсказки."""
    rows = [{"at": u["started_at"], "line": context.format_transcript([u])}
            for u in db.call_transcript(call_id)]
    rows += [{"at": a["created_at"], "line": f"    [подсказка суфлёра] {a['text']}"}
             for a in db.all("SELECT text, created_at FROM answers WHERE call_id = ?", [call_id])]
    return "\n".join(r["line"] for r in sorted(rows, key=lambda r: r["at"]))


async def make_brief(d: "Daemon", iid: int) -> None:
    iv = d.db.one("SELECT * FROM interviews WHERE id = ?", [iid])
    if not iv or not (iv["vacancy"] or "").strip():
        return
    prompt = (prompts.load("brief") + "\n\nПрофиль кандидата:\n" + (prompts.profile() or "(не заполнен)")
              + f"\n\nКомпания: {iv['company']}\nРоль: {iv['role'] or '—'}\nЭтап: {iv['stage'] or '—'}"
              + "\n\nВакансия:\n" + iv["vacancy"][:20000])
    try:
        text, _ = await d.agents.run(prompt, task=f"бриф: {iv['company']}")
    except AgentError as e:
        d.bus.publish("notice", {"text": f"Бриф не составлен: {e}"[:200]})
        return
    d.db.update("interviews", iid, brief=text.strip(), updated_at=time.time())
    d.bus.publish("notice", {"text": f"Бриф готов: {iv['company']}"})


async def parse_invite(d: "Daemon", text: str) -> dict[str, Any]:
    """Приглашение из письма или мессенджера → поля собеседования."""
    now = datetime.now()
    prompt = prompts.render(prompts.load("invite"), now=now.strftime("%Y-%m-%d %H:%M, %A"), text=text[:8000])
    try:
        if d.cfg.api_key:
            res = await d.llm.chat(d.cfg.models.answer, [{"role": "user", "content": prompt}],
                                   max_tokens=500, json_mode=True)
            data = json.loads(res.text[res.text.find("{"):res.text.rfind("}") + 1])
        else:
            data, _ = await d.agents.run_json(prompt, task="приглашение на собеседование")
    except Exception as e:  # noqa: BLE001 — не разобрали, пусть заполнит руками
        log.warning("Приглашение не разобрано: %s", e)
        return {}
    return data if isinstance(data, dict) else {}


async def analyze_rejection(d: "Daemon", iid: int) -> None:
    db = d.db
    iv = db.one("SELECT * FROM interviews WHERE id = ?", [iid])
    if not iv:
        return
    calls = company_calls(db, iv)
    if not calls and not (iv["outcome"] or "").strip():
        db.update("interviews", iid, analysis="Разбирать нечего: нет ни записанных звонков с компанией, ни текста отказа.")
        return
    talks = "\n\n".join(f"## Звонок {datetime.fromtimestamp(c['started_at']):%d.%m %H:%M}"
                        f"{' — ' + c['title'] if c['title'] else ''}\n{_transcript_with_hints(db, c['id'])}"
                        for c in calls)
    history = "\n".join(f"- {r['stage'] or 'этап'} {datetime.fromtimestamp(r['scheduled_at']):%d.%m}: "
                        f"{STATUSES.get(r['status'], r['status'])}"
                        for r in db.all("""SELECT * FROM interviews WHERE lower(trim(company)) = ?
                                           AND scheduled_at IS NOT NULL ORDER BY scheduled_at""",
                                        [norm_company(iv["company"])]))
    prompt = prompts.render(
        prompts.load("rejection"), company=iv["company"], role=iv["role"] or "—", stage=iv["stage"] or "—",
        outcome=(iv["outcome"] or "").strip() or "(текста отказа нет)", history=history or "—",
        vacancy=(iv["brief"] or iv["vacancy"] or "(нет)")[:6000], profile=prompts.profile() or "(не заполнен)",
        transcripts=talks[-60000:] or "(звонки не записаны)")
    db.update("interviews", iid, analysis="⏳ Разбираю…")
    try:
        data, agent = await d.agents.run_json(prompt, task=f"разбор отказа: {iv['company']}")
    except AgentError as e:
        db.update("interviews", iid, analysis=f"Разбор не получился: {e}")
        d.bus.publish("notice", {"text": f"Разбор отказа: {e}"[:200]})
        return
    if not isinstance(data, dict):
        db.update("interviews", iid, analysis="Разбор не получился: агент вернул не JSON")
        return
    db.update("interviews", iid, analysis=str(data.get("analysis") or "").strip(), updated_at=time.time())
    for kind, key in (("knowledge", "topics"), ("english", "english")):
        for item in data.get(key) or []:
            item = str(item).strip()[:200]
            if item:
                db.execute("""INSERT INTO trainer_items(kind, item, source, due_at) VALUES(?, ?, 'rejections', ?)
                              ON CONFLICT(kind, item) DO UPDATE SET due_at = excluded.due_at""",
                           [kind, item, time.time()])
    from .trainer.topics import refresh_soon
    refresh_soon(d, force=True)  # темы тренера — с учётом того, на чём поплыл
    log.info("Разбор отказа %s готов (%s)", iv["company"], agent)
    d.bus.publish("notice", {"text": f"Разбор отказа готов: {iv['company']}"})


async def analyze_all(d: "Daemon") -> None:
    rows = d.db.all("""SELECT * FROM interviews WHERE status = 'rejected' ORDER BY scheduled_at""")
    if not rows:
        d.db.set_meta("rejections_summary", "Отказов пока нет.")
        return
    items = "\n\n".join(
        f"## {r['company']} — {r['role'] or 'роль ?'}, этап: {r['stage'] or '?'}"
        f"{', ' + datetime.fromtimestamp(r['scheduled_at']).strftime('%d.%m.%Y') if r['scheduled_at'] else ''}\n"
        f"Текст отказа: {(r['outcome'] or '').strip() or '—'}\n\nРазбор:\n{(r['analysis'] or '(не разобран)')[:4000]}"
        for r in rows)
    stats = d.db.all("SELECT status, COUNT(*) AS n FROM interviews GROUP BY status")
    funnel = ", ".join(f"{STATUSES.get(s['status'], s['status'])}: {s['n']}" for s in stats)
    prompt = prompts.render(prompts.load("rejections_summary"), funnel=funnel, items=items[-60000:],
                            profile=prompts.profile() or "(не заполнен)")
    d.db.set_meta("rejections_summary", "⏳ Разбираю…")
    try:
        text, _ = await d.agents.run(prompt, task="общий разбор отказов")
    except AgentError as e:
        d.db.set_meta("rejections_summary", f"Не получилось: {e}")
        return
    d.db.set_meta("rejections_summary", text.strip())
    d.db.set_meta("rejections_summary_at", str(time.time()))
    d.bus.publish("notice", {"text": "Общий разбор отказов готов"})


def set_status(d: "Daemon", iid: int, status: str, outcome: str | None = None) -> None:
    if status not in STATUSES:
        return
    values: dict[str, Any] = {"status": status, "updated_at": time.time()}
    if outcome is not None:
        values["outcome"] = outcome.strip() or None
    d.db.update("interviews", iid, **values)
    if status in CLOSED:
        n = d.db.cleanup_legend()
        if n:
            log.info("Процесс закрыт — удалено пунктов легенды: %s", n)
    if status == "rejected":
        d._spawn(analyze_rejection(d, iid))


def upcoming(db: DB, limit: int = 5) -> list[dict[str, Any]]:
    return db.all("""SELECT * FROM interviews WHERE status = 'planned' AND
                     (scheduled_at IS NULL OR scheduled_at > ?) ORDER BY scheduled_at IS NULL, scheduled_at LIMIT ?""",
                  [time.time() - AFTER, limit])


def nearest_vacancy(db: DB) -> str:
    """Вакансия ближайшего собеседования — для тренировок «как будто это оно»."""
    for iv in upcoming(db):
        if (iv["vacancy"] or "").strip():
            return iv["vacancy"]
    return ""


async def reminders(d: "Daemon") -> None:
    """Раз в минуту: напоминание за 15 минут до собеседования."""
    while not d._stop.is_set():
        now = time.time()
        for iv in d.db.all("""SELECT * FROM interviews WHERE status = 'planned' AND reminded = 0
                              AND scheduled_at BETWEEN ? AND ?""", [now - 60, now + REMIND]):
            d.db.update("interviews", iv["id"], reminded=1)
            mins = max(0, round((iv["scheduled_at"] - now) / 60))
            text = f"Через {mins} мин собеседование: {iv['company']}" + (f" ({iv['stage']})" if iv["stage"] else "")
            d.bus.publish("notice", {"text": text})
            if d.with_overlay and shutil.which("notify-send"):
                subprocess.Popen(["notify-send", "-a", "Sotto", "Sotto", text + (f"\n{iv['link']}" if iv["link"] else "")],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        await asyncio.sleep(60)
