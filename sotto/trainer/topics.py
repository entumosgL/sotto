"""Темы для тренировок подбираются сами — руками ничего задавать не нужно.

Источники, по убыванию веса:
  1) ближайшие собеседования и их вакансии;
  2) вопросы и темы с твоих настоящих собеседований и разборов отказов;
  3) требования из собранных вакансий;
  4) знания агента о том, что обычно спрашивают на такую роль.

Обновляются: при запуске (если список пуст или старше STALE), после собеседования, после разбора
отказа и когда запланировано новое собеседование с вакансией.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING

from ..agents import AgentError
from ..copilot import prompts

if TYPE_CHECKING:
    from ..daemon import Daemon

STALE = 3 * 86400
WHAT = {
    "knowledge": "15 тем знаний для технических собеседований",
    "english": "12 тем разговорной практики английского: рассказ о себе и проектах, behavioral-вопросы, "
               "технические темы из его стека, small talk с командой, обсуждение задач",
}
# Вопросы со звонков и из разборов отказов — сырьё для агента, а не темы: в списке и в выборе темы их нет.
CURATED = "source NOT IN ('calls', 'rejections')"
RETRY = 15 * 60       # агент не ответил — повторяем не раньше чем через 15 минут
_running: set[str] = set()
_tried: dict[str, float] = {}


def _lines(items: list[str]) -> str:
    return "\n".join(f"- {x}" for x in items) or "(пока нет)"


async def refresh_topics(d: "Daemon", kind: str, quiet: bool = False) -> int:
    if kind in _running:
        return 0
    _running.add(kind)
    try:
        return await _refresh(d, kind, quiet)
    finally:
        _running.discard(kind)


async def _refresh(d: "Daemon", kind: str, quiet: bool) -> int:
    db = d.db
    from ..interviews import upcoming
    soon = "\n".join(
        "- " + (f"{datetime.fromtimestamp(iv['scheduled_at']):%d.%m %H:%M} " if iv["scheduled_at"] else "")
        + f"{iv['company']}, {iv['role'] or 'роль ?'}, этап {iv['stage'] or '?'}\n"
        + f"  {(iv['brief'] or iv['vacancy'] or '')[:1500]}" for iv in upcoming(db)) or "(не запланировано)"
    from_calls = [r["item"] for r in db.all(
        """SELECT item FROM trainer_items WHERE kind = ? AND source IN ('calls', 'rejections')
           ORDER BY source = 'rejections' DESC, id DESC LIMIT 80""", [kind])]
    clusters = db.all("SELECT name, description FROM clusters ORDER BY id")
    vacancies = db.all("SELECT title, text FROM vacancies WHERE status != 'hidden' ORDER BY found_at DESC LIMIT 15")
    weak = [r["item"] for r in db.all(
        "SELECT item FROM trainer_items WHERE kind = ? AND seen > 0 AND score < 0.6 ORDER BY score LIMIT 15", [kind])]
    strong = [r["item"] for r in db.all(
        "SELECT item FROM trainer_items WHERE kind = ? AND seen > 0 AND score >= 0.8 LIMIT 30", [kind])]
    prompt = prompts.render(
        prompts.load("trainer_topics"), what=WHAT[kind], upcoming=soon, from_calls=_lines(from_calls),
        weak=_lines(weak), strong=_lines(strong),
        clusters=_lines([f"{c['name']}: {c['description'] or ''}" for c in clusters]),
        vacancies=_lines([f"{v['title']}: {v['text'][:300]}" for v in vacancies]),
        profile=prompts.profile() or "(не заполнен)")
    try:
        items, _agent = await d.agents.run_json(prompt, task=f"темы тренера {kind}")
    except AgentError as e:
        if not quiet:
            d.bus.publish("notice", {"text": f"Темы: {e}"[:200]})
        return 0
    topics = [str(it.get("topic") or "").strip()[:200] for it in (items if isinstance(items, list) else [])
              if isinstance(it, dict)]
    topics = [t for t in topics if t]
    if not topics:
        return 0
    # Прошлый список агента, к которому ещё не приступали, заменяем новым — список не разрастается.
    db.execute("DELETE FROM trainer_items WHERE kind = ? AND source = 'agent' AND seen = 0", [kind])
    now = time.time()
    for i, topic in enumerate(topics):  # порядок агента = очерёдность
        db.execute("INSERT OR IGNORE INTO trainer_items(kind, item, source, due_at) VALUES(?, ?, 'agent', ?)",
                   [kind, topic, now + i])
    db.set_meta(f"topics_at_{kind}", str(now))
    d.bus.publish("trainer.topics", {"kind": kind, "n": len(topics)})
    return len(topics)


def refresh_soon(d: "Daemon", force: bool = False) -> None:
    """Обновить темы в фоне: force — после событий (собеседование, отказ), иначе — только устаревшие."""
    for kind in ("knowledge", "english"):
        at = float(d.db.get_meta(f"topics_at_{kind}") or 0)
        empty = not d.db.one(f"SELECT 1 AS x FROM trainer_items WHERE kind = ? AND {CURATED} LIMIT 1", [kind])
        if (force or empty or time.time() - at > STALE) and (force or time.time() - _tried.get(kind, 0) > RETRY):
            _tried[kind] = time.time()
            d._spawn(refresh_topics(d, kind, quiet=True))


def pick(d: "Daemon", kind: str) -> str:
    due = d.db.one(f"SELECT item FROM trainer_items WHERE kind = ? AND {CURATED} ORDER BY due_at ASC, score ASC LIMIT 1",
                   [kind])
    return due["item"] if due else ("рассказ о себе и опыте" if kind == "english" else "темы из профиля кандидата")

