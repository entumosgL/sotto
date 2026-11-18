"""Собеседования: план, привязка звонков, напоминания и разбор отказов (команды).

Звонок, начавшийся в окне собеседования, привязывается к нему сам: суфлёр получает бриф, а придуманный
опыт (легенда) — компанию. Отказ разбирает CLI-агент по расшифровкам всех звонков с компанией и
подсказкам суфлёра; слабые темы уходят в тренер. Общий разбор ищет повторяющиеся причины по всем отказам.
Правила — в domain/interviews.py, страницы читают данные через interfaces/web → queries.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from ..domain import interviews as rules
from ..domain.interviews import Interview, Status
from ..domain.transcript import with_hints
from .legend import LegendService
from .ports import LLM, AgentError, Agents, CallRepository, Events, InterviewRepository, Meta, Prompts
from .prompting import parse_json_object, render
from .tasks import Background
from .topics import TopicService

log = logging.getLogger(__name__)


def parse_when(value: str) -> float | None:
    """Поле datetime-local («2026-11-20T15:00») → время. Пусто или мусор — без даты."""
    value = (value or "").strip()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


class InterviewService:
    def __init__(self, interviews: InterviewRepository, calls: CallRepository, meta: Meta, legend: LegendService,
                 topics: TopicService, agents: Agents, llm: LLM, prompts: Prompts, events: Events,
                 background: Background, invite_model: Callable[[], str],
                 notify: Callable[[str, str], None] | None = None):
        self.interviews = interviews
        self.calls = calls
        self.meta = meta
        self.legend = legend
        self.topics = topics
        self.agents = agents
        self.llm = llm
        self.prompts = prompts
        self.events = events
        self.background = background
        self.invite_model = invite_model
        self.notify = notify
        self.draft: dict[str, Any] = {}  # разобранное приглашение, пока его не сохранили

    def _get(self, iid: int) -> Interview | None:
        return self.interviews.get(iid)

    # --- план ---

    def schedule(self, company: str, **fields: Any) -> int:
        iv = Interview.new(company, time.time(), **fields)
        iid = self.interviews.add(iv)
        self.draft = {}
        if iv.vacancy:
            self.background.spawn(self.make_brief(iid))
            self.topics.refresh_soon(force=True)  # темы тренера — под новую вакансию
        return iid

    def edit(self, iid: int, **fields: Any) -> None:
        iv = self._get(iid)
        if iv is None:
            return
        renamed = iv.edit(**fields)
        self.interviews.save(iv)
        if renamed:  # звонки и легенда — за компанией
            for call in self.calls.for_interview(iv):
                if call.interview_id == iid:
                    self.legend.assign_company(call.id, iv.company, iid)
        if iv.needs_brief:
            self.background.spawn(self.make_brief(iid))

    def change_status(self, iid: int, status: str, outcome: str | None = None) -> None:
        iv = self._get(iid)
        if iv is None or not rules.is_status(status):
            return
        iv.change_status(status, outcome)
        self.interviews.save(iv)
        if iv.is_closed:
            self.legend.cleanup()  # процесс с компанией закрыт — её легенда больше не нужна
        if iv.status == Status.REJECTED:
            self.background.spawn(self.analyze_rejection(iid))

    def next_stage(self, iid: int) -> int | None:
        iv = self._get(iid)
        if iv is None:
            return None
        nxt = iv.next_stage(time.time())
        self.interviews.save(iv)
        return self.interviews.add(nxt)

    def delete(self, iid: int) -> None:
        self.calls.unlink_interview(iid)
        self.interviews.delete(iid)
        self.legend.cleanup()

    async def parse_invite(self, text: str) -> dict[str, Any]:
        """Приглашение из письма или мессенджера → поля формы (ты проверишь и сохранишь)."""
        prompt = render(self.prompts.load("invite"), now=datetime.now().strftime("%Y-%m-%d %H:%M, %A"),
                        text=text[:8000])
        try:
            if self.llm.configured:
                res = await self.llm.chat(self.invite_model(), [{"role": "user", "content": prompt}],
                                          max_tokens=500, json_mode=True)
                data = parse_json_object(res.text)
            else:
                data, _ = await self.agents.run_json(prompt, task="приглашение на собеседование")
        except Exception as e:  # noqa: BLE001 — не разобрали, заполнит руками
            log.warning("Приглашение не разобрано: %s", e)
            data = {}
        data = data if isinstance(data, dict) else {}
        data["notes"] = (str(data.get("notes") or "") + "\n\n" + text.strip()).strip()
        self.draft = data
        return data

    # --- звонки ---

    def attach_starting_call(self, call_id: int, at: float) -> Interview | None:
        """Звонок начался в окне запланированного собеседования — бриф и компания оттуда."""
        iv = rules.for_call(self.interviews.all(), at)
        if iv is None:
            return None
        self.legend.assign_company(call_id, iv.company, iv.id)
        call = self.calls.get(call_id)
        if call and not call.brief:
            self.calls.update(call_id, brief=iv.call_brief())
        log.info("Звонок привязан к собеседованию: %s", iv.company)
        return iv

    def link_after_call(self, call_id: int, company: str | None) -> None:
        """После звонка: компания из итогов. Не привязан — ищем собеседование с этой компанией."""
        call = self.calls.get(call_id)
        if call is None or call.is_training:
            return
        iv = self._get(call.interview_id) if call.interview_id else None
        if iv is None and company:
            iv = rules.by_company(self.interviews.all(), company, call.started_at)
            if iv is None:
                self.legend.assign_company(call_id, company)
                return
            self.legend.assign_company(call_id, iv.company, iv.id)
        if iv is not None and iv.is_planned:
            iv.call_happened()
            self.interviews.save(iv)

    def link_call(self, call_id: int, interview_id: int | None, company: str = "") -> None:
        """Ручная привязка звонка к собеседованию (или просто к компании) — легенда переезжает туда же."""
        iv = self._get(interview_id) if interview_id else None
        if iv is not None:
            self.legend.assign_company(call_id, iv.company, iv.id)
            if iv.is_planned:
                iv.call_happened()
                self.interviews.save(iv)
        else:
            self.legend.assign_company(call_id, company)
            self.calls.update(call_id, interview_id=None)

    # --- напоминания ---

    def remind(self, now: float) -> int:
        due = rules.due_reminders(self.interviews.all(), now)
        for iv in due:
            iv.mark_reminded()
            self.interviews.save(iv)
            text = rules.reminder_text(iv, now)
            self.events.publish("notice", {"text": text})
            if self.notify:
                self.notify("Sotto", text + (f"\n{iv.link}" if iv.link else ""))
        return len(due)

    # --- агенты: бриф и разборы ---

    async def make_brief(self, iid: int) -> None:
        iv = self._get(iid)
        if iv is None or not (iv.vacancy or "").strip():
            return
        prompt = (self.prompts.load("brief") + "\n\nПрофиль кандидата:\n" + (self.prompts.profile() or "(не заполнен)")
                  + f"\n\nКомпания: {iv.company}\nРоль: {iv.role or '—'}\nЭтап: {iv.stage or '—'}"
                  + "\n\nВакансия:\n" + (iv.vacancy or "")[:20000])
        try:
            text, _ = await self.agents.run(prompt, task=f"бриф: {iv.company}")
        except AgentError as e:
            self.events.publish("notice", {"text": f"Бриф не составлен: {e}"[:200]})
            return
        iv = self._get(iid)
        if iv is None:
            return
        iv.brief = text.strip()
        self.interviews.save(iv)
        self.events.publish("notice", {"text": f"Бриф готов: {iv.company}"})

    async def brief_for_vacancy(self, vacancy: str) -> str:
        """Бриф к следующему звонку без собеседования в плане."""
        prompt = (self.prompts.load("brief") + "\n\nПрофиль кандидата:\n" + self.prompts.profile()
                  + "\n\nВакансия:\n" + vacancy)
        try:
            text, _ = await self.agents.run(prompt, task="бриф перед звонком")
        except AgentError as e:
            text = f"(не удалось: {e})"
        self.meta.set_meta("next_call_brief", text)
        return text

    def _set_analysis(self, iv: Interview, text: str) -> None:
        iv.analysis = text
        self.interviews.save(iv)

    async def analyze_rejection(self, iid: int) -> None:
        iv = self._get(iid)
        if iv is None:
            return
        calls = self.calls.for_interview(iv)
        if not calls and not (iv.outcome or "").strip():
            self._set_analysis(iv, "Разбирать нечего: нет ни записанных звонков с компанией, ни текста отказа.")
            return
        talks = "\n\n".join(
            f"## Звонок {datetime.fromtimestamp(c.started_at):%d.%m %H:%M}{' — ' + c.title if c.title else ''}\n"
            + with_hints(self.calls.transcript(c.id), self.calls.hints(c.id)) for c in calls)
        history = "\n".join(f"- {r.stage or 'этап'} {datetime.fromtimestamp(r.scheduled_at):%d.%m}: {r.status_label}"
                            for r in self.interviews.all()
                            if r.company_key == iv.company_key and r.scheduled_at is not None)
        prompt = render(
            self.prompts.load("rejection"), company=iv.company, role=iv.role or "—", stage=iv.stage or "—",
            outcome=(iv.outcome or "").strip() or "(текста отказа нет)", history=history or "—",
            vacancy=(iv.brief or iv.vacancy or "(нет)")[:6000], profile=self.prompts.profile() or "(не заполнен)",
            transcripts=talks[-60000:] or "(звонки не записаны)")
        self._set_analysis(iv, "⏳ Разбираю…")
        try:
            data, agent = await self.agents.run_json(prompt, task=f"разбор отказа: {iv.company}")
        except AgentError as e:
            self._set_analysis(iv, f"Разбор не получился: {e}")
            self.events.publish("notice", {"text": f"Разбор отказа: {e}"[:200]})
            return
        if not isinstance(data, dict):
            self._set_analysis(iv, "Разбор не получился: агент вернул не JSON")
            return
        self._set_analysis(iv, str(data.get("analysis") or "").strip())
        for kind, key in (("knowledge", "topics"), ("english", "english")):
            items = [str(x).strip()[:200] for x in data.get(key) or [] if str(x).strip()]
            self.topics.store.add_items(kind, items, "rejections", time.time(), bump=True)
        self.topics.refresh_soon(force=True)  # темы тренера — с учётом того, на чём поплыл
        log.info("Разбор отказа %s готов (%s)", iv.company, agent)
        self.events.publish("notice", {"text": f"Разбор отказа готов: {iv.company}"})

    async def analyze_all(self) -> None:
        all_ = self.interviews.all()
        rejected = [r for r in all_ if r.status == Status.REJECTED]
        if not rejected:
            self.meta.set_meta("rejections_summary", "Отказов пока нет.")
            return
        items = "\n\n".join(
            f"## {r.company} — {r.role or 'роль ?'}, этап: {r.stage or '?'}"
            f"{', ' + datetime.fromtimestamp(r.scheduled_at).strftime('%d.%m.%Y') if r.scheduled_at else ''}\n"
            f"Текст отказа: {(r.outcome or '').strip() or '—'}\n\nРазбор:\n{(r.analysis or '(не разобран)')[:4000]}"
            for r in rejected)
        counts: dict[str, int] = {}
        for r in all_:
            counts[r.status_label] = counts.get(r.status_label, 0) + 1
        prompt = render(self.prompts.load("rejections_summary"),
                        funnel=", ".join(f"{k}: {n}" for k, n in counts.items()), items=items[-60000:],
                        profile=self.prompts.profile() or "(не заполнен)")
        self.meta.set_meta("rejections_summary", "⏳ Разбираю…")
        try:
            text, _ = await self.agents.run(prompt, task="общий разбор отказов")
        except AgentError as e:
            self.meta.set_meta("rejections_summary", f"Не получилось: {e}")
            return
        self.meta.set_meta("rejections_summary", text.strip())
        self.meta.set_meta("rejections_summary_at", str(time.time()))
        self.events.publish("notice", {"text": "Общий разбор отказов готов"})
