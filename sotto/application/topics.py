"""Темы для тренировок подбираются сами — руками ничего задавать не нужно.

Источники, по убыванию веса:
  1) ближайшие собеседования и их вакансии;
  2) вопросы и темы с твоих настоящих собеседований и разборов отказов;
  3) требования из собранных вакансий;
  4) знания агента о том, что обычно спрашивают на такую роль.

Обновляются: при запуске (если список пуст или старше трёх дней), после собеседования, после разбора
отказа и когда запланировано новое собеседование с вакансией.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Protocol

from ..domain import interviews as iv_rules
from ..domain import training
from .ports import AgentError, Agents, Events, InterviewRepository, Meta, Prompts
from .prompting import render
from .tasks import Background

WHAT = {
    "knowledge": "15 тем знаний для технических собеседований",
    "english": "12 тем разговорной практики английского: рассказ о себе и проектах, behavioral-вопросы, "
               "технические темы из его стека, small talk с командой, обсуждение задач",
}


class TopicStore(Protocol):
    def raw_signals(self, kind: str, limit: int = 80) -> list[str]: ...
    def by_score(self, kind: str, weak: bool, limit: int) -> list[str]: ...
    def replace_agent_topics(self, kind: str, topics: list[str], now: float) -> None: ...
    def has_topics(self, kind: str) -> bool: ...
    def next_topic(self, kind: str) -> str | None: ...
    def add_items(self, kind: str, items: list[str], source: str, due_at: float, bump: bool = False) -> None: ...
    def delete_item(self, item_id: int) -> None: ...


class Market(Protocol):
    """Что известно о рынке из собранных вакансий."""
    def clusters(self) -> list[dict[str, Any]]: ...
    def recent_vacancies(self, limit: int) -> list[dict[str, Any]]: ...


def _lines(items: list[str]) -> str:
    return "\n".join(f"- {x}" for x in items) or "(пока нет)"


class TopicService:
    def __init__(self, store: TopicStore, interviews: InterviewRepository, market: Market, meta: Meta,
                 agents: Agents, prompts: Prompts, events: Events, background: Background):
        self.store = store
        self.interviews = interviews
        self.market = market
        self.meta = meta
        self.agents = agents
        self.prompts = prompts
        self.events = events
        self.background = background
        self._running: set[str] = set()
        self._tried: dict[str, float] = {}

    def updated_at(self, kind: str) -> float:
        return float(self.meta.get_meta(f"topics_at_{kind}") or 0)

    def drop(self, item_id: int) -> None:
        """Тему убрали руками — больше не предлагать (до следующего подбора агентом)."""
        self.store.delete_item(item_id)

    def pick(self, kind: str) -> str:
        return self.store.next_topic(kind) or training.FALLBACK[kind]

    def refresh_soon(self, force: bool = False) -> None:
        """Обновить в фоне: force — после событий (собеседование, отказ), иначе — только пустые и устаревшие."""
        now = time.time()
        for kind in training.KINDS:
            if training.needs_refresh(self.updated_at(kind), not self.store.has_topics(kind),
                                      self._tried.get(kind, 0), now, force):
                self._tried[kind] = now
                self.background.spawn(self.refresh(kind, quiet=True))

    async def refresh(self, kind: str, quiet: bool = False) -> int:
        if kind in self._running:
            return 0
        self._running.add(kind)
        try:
            return await self._refresh(kind, quiet)
        finally:
            self._running.discard(kind)

    async def _refresh(self, kind: str, quiet: bool) -> int:
        soon = "\n".join(
            "- " + (f"{datetime.fromtimestamp(iv.scheduled_at):%d.%m %H:%M} " if iv.scheduled_at else "")
            + f"{iv.company}, {iv.role or 'роль ?'}, этап {iv.stage or '?'}\n"
            + f"  {(iv.brief or iv.vacancy or '')[:1500]}"
            for iv in iv_rules.upcoming(self.interviews.all(), time.time())) or "(не запланировано)"
        prompt = render(
            self.prompts.load("trainer_topics"), what=WHAT[kind], upcoming=soon,
            from_calls=_lines(self.store.raw_signals(kind)),
            weak=_lines(self.store.by_score(kind, weak=True, limit=15)),
            strong=_lines(self.store.by_score(kind, weak=False, limit=30)),
            clusters=_lines([f"{c['name']}: {c['description'] or ''}" for c in self.market.clusters()]),
            vacancies=_lines([f"{v['title']}: {v['text'][:300]}" for v in self.market.recent_vacancies(15)]),
            profile=self.prompts.profile() or "(не заполнен)")
        try:
            items, _agent = await self.agents.run_json(prompt, task=f"темы тренера {kind}")
        except AgentError as e:
            if not quiet:
                self.events.publish("notice", {"text": f"Темы: {e}"[:200]})
            return 0
        topics = [str(it.get("topic") or "").strip()[:200] for it in (items if isinstance(items, list) else [])
                  if isinstance(it, dict)]
        topics = [t for t in topics if t]
        if not topics:
            return 0
        now = time.time()
        self.store.replace_agent_topics(kind, topics, now)
        self.meta.set_meta(f"topics_at_{kind}", str(now))
        self.events.publish("trainer.topics", {"kind": kind, "n": len(topics)})
        return len(topics)
