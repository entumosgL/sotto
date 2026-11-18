"""После звонка: название, краткое описание, компания, вопросы и темы — через CLI-агента (не тратит ключ).

Вопросы с настоящих собеседований попадают в тренер как сырьё для подбора тем; после собеседования
темы обновляются. Тренировочные звонки сигналом не считаются.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from ..domain.transcript import format_transcript
from .interviews import InterviewService
from .legend import LegendService
from .ports import AgentError, Agents, CallRepository, Events, Prompts, SearchIndex
from .topics import TopicService

log = logging.getLogger(__name__)

MIN_UTTERANCES = 4  # совсем короткие «звонки» не разбираем


class PostCall:
    def __init__(self, calls: CallRepository, interviews: InterviewService, legend: LegendService,
                 topics: TopicService, agents: Agents, prompts: Prompts, search: SearchIndex | None, events: Events):
        self.calls = calls
        self.interviews = interviews
        self.legend = legend
        self.topics = topics
        self.agents = agents
        self.prompts = prompts
        self.search = search
        self.events = events

    async def _index(self, call_id: int) -> None:
        if self.search is not None:
            await asyncio.to_thread(self.search.index_call, call_id)

    async def process(self, call_id: int) -> dict[str, Any] | None:
        """Итоги звонка. Возвращает разобранный ответ агента (None — не разбирали)."""
        rows = self.calls.transcript(call_id)
        if len(rows) < MIN_UTTERANCES:
            await self._index(call_id)
            return None
        prompt = self.prompts.load("postcall") + "\n\nРасшифровка:\n" + format_transcript(rows)[-60000:]
        try:
            data, agent = await self.agents.run_json(prompt, task=f"итоги звонка #{call_id}")
        except AgentError as e:
            log.warning("Итоги звонка #%s не получены: %s", call_id, e)
            self.events.publish("notice", {"text": f"Итоги звонка не готовы: {e}"[:200]})
            await self._index(call_id)
            return None
        if not isinstance(data, dict):
            log.warning("Итоги звонка #%s: агент вернул не объект", call_id)
            await self._index(call_id)
            return None
        self.calls.update(call_id, title=str(data.get("title") or "")[:200], summary=str(data.get("summary") or ""),
                          language=data.get("language"))
        interview = data.get("kind") == "interview"
        self.interviews.link_after_call(call_id, data.get("company") if interview else None)
        self.legend.cleanup()
        call = self.calls.get(call_id)
        if interview and call is not None and not call.is_training:
            items = [str(t).strip()[:300] for t in (data.get("topics") or []) + (data.get("questions") or [])
                     if str(t).strip()]
            self.topics.store.add_items("knowledge", items, "calls", time.time())
            self.topics.refresh_soon(force=True)  # темы тренера — под то, что спрашивали
        await self._index(call_id)
        log.info("Итоги звонка #%s готовы (%s)", call_id, agent)
        self.events.publish("call.processed", {"call_id": call_id, "title": data.get("title")})
        return data
