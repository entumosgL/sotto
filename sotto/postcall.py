"""После звонка: название, краткое описание, вопросы и темы — через CLI-агента (не тратит ключ).

Темы и вопросы из реальных собеседований попадают в тренер знаний.
"""

from __future__ import annotations

import asyncio
import logging
import time

from .agents import AgentError, Agents
from .copilot import context, prompts
from .db import DB
from .events import EventBus
from .search import Search

log = logging.getLogger(__name__)

MIN_UTTERANCES = 4  # совсем короткие «звонки» не разбираем


async def process_call(call_id: int, db: DB, agents: Agents, search: Search | None, bus: EventBus) -> dict | None:
    """Итоги звонка. Возвращает разобранный ответ агента (None — не разбирали)."""
    rows = db.call_transcript(call_id)
    if len(rows) < MIN_UTTERANCES:
        if search:
            await asyncio.to_thread(search.index_call, call_id)
        return None
    transcript = context.format_transcript(rows)
    prompt = prompts.load("postcall") + "\n\nРасшифровка:\n" + transcript[-60000:]
    try:
        data, agent = await agents.run_json(prompt, task=f"итоги звонка #{call_id}")
    except AgentError as e:
        log.warning("Итоги звонка #%s не получены: %s", call_id, e)
        bus.publish("notice", {"text": f"Итоги звонка не готовы: {e}"[:200]})
        if search:
            await asyncio.to_thread(search.index_call, call_id)
        return None
    if not isinstance(data, dict):
        log.warning("Итоги звонка #%s: агент вернул не объект", call_id)
        if search:
            await asyncio.to_thread(search.index_call, call_id)
        return None
    db.update("calls", call_id, title=str(data.get("title") or "")[:200],
              summary=str(data.get("summary") or ""), language=data.get("language"))
    from .interviews import link_after_call
    link_after_call(db, call_id, data.get("company") if data.get("kind") == "interview" else None)
    db.cleanup_legend()
    training = (db.one("SELECT app FROM calls WHERE id = ?", [call_id]) or {}).get("app") == "training"
    if data.get("kind") == "interview" and not training:  # вопросы тренировочного интервьюера — не сигнал
        for topic in (data.get("topics") or []) + (data.get("questions") or []):
            topic = str(topic).strip()
            if topic:
                db.execute("INSERT OR IGNORE INTO trainer_items(kind, item, source, due_at) VALUES('knowledge', ?, 'calls', ?)",
                           [topic[:300], time.time()])
    if search:
        await asyncio.to_thread(search.index_call, call_id)
    log.info("Итоги звонка #%s готовы (%s)", call_id, agent)
    bus.publish("call.processed", {"call_id": call_id, "title": data.get("title")})
    return data
