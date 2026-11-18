"""Текстовый режим: проверить подсказки суфлёра без звука.

Два вида сессии:
  self        — ты сам пишешь вопросы за собеседника, суфлёр подсказывает ответ;
  interviewer — вопросы задаёт интервьюер-тренер (тот же, что в `sotto sim`), ты отвечаешь текстом
                (по умолчанию — подсказкой суфлёра).

Каждая сессия — отдельный звонок с app="training": её легенда не попадает в настоящие звонки, а сама
сессия видна в истории звонков. У сессии свой суфлёр и своя шина событий, поэтому она не мешает
голосовому суфлёру и не показывается в оверлее.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import time
from typing import TYPE_CHECKING, Any

from .copilot import prompts
from .copilot.engine import Copilot
from .events import EventBus
from .sim import interview as iv

if TYPE_CHECKING:
    from .daemon import Daemon

log = logging.getLogger(__name__)


class TextSession:
    def __init__(self, d: "Daemon", mode: str, lang: str, role: str, vacancy: str):
        self.d = d
        self.id = secrets.token_hex(4)
        self.mode = mode
        self.lang = lang if lang in iv.LANGS else "en"
        self.role = role.strip() or iv.LANGS[self.lang]["default_role"]
        self.vacancy = vacancy.strip()
        self.bus = EventBus()
        self.bus.bind(asyncio.get_running_loop())
        self.copilot = Copilot(d.cfg, d.db, self.bus, d.llm, them_speaking=lambda: False, stage_name=lambda: None)
        self.call_id = d.db.insert("calls", started_at=time.time(), app="training", language=self.lang,
                                   title=f"Текстовая проверка · {self.role}")
        self.copilot.set_call(self.call_id)
        self.them_speaker = d.db.insert("speakers", call_id=self.call_id, label="Собеседник 1",
                                        name=iv.LANGS[self.lang]["name"] if mode == "interviewer" else None)
        self.history: list[dict[str, str]] = []   # для интервьюера: assistant — он, user — кандидат
        self.plan = ""
        self.context = ""
        self._plan_task: asyncio.Task | None = None
        self.hint = ""          # последняя подсказка (без [[ ]]), «∅» — суфлёр решил промолчать
        self.answered = True    # кандидат уже ответил на последнюю реплику собеседника
        self.busy = False
        self.ended = False
        self._lock = asyncio.Lock()
        self._tasks: set[asyncio.Task] = set()

    # --- общее ---

    def state(self) -> dict[str, Any]:
        return {"id": self.id, "mode": self.mode, "lang": self.lang, "role": self.role, "call_id": self.call_id,
                "busy": self.busy, "ended": self.ended, "hint": self.hint}

    def _spawn(self, coro) -> None:
        task = asyncio.ensure_future(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        task.add_done_callback(self._report)

    def _report(self, task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception():
            log.error("Текстовая сессия: %s", task.exception())
            self.bus.publish("chat.error", {"text": str(task.exception())[:300]})
            self._set_busy(False)

    def _set_busy(self, busy: bool, what: str = "") -> None:
        self.busy = busy
        self.bus.publish("chat.state", {"busy": busy, "what": what})

    def _line(self, channel: str, text: str) -> None:
        self.d.db.insert("utterances", call_id=self.call_id, channel=channel,
                         speaker_id=self.them_speaker if channel == "them" else None,
                         started_at=time.time(), ended_at=time.time(), text=text, language=self.lang)
        self.bus.publish("chat.line", {"channel": channel, "text": text})

    async def _hint(self) -> str:
        """Подсказка на последнюю реплику собеседника — та же, что в звонке (авто-режим)."""
        self._set_busy(True, "suffler")
        self.copilot.last_hint = ""
        await self.copilot.answer(kind="auto")
        task = self.copilot._tasks.get("main")
        if task is not None:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self.hint = self.copilot.reading_hint().strip() or "∅"
        self.answered = False
        self._set_busy(False)
        return self.hint

    # --- ты сам задаёшь вопросы ---

    async def ask(self, text: str) -> None:
        async with self._lock:
            self._candidate_said_hint()
            self._line("them", text)
            await self._hint()

    def _candidate_said_hint(self) -> None:
        """Ты не написал ответ кандидата — считаем, что он сказал подсказку (как на настоящем звонке)."""
        if not self.answered and self.hint and self.hint != "∅":
            self._line("me", self.hint)
        self.answered = True

    async def candidate(self, text: str) -> None:
        """Ответ кандидата. В режиме интервьюера после него отвечает интервьюер."""
        async with self._lock:
            text = text.strip() or (self.hint if self.hint != "∅" else "")
            if text:
                self._line("me", text)
            self.answered = True
            if self.mode == "interviewer":
                self.history.append({"role": "user", "content": text or iv.LANGS[self.lang]["silence"]})
                await self._interviewer_turn()

    # --- вопросы задаёт интервьюер ---

    async def start_interviewer(self) -> None:
        async with self._lock:
            self._set_busy(True, "interviewer")
            resume = prompts.profile()
            brief = ""
            if self.vacancy:
                info = await iv.digest_vacancy(self.vacancy, self.d.cfg)
                if info and info.get("role"):
                    self.role = info["role"] + (f" ({info['company']})" if info.get("company") else "")
                    brief = info.get("brief", "")
                    self.d.db.update("calls", self.call_id, brief=brief, title=f"Текстовая тренировка · {self.role}")
                    self.bus.publish("chat.role", {"role": self.role})
            ctx = iv.CONTEXT[self.lang]
            self.context = ((ctx["vacancy"].format(text=brief or self.vacancy) if self.vacancy else "")
                            + (ctx["resume"].format(text=resume) if resume else ""))
            self._plan_task = asyncio.create_task(self._make_plan())
            await self._interviewer_turn()       # приветствие, пока пишется план

    async def _make_plan(self) -> str:
        prompt = iv.plan_prompt(self.lang, name=iv.LANGS[self.lang]["name"], role=self.role, context=self.context)
        res = await self.d.llm.chat(self.d.cfg.models.answer, [{"role": "user", "content": prompt}],
                                    max_tokens=1200, temperature=0.7)
        return res.text.strip()

    async def _interviewer_turn(self) -> None:
        self._set_busy(True, "interviewer")
        if self.history and self._plan_task is not None:  # план пишется, пока кандидат представляется
            try:
                self.plan = await asyncio.wait_for(self._plan_task, timeout=20)
            except Exception:  # noqa: BLE001 — без плана интервьюер справится
                self.plan = ""
            self._plan_task = None
        plan = iv.PLAN_RULES[self.lang].format(plan=self.plan) if self.plan else ""
        system = iv.system_prompt(self.lang, name=iv.LANGS[self.lang]["name"], role=self.role, end=iv.END_MARK,
                                             context=self.context + plan)
        messages = [{"role": "system", "content": system}, *self.history]
        if not self.history:
            messages.append({"role": "user", "content": iv.LANGS[self.lang]["joined"]})
        res = await self.d.llm.chat(self.d.cfg.models.answer, messages, max_tokens=300, temperature=0.7)
        text = res.text.replace(iv.END_MARK, "").strip()
        self.history.append({"role": "assistant", "content": text})
        self._line("them", text)
        if iv.END_MARK in res.text:
            self._set_busy(False)
            await self.end()
            return
        await self._hint()

    # --- конец ---

    async def end(self) -> None:
        if self.ended:
            return
        self.ended = True
        self.copilot.cancel_all()
        self.d.db.update("calls", self.call_id, ended_at=time.time())
        self.bus.publish("chat.ended", {"call_id": self.call_id})
        from .postcall import process_call
        self.d._spawn(process_call(self.call_id, self.d.db, self.d.agents, self.d.search, self.d.bus))

    def transcript(self) -> list[dict[str, Any]]:
        """Реплики и подсказки по порядку — для страницы после перезагрузки."""
        rows = [{"kind": u["channel"], "text": u["text"], "at": u["started_at"]}
                for u in self.d.db.call_transcript(self.call_id)]
        rows += [{"kind": "hint", "text": a["text"], "at": a["created_at"]}
                 for a in self.d.db.all("SELECT text, created_at FROM answers WHERE call_id = ?", [self.call_id])]
        return sorted(rows, key=lambda r: r["at"])


class TextSessions:
    def __init__(self, d: "Daemon"):
        self.d = d
        self.items: dict[str, TextSession] = {}

    def start(self, mode: str, lang: str, role: str, vacancy: str) -> TextSession:
        s = TextSession(self.d, mode, lang, role, vacancy)
        self.items[s.id] = s
        if mode == "interviewer":
            s._spawn(s.start_interviewer())
        return s

    def get(self, sid: str) -> TextSession | None:
        return self.items.get(sid)

    async def close_all(self) -> None:
        for s in self.items.values():
            if not s.ended:
                s.ended = True
                s.copilot.cancel_all()
                self.d.db.update("calls", s.call_id, ended_at=time.time())
