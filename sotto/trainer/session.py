"""Голосовая тренировка: тренер говорит (локальный синтез), ты отвечаешь голосом (локальное
распознавание), ход разговора и разбор ведёт CLI-агент. Текст — только как обратная связь.

Два тренера:
  knowledge — собеседование по знаниям (mode=interview) или разбор темы (mode=lesson);
  english   — разговорная практика / собеседование на английском с разбором ошибок.

События: trainer.state {active, kind, topic, phase}, trainer.turn {role, text, feedback}.
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from ..agents import AgentError
from ..audio import local_stt, tts
from ..audio.vad import ChannelVAD
from ..copilot import prompts

if TYPE_CHECKING:
    from ..daemon import Daemon

log = logging.getLogger(__name__)

STOP_WORDS = re.compile(r"^\W*(стоп|хватит|закончим|заканчиваем|stop|that'?s all|let'?s stop|finish)\W*$", re.I)



def next_due(score: float) -> float:
    """Интервальное повторение: чем лучше владеешь темой, тем реже она возвращается."""
    days = 0.5 * math.pow(2, score * 5)  # от 12 часов до ~16 дней
    return time.time() + days * 86400


class TrainerSession:
    def __init__(self, d: "Daemon", kind: str, topic: str, mode: str = "interview"):
        self.d = d
        self.kind = kind
        self.mode = mode
        self.topic = topic
        self.session_id: int | None = None
        self.history: list[dict[str, str]] = []
        self.listening = False
        self.speaking = False
        self.stopped = False
        self._segments: asyncio.Queue = asyncio.Queue()
        self._vad = ChannelVAD("me", min_silence=d.cfg.trainer.end_of_turn_silence, max_speech=90)
        self.phase = "starting"
        self._speech = None  # процесс pw-play, пока тренер говорит — чтобы stop() мог его оборвать

    # --- звук из демона (поток захвата) ---

    def on_mic(self, samples: np.ndarray, ts: float) -> None:
        if not self.listening or self.speaking or self.d.loop is None:
            return
        for seg in self._vad.feed(samples, ts):
            self.d.loop.call_soon_threadsafe(self._segments.put_nowait, seg)

    # --- жизненный цикл ---

    def state(self) -> dict[str, Any]:
        return {"active": not self.stopped, "kind": self.kind, "mode": self.mode, "topic": self.topic,
                "phase": self.phase, "session_id": self.session_id}

    def _publish_state(self, phase: str) -> None:
        self.phase = phase
        self.d.bus.publish("trainer.state", self.state())

    def stop(self) -> None:
        """Можно вызывать откуда угодно в потоке event loop: прерывает ожидание ответа и озвучку."""
        self.stopped = True
        self.listening = False
        self._segments.put_nowait(None)
        if self._speech is not None:
            self._speech.terminate()

    async def run(self) -> None:
        self.session_id = self.d.db.insert("trainer_sessions", kind=self.kind, topic=self.topic, started_at=time.time())
        user_text: str | None = None
        try:
            while not self.stopped:
                self._publish_state("thinking")
                reply = await self._coach_turn(user_text)
                if reply is None or self.stopped:  # пока агент думал, тренировку могли остановить (звонок!)
                    break
                await self._speak(reply.get("say") or "")
                if reply.get("done") or self.stopped:
                    break
                self._publish_state("listening")
                user_text = await self._listen()
                if user_text is None or self.stopped:
                    break
                if STOP_WORDS.match(user_text):
                    self.history.append({"role": "user", "text": user_text})
                    user_text = "(пользователь просит закончить)"
        finally:
            await self._finish()

    # --- шаги ---

    async def _coach_turn(self, user_text: str | None) -> dict[str, Any] | None:
        persona = prompts.render(prompts.load(f"trainer_{self.kind}_{self.mode}"), topic=self.topic)
        dialog = "\n".join(f"{'Тренер' if t['role'] == 'coach' else 'Пользователь'}: {t['text']}" for t in self.history)
        prompt = (f"{persona}\n\nПрофиль пользователя:\n{prompts.profile() or '(не заполнен)'}\n\n"
                  f"Разговор до сих пор:\n{dialog or '(ещё не начинали — начни: поприветствуй коротко и задай первый вопрос)'}\n"
                  f"\n{prompts.load('trainer_turn')}")
        try:
            reply, _agent = await self.d.agents.run_json(prompt, task=f"тренер {self.kind}")
        except AgentError as e:
            self.d.bus.publish("notice", {"text": f"Тренер: {e}"[:200]})
            return None
        if not isinstance(reply, dict):
            return None
        if user_text is not None and self.history:
            # Разбор относится к последнему ответу пользователя.
            last = self.history[-1]
            last["feedback"] = reply.get("feedback") or ""
            self.d.db.execute("UPDATE trainer_turns SET feedback = ? WHERE id = (SELECT MAX(id) FROM trainer_turns "
                              "WHERE session_id = ? AND role = 'user')", [last["feedback"], self.session_id])
            self.d.bus.publish("trainer.feedback", {"session_id": self.session_id, "feedback": last["feedback"],
                                                    "mastery": reply.get("mastery")})
            self._update_items(reply)
        say = (reply.get("say") or "").strip()
        if say:
            self.history.append({"role": "coach", "text": say})
            self.d.db.insert("trainer_turns", session_id=self.session_id, role="coach", text=say, created_at=time.time())
            self.d.bus.publish("trainer.turn", {"session_id": self.session_id, "role": "coach", "text": say})
        return reply

    async def _speak(self, text: str) -> None:
        if not text:
            return
        self._publish_state("speaking")
        self.speaking = True
        voice = self.d.cfg.trainer.voice_en if self.kind == "english" else tts.pick_voice(
            text, self.d.cfg.trainer.voice_ru, self.d.cfg.trainer.voice_en)
        try:
            samples, sr = await asyncio.to_thread(tts.synthesize, text, voice)
            if self.stopped:
                return
            self._speech = tts.start_playback(samples, sr)
            await asyncio.to_thread(self._speech.wait)
        finally:
            self._speech = None
            await asyncio.sleep(0.3)  # хвост эха из колонок
            self._vad.request_reset()
            self.speaking = False

    async def _listen(self) -> str | None:
        # Старые куски (эхо тренера и т.п.) выбрасываем, но сигнал остановки (None) сохраняем.
        while not self._segments.empty():
            if self._segments.get_nowait() is None:
                return None
        if self.stopped:
            return None
        self.listening = True
        parts: list[str] = []
        try:
            while True:
                seg = await self._segments.get()
                if seg is None:
                    return None
                self._publish_state("recognizing")
                lang = "en" if self.kind == "english" else None
                text, _ = await asyncio.to_thread(local_stt.transcribe, seg.samples, self.d.cfg.trainer.stt_model, lang)
                if text.strip():
                    parts.append(text.strip())
                # Длинный ответ мог разбиться паузой: если ты продолжил говорить — ждём ещё кусок.
                await asyncio.sleep(0.2)
                if self._segments.empty() and not self._vad.speaking:
                    break
                self._publish_state("listening")
        finally:
            self.listening = False
        text = " ".join(parts).strip()
        if not text:
            return None if self.stopped else await self._listen()
        self.history.append({"role": "user", "text": text})
        self.d.db.insert("trainer_turns", session_id=self.session_id, role="user", text=text, created_at=time.time())
        self.d.bus.publish("trainer.turn", {"session_id": self.session_id, "role": "user", "text": text})
        return text

    def _update_items(self, reply: dict[str, Any]) -> None:
        db = self.d.db
        mastery = reply.get("mastery")
        if isinstance(mastery, (int, float)):
            db.execute("INSERT OR IGNORE INTO trainer_items(kind, item, source) VALUES(?, ?, 'session')",
                       [self.kind, self.topic])
            row = db.one("SELECT id, score, seen FROM trainer_items WHERE kind = ? AND item = ?", [self.kind, self.topic])
            score = row["score"] * 0.6 + float(mastery) * 0.4 if row["seen"] else float(mastery)
            db.update("trainer_items", row["id"], score=score, seen=row["seen"] + 1, due_at=next_due(score))
        for m in reply.get("mistakes") or []:
            item = str(m).strip()[:200]
            if item:
                db.execute("INSERT OR IGNORE INTO trainer_items(kind, item, source, due_at) VALUES(?, ?, 'session', ?)",
                           [self.kind, item, time.time()])

    async def _finish(self) -> None:
        self.stopped = True
        self.listening = False
        if self.session_id is not None:
            notes = "\n\n".join(f"**{t['text']}**\n{t.get('feedback', '')}" for t in self.history
                                if t["role"] == "user" and t.get("feedback"))
            self.d.db.update("trainer_sessions", self.session_id, ended_at=time.time(), feedback=notes or None)
        self._publish_state("finished")
