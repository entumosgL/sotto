"""Конвейер переводчика: два независимых направления, каждое — упорядоченная очередь.

  кусок звука → VAD (фраза) → [распознавание стартует сразу] → перевод → очередь озвучки → устройство

Распознавание следующей фразы идёт параллельно с переводом и озвучкой предыдущей, но порядок фраз
сохраняется. Если озвучка не успевает (копится очередь), речь ускоряется — задержка не растёт бесконечно.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ...config import Config
from ...infrastructure.audio.speakers import Embedder, VoiceRegistry
from ...infrastructure.audio.vad import ChannelVAD, Segment
from ...infrastructure.llm.openrouter import OpenRouter
from ...infrastructure.platform.linux.pipewire import Stream
from . import devices
from .translate import Translator
from .voices import MyVoice, Player, TheirVoices

log = logging.getLogger(__name__)


@dataclass
class Phrase:
    direction: str          # "in" — собеседник → ты, "out" — ты → собеседник
    seg: Segment
    speaker: int = 0
    text: str = ""
    translation: str = ""
    t_stt: float = 0.0      # секунды от конца фразы
    t_translate: float = 0.0
    t_audio: float = 0.0
    timings: dict = field(default_factory=dict)


class TranslatorEngine:
    def __init__(self, cfg: Config, on_phrase: Callable[[Phrase], None]):
        self.cfg = cfg
        t = cfg.translator
        self.on_phrase = on_phrase
        self.llm = OpenRouter(cfg.api_key, cfg.openrouter.base_url, cfg.openrouter.provider_sort)
        self.translator = Translator(self.llm, t.translate_model, t.their_language, t.my_language)
        self.my_voice = MyVoice(cfg.api_key, cfg.openrouter.base_url, t.clone_model, t.fallback_my_voice)
        self.their_voices = TheirVoices(t.voices_male, t.voices_female)
        self.vad = {"in": ChannelVAD("in", t.min_silence_in, t.max_segment),
                    "out": ChannelVAD("out", t.min_silence_out, t.max_segment)}
        self.embedder: Embedder | None = None
        self.registry = VoiceRegistry(cfg.audio.speaker_threshold)
        self.translate_out = True     # False — твой настоящий голос идёт в звонок без перевода
        self.translate_in = True
        self.loop: asyncio.AbstractEventLoop | None = None
        self.queues: dict[str, asyncio.Queue] = {}
        self.speech: dict[str, asyncio.Queue] = {}
        self.players = {"in": Player(None), "out": Player(devices.MIC_FEED)}
        self._passthrough: subprocess.Popen | None = None
        self._streams = [
            Stream("in", devices.CALL_SINK, monitor=True, on_chunk=self._on_chunk),
            Stream("out", cfg.audio.mic, monitor=False, on_chunk=self._on_chunk),
        ]

    # --- звук (потоки захвата) ---

    def _on_chunk(self, channel: str, samples: np.ndarray, ts: float) -> None:
        if channel == "out" and not self.translate_out:
            self._pass_mic(samples)
            return
        if channel == "in" and not self.translate_in:
            return
        for seg in self.vad[channel].feed(samples, ts):
            if self.loop:
                self.loop.call_soon_threadsafe(self._enqueue, seg)

    def _pass_mic(self, samples: np.ndarray) -> None:
        """Перевод выключен: твой микрофон как есть идёт в виртуальный микрофон."""
        if self._passthrough is None or self._passthrough.poll() is not None:
            self._passthrough = subprocess.Popen(
                ["pw-play", "--raw", "--rate", "16000", "--channels", "1", "--format", "s16", "--target", devices.MIC_FEED, "-"],
                stdin=subprocess.PIPE)
        try:
            assert self._passthrough.stdin is not None
            self._passthrough.stdin.write((samples * 32767).astype(np.int16).tobytes())
        except BrokenPipeError:
            self._passthrough = None

    def stop_passthrough(self) -> None:
        if self._passthrough and self._passthrough.poll() is None:
            self._passthrough.stdin.close()  # type: ignore[union-attr]
            self._passthrough.terminate()
        self._passthrough = None

    # --- конвейер ---

    def _enqueue(self, seg: Segment) -> None:
        phrase = Phrase(direction=seg.channel, seg=seg)
        stt = asyncio.ensure_future(self._transcribe(phrase))
        self.queues[seg.channel].put_nowait((phrase, stt))

    async def _transcribe(self, p: Phrase) -> str:
        lang = self.cfg.translator.their_language if p.direction == "in" else self.cfg.translator.my_language
        mode = self.cfg.translator.stt
        from ...infrastructure.audio import local_stt
        if mode == "auto" and local_stt.has_cuda() or mode.startswith("local") or not self.llm.configured:
            text, _ = await asyncio.to_thread(local_stt.transcribe, p.seg.samples, "auto", lang)
        else:
            text, _lang, _cost = await self.llm.transcribe(self.cfg.models.stt, p.seg.samples, language=lang)
        p.t_stt = time.time() - p.seg.ended_at
        return text

    async def _translate_worker(self, direction: str) -> None:
        q = self.queues[direction]
        while True:
            p, stt = await q.get()
            try:
                p.text = (await stt).strip()
                if not p.text or len(p.text) < 2:
                    continue
                if direction == "in":
                    p.speaker = await self._speaker(p.seg)
                who = f"собеседник {p.speaker + 1}" if direction == "in" else "кандидат"
                p.translation, _ = await self.translator.translate(direction, p.text, who)
                p.t_translate = time.time() - p.seg.ended_at
                if p.translation:
                    self.speech[direction].put_nowait(p)
                else:
                    self.on_phrase(p)
            except Exception as e:  # noqa: BLE001 — одна фраза не должна ронять переводчик
                log.exception("Фраза не переведена")
                p.translation = f"(ошибка: {e})"
                self.on_phrase(p)

    async def _speech_worker(self, direction: str) -> None:
        q = self.speech[direction]
        player = self.players[direction]
        while True:
            p: Phrase = await q.get()
            # Очередь копится — говорим быстрее, чтобы догнать разговор.
            speed = 1.0 + min(0.25, 0.1 * q.qsize())
            try:
                if direction == "out":
                    first = await asyncio.to_thread(self._speak_mine, p.translation, player)
                else:
                    first = await asyncio.to_thread(self._speak_theirs, p, player, speed)
                p.t_audio = (time.time() - p.seg.ended_at) if first is None else p.t_translate + first
            except Exception as e:  # noqa: BLE001
                log.exception("Озвучка не удалась")
                p.translation += f"  (озвучка не удалась: {e})"
            self.on_phrase(p)

    def _speak_mine(self, text: str, player: Player) -> float:
        return self.my_voice.synthesize_to(text, player)

    def _speak_theirs(self, p: Phrase, player: Player, speed: float) -> None:
        voice = self.their_voices.for_speaker(p.speaker, p.seg.samples)
        self.their_voices.speak(p.translation, voice, player, speed=speed)

    async def _speaker(self, seg: Segment) -> int:
        if self.embedder is None:
            try:
                self.embedder = await asyncio.to_thread(Embedder)
            except Exception:  # noqa: BLE001
                return 0
        emb = await asyncio.to_thread(self.embedder.embed, seg.samples)
        sid, _ = self.registry.match(emb, seg.duration)
        if sid is None:
            sid = len(self.registry.voices)
            self.registry.add(sid, emb)
        return sid

    # --- запуск ---

    async def run(self) -> None:
        self.loop = asyncio.get_running_loop()
        tasks = []
        for d in ("in", "out"):
            self.queues[d] = asyncio.Queue()
            self.speech[d] = asyncio.Queue()
            tasks += [asyncio.create_task(self._translate_worker(d)), asyncio.create_task(self._speech_worker(d))]
        for s in self._streams:
            s.start()
        try:
            await asyncio.gather(*tasks)
        finally:
            for s in self._streams:
                s.stop()
            for p in self.players.values():
                p.stop()
            self.stop_passthrough()
            await self.llm.close()
