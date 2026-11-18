"""От звука к репликам: VAD → кто говорит → распознавание → база → события.

Работает только во время звонка (вне звонка звук лишь пишется в архив, и мы не тратим деньги).
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import re
import time
from typing import Callable

import numpy as np

from ..audio.speakers import Embedder, VoiceRegistry
from ..audio.vad import ChannelVAD, Segment
from ..config import Config
from ..db import DB
from ..events import EventBus
from ..openrouter import OpenRouter, OpenRouterError
from . import prompts

log = logging.getLogger(__name__)

# Признаки самопредставления — чтобы не гонять модель на каждой реплике.
INTRO = re.compile(
    r"(меня зовут|зовут меня|я\s+[—–-]\s*[А-ЯЁ]|это\s+[А-ЯЁ][а-яё]+[,.]|представлюсь|с вами\s+[А-ЯЁ]|"
    r"my name is|i'?m\s+[A-Z][a-z]+|this is\s+[A-Z][a-z]+|i am\s+[A-Z][a-z]+)",
    re.I,
)

# Без ключа имя достаём регуляркой: «меня зовут Анна», «я — Дмитрий», «I'm Sarah».
NAME_RX = re.compile(
    r"(?:меня зовут|зовут меня|я\s*[—–-]|^(?:\w+[,!.]\s*)?я|my name is|i'?m|i am|this is)\s+([A-ZА-ЯЁ][a-zа-яё]{1,20})",
    re.I | re.M)
NOT_NAMES = {"рад", "рада", "тоже", "здесь", "очень", "буду", "hr", "the", "here", "glad", "happy", "going", "from"}


def name_from_intro(text: str) -> str | None:
    for m in NAME_RX.finditer(text):
        name = m.group(1)
        if name.lower() not in NOT_NAMES and name[0].isupper():
            return name
    return None


# Whisper иногда «слышит» в тишине типовые фразы из субтитров — выбрасываем.
HALLUCINATIONS = {
    "продолжение следует", "субтитры сделал dimatorzok", "спасибо за просмотр", "редактор субтитров",
    "thank you for watching", "thanks for watching", "you", "субтитры создавал dimatorzok",
}


_TAIL_JUNK = re.compile(
    r"\s*(продолжение следует|спасибо за просмотр|субтитры (сделал|создавал) \w+|редактор субтитров[^.]*|"
    r"thanks? (you )?for watching)[.!…]*\s*$", re.I)


def strip_hallucinations(text: str) -> str:
    """Whisper дописывает «Продолжение следует...» в конец настоящей фразы — срезаем хвост."""
    prev = None
    while prev != text:
        prev, text = text, _TAIL_JUNK.sub("", text).strip()
    return text


def is_hallucination(text: str) -> bool:
    t = re.sub(r"[^\w\s]", "", text.lower()).strip()
    # «足を» и прочие обрывки чужой письменности — Whisper галлюцинирует их на шуме
    if not re.search(r"[a-zа-яё]", t):
        return True
    return not t or t in HALLUCINATIONS or any(h in t for h in ("dimatorzok", "amara.org"))


EARLY_SILENCE = 0.2


class Listener:
    def __init__(self, cfg: Config, db: DB, bus: EventBus, llm: OpenRouter,
                 on_utterance: Callable[[str], None]):
        self.cfg = cfg
        self.db = db
        self.bus = bus
        self.llm = llm
        self.on_utterance = on_utterance
        self.loop: asyncio.AbstractEventLoop | None = None
        self.vad = {"me": ChannelVAD("me", cfg.audio.min_silence, cfg.audio.max_utterance),
                    # у собеседника — упреждение: подсказка стартует через 0.2 с паузы, а не 0.5 с + распознавание
                    "them": ChannelVAD("them", cfg.audio.min_silence, cfg.audio.max_utterance, early_silence=EARLY_SILENCE)}
        # Упреждающий старт подсказки (см. Copilot.speculate): текст предварительной реплики и «речь продолжилась».
        self.on_early: Callable[[str], None] | None = None
        self.on_resume: Callable[[], None] | None = None
        self._early_task: asyncio.Task | None = None
        self.embedder: Embedder | None = None
        self.call_id: int | None = None
        self.voices = VoiceRegistry(cfg.audio.speaker_threshold)
        self.me_speaker: int | None = None
        self._level = {"me": -90.0, "them": -90.0}
        self._last_level_pub = 0.0
        self.reading_hint: Callable[[], str] | None = None  # подсказка, которую кандидат сейчас читает
        self._me_last_voice = 0.0  # когда кандидат последний раз говорил
        self._last_lang: dict[str, str] = {}  # язык прошлой фразы канала — подсказка для Whisper
        # Реплики распознаются параллельно (ради скорости), но «кто говорит» и запись в базу идут
        # строго по очереди — иначе гонки: два «Собеседника» на одного человека, перепутанный порядок.
        self._queue: asyncio.Queue | None = None
        self._tasks: set[asyncio.Task] = set()

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop
        self._queue = asyncio.Queue()
        self._keep(loop.create_task(self._worker()))

    def _keep(self, task: asyncio.Task) -> None:
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def them_speaking(self) -> bool:
        return self.vad["them"].speaking

    def me_speaking(self) -> bool:
        """Кандидат говорит сейчас или замолчал меньше 1.5 с назад (пауза посреди ответа)."""
        return self.vad["me"].speaking or time.time() - self._me_last_voice < 1.5

    # --- звонок ---

    def start_call(self, call_id: int) -> None:
        self.call_id = call_id
        self.voices = VoiceRegistry(self.cfg.audio.speaker_threshold)
        self.me_speaker = self.db.insert("speakers", call_id=call_id, label="Я", is_me=1)
        for v in self.vad.values():
            v.request_reset()  # сбросит поток захвата сам — без гонки с feed()
        if self.embedder is None:
            try:
                self.embedder = Embedder()
            except Exception:
                log.exception("Не удалось загрузить модель голосовых отпечатков — все собеседники будут одним")

    def end_call(self) -> None:
        self.call_id = None

    # --- поток аудио (вызывается из потоков захвата) ---

    def on_chunk(self, channel: str, samples: np.ndarray, ts: float) -> None:
        level = float(20 * np.log10(np.sqrt(np.mean(samples ** 2)) + 1e-9))
        self._level[channel] = max(level, self._level[channel] * 0.8 + level * 0.2)
        if time.time() - self._last_level_pub > 0.25:
            self._last_level_pub = time.time()
            self.bus.publish("audio.level", {"me": round(self._level["me"], 1), "them": round(self._level["them"], 1),
                                             "me_speaking": self.vad["me"].speaking,
                                             "them_speaking": self.vad["them"].speaking})
            self._level = {"me": -90.0, "them": -90.0}
        call_id = self.call_id
        if call_id is None or self.loop is None:
            return
        for seg in self.vad[channel].feed(samples, ts):
            self.loop.call_soon_threadsafe(self._enqueue, seg, call_id)
        if channel == "me" and self.vad["me"].speaking:
            self._me_last_voice = time.time()
        for kind, seg in self.vad[channel].take_events():
            if kind == "early":
                self.loop.call_soon_threadsafe(self._early, seg)
            else:
                self.loop.call_soon_threadsafe(self._resume)

    # --- обработка реплики ---

    def _early(self, seg: Segment) -> None:
        """Собеседник замолчал на 0.2 с: распознаём сказанное заранее — вдруг это конец вопроса.
        Только с локальным Whisper: облачное распознавание медленное и платное, выигрыша не будет."""
        if self.on_early is None or not self.cfg.answer.early or not self.use_local_stt():
            return
        if self._early_task and not self._early_task.done():
            self._early_task.cancel()
        self._early_task = asyncio.ensure_future(self._early_run(seg))
        self._keep(self._early_task)

    async def _early_run(self, seg: Segment) -> None:
        log.debug("Пауза собеседника: распознаю заранее %.1f с речи", seg.duration)
        text, _lang, _cost = await self._transcribe(seg.samples, seg.channel)
        text = strip_hallucinations(text)
        if text and not is_hallucination(text) and self.on_early:
            self.on_early(text)

    def _resume(self) -> None:
        if self._early_task and not self._early_task.done():
            self._early_task.cancel()
        if self.on_resume:
            self.on_resume()

    def _enqueue(self, seg: Segment, call_id: int) -> None:
        assert self._queue is not None
        stt = asyncio.ensure_future(self._transcribe(seg.samples, seg.channel))  # распознавание стартует сразу
        self._queue.put_nowait((seg, call_id, stt))

    async def _worker(self) -> None:
        assert self._queue is not None
        while True:
            seg, call_id, stt = await self._queue.get()
            try:
                await self._process(seg, call_id, stt)
            except Exception as e:  # noqa: BLE001 — реплика не должна теряться молча
                log.exception("Реплика не обработана")
                self.bus.publish("notice", {"text": f"Распознавание речи: {e}"[:200]})

    async def _process(self, seg: Segment, call_id: int, stt: asyncio.Future) -> None:
        speaker_id = self.me_speaker if seg.channel == "me" else await self._identify(seg, call_id)
        text, lang, cost = await stt
        text = strip_hallucinations(text)
        if is_hallucination(text):
            return
        if seg.channel == "me" and self._is_echo(text, call_id, seg.started_at):
            return  # микрофон поймал звук из колонок
        utt_id = self.db.insert("utterances", call_id=call_id, speaker_id=speaker_id, channel=seg.channel,
                                started_at=seg.started_at, ended_at=seg.ended_at, text=text, language=lang,
                                cost=cost)
        speaker = self.db.one("SELECT COALESCE(name, label) AS name, role FROM speakers WHERE id = ?", [speaker_id]) or {}
        self.bus.publish("utterance", {"id": utt_id, "call_id": call_id, "channel": seg.channel,
                                       "speaker_id": speaker_id, "speaker": speaker.get("name"),
                                       "text": text, "started_at": seg.started_at, "language": lang})
        if seg.channel == "them" and INTRO.search(text):
            self._keep(asyncio.create_task(self._maybe_name(speaker_id, text)))
        self.on_utterance(seg.channel)

    async def _transcribe(self, samples: np.ndarray, channel: str = "them") -> tuple[str, str | None, float | None]:
        """Локальный Whisper на видеокарте или OpenRouter — см. [audio].stt."""
        model = self.cfg.models.stt
        if self.use_local_stt():
            from ..audio import local_stt
            name = model.removeprefix("local:") if model.startswith("local:") else self.cfg.trainer.stt_model
            prompt = self.reading_hint() if channel == "me" and self.reading_hint else None
            text, lang = await asyncio.to_thread(local_stt.transcribe, samples, name, None,
                                                 self._last_lang.get(channel), prompt or None)
            if text and lang:
                self._last_lang[channel] = lang
            return text, lang, 0.0
        return await self.llm.transcribe(model, samples)

    def use_local_stt(self) -> bool:
        mode = self.cfg.audio.stt
        if mode == "local" or self.cfg.models.stt.startswith("local:") or not self.llm.configured:
            return True
        if mode == "auto":
            from ..audio import local_stt
            return local_stt.has_cuda()
        return False

    async def _identify(self, seg: Segment, call_id: int) -> int:
        if self.embedder is None:
            return self._new_speaker(call_id, None)
        emb = await asyncio.to_thread(self.embedder.embed, seg.samples)
        speaker_id, _sim = self.voices.match(emb, seg.duration)
        if speaker_id is None:
            speaker_id = self._new_speaker(call_id, emb)
            self.voices.add(speaker_id, emb)
        return speaker_id

    def _new_speaker(self, call_id: int, emb: np.ndarray | None) -> int:
        n = self.db.one("SELECT COUNT(*) AS n FROM speakers WHERE call_id = ? AND is_me = 0", [call_id])["n"]
        if emb is None and n:
            return self.db.one("SELECT id FROM speakers WHERE call_id = ? AND is_me = 0", [call_id])["id"]
        sid = self.db.insert("speakers", call_id=call_id, label=f"Собеседник {n + 1}",
                             embedding=emb.tobytes() if emb is not None else None)
        self.bus.publish("speaker", {"id": sid, "call_id": call_id, "label": f"Собеседник {n + 1}"})
        return sid

    def _is_echo(self, text: str, call_id: int, started_at: float) -> bool:
        recent = self.db.all("SELECT text FROM utterances WHERE call_id = ? AND channel = 'them' "
                             "AND ended_at > ? ORDER BY id DESC LIMIT 3", [call_id, started_at - 15])
        return any(difflib.SequenceMatcher(None, text.lower(), r["text"].lower()).ratio() > 0.6 for r in recent)

    async def _maybe_name(self, speaker_id: int, text: str) -> None:
        row = self.db.one("SELECT name FROM speakers WHERE id = ?", [speaker_id])
        if not row or row["name"]:
            return  # уже названы (в том числе вручную) — не трогаем
        if not self.llm.configured:
            name = name_from_intro(text)
            if name:
                self.db.update("speakers", speaker_id, name=name)
                self.bus.publish("speaker", {"id": speaker_id, "name": name, "role": None})
            return
        try:
            res = await self.llm.chat(self.cfg.models.utility, [
                {"role": "system", "content": prompts.load("names")},
                {"role": "user", "content": text}], max_tokens=60, json_mode=True)
            data = json.loads(res.text[res.text.find("{"):res.text.rfind("}") + 1])
        except (OpenRouterError, ValueError):
            return
        if data.get("name"):
            self.db.update("speakers", speaker_id, name=data["name"], role=data.get("role"))
            self.bus.publish("speaker", {"id": speaker_id, "name": data["name"], "role": data.get("role")})
