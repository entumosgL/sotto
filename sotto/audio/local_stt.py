"""Локальное распознавание речи (faster-whisper) — для тренеров и офлайн-режима, без ключей."""

from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache

import numpy as np

log = logging.getLogger(__name__)
_lock = threading.Lock()


def has_cuda() -> bool:
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:  # noqa: BLE001
        return False


def resolve(name: str) -> str:
    """"auto": на видеокарте — точная large-v3-turbo, на процессоре — быстрая small."""
    if name != "auto":
        return name
    return "large-v3-turbo" if has_cuda() else "small"


@lru_cache(maxsize=2)
def _model(name: str):
    from faster_whisper import WhisperModel
    name = resolve(name)
    if not has_cuda():
        log.info("faster-whisper %s на CPU (для GPU: nix develop .#cuda / packages.cuda)", name)
        return WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)
    try:
        m = WhisperModel(name, device="cuda", compute_type="int8_float16")
        log.info("faster-whisper %s на GPU", name)
        return m
    except Exception as e:  # нет CUDA в сборке ctranslate2 — работаем на CPU
        log.info("faster-whisper %s на CPU (%s)", name, e)
        return WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=os.cpu_count() or 4)


def transcribe(samples: np.ndarray, model: str = "auto", language: str | None = None,
               hint: str | None = None, prompt: str | None = None) -> tuple[str, str | None]:
    """language — язык точно известен. hint — язык прошлой фразы: пробуем его без определения языка
    (экономит 0.2–0.5 с), а если распознавание вышло неуверенным — определяем язык заново.
    prompt — текст, который человек, скорее всего, произносит (подсказка суфлёра, которую он читает):
    Whisper с ним слышит термины и акцент правильно («bought»→«built», «Reptmq»→«RabbitMQ»)."""
    beam = 5 if prompt else 1
    with _lock:
        if language is None and hint:
            text, confidence = _run(model, samples, hint, prompt, beam)
            if text and confidence > -0.7:
                return text, _script_language(text, hint)
        segments, info = _model(model).transcribe(
            samples, language=language, beam_size=beam, vad_filter=False,
            condition_on_previous_text=False, initial_prompt=prompt)
        text = " ".join(s.text.strip() for s in segments).strip()
    return text, info.language


def _script_language(text: str, default: str) -> str:
    """Whisper с подсказкой «ru» английскую речь всё равно пишет латиницей — метку берём по алфавиту."""
    latin = sum(c.isascii() and c.isalpha() for c in text)
    cyr = sum("а" <= c.lower() <= "я" or c.lower() == "ё" for c in text)
    if latin > 2 * cyr:
        return "en"
    if cyr > 2 * latin:
        return "ru"
    return default


def _run(model: str, samples: np.ndarray, language: str, prompt: str | None = None,
         beam: int = 1) -> tuple[str, float]:
    segments, _info = _model(model).transcribe(
        samples, language=language, beam_size=beam, vad_filter=False, condition_on_previous_text=False,
        initial_prompt=prompt)
    segs = list(segments)
    if not segs:
        return "", -9.0
    return " ".join(s.text.strip() for s in segs).strip(), float(np.mean([s.avg_logprob for s in segs]))


def warm_up(model: str = "auto") -> None:
    _model(model)
