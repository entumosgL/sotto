"""Локальный синтез речи (Piper через sherpa-onnx) и воспроизведение через PipeWire."""

from __future__ import annotations

import os
import re
import subprocess
import threading
from functools import lru_cache

import numpy as np
import sherpa_onnx

from . import models
from .. import proc as proc_mod

CYRILLIC = re.compile(r"[а-яё]", re.I)


@lru_cache(maxsize=4)
def _engine(voice: str) -> sherpa_onnx.OfflineTts:
    d = models.piper_voice(voice)
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=str(d / f"{voice}.onnx"), tokens=str(d / "tokens.txt"),
                data_dir=str(d / "espeak-ng-data")),
            num_threads=4),
        max_num_sentences=2)
    return sherpa_onnx.OfflineTts(cfg)


def pick_voice(text: str, voice_ru: str, voice_en: str) -> str:
    return voice_ru if CYRILLIC.search(text) else voice_en


def synthesize(text: str, voice: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
    audio = _engine(voice).generate(text, sid=0, speed=speed)
    return np.asarray(audio.samples, dtype=np.float32), int(audio.sample_rate)


def start_playback(samples: np.ndarray, sample_rate: int, target: str | None = None) -> subprocess.Popen:
    """Запускает воспроизведение и сразу возвращает процесс (его можно оборвать terminate())."""
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes()
    cmd = ["pw-play", "--raw", "--rate", str(sample_rate), "--channels", "1", "--format", "s16"]
    target = os.environ.get("SOTTO_TTS_TARGET") or target  # тесты: весь синтез — в беззвучный выход
    if target:
        cmd += ["--target", target]
    proc = subprocess.Popen([*cmd, "-"], stdin=subprocess.PIPE, **proc_mod.CHILD)

    def feed() -> None:  # pw-play читает со скоростью воспроизведения — пишем в отдельном потоке
        assert proc.stdin is not None
        try:
            proc.stdin.write(pcm)
            proc.stdin.close()
        except (BrokenPipeError, ValueError):
            pass

    threading.Thread(target=feed, daemon=True).start()
    return proc


def play(samples: np.ndarray, sample_rate: int, target: str | None = None) -> None:
    """Проигрывает звук и ждёт окончания. target — имя выхода PipeWire (по умолчанию системный)."""
    start_playback(samples, sample_rate, target).wait()


def say(text: str, voice: str, target: str | None = None, speed: float = 1.0) -> float:
    """Синтезирует и проигрывает; возвращает длительность в секундах."""
    samples, sr = synthesize(text, voice, speed)
    play(samples, sr, target)
    return len(samples) / sr
