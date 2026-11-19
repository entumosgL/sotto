"""Локальный синтез речи (Piper через sherpa-onnx) и воспроизведение.

Linux — `pw-play` (PipeWire), Windows — WASAPI через `soundcard`.
Наружу отдаётся «процесс-подобный» объект с poll()/wait()/terminate(): тренеру нужно
оборвать речь на полуслове, когда тренировку останавливают.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import threading
from functools import lru_cache
from typing import Any

import numpy as np
import sherpa_onnx

from . import models
from .. import proc as proc_mod

log = logging.getLogger(__name__)

CYRILLIC = re.compile(r"[а-яё]", re.I)
WINDOWS = sys.platform == "win32"


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
    samples = np.asarray(audio.samples, dtype=np.float32)
    if samples.size == 0:
        # sherpa-onnx 2.x убрал espeak-фронтенд: Piper-голоса без готовых фонем молчат.
        raise RuntimeError(
            "Озвучка не получилась: установленная версия sherpa-onnx не умеет голоса Piper. "
            "Поставь sherpa-onnx<2:  pip install 'sherpa-onnx<2'")
    return samples, int(audio.sample_rate)


def _target() -> str | None:
    return os.environ.get("SOTTO_TTS_TARGET") or None


if WINDOWS:  # pragma: no cover - выбирается на Windows
    import soundcard as sc

    class Playback:
        """Воспроизведение в отдельном потоке: куски по 50 мс, terminate() обрывает сразу."""

        def __init__(self, samples: np.ndarray, sample_rate: int, target: str | None = None):
            self.samples = np.ascontiguousarray(samples, dtype=np.float32)
            self.sample_rate = int(sample_rate)
            self.target = target or _target()
            self.returncode: int | None = None
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._run, name="tts-play", daemon=True)
            self._thread.start()

        def _speaker(self):
            if self.target:
                try:
                    return sc.get_speaker(id=self.target)
                except Exception:  # noqa: BLE001 — нет такого выхода, играем в общий
                    log.debug("Нет устройства вывода «%s», играю в устройство по умолчанию", self.target)
            return sc.default_speaker()

        def _run(self) -> None:
            block = max(256, int(self.sample_rate * 0.05))
            try:
                speaker = self._speaker()
                if speaker is None:
                    log.error("Нет устройства воспроизведения — речь тренера не слышна")
                    return
                with speaker.player(samplerate=self.sample_rate, channels=1, blocksize=block) as player:
                    for start in range(0, len(self.samples), block):
                        if self._stop.is_set():
                            break
                        player.play(self.samples[start:start + block])
            except Exception:  # noqa: BLE001
                log.exception("Ошибка воспроизведения речи")
            finally:
                self.returncode = 0

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int | None:
            self._thread.join(timeout)
            return self.returncode

        def terminate(self) -> None:
            self._stop.set()
            self._thread.join(timeout=2)

        def send_signal(self, _sig: int) -> None:  # совместимость с Popen.terminate()
            self.terminate()

        def kill(self) -> None:
            self.terminate()

else:  # pragma: no cover - выбирается на Linux

    def _pw_play(samples: np.ndarray, sample_rate: int, target: str | None = None) -> subprocess.Popen:
        """pw-play подпроцессом: возвращает процесс, который можно оборвать terminate()."""
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


def start_playback(samples: np.ndarray, sample_rate: int, target: str | None = None) -> Any:
    """Начать воспроизведение; вернуть объект с poll()/wait()/terminate()."""
    if WINDOWS:
        return Playback(samples, sample_rate, target)
    return _pw_play(samples, sample_rate, target)


def play(samples: np.ndarray, sample_rate: int, target: str | None = None) -> None:
    """Проигрывает звук и ждёт окончания. target — имя выхода (по умолчанию системный)."""
    start_playback(samples, sample_rate, target).wait()


def say(text: str, voice: str, target: str | None = None, speed: float = 1.0) -> float:
    """Синтезирует и проигрывает; возвращает длительность в секундах."""
    samples, sr = synthesize(text, voice, speed)
    play(samples, sr, target)
    return len(samples) / sr
