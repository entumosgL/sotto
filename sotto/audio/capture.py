"""Захват звука через PipeWire.

Два независимых потока:
* "me"   — микрофон;
* "them" — монитор системного выхода, т.е. всё, что ты слышишь (голос собеседника).

Раздельные каналы дают бесплатную и точную разметку «я / не я».
Используем `pw-record` как подпроцесс: никаких нативных биндингов, легко отлаживать.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from typing import Callable

import numpy as np

from .. import proc

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK = 512  # 32 мс — размер окна Silero VAD

# callback(канал, отсчёты float32, время начала куска по time.time())
ChunkCallback = Callable[[str, np.ndarray, float], None]


class Stream:
    """Один pw-record, читаемый в отдельном потоке. Перезапускается при падении."""

    def __init__(self, channel: str, target: str, monitor: bool, on_chunk: ChunkCallback):
        self.channel = channel
        self.target = target
        self.monitor = monitor
        self.on_chunk = on_chunk
        self._proc: subprocess.Popen | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"audio-{channel}", daemon=True)

    def command(self) -> list[str]:
        cmd = ["pw-record", "--raw", "--rate", str(SAMPLE_RATE), "--channels", "1", "--format", "s16"]
        if self.target:  # пусто — устройство по умолчанию (для монитора — выход по умолчанию)
            cmd += ["--target", self.target]
        cmd += ["-P", "{ node.name = sotto-" + self.channel + ", media.role = Production"
               + (", stream.capture.sink = true" if self.monitor else "") + " }",
               "-"]
        return cmd

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()

    def _run(self) -> None:
        bytes_per_chunk = CHUNK * 2
        while not self._stop.is_set():
            try:
                # Своя сессия: Ctrl+C в терминале не должен убивать захват мимо демона.
                self._proc = subprocess.Popen(self.command(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                              **proc.CHILD)
            except FileNotFoundError:
                log.error("pw-record не найден — нужен PipeWire")
                return
            assert self._proc.stdout is not None
            # Время по количеству прочитанных отсчётов: точнее, чем time.time() на каждый кусок.
            t0 = time.time()
            n = 0
            while not self._stop.is_set():
                raw = self._proc.stdout.read(bytes_per_chunk)
                if len(raw) < bytes_per_chunk:
                    break
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                ts = t0 + n / SAMPLE_RATE
                n += CHUNK
                # Если накопился дрейф (например, система спала) — пересинхронизируемся.
                if abs(time.time() - (ts + CHUNK / SAMPLE_RATE)) > 2.0:
                    t0, n = time.time() - CHUNK / SAMPLE_RATE, CHUNK
                    ts = t0
                try:
                    self.on_chunk(self.channel, samples, ts)
                except Exception:
                    log.exception("Ошибка обработки аудио (%s)", self.channel)
            if self._proc.poll() is None:
                self._proc.terminate()
            try:
                self._proc.wait(timeout=3)  # иначе остаётся процесс-зомби
            except subprocess.TimeoutExpired:
                self._proc.kill()
            if not self._stop.is_set():
                log.warning("pw-record (%s) завершился, перезапуск через 1 с", self.channel)
                time.sleep(1)


class AudioCapture:
    def __init__(self, mic: str, system: str, on_chunk: ChunkCallback):
        self.streams = [
            Stream("me", mic, monitor=False, on_chunk=on_chunk),
            Stream("them", system, monitor=True, on_chunk=on_chunk),
        ]

    def start(self) -> None:
        for s in self.streams:
            s.start()

    def stop(self) -> None:
        for s in self.streams:
            s.stop()
