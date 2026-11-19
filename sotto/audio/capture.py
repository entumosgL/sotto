"""Захват звука: PipeWire на Linux, WASAPI на Windows.

Два независимых потока:
* "me"   — микрофон;
* "them" — системный выход, т.е. всё, что ты слышишь (голос собеседника).

Раздельные каналы дают бесплатную и точную разметку «я / не я».

Linux: `pw-record` подпроцессом — никаких нативных биндингов, легко отлаживать.
Windows: WASAPI через `soundcard` одним потоком на канал; loopback встроенного выхода
Windows отдаёт сам (аналог PipeWire-монитора), пересэмплирование в 16 кГц делает WASAPI.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from typing import Callable

import numpy as np

from .. import proc
from .select import LoudestPicker

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK = 512  # 32 мс — размер окна Silero VAD
WINDOWS = sys.platform == "win32"

# callback(канал, отсчёты float32, время начала куска по time.time())
ChunkCallback = Callable[[str, np.ndarray, float], None]


if WINDOWS:  # pragma: no cover - выбирается на Windows
    import soundcard as sc

    def devices() -> dict[str, list[dict[str, str]]]:
        """Устройства для панели настроек: микрофоны и выходы (для loopback)."""
        mics = [{"id": m.id, "name": m.name} for m in sc.all_microphones()]
        speakers = [{"id": s.id, "name": s.name} for s in sc.all_speakers()]
        return {"microphones": mics, "speakers": speakers}

    def _resolve_microphone(target: str):
        """target — id устройства или часть имени; пусто — устройство по умолчанию."""
        if target:
            return sc.get_microphone(id=target)
        mic = sc.default_microphone()
        if mic is None:
            raise RuntimeError("в системе нет микрофона")
        return mic

    def _resolve_loopback(target: str):
        """Loopback конкретного выхода: id устройства или часть имени."""
        return sc.get_microphone(id=target, include_loopback=True)

    class MultiLoopback:
        """Системный звук со всех устройств вывода сразу: отдаём блоки того, где громче.

        Приложение (Discord, Zoom, браузер) само выбирает, в какое устройство играть, и это
        не всегда устройство по умолчанию — из-за этого «собеседники» раньше оставались
        пустыми. Одно устройство в настройках `[audio].system` ограничивает запись только им.
        """

        def __init__(self, channel: str, on_chunk: ChunkCallback):
            self.channel = channel
            self.on_chunk = on_chunk
            self.picker = LoudestPicker()
            self._stop = threading.Event()
            self._devices: dict[str, threading.Thread] = {}
            self._supervisor = threading.Thread(target=self._run_supervisor, name="loopback-watch", daemon=True)
            self._emitter = threading.Thread(target=self._run_emitter, name=f"audio-{channel}", daemon=True)

        def start(self) -> None:
            self._emitter.start()
            self._supervisor.start()

        def stop(self) -> None:
            self._stop.set()

        def _run_supervisor(self) -> None:
            """Заводит по потоку на каждое устройство вывода; новые (наушники включили) — появятся сами."""
            while not self._stop.is_set():
                try:
                    for speaker in sc.all_speakers():
                        if speaker.id in self._devices or self._stop.is_set():
                            continue
                        log.info("Слушаю системный звук: %s", speaker.name)
                        thread = threading.Thread(target=self._run_device, args=(speaker,),
                                                  name=f"loopback-{speaker.id[-6:]}", daemon=True)
                        self._devices[speaker.id] = thread
                        thread.start()
                except Exception:  # noqa: BLE001 — перечисление устройств не должно ронять захват
                    log.exception("Не удалось перечислить устройства вывода")
                self._stop.wait(5.0)

        def _run_device(self, speaker) -> None:
            while not self._stop.is_set():
                try:
                    loop = sc.get_microphone(id=speaker.id, include_loopback=True)
                    if loop is None:
                        log.warning("У устройства «%s» нет loopback — пропускаю", speaker.name)
                        return
                    with loop.recorder(samplerate=SAMPLE_RATE, channels=1, blocksize=CHUNK) as rec:
                        while not self._stop.is_set():
                            block = np.asarray(rec.record(numframes=CHUNK), dtype=np.float32).reshape(-1)
                            rms = float(np.sqrt(np.mean(block ** 2))) if block.size else 0.0
                            self.picker.observe(speaker.id, rms, block, time.time())
                except Exception as e:  # noqa: BLE001
                    if self._stop.is_set():
                        return
                    log.warning("Loopback «%s»: %s — переоткрываю", speaker.name, e)
                    self.picker.forget(speaker.id)
                    self._stop.wait(2)

        def _run_emitter(self) -> None:
            """Отдаёт куски ровно по 32 мс — как pw-record, чтобы VAD и тайминги не поехали."""
            t0 = time.time()
            n = 0
            silence = np.zeros(CHUNK, dtype=np.float32)
            while not self._stop.is_set():
                scheduled = t0 + n / SAMPLE_RATE
                delay = scheduled - time.time()
                if delay > 0:
                    time.sleep(min(delay, 0.05))
                picked = self.picker.pick(time.time())
                block = picked[1] if picked and picked[1].size == CHUNK else silence
                if abs(time.time() - (scheduled + CHUNK / SAMPLE_RATE)) > 2.0:  # система спала
                    t0, n = time.time() - CHUNK / SAMPLE_RATE, CHUNK
                    scheduled = t0
                try:
                    self.on_chunk(self.channel, block, scheduled)
                except Exception:  # noqa: BLE001
                    log.exception("Ошибка обработки аудио (%s)", self.channel)
                n += CHUNK

    def system_stream(target: str, on_chunk: ChunkCallback) -> "Stream | MultiLoopback":
        """Системный звук: со всех выходов (по умолчанию) или с одного выбранного."""
        if target:
            return Stream("them", target, monitor=True, on_chunk=on_chunk)
        return MultiLoopback("them", on_chunk)

    class Stream:
        """Один поток WASAPI, читаемый кусками по 32 мс. Переоткрывается при ошибке.

        `blocksize=CHUNK` просим у WASAPI, а время считаем по числу прочитанных отсчётов —
        AudioCapture ждёт от нас ту же частоту 16 кГц и те же 512 отсчётов, что и pw-record.
        """

        def __init__(self, channel: str, target: str, monitor: bool, on_chunk: ChunkCallback):
            self.channel = channel
            self.target = target
            self.monitor = monitor
            self.on_chunk = on_chunk
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._run, name=f"audio-{channel}", daemon=True)

        def start(self) -> None:
            self._thread.start()

        def stop(self) -> None:
            self._stop.set()

        def _open(self):
            return _resolve_loopback(self.target) if self.monitor else _resolve_microphone(self.target)

        def _run(self) -> None:
            while not self._stop.is_set():
                try:
                    device = self._open()
                except Exception as e:  # noqa: BLE001 — устройство могли выдернуть
                    log.error("Звук (%s): устройство не открылось: %s", self.channel, e)
                    self._stop.wait(3)
                    continue
                t0 = time.time()
                n = 0
                try:
                    with device.recorder(samplerate=SAMPLE_RATE, channels=1, blocksize=CHUNK) as rec:
                        while not self._stop.is_set():
                            data = rec.record(numframes=CHUNK)
                            samples = np.asarray(data, dtype=np.float32).reshape(-1)
                            if samples.size == 0:
                                break
                            ts = t0 + n / SAMPLE_RATE
                            n += samples.size
                            # Если накопился дрейф (система спала, звук перезапускался) — пересинхронизируемся.
                            if abs(time.time() - (ts + samples.size / SAMPLE_RATE)) > 2.0:
                                t0, n = time.time() - CHUNK / SAMPLE_RATE, CHUNK
                                ts = t0
                            try:
                                self.on_chunk(self.channel, samples, ts)
                            except Exception:  # noqa: BLE001
                                log.exception("Ошибка обработки аудио (%s)", self.channel)
                except Exception as e:  # noqa: BLE001
                    if not self._stop.is_set():
                        log.warning("Звук (%s): поток прервался: %s — переоткрываю", self.channel, e)
                        self._stop.wait(1)

else:  # pragma: no cover - выбирается на Linux

    def devices() -> dict[str, list[dict[str, str]]]:
        """Устройства PipeWire для панели настроек."""
        import json
        try:
            out = subprocess.run(["pw-dump"], capture_output=True, text=True, timeout=5).stdout
            objects = json.loads(out or "[]")
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            return {"microphones": [], "speakers": []}
        mics, speakers = [], []
        for o in objects:
            props = ((o.get("info") or {}).get("props") or {})
            media_class = props.get("media.class")
            node = str(props.get("node.name") or "")
            if media_class == "Audio/Source" and node:
                mics.append({"id": node, "name": str(props.get("node.description") or node)})
            elif media_class == "Audio/Sink" and node:
                speakers.append({"id": node, "name": str(props.get("node.description") or node)})
        return {"microphones": mics, "speakers": speakers}

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
                    self._proc = subprocess.Popen(self.command(), stdout=subprocess.PIPE,
                                                  stderr=subprocess.DEVNULL, **proc.CHILD)
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
                    except Exception:  # noqa: BLE001
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

    def system_stream(target: str, on_chunk: ChunkCallback) -> "Stream":
        """Системный звук: монитор выхода PipeWire (в Linux он всегда один — выход по умолчанию)."""
        return Stream("them", target, monitor=True, on_chunk=on_chunk)


class AudioCapture:
    def __init__(self, mic: str, system: str, on_chunk: ChunkCallback):
        self.streams = [
            Stream("me", mic, monitor=False, on_chunk=on_chunk),
            system_stream(system, on_chunk),
        ]

    def start(self) -> None:
        for s in self.streams:
            s.start()

    def stop(self) -> None:
        for s in self.streams:
            s.stop()
