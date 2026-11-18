"""Нарезка потока на реплики детектором речи Silero (через sherpa-onnx)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import sherpa_onnx

from . import CHUNK, SAMPLE_RATE, models


@dataclass
class Segment:
    channel: str
    samples: np.ndarray
    started_at: float
    ended_at: float

    @property
    def duration(self) -> float:
        return len(self.samples) / SAMPLE_RATE


PRE_ROLL = 0.3  # с звука перед началом речи: детектор срабатывает с опозданием и съедает первый слог


class ChannelVAD:
    """VAD одного канала. feed() возвращает законченные реплики."""

    def __init__(self, channel: str, min_silence: float = 0.5, max_speech: float = 25.0,
                 threshold: float = 0.5, early_silence: float | None = None):
        """early_silence — упреждение: после такой короткой паузы (меньше min_silence) отдаём
        предварительную реплику, чтобы распознавание и подсказка стартовали, не дожидаясь конца фразы.
        Если речь продолжилась — событие «resume», предварительную реплику надо забыть."""
        self.channel = channel
        self.early_silence = early_silence
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = str(models.file("silero_vad.onnx"))
        cfg.silero_vad.min_silence_duration = min_silence
        cfg.silero_vad.min_speech_duration = 0.25
        cfg.silero_vad.max_speech_duration = max_speech
        cfg.silero_vad.threshold = threshold
        cfg.sample_rate = SAMPLE_RATE
        assert cfg.silero_vad.window_size == CHUNK
        self._cfg = cfg
        self._vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=max_speech + 5)
        self._base_ts: float | None = None   # время отсчёта №0 внутри VAD
        self._fed = 0                         # сколько отсчётов уже скормлено
        self._history: deque[tuple[int, np.ndarray]] = deque()  # (номер первого отсчёта, кусок)
        self._reset_requested = False
        self._fast: sherpa_onnx.VoiceActivityDetector | None = None
        self._speech_from: int | None = None  # с какого отсчёта идёт текущая фраза (с запасом)
        self._early_sent = False               # предварительная реплика отдана, ждём конца или продолжения
        self._was_speaking = False
        self.events: list[tuple[str, Segment | None]] = []  # ("early", сегмент) / ("resume", None)
        if early_silence:
            self._fast = self._make_fast()

    def _make_fast(self) -> sherpa_onnx.VoiceActivityDetector:
        fast = sherpa_onnx.VadModelConfig()
        fast.silero_vad.model = self._cfg.silero_vad.model
        fast.silero_vad.min_silence_duration = self.early_silence
        fast.silero_vad.min_speech_duration = 0.25
        fast.silero_vad.max_speech_duration = self._cfg.silero_vad.max_speech_duration
        fast.silero_vad.threshold = self._cfg.silero_vad.threshold
        fast.sample_rate = SAMPLE_RATE
        return sherpa_onnx.VoiceActivityDetector(fast, buffer_size_in_seconds=self._cfg.silero_vad.max_speech_duration + 5)

    def take_events(self) -> list[tuple[str, Segment | None]]:
        out, self.events = self.events, []
        return out

    def request_reset(self) -> None:
        """Безопасно из другого потока: сброс выполнится в начале следующего feed()."""
        self._reset_requested = True

    def reset(self) -> None:
        self._vad = sherpa_onnx.VoiceActivityDetector(self._cfg, buffer_size_in_seconds=self._cfg.silero_vad.max_speech_duration + 5)
        self._base_ts = None
        self._fed = 0
        self._history.clear()
        if self._fast is not None:
            self._fast = self._make_fast()
        self._speech_from = None
        self._early_sent = False
        self._was_speaking = False

    def _before(self, start: int) -> np.ndarray:
        """Звук [start - PRE_ROLL, start) из истории — то, что VAD не включил в реплику."""
        lo = max(0, start - int(PRE_ROLL * SAMPLE_RATE))
        parts = []
        for first, chunk in self._history:
            a, b = max(lo, first), min(start, first + len(chunk))
            if a < b:
                parts.append(chunk[a - first:b - first])
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    def _slice(self, start: int, end: int) -> np.ndarray:
        parts = []
        for first, chunk in self._history:
            a, b = max(start, first), min(end, first + len(chunk))
            if a < b:
                parts.append(chunk[a - first:b - first])
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    @property
    def speaking(self) -> bool:
        return bool(self._vad.is_speech_detected())

    def feed(self, samples: np.ndarray, ts: float) -> list[Segment]:
        if self._reset_requested:
            self._reset_requested = False
            self.reset()
        expected = None if self._base_ts is None else self._base_ts + self._fed / SAMPLE_RATE
        if expected is None or abs(ts - expected) > 0.5:
            # Первый кусок или разрыв во времени — начинаем отсчёт заново.
            if expected is not None:
                self.reset()
            self._base_ts = ts
        self._vad.accept_waveform(samples)
        self._history.append((self._fed, np.asarray(samples, dtype=np.float32)))
        self._fed += len(samples)
        keep = int((self._cfg.silero_vad.max_speech_duration + 6) * SAMPLE_RATE)
        while self._history and self._history[0][0] + len(self._history[0][1]) < self._fed - keep:
            self._history.popleft()
        out: list[Segment] = []
        while not self._vad.empty():
            seg = self._vad.front
            pre = self._before(seg.start)
            data = np.concatenate([pre, np.array(seg.samples, dtype=np.float32)])
            start = self._base_ts + (seg.start - len(pre)) / SAMPLE_RATE
            out.append(Segment(self.channel, data, start, start + len(data) / SAMPLE_RATE))
            self._vad.pop()
        if self._fast is not None:
            self._feed_fast(samples, bool(out))
        return out

    def _feed_fast(self, samples: np.ndarray, closed: bool) -> None:
        speaking = self.speaking
        if speaking and not self._was_speaking:
            # Детектор замечает речь с опозданием (min_speech + окно) — берём с запасом.
            self._speech_from = max(0, self._fed - len(samples) - int(0.8 * SAMPLE_RATE))
        self._was_speaking = speaking
        self._fast.accept_waveform(samples)
        paused = False
        while not self._fast.empty():
            self._fast.pop()
            paused = True
        if closed or not speaking:
            self._early_sent = False
            if not speaking:
                self._speech_from = None
            return
        if self._early_sent and self._fast.is_speech_detected():
            self._early_sent = False
            self.events.append(("resume", None))
        if paused and self._speech_from is not None and not self._early_sent:
            data = self._slice(self._speech_from, self._fed)
            start = self._base_ts + self._speech_from / SAMPLE_RATE
            self.events.append(("early", Segment(self.channel, data, start, start + len(data) / SAMPLE_RATE)))
            self._early_sent = True


def rms_db(samples: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(samples)) + 1e-12))
    return 20 * np.log10(rms)
