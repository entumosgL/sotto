"""Кто говорит: голосовые отпечатки и онлайн-кластеризация собеседников внутри звонка.

Микрофон — всегда «Я». Канал собеседников может содержать нескольких людей:
каждую реплику сравниваем с уже известными голосами по косинусной близости.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np
import sherpa_onnx

from . import models
from .capture import SAMPLE_RATE

MIN_SECONDS_FOR_NEW_VOICE = 1.2  # по очень коротким репликам («ага») нового человека не заводим


class Embedder:
    def __init__(self) -> None:
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(models.file("wespeaker_en_voxceleb_resnet34_LM.onnx")), num_threads=2)
        self._ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)
        self._lock = threading.Lock()

    def embed(self, samples: np.ndarray) -> np.ndarray:
        with self._lock:
            stream = self._ex.create_stream()
            stream.accept_waveform(SAMPLE_RATE, samples)
            stream.input_finished()
            vec = np.array(self._ex.compute(stream), dtype=np.float32)
        return vec / (np.linalg.norm(vec) + 1e-9)


@dataclass
class Voice:
    speaker_id: int
    centroid: np.ndarray
    count: int = 1


@dataclass
class VoiceRegistry:
    """Голоса собеседников одного звонка."""
    threshold: float = 0.6
    voices: list[Voice] = field(default_factory=list)

    def match(self, emb: np.ndarray, duration: float) -> tuple[int | None, float]:
        """Возвращает (speaker_id или None, близость). None — похоже на нового человека."""
        best, best_sim = None, -1.0
        for v in self.voices:
            sim = float(v.centroid @ emb)
            if sim > best_sim:
                best, best_sim = v, sim
        if best is not None and (best_sim >= self.threshold or duration < MIN_SECONDS_FOR_NEW_VOICE):
            self._update(best, emb)
            return best.speaker_id, best_sim
        return None, best_sim

    def add(self, speaker_id: int, emb: np.ndarray) -> None:
        self.voices.append(Voice(speaker_id, emb.copy()))

    @staticmethod
    def _update(v: Voice, emb: np.ndarray) -> None:
        v.centroid = (v.centroid * v.count + emb) / (v.count + 1)
        v.centroid /= np.linalg.norm(v.centroid) + 1e-9
        v.count += 1
