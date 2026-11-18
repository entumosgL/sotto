"""Звонок — отрезок времени, когда приложение держало микрофон (или включён вручную).

Тренировочный звонок (app = "training") — разговор с ИИ-интервьюером или текстовая проверка. Его легенда
нигде не используется, а вопросы не считаются сигналом для тренера: каждая тренировка — с чистого листа.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

TRAINING_APP = "training"
ME, THEM = "me", "them"   # каналы звука: микрофон и всё, что слышишь


@dataclass
class Call:
    id: int
    started_at: float
    ended_at: float | None = None
    app: str | None = None
    title: str | None = None
    summary: str | None = None
    language: str | None = None
    record_only: int = 0
    brief: str | None = None
    company: str | None = None
    interview_id: int | None = None

    @property
    def is_training(self) -> bool:
        return self.app == TRAINING_APP

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Call:
        return cls(**{k: row.get(k) for k in cls.__dataclass_fields__ if k in row})  # type: ignore[arg-type]
