"""Как разговор показывается модели и человеку."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any


def speaker_label(row: dict[str, Any]) -> str:
    if row["channel"] == "me":
        return "Я (кандидат)"
    who = row.get("speaker") or "Собеседник"
    return f"{who}, {row['speaker_role']}" if row.get("speaker_role") else who


def format_transcript(rows: Iterable[dict[str, Any]]) -> str:
    return "\n".join(f"[{datetime.fromtimestamp(r['started_at']):%H:%M:%S}] {speaker_label(r)}: {r['text']}"
                     for r in rows)


def transcript_message(rows: list[dict[str, Any]], instruction: str) -> str:
    body = format_transcript(rows) or "(расшифровки пока нет)"
    return f"Расшифровка разговора (последние реплики внизу):\n{body}\n\n{instruction}"


def last_question(rows: list[dict[str, Any]]) -> str:
    """Последняя реплика собеседника."""
    return next((r["text"] for r in reversed(rows) if r["channel"] == "them"), "")


def with_hints(utterances: list[dict[str, Any]], hints: list[dict[str, Any]]) -> str:
    """Расшифровка, где после вопроса видно, что подсказал суфлёр, — для разбора отказа."""
    rows = [(u["started_at"], format_transcript([u])) for u in utterances]
    rows += [(h["created_at"], f"    [подсказка суфлёра] {h['text']}") for h in hints]
    return "\n".join(line for _, line in sorted(rows, key=lambda r: r[0]))
