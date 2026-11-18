"""Сборка сообщений для модели.

Порядок важен для кэша провайдера: сначала неизменное (инструкция, профиль, легенда, бриф),
в самом конце — меняющаяся расшифровка. Тогда при каждом вопросе платим полную цену
только за новый хвост разговора.
"""

from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Any

from . import prompts

LEGEND_MARK = re.compile(r"\[\[(.+?)\]\]", re.S)


def system_prompt(kind: str, *, allow_invention: bool, profile: str,
                  legend: list[dict[str, Any]], brief: str | None) -> str:
    rule = prompts.INVENTION_ALLOWED if allow_invention else prompts.INVENTION_FORBIDDEN
    parts = [prompts.render(prompts.load(kind), invention_rule=rule)]
    parts.append("# Профиль кандидата\n" + (profile or "(профиль пока не заполнен — кандидат: сильный бэкенд-разработчик уровня мидл+/сеньор; конкретику опыта придумывай по правилам легенды)"))
    if legend:
        lines = []
        for item in legend:
            line = f"- {item['claim']}"
            if item.get("note"):
                line += f" (уточнение: {item['note']})"
            lines.append(line)
        parts.append("# Легенда — уже рассказанное ранее, придерживайся этого\n" + "\n".join(lines))
    if brief:
        parts.append("# Бриф о компании и вакансии\n" + brief)
    return "\n\n".join(parts)


def format_transcript(rows: list[dict[str, Any]]) -> str:
    out = []
    for r in rows:
        ts = datetime.fromtimestamp(r["started_at"]).strftime("%H:%M:%S")
        if r["channel"] == "me":
            who = "Я (кандидат)"
        else:
            who = r.get("speaker") or "Собеседник"
            if r.get("speaker_role"):
                who = f"{who}, {r['speaker_role']}"
        out.append(f"[{ts}] {who}: {r['text']}")
    return "\n".join(out)


def transcript_message(rows: list[dict[str, Any]], instruction: str) -> str:
    body = format_transcript(rows) or "(расшифровки пока нет)"
    return f"Расшифровка разговора (последние реплики внизу):\n{body}\n\n{instruction}"


def extract_legend(text: str) -> list[str]:
    """Достаёт придуманные факты, отмеченные [[...]]."""
    seen, out = set(), []
    for m in LEGEND_MARK.finditer(text):
        claim = " ".join(m.group(1).split())
        if claim and claim.lower() not in seen:
            seen.add(claim.lower())
            out.append(claim)
    return out


def since(minutes: int) -> float:
    return time.time() - minutes * 60
