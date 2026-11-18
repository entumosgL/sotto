"""Сборка сообщений для модели.

Порядок важен для кэша провайдера: сначала неизменное (инструкция, профиль, легенда, бриф),
в самом конце — меняющаяся расшифровка. Тогда при каждом вопросе платим полную цену
только за новый хвост разговора.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from ..domain.legend import LegendItem

INVENTION_ALLOWED = (
    "Если для сильного ответа не хватает опыта из профиля (или профиль пуст), ты САМ решаешь, стоит ли "
    "дополнить его правдоподобным конкретным опытом, согласованным с профилем (стек, годы, масштаб компаний, "
    "роль). Лучше конкретный придуманный факт, чем заглушка или общие слова. Без нужды не придумывай: если "
    "профиль или легенда уже покрывают вопрос — используй их. Каждый придуманный факт "
    "оберни в двойные квадратные скобки: [[в 2023 вёл эту миграцию в команде из пяти человек]]. Скобки увидит "
    "только кандидат — это нужно, чтобы он запомнил легенду."
)
INVENTION_FORBIDDEN = (
    "Не придумывай опыт, которого нет в профиле и легенде. Если опыта нет — сформулируй, как кандидат "
    "подошёл бы к задаче, опираясь на смежный опыт."
)
EMPTY_PROFILE = ("(профиль пока не заполнен — кандидат: сильный бэкенд-разработчик уровня мидл+/сеньор; "
                 "конкретику опыта придумывай по правилам легенды)")


def render(template: str, **values: str) -> str:
    """Подставляет только переданные {ключи}; остальные фигурные скобки не трогает."""
    for key, value in values.items():
        template = template.replace("{" + key + "}", value)
    return template


def system_prompt(template: str, *, allow_invention: bool, profile: str,
                  legend: Iterable[LegendItem], brief: str | None) -> str:
    rule = INVENTION_ALLOWED if allow_invention else INVENTION_FORBIDDEN
    parts = [render(template, invention_rule=rule), "# Профиль кандидата\n" + (profile or EMPTY_PROFILE)]
    lines = [f"- {it.claim}" + (f" (уточнение: {it.note})" if it.note else "") for it in legend]
    if lines:
        parts.append("# Легенда — уже рассказанное ранее, придерживайся этого\n" + "\n".join(lines))
    if brief:
        parts.append("# Бриф о компании и вакансии\n" + brief)
    return "\n\n".join(parts)


def parse_json_object(text: str) -> dict[str, Any]:
    """JSON-объект из ответа модели (вокруг может быть текст или ```)."""
    data = json.loads(text[text.find("{"):text.rfind("}") + 1])
    if not isinstance(data, dict):
        raise ValueError("не объект")
    return data
