"""Тренер: темы и интервальное повторение.

Темы подбирает агент; вопросы с настоящих собеседований и из разборов отказов — сырьё для него, а не
темы: в списке и в выборе следующей темы их нет. Тема, которой владеешь лучше, возвращается реже.
"""

from __future__ import annotations

import math
import re

KINDS = ("knowledge", "english")
RAW_SOURCES = ("calls", "rejections")       # сырьё для агента, не темы
STALE = 3 * 86400                            # список тем старше трёх дней — обновить
RETRY = 15 * 60                              # агент не ответил — снова не раньше чем через 15 минут
FALLBACK = {"knowledge": "темы из профиля кандидата", "english": "рассказ о себе и опыте"}
SOURCE_LABELS = {"agent": "подобрано", "session": "из тренировки", "manual": "добавлено", "vacancies": "из вакансий"}

STOP_WORDS = re.compile(r"^\W*(стоп|хватит|закончим|заканчиваем|stop|that'?s all|let'?s stop|finish)\W*$", re.I)


def next_due(score: float, now: float) -> float:
    """Чем лучше владеешь темой, тем реже она возвращается: от 12 часов до ~16 дней."""
    return now + 0.5 * math.pow(2, score * 5) * 86400


def blend_score(old: float, seen: int, mastery: float) -> float:
    """Новая оценка владения темой: первая — как есть, дальше — с памятью о прошлых."""
    return old * 0.6 + mastery * 0.4 if seen else mastery


def needs_refresh(updated_at: float, empty: bool, last_try: float, now: float, force: bool) -> bool:
    if force:
        return True
    return (empty or now - updated_at > STALE) and now - last_try > RETRY


def wants_to_stop(text: str) -> bool:
    return bool(STOP_WORDS.match(text))
