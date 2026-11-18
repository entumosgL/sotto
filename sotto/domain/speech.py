"""Распознанная речь: мусор Whisper, эхо из колонок, имена из самопредставлений."""

from __future__ import annotations

import difflib
import re
from collections.abc import Iterable

# Признаки самопредставления — чтобы не гонять модель на каждой реплике.
INTRO = re.compile(
    r"(меня зовут|зовут меня|я\s+[—–-]\s*[А-ЯЁ]|это\s+[А-ЯЁ][а-яё]+[,.]|представлюсь|с вами\s+[А-ЯЁ]|"
    r"my name is|i'?m\s+[A-Z][a-z]+|this is\s+[A-Z][a-z]+|i am\s+[A-Z][a-z]+)",
    re.I,
)

# Без модели имя достаём регуляркой: «меня зовут Анна», «я — Дмитрий», «I'm Sarah».
NAME_RX = re.compile(
    r"(?:меня зовут|зовут меня|я\s*[—–-]|^(?:\w+[,!.]\s*)?я|my name is|i'?m|i am|this is)\s+([A-ZА-ЯЁ][a-zа-яё]{1,20})",
    re.I | re.M)
NOT_NAMES = {"рад", "рада", "тоже", "здесь", "очень", "буду", "hr", "the", "here", "glad", "happy", "going", "from"}

# Whisper иногда «слышит» в тишине типовые фразы из субтитров.
HALLUCINATIONS = {
    "продолжение следует", "субтитры сделал dimatorzok", "спасибо за просмотр", "редактор субтитров",
    "thank you for watching", "thanks for watching", "you", "субтитры создавал dimatorzok",
}
_TAIL_JUNK = re.compile(
    r"\s*(продолжение следует|спасибо за просмотр|субтитры (сделал|создавал) \w+|редактор субтитров[^.]*|"
    r"thanks? (you )?for watching)[.!…]*\s*$", re.I)


def is_intro(text: str) -> bool:
    return bool(INTRO.search(text))


def name_from_intro(text: str) -> str | None:
    for m in NAME_RX.finditer(text):
        name = m.group(1)
        if name.lower() not in NOT_NAMES and name[0].isupper():
            return name
    return None


def strip_hallucinations(text: str) -> str:
    """Whisper дописывает «Продолжение следует...» в конец настоящей фразы — срезаем хвост."""
    prev = None
    while prev != text:
        prev, text = text, _TAIL_JUNK.sub("", text).strip()
    return text


def is_hallucination(text: str) -> bool:
    t = re.sub(r"[^\w\s]", "", text.lower()).strip()
    if not re.search(r"[a-zа-яё]", t):  # «足を» и прочие обрывки чужой письменности — шум
        return True
    return not t or t in HALLUCINATIONS or any(h in t for h in ("dimatorzok", "amara.org"))


def is_echo(text: str, recent_them: Iterable[str]) -> bool:
    """Микрофон поймал звук из колонок: фраза кандидата почти повторяет недавнюю фразу собеседника."""
    return any(difflib.SequenceMatcher(None, text.lower(), r.lower()).ratio() > 0.6 for r in recent_them)
