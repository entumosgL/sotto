"""Когда суфлёр отвечает, а когда молчит.

Модель сама решает, нужен ли ответ (тогда пишет «∅»), но часть реплик отсекается раньше — без запроса:
поддакивание, обрывки фраз, короткие реплики собеседника посреди ответа кандидата.
"""

from __future__ import annotations

import difflib
import re

SKIP = "∅"   # модель решила: отвечать не нужно

BACKCHANNEL_WORDS = {
    "угу", "ага", "ну", "да", "так", "понятно", "ясно", "хорошо", "окей", "ок", "отлично", "супер", "ладно",
    "спасибо", "понял", "поняла", "принято", "здорово", "класс", "хм", "ммм", "мм", "аха", "интересно",
    "вот", "это", "уже", "большое", "отличный", "хороший", "ответ", "круто",
    "ok", "okay", "mhm", "uh", "huh", "yeah", "yes", "yep", "right", "sure", "great", "got", "it", "i", "see",
    "cool", "nice", "thanks", "thank", "you", "good", "perfect", "alright", "understood", "makes", "sense",
    "interesting", "awesome",
}

_HANGING = re.compile(r"(,|\b(и|а|но|или|что|чтобы|потому|если|когда|как|в|на|с|по|для|and|or|but|so|the|a|to|of|"
                      r"with|for|that|if|when)|\.\.\.|…|-)$", re.I)


def is_backchannel(text: str) -> bool:
    """«Угу, понятно», «Okay, got it» — поддакивание без вопроса."""
    if "?" in text:
        return False
    words = re.findall(r"[\w']+", text.lower())
    return 0 < len(words) <= 5 and all(w in BACKCHANNEL_WORDS for w in words)


def is_interjection(text: str) -> bool:
    """Короткая реплика собеседника посреди ответа кандидата — кашель, «ага», обрывок. Не вопрос:
    перегенерировать подсказку, которую кандидат сейчас читает, нельзя."""
    return "?" not in text and len(re.findall(r"[\w']+", text)) <= 3


def looks_unfinished(text: str) -> bool:
    """«Расскажите, как вы…», «and the» — собеседник явно не договорил, упреждать нечего."""
    return bool(_HANGING.search(text.strip()))


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def same_text(a: str, b: str) -> bool:
    """Whisper на том же звуке плюс 0.3 с тишины пишет почти то же самое — сравниваем без пунктуации."""
    a, b = _norm(a), _norm(b)
    return a == b or difflib.SequenceMatcher(None, a, b).ratio() >= 0.9


def worth_speculating(text: str, candidate_speaking: bool) -> bool:
    """Стоит ли начинать подсказку заранее, по предварительному тексту реплики."""
    return not (is_backchannel(text) or len(text.split()) < 3 or looks_unfinished(text)
                or (candidate_speaking and is_interjection(text)))


_LATIN = re.compile(r"[A-Za-z]")
_CYRILLIC = re.compile(r"[А-Яа-яЁё]")
_NATIVE = re.compile(r"([A-Za-zА-Яа-яЁё]+)\s*[—–-]\s*(?:родной|родная|native)", re.I)
_ENGLISH = {"english", "английский", "английская"}
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[\w'-]+")


def native_language(profile: str) -> str:
    """«Русский — родной», «English — native» → 'русский' / 'english'; '' — в профиле не указано."""
    m = _NATIVE.search(profile)
    return m.group(1).lower() if m else ""


def trim_nonnative(text: str, profile: str, limit: int) -> str:
    """Подсказку на неродном языке кандидат читает вслух: сверх лимита слов отрезаем хвостовые
    предложения (первое оставляем целым). Английский считаем неродным, если родной — другой."""
    native = native_language(profile)
    if limit <= 0 or not native or native in _ENGLISH:
        return text
    if len(_LATIN.findall(text)) <= len(_CYRILLIC.findall(text)):  # подсказка не на английском
        return text
    if len(_WORD.findall(text)) <= limit:
        return text
    sentences = _SENTENCE.split(text)
    kept = sentences[:1]
    for s in sentences[1:]:
        if len(_WORD.findall(" ".join([*kept, s]))) > limit:
            break
        kept.append(s)
    return " ".join(kept)


def is_skip(text: str) -> bool:
    return not text.strip() or text.strip().startswith(SKIP)
