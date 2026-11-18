"""Промпты: встроенные (sotto/prompts/*.md) и свои (~/.config/sotto/prompts/*.md — перекрывают встроенные).

Свой промпт = файл с тем же именем. Удалишь файл — вернётся встроенный. Плейсхолдеры в фигурных скобках
({profile}, {topic}…) подставляет код; остальные фигурные скобки (например, JSON-примеры) остаются как есть."""

from __future__ import annotations

from .. import paths

# Все промпты: имя файла в sotto/prompts → (раздел, что делает). Свой вариант кладётся в
# ~/.config/sotto/prompts/<имя>.md (или правится на странице «Промпты») и перекрывает встроенный.
CATALOG: dict[str, tuple[str, str]] = {
    "answer": ("Суфлёр", "основной ответ на вопрос собеседника"),
    "answer_short": ("Суфлёр", "короткая подсказка-тезисы (режим dual)"),
    "answer_detailed": ("Суфлёр", "подробный ответ (режим dual)"),
    "screen": ("Суфлёр", "ответ по скриншоту экрана (задача, код, тест)"),
    "questions": ("Суфлёр", "вопросы, которые кандидату стоит задать собеседнику"),
    "names": ("Суфлёр", "извлечение имени и роли из самопредставления"),
    "brief": ("Собеседования", "бриф к собеседованию по тексту вакансии"),
    "invite": ("Собеседования", "разбор приглашения на собеседование в поля плана"),
    "postcall": ("Собеседования", "итоги звонка: название, описание, компания, вопросы и темы"),
    "rejection": ("Собеседования", "разбор одного отказа по расшифровкам и подсказкам"),
    "rejections_summary": ("Собеседования", "общий разбор всех отказов"),
    "interviewer_en": ("Тренировочный интервьюер", "интервьюер на английском"),
    "interviewer_ru": ("Тренировочный интервьюер", "интервьюер на русском"),
    "interviewer_plan_en": ("Тренировочный интервьюер", "скрытый план вопросов (английский)"),
    "interviewer_plan_ru": ("Тренировочный интервьюер", "скрытый план вопросов (русский)"),
    "vacancy": ("Тренировочный интервьюер", "выжимка вакансии: роль, компания, бриф"),
    "trainer_knowledge_interview": ("Голосовой тренер", "знания: пробное собеседование по теме"),
    "trainer_knowledge_lesson": ("Голосовой тренер", "знания: разбор темы с наставником"),
    "trainer_english_interview": ("Голосовой тренер", "английский: собеседование с разбором ошибок"),
    "trainer_english_lesson": ("Голосовой тренер", "английский: разговорная практика"),
    "trainer_turn": ("Голосовой тренер", "формат хода тренера (JSON)"),
    "trainer_topics": ("Голосовой тренер", "подбор тем для тренировок"),
    "prompt_author": ("Служебные", "как агент переписывает промпт под тебя"),
}
KINDS = {k: v[1] for k, v in CATALOG.items()}


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


def render(template: str, **values: str) -> str:
    """Подставляет только переданные {ключи}; остальные фигурные скобки не трогает."""
    for key, value in values.items():
        template = template.replace("{" + key + "}", value)
    return template


def load(kind: str) -> str:
    user = paths.PROMPTS_DIR / f"{kind}.md"
    if user.exists() and user.read_text(encoding="utf-8").strip():
        return user.read_text(encoding="utf-8")
    return (paths.BUILTIN_PROMPTS_DIR / f"{kind}.md").read_text(encoding="utf-8")


def builtin(kind: str) -> str:
    return (paths.BUILTIN_PROMPTS_DIR / f"{kind}.md").read_text(encoding="utf-8")


def is_customized(kind: str) -> bool:
    return (paths.PROMPTS_DIR / f"{kind}.md").exists()


def save(kind: str, text: str) -> None:
    paths.PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    (paths.PROMPTS_DIR / f"{kind}.md").write_text(text.strip() + "\n", encoding="utf-8")


def reset(kind: str) -> None:
    (paths.PROMPTS_DIR / f"{kind}.md").unlink(missing_ok=True)


def profile() -> str:
    """Профиль кандидата. Незаполненный шаблон (одни заголовки и «…») считается пустым:
    иначе модель копирует из него заглушки."""
    if not paths.PROFILE_FILE.exists():
        return ""
    text = paths.PROFILE_FILE.read_text(encoding="utf-8").strip()
    return text if is_filled(text) else ""


def is_filled(text: str) -> bool:
    """Отличается ли профиль от шаблона, который создаёт `sotto setup`."""
    from ..setup import PROFILE_TEMPLATE
    norm = lambda t: " ".join(t.split())  # noqa: E731
    return bool(text.strip()) and norm(text) != norm(PROFILE_TEMPLATE)
