"""Из сырых сообщений/постов — в вакансии.

1) Дешёвый локальный фильтр: длинный текст со словами найма.
2) Контакты достаём регулярками (@username, t.me, email, linkedin) — агенту не доверяем адреса.
3) Агент пачками решает «это вакансия?» и раскладывает поля.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..agents import AgentError, Agents
from ..copilot import prompts
from ..db import DB

if TYPE_CHECKING:
    from ..search import Search

HIRING = re.compile(
    r"(ваканси|ищем|ищу|требуется|в поиске|открыта позиция|нанимаем|присоединяйся|зарплат|вилка|оклад|"
    r"hiring|we'?re looking|looking for|join (our|us)|open (position|role)|vacancy|job opening|salary|"
    r"#job|#ваканси|#hiring|remote|удал[её]нк)", re.I)

TG_USER = re.compile(r"(?<![\w.])@([A-Za-z][A-Za-z0-9_]{4,31})\b")
TG_LINK = re.compile(r"(?:https?://)?t\.me/([A-Za-z][A-Za-z0-9_]{4,31})(?![/\w])")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
LINKEDIN = re.compile(r"https?://(?:[a-z]{2,3}\.)?linkedin\.com/(?:in|company|jobs/view)/[^\s)\]]+", re.I)
URL = re.compile(r"https?://[^\s)\]]+")

# Ботов и служебные аккаунты контактами не считаем.
NOT_CONTACTS = {"gmail", "mail", "yandex", "telegram", "username"}


@dataclass
class RawPost:
    source: str        # telegram | linkedin_jobs | linkedin_posts
    source_ref: str    # уникальный id внутри источника
    text: str
    url: str | None = None
    posted_at: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # автор поста и т.п.


def looks_like_job(text: str) -> bool:
    return len(text) >= 150 and bool(HIRING.search(text))


def contacts(text: str, author: str | None = None) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen = set()

    def add(kind: str, value: str) -> None:
        key = (kind, value.lower())
        if key not in seen:
            seen.add(key)
            out.append({"type": kind, "value": value})

    for m in EMAIL.finditer(text):
        add("email", m.group(0).rstrip("."))
    emails_users = {e["value"].split("@")[0].lower() for e in out}
    for rx in (TG_USER, TG_LINK):
        for m in rx.finditer(text):
            name = m.group(1)
            if name.lower() not in NOT_CONTACTS and name.lower() not in emails_users and not name.lower().endswith("bot"):
                add("telegram", "@" + name)
    for m in LINKEDIN.finditer(text):
        add("linkedin", m.group(0).rstrip(".,"))
    if author:
        add("linkedin" if "linkedin.com" in author else "telegram", author)
    for m in URL.finditer(text):
        u = m.group(0).rstrip(".,")
        if "t.me/" not in u and "linkedin.com" not in u:
            add("url", u)
    return out


def text_hash(text: str) -> str:
    norm = re.sub(r"\W+", " ", text.lower()).strip()[:400]
    return hashlib.sha1(norm.encode()).hexdigest()


EXTRACT_PROMPT = """Ниже пронумерованные сообщения из Telegram-каналов и LinkedIn. Для КАЖДОГО реши, является ли оно
вакансией или постом о найме (компания или рекрутер ищет сотрудника). Резюме соискателей, курсы, реклама,
новости, мемы — не вакансии.

Для вакансий заполни поля по тексту (не выдумывай; нет данных — null):
title (должность), company, location, remote ("remote" | "hybrid" | "office" | null), salary (как в тексте),
language (язык текста: "ru" | "en"), summary (2 строки: стек, уровень, ключевые требования), fit (0..1 —
насколько подходит кандидату с профилем ниже).

Профиль кандидата:
{profile}

Верни ТОЛЬКО JSON-массив: [{"n": номер, "is_vacancy": true/false, "title": ..., "company": ..., "location": ...,
"remote": ..., "salary": ..., "language": ..., "summary": ..., "fit": ...}]

Сообщения:
{posts}
"""


async def ingest(posts: list[RawPost], db: DB, agents: Agents, search: "Search | None" = None,
                 batch: int = 15) -> tuple[int, int]:
    """Фильтрует, разбирает агентом и сохраняет новые вакансии.
    Возвращает (добавлено, пачек не разобрано из-за агентов)."""
    fresh = []
    for p in posts:
        if not looks_like_job(p.text):
            continue
        if db.one("SELECT 1 FROM vacancies WHERE source = ? AND source_ref = ?", [p.source, p.source_ref]):
            continue
        h = text_hash(p.text)
        if db.get_meta(f"vh:{h}") or any(text_hash(f.text) == h for f in fresh):
            continue  # тот же текст уже видели в другом канале (репост)
        fresh.append(p)

    added = failed = 0
    for i in range(0, len(fresh), batch):
        chunk = fresh[i:i + batch]
        listing = "\n\n".join(f"### {n}\n{p.text[:2500]}" for n, p in enumerate(chunk))
        prompt = prompts.render(EXTRACT_PROMPT, profile=prompts.profile() or "(не заполнен)", posts=listing)
        try:
            parsed, _agent = await agents.run_json(prompt, task="разбор вакансий")
        except AgentError:
            failed += 1
            continue  # пачку не помечаем просмотренной — разберём при следующем сборе
        if not isinstance(parsed, list):
            failed += 1
            continue
        for p in chunk:  # разобрано — больше не показываем агенту (в том числе «не вакансии»)
            db.set_meta(f"vh:{text_hash(p.text)}", p.source_ref)
        used: set[int] = set()
        for item in parsed:
            if not isinstance(item, dict) or not item.get("is_vacancy"):
                continue
            n = item.get("n")
            if not isinstance(n, int) or not 0 <= n < len(chunk) or n in used:
                continue
            used.add(n)
            p = chunk[n]
            vid = db.insert(
                "vacancies", source=p.source, source_ref=p.source_ref, url=p.url,
                title=item.get("title"), company=item.get("company"), location=item.get("location"),
                salary=item.get("salary"), remote=item.get("remote"), language=item.get("language"),
                text=p.text, summary=item.get("summary"),
                fit=item.get("fit") if isinstance(item.get("fit"), (int, float)) else None,
                contacts=json.dumps(contacts(p.text, p.extra.get("author")), ensure_ascii=False),
                found_at=time.time(), posted_at=p.posted_at)
            if search is not None:
                await asyncio.to_thread(search.index, "vacancy", vid, f"{item.get('title') or ''}. {p.text[:1500]}")
            added += 1
    return added, failed
