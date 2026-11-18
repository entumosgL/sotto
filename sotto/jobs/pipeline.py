"""Сбор вакансий целиком: источники → разбор → кластеры → новые поисковые запросы."""

from __future__ import annotations

import logging
import time
from typing import Any

from ..agents import AgentError, Agents
from ..config import Config
from ..copilot import prompts
from ..db import DB
from ..events import EventBus
from ..search import Search
from . import linkedin, telegram
from .cluster import update_clusters
from .extract import ingest

log = logging.getLogger(__name__)


def active_queries(db: DB, cfg: Config) -> list[str]:
    for q in (x.strip() for x in cfg.jobs.queries):  # запросы из конфига — как ручные
        if not q:
            continue
        db.execute("INSERT OR IGNORE INTO search_queries(query, source, created_at) VALUES(?, 'manual', ?)", [q, time.time()])
    return [r["query"] for r in db.all("SELECT query FROM search_queries WHERE active = 1 ORDER BY hits DESC, id")]


async def suggest_queries(db: DB, agents: Agents) -> int:
    """Агент пополняет список запросов: синонимы ролей, RU/EN, хэштеги, по найденным вакансиям."""
    existing = [r["query"] for r in db.all("SELECT query FROM search_queries")]
    good = db.all("SELECT title FROM vacancies WHERE status != 'hidden' AND COALESCE(fit, 0.5) >= 0.6 "
                  "ORDER BY found_at DESC LIMIT 40")
    prompt = ("Предложи до 8 НОВЫХ поисковых запросов для поиска вакансий этого кандидата в Telegram-каналах и "
              "постах LinkedIn (там люди пишут «ищем», «hiring», хэштеги вроде #вакансия #hiring). Смешай русские и "
              "английские формулировки, синонимы названия роли, ключевой стек. Не повторяй существующие.\n\n"
              f"Профиль:\n{prompts.profile() or '(не заполнен)'}\n\nУже есть:\n" + "\n".join(existing) +
              "\n\nПодходящие найденные вакансии:\n" + "\n".join(v["title"] or "" for v in good) +
              '\n\nВерни ТОЛЬКО JSON-массив строк.')
    try:
        items, _ = await agents.run_json(prompt, task="новые поисковые запросы")
    except AgentError:
        return 0
    n = 0
    for q in items if isinstance(items, list) else []:
        q = str(q).strip()
        if q and q.lower() not in {e.lower() for e in existing}:
            db.execute("INSERT OR IGNORE INTO search_queries(query, source, created_at) VALUES(?, 'agent', ?)", [q, time.time()])
            n += 1
    return n


async def fetch_all(cfg: Config, db: DB, agents: Agents, search: Search, bus: EventBus | None = None) -> dict[str, Any]:
    def note(text: str) -> None:
        log.info(text)
        if bus:
            bus.publish("notice", {"text": text})

    queries = active_queries(db, cfg)
    if not queries and prompts.profile():
        await suggest_queries(db, agents)
        queries = active_queries(db, cfg)
    report: dict[str, Any] = {"telegram": 0, "linkedin": 0, "errors": []}
    raw = []
    tg_cursors: dict[str, str] = {}
    if cfg.jobs.telegram.api_id and telegram.logged_in():
        try:
            tg, tg_cursors = await telegram.fetch(cfg.jobs.telegram, db, queries)
            raw += tg
            report["telegram"] = len(tg)
        except Exception as e:  # noqa: BLE001 — источник упал, остальные продолжают
            report["errors"].append(f"Telegram: {e}")
    if cfg.jobs.linkedin.enabled and linkedin.logged_in():
        try:
            li = await linkedin.fetch(cfg.jobs.linkedin, db, queries)
            raw += li
            report["linkedin"] = len(li)
        except Exception as e:  # noqa: BLE001
            report["errors"].append(f"LinkedIn: {e}")
    note(f"Собрано сообщений: Telegram {report['telegram']}, LinkedIn {report['linkedin']}. Разбираю…")
    report["added"], failed = await ingest(raw, db, agents, search)
    if failed:
        report["errors"].append(f"агенты не разобрали {failed} пачек — повторю при следующем сборе")
    else:  # курсоры каналов двигаем, только если всё разобрано — иначе сообщения потерялись бы
        for key, value in tg_cursors.items():
            db.set_meta(key, value)
    report["clusters"] = await update_clusters(db, search, agents)
    for q in queries:  # какие запросы приносят вакансии — те поднимаются выше
        words = q.split()
        if not words:
            continue
        hits = db.one("SELECT COUNT(*) AS n FROM vacancies WHERE text LIKE ?", [f"%{words[0]}%"])["n"]
        db.execute("UPDATE search_queries SET hits = ? WHERE query = ?", [hits, q])
    report["new_queries"] = await suggest_queries(db, agents)
    note(f"Новых вакансий: {report['added']}; кластеры: {report['clusters']}; новых запросов: {report['new_queries']}"
         + (f"; ошибки: {'; '.join(report['errors'])}" if report["errors"] else ""))
    if bus:
        bus.publish("job.done", {"text": f"Вакансии обновлены: +{report['added']}"})
    return report
