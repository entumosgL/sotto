"""`sotto jobs …` — вход в Telegram/LinkedIn и сбор вакансий из терминала."""

from __future__ import annotations

from .. import config as config_mod
from .. import paths
from ..agents import Agents
from ..db import DB
from ..search import Search
from . import linkedin, telegram
from .cluster import update_clusters
from .pipeline import fetch_all


async def run_jobs(action: str) -> int:
    paths.ensure_dirs()
    cfg = config_mod.load()
    if action == "login-telegram":
        await telegram.login(cfg.jobs.telegram)
        return 0
    if action == "login-linkedin":
        await linkedin.login()
        cfg.jobs.linkedin.enabled = True
        config_mod.save(cfg)
        return 0
    db = DB()
    agents = Agents(db, cfg.agents)
    search = Search(db)
    if action == "fetch":
        report = await fetch_all(cfg, db, agents, search)
        print(report)
    elif action == "cluster":
        print(await update_clusters(db, search, agents))
    return 0
