"""Раздел «Тренеры»: темы подбираются сами, тренировка голосом."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from ...domain.training import KINDS, SOURCE_LABELS

if TYPE_CHECKING:
    from ...bootstrap import Sotto


def register(app: FastAPI, s: Sotto, page: Callable) -> None:

    @app.get("/trainer", response_class=HTMLResponse)
    async def trainer_page(request: Request):
        trainer = s.voice.trainer
        current = trainer.state() if trainer and not trainer.stopped else None
        s.topics.refresh_soon()  # пустой или устаревший список — агент обновит в фоне
        data = s.queries.trainer_page(current.get("session_id") if current else None)
        return page(request, "trainer.html", current=current, now=time.time(), sources=SOURCE_LABELS,
                    updated={k: s.topics.updated_at(k) for k in KINDS},
                    next_topic={k: s.topics.pick(k) for k in KINDS}, **data)

    @app.get("/trainer/sessions/{sid}", response_class=HTMLResponse)
    async def trainer_session(request: Request, sid: int):
        session, turns = s.queries.trainer_session(sid)
        return page(request, "trainer_session.html", s=session, turns=turns)

    def start(kind: str, topic: str, mode: str) -> dict:
        if kind not in KINDS:
            return {"error": "Нет такого тренера"}
        return s.voice.start_trainer(s.trainer_session(kind, topic, mode))

    @app.post("/trainer/start")
    async def trainer_start(kind: str = Form(...), topic: str = Form(""), mode: str = Form("interview")):
        result = start(kind, topic, mode)
        return JSONResponse(result) if "error" in result else RedirectResponse("/trainer", status_code=303)

    @app.post("/api/trainer/start")
    async def api_trainer_start(kind: str, topic: str = "", mode: str = "interview"):
        return start(kind, topic, mode)

    @app.post("/trainer/stop")
    async def trainer_stop():
        s.voice.stop_trainer()
        return RedirectResponse("/trainer", status_code=303)

    @app.post("/api/trainer/stop")
    async def api_trainer_stop():
        s.voice.stop_trainer()
        return {"ok": True}

    @app.post("/trainer/topics/{kind}")
    async def trainer_topics(kind: str):
        if kind in KINDS:
            s.background.spawn(s.topics.refresh(kind))
        return RedirectResponse("/trainer", status_code=303)

    @app.post("/trainer/items/{item_id}/delete")
    async def trainer_item_delete(item_id: int):
        s.topics.drop(item_id)
        return RedirectResponse("/trainer", status_code=303)
