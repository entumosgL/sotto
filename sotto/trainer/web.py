"""Раздел «Тренеры» в панели."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Callable

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from .session import TrainerSession
from .topics import CURATED, pick, refresh_soon, refresh_topics

if TYPE_CHECKING:
    from ..daemon import Daemon


def register(app: FastAPI, d: "Daemon", page: Callable) -> None:

    @app.get("/trainer", response_class=HTMLResponse)
    async def trainer_page(request: Request):
        items = {k: d.db.all(f"""SELECT * FROM trainer_items WHERE kind = ? AND {CURATED}
                                 ORDER BY due_at ASC, score ASC LIMIT 60""", [k])
                 for k in ("knowledge", "english")}
        sessions = d.db.all("SELECT * FROM trainer_sessions ORDER BY started_at DESC LIMIT 20")
        current = d.trainer.state() if d.trainer and not d.trainer.stopped else None
        turns = d.db.all("SELECT * FROM trainer_turns WHERE session_id = ? ORDER BY id",
                         [current["session_id"]]) if current and current.get("session_id") else []
        refresh_soon(d)  # пустой или устаревший список — агент обновит в фоне
        updated = {k: float(d.db.get_meta(f"topics_at_{k}") or 0) for k in items}
        return page(request, "trainer.html", items=items, sessions=sessions, current=current, turns=turns,
                    now=time.time(), updated=updated, next_topic={k: pick(d, k) for k in items})

    @app.get("/trainer/sessions/{sid}", response_class=HTMLResponse)
    async def trainer_session(request: Request, sid: int):
        s = d.db.one("SELECT * FROM trainer_sessions WHERE id = ?", [sid])
        turns = d.db.all("SELECT * FROM trainer_turns WHERE session_id = ? ORDER BY id", [sid])
        return page(request, "trainer_session.html", s=s, turns=turns)

    @app.post("/trainer/start")
    async def trainer_start(kind: str = Form(...), topic: str = Form(""), mode: str = Form("interview")):
        result = start(d, kind, topic, mode)
        return JSONResponse(result) if "error" in result else RedirectResponse("/trainer", status_code=303)

    @app.post("/api/trainer/start")
    async def api_trainer_start(kind: str, topic: str = "", mode: str = "interview"):
        return start(d, kind, topic, mode)

    @app.post("/trainer/stop")
    async def trainer_stop():
        if d.trainer:
            d.trainer.stop()
        return RedirectResponse("/trainer", status_code=303)

    @app.post("/api/trainer/stop")
    async def api_trainer_stop():
        if d.trainer:
            d.trainer.stop()
        return {"ok": True}

    @app.post("/trainer/topics/{kind}")
    async def trainer_topics(kind: str):
        d._spawn(refresh_topics(d, kind))
        return RedirectResponse("/trainer", status_code=303)

    @app.post("/trainer/items/{item_id}/delete")
    async def trainer_item_delete(item_id: int):
        d.db.execute("DELETE FROM trainer_items WHERE id = ?", [item_id])
        return RedirectResponse("/trainer", status_code=303)


def start(d: "Daemon", kind: str, topic: str, mode: str) -> dict:
    if d.call_id is not None:
        return {"error": "Идёт звонок — тренер не запускается"}
    if d.trainer and not d.trainer.stopped:
        return {"error": "Тренировка уже идёт"}
    topic = topic.strip() or pick(d, kind)
    d.trainer = TrainerSession(d, kind, topic, mode)
    d._spawn(d.trainer.run())
    return {"ok": True, "topic": topic}
