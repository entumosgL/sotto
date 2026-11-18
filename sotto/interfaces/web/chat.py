"""Страницы текстового режима: /chat — начать, /chat/{id} — переписка с подсказками."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from ... import paths
from .app import sse

if TYPE_CHECKING:
    from ...bootstrap import Sotto


def register(app: FastAPI, d: Sotto, page: Callable[..., HTMLResponse]) -> None:

    @app.get("/chat", response_class=HTMLResponse)
    async def chat_home(request: Request):
        vacancy = d.queries.nearest_vacancy()
        from_interview = bool(vacancy)
        vacancy_file = paths.CONFIG_DIR / "vacancy.txt"
        if not vacancy and vacancy_file.exists():
            vacancy = vacancy_file.read_text(encoding="utf-8")
        sessions = [s.state() for s in reversed(d.chats.items.values())]
        past = d.queries.past_text_sessions()
        return page(request, "chat.html", vacancy=vacancy, from_interview=from_interview, sessions=sessions,
                    past=past)

    @app.post("/chat/start")
    async def chat_start(mode: str = Form("self"), lang: str = Form("en"), role: str = Form(""),
                         vacancy: str = Form("")):
        if mode == "interviewer" and not d.cfg.api_key:
            d.bus.publish("notice", {"text": "Нужен ключ OpenRouter: интервьюер — языковая модель"})
            return RedirectResponse("/chat", status_code=303)
        if vacancy.strip():
            (paths.CONFIG_DIR / "vacancy.txt").write_text(vacancy.strip() + "\n", encoding="utf-8")
        s = d.chats.start(mode, lang, role, vacancy)
        return RedirectResponse(f"/chat/{s.id}", status_code=303)

    @app.get("/chat/{sid}", response_class=HTMLResponse)
    async def chat_page(request: Request, sid: str):
        s = d.chats.get(sid)
        if s is None:
            return RedirectResponse("/chat", status_code=303)
        return page(request, "chat_session.html", s=s.state(), lines=s.transcript())

    def _session(sid: str):
        s = d.chats.get(sid)
        if s is None or s.ended:
            return None, JSONResponse({"ok": False, "error": "сессия закончена"}, status_code=404)
        return s, None

    @app.post("/api/chat/{sid}/ask")
    async def chat_ask(sid: str, request: Request) -> JSONResponse:
        s, err = _session(sid)
        if err:
            return err
        text = str((await request.json()).get("text", "")).strip()
        if not text or s.mode != "self":
            return JSONResponse({"ok": False})
        s._spawn(s.ask(text))
        return JSONResponse({"ok": True})

    @app.post("/api/chat/{sid}/answer")
    async def chat_answer(sid: str, request: Request) -> JSONResponse:
        s, err = _session(sid)
        if err:
            return err
        s._spawn(s.candidate(str((await request.json()).get("text", ""))))
        return JSONResponse({"ok": True})

    @app.post("/api/chat/{sid}/end")
    async def chat_end(sid: str) -> JSONResponse:
        s, err = _session(sid)
        if err:
            return err
        await s.end()
        return JSONResponse({"ok": True, "call_id": s.call_id})

    @app.get("/api/chat/{sid}/events")
    async def chat_events(request: Request, sid: str) -> Any:
        s = d.chats.get(sid)
        if s is None:
            return JSONResponse({"error": "нет сессии"}, status_code=404)
        return sse(request, s.bus, d.stopping, first=[{"type": "chat.state", "busy": s.busy}])
