"""Локальная веб-панель и API демона (только 127.0.0.1).

Страницы — обычный HTML с формами (Jinja2), без сборки фронтенда.
Живые обновления — Server-Sent Events (/api/events).
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import markdown as md
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .. import config as config_mod
from .. import desktop, paths
from ..agents import AgentError
from ..copilot import prompts
from ..openrouter import OpenRouterError
from ..interviews import upcoming as upcoming_interviews

if TYPE_CHECKING:
    from ..daemon import Daemon

HERE = Path(__file__).parent
templates = Jinja2Templates(directory=str(HERE / "templates"))
templates.env.filters["dt"] = lambda ts: datetime.fromtimestamp(ts).strftime("%d.%m.%Y %H:%M") if ts else "—"
templates.env.filters["time"] = lambda ts: datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else ""
templates.env.filters["md"] = lambda text: md.markdown(text or "", extensions=["fenced_code", "nl2br"])
templates.env.filters["duration"] = lambda s: f"{int(s // 60)} мин" if s else "—"
templates.env.filters["size"] = lambda b: f"{(b or 0) / 1e6:.0f} МБ"


# Разделы конфига, которые можно менять формой настроек (остальное — только в config.toml).
EDITABLE_SECTIONS = {"openrouter", "models", "answer", "audio", "recording", "calls", "agents"}


def back(request: Request, default: str = "/") -> RedirectResponse:
    return RedirectResponse(request.headers.get("referer") or default, status_code=303)


def audio_devices() -> dict[str, list[dict[str, str]]]:
    """Устройства звука для панели настроек (PipeWire в Linux, WASAPI в Windows)."""
    from ..audio.capture import devices
    try:
        return devices()
    except Exception:  # noqa: BLE001 — без списка устройств страница должна открываться
        return {"microphones": [], "speakers": []}


def build_app(d: "Daemon") -> FastAPI:
    app = FastAPI(title="Sotto", docs_url=None, redoc_url=None)
    allowed_hosts = {f"{h}:{d.cfg.web.port}" for h in ("127.0.0.1", "localhost", d.cfg.web.host)}

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        """Защита от чужих сайтов в браузере: DNS rebinding (Host) и CSRF (Origin у изменяющих запросов).
        `sotto ctl` и хоткеи шлют запросы без Origin — их пропускаем."""
        if request.headers.get("host") not in allowed_hosts:
            return JSONResponse({"error": "forbidden host"}, status_code=403)
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD", "OPTIONS") and origin is not None:
            if origin.removeprefix("http://") not in allowed_hosts:
                return JSONResponse({"error": "forbidden origin"}, status_code=403)
        return await call_next(request)
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    def page(request: Request, name: str, **ctx: Any) -> HTMLResponse:
        return templates.TemplateResponse(request, name, {"status": d.status(), "cfg": d.cfg, "paths": paths, **ctx})

    # ---------------- API ----------------

    @app.get("/api/status")
    async def api_status() -> dict[str, Any]:
        return d.status()

    @app.post("/api/ctl/{action}")
    async def api_ctl(action: str) -> JSONResponse:
        return JSONResponse(await d.action(action))

    @app.get("/api/events")
    async def api_events(request: Request) -> StreamingResponse:
        q = d.bus.subscribe()

        async def stream():
            try:
                yield "retry: 1000\n\n"
                while not await request.is_disconnected() and not d._stop.is_set():
                    try:
                        event = await asyncio.wait_for(q.get(), timeout=15)
                        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": ping\n\n"
            finally:
                d.bus.unsubscribe(q)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @app.post("/api/reload")
    async def api_reload() -> dict[str, bool]:
        d.reload_config()
        return {"ok": True}

    # ---------------- главная ----------------

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        calls = d.db.all("SELECT * FROM calls ORDER BY started_at DESC LIMIT 8")
        credits = await d.llm.credits() if d.cfg.api_key else None
        month = time.time() - 30 * 86400
        spent = d.db.one("SELECT COALESCE(SUM(cost), 0) AS s, COUNT(*) AS n FROM answers WHERE created_at > ?", [month])
        spent["stt"] = d.db.one("SELECT COALESCE(SUM(cost), 0) AS s FROM utterances WHERE started_at > ?", [month])["s"]
        return page(request, "index.html", calls=calls, credits=credits, spent=spent,
                    agents=d.agents.status(), profile_filled=bool(prompts.profile()),
                    brief=d.db.get_meta("next_call_brief") or "", upcoming=upcoming_interviews(d.db, 3),
                    now=time.time())

    @app.post("/brief")
    async def set_brief(request: Request, brief: str = Form("")):
        d.db.set_meta("next_call_brief", brief.strip())
        if d.call_id is not None:
            d.db.update("calls", d.call_id, brief=brief.strip())
        return back(request)

    @app.post("/brief/generate")
    async def gen_brief(request: Request, vacancy: str = Form(...)):
        prompt = prompts.load("brief") + "\n\nПрофиль кандидата:\n" + prompts.profile() + "\n\nВакансия:\n" + vacancy
        try:
            text, _ = await d.agents.run(prompt, task="бриф перед звонком")
        except AgentError as e:
            text = f"(не удалось: {e})"
        d.db.set_meta("next_call_brief", text)
        return back(request)

    # ---------------- звонки и поиск ----------------

    @app.get("/calls", response_class=HTMLResponse)
    async def calls_page(request: Request, q: str = ""):
        results = await asyncio.to_thread(d.search.search_calls, q) if q else None
        calls = d.db.all("""SELECT c.*, (SELECT COUNT(*) FROM utterances u WHERE u.call_id = c.id) AS n
                            FROM calls c ORDER BY started_at DESC LIMIT 200""") if not q else []
        return page(request, "calls.html", calls=calls, results=results, q=q)

    @app.get("/calls/{call_id}", response_class=HTMLResponse)
    async def call_page(request: Request, call_id: int):
        call = d.db.one("SELECT * FROM calls WHERE id = ?", [call_id])
        if not call:
            return RedirectResponse("/calls", status_code=303)
        speakers = d.db.all("SELECT * FROM speakers WHERE call_id = ? ORDER BY is_me DESC, id", [call_id])
        transcript = d.db.call_transcript(call_id)
        answers = d.db.all("SELECT * FROM answers WHERE call_id = ? ORDER BY created_at", [call_id])
        legend = d.db.all("SELECT * FROM legend WHERE call_id = ?", [call_id])
        recs = d.db.all("""SELECT * FROM recordings WHERE started_at < ? AND COALESCE(ended_at, ?) > ?
                           ORDER BY started_at""",
                        [call["ended_at"] or time.time(), time.time(), call["started_at"]])
        for u in transcript:  # в какой файл записи попала реплика — чтобы открыть видео с этого места
            u["rec_id"] = next((r["id"] for r in recs if r["started_at"] <= u["started_at"]
                                and u["started_at"] < (r["ended_at"] or time.time())), None)
        ivs = d.db.all("SELECT id, company, role, stage, scheduled_at FROM interviews ORDER BY COALESCE(scheduled_at, created_at) DESC LIMIT 50")
        return page(request, "call.html", call=call, speakers=speakers, transcript=transcript,
                    answers=answers, legend=legend, recordings=recs, interviews=ivs)

    @app.post("/calls/{call_id}/edit")
    async def call_edit(request: Request, call_id: int, title: str = Form(""), summary: str = Form("")):
        d.db.update("calls", call_id, title=title, summary=summary)
        await asyncio.to_thread(d.search.index_call, call_id)
        return back(request)

    @app.post("/calls/{call_id}/process")
    async def call_process(request: Request, call_id: int):
        d._spawn(d.process_call(call_id))
        return back(request)

    @app.post("/calls/{call_id}/delete")
    async def call_delete(call_id: int):
        utt = [u["id"] for u in d.db.all("SELECT id FROM utterances WHERE call_id = ?", [call_id])]
        d.search.remove("utterance", utt)
        d.search.remove("call", [call_id])
        d.db.execute("DELETE FROM calls WHERE id = ?", [call_id])
        return RedirectResponse("/calls", status_code=303)

    @app.post("/speakers/{speaker_id}")
    async def speaker_edit(request: Request, speaker_id: int, name: str = Form(""), role: str = Form("")):
        d.db.update("speakers", speaker_id, name=name.strip() or None, role=role.strip() or None)
        d.bus.publish("speaker", {"id": speaker_id, "name": name.strip(), "role": role.strip()})
        return back(request)

    @app.post("/recordings/{rec_id}/open")
    async def recording_open(request: Request, rec_id: int, at: float = Form(0)):
        rec = d.db.one("SELECT * FROM recordings WHERE id = ?", [rec_id])
        if rec and Path(rec["path"]).exists():
            offset = max(0, at - rec["started_at"]) if at else 0
            try:
                desktop.open_path(rec["path"], offset)
            except OSError:
                pass
        return back(request)

    @app.get("/recordings", response_class=HTMLResponse)
    async def recordings_page(request: Request):
        recs = d.db.all("SELECT * FROM recordings ORDER BY started_at DESC LIMIT 500")
        total = d.db.one("SELECT COALESCE(SUM(size_bytes), 0) AS s FROM recordings")["s"]
        current = d.recorder._row_id if d.recorder.segment is not None else None
        return page(request, "recordings.html", recordings=recs, total=total, current=current)

    @app.post("/recordings/{rec_id}/delete")
    async def recording_delete(request: Request, rec_id: int):
        rec = d.db.one("SELECT * FROM recordings WHERE id = ?", [rec_id])
        if rec and not (d.recorder.segment is not None and d.recorder._row_id == rec_id):  # пишущийся — нельзя
            base = Path(rec["path"]).with_suffix("")
            for suffix in (".mkv", ".video.mkv", ".me.opus", ".them.opus"):
                base.with_suffix(suffix).unlink(missing_ok=True)
            d.db.execute("DELETE FROM recordings WHERE id = ?", [rec_id])
        return back(request)

    # ---------------- легенда ----------------

    @app.get("/legend", response_class=HTMLResponse)
    async def legend_page(request: Request):
        items = d.db.all("""SELECT l.*, c.title AS call_title FROM legend l
                            LEFT JOIN calls c ON c.id = l.call_id
                            ORDER BY COALESCE(l.company, '') = '', lower(l.company), l.created_at DESC""")
        groups: dict[str, list[dict[str, Any]]] = {}
        for i in items:
            key = (i["company"] or "").strip() or ("Без компании: только этот звонок" if i["call_id"] else "Общее (добавлено вручную)")
            groups.setdefault(key, []).append(i)
        return page(request, "legend.html", groups=groups)

    @app.post("/legend/add")
    async def legend_add(request: Request, claim: str = Form(...), note: str = Form(""), company: str = Form("")):
        d.db.insert("legend", claim=claim.strip(), note=note.strip() or None, company=company.strip() or None,
                    created_at=time.time())
        return back(request)

    @app.post("/legend/{item_id}")
    async def legend_edit(request: Request, item_id: int, claim: str = Form(...), note: str = Form(""),
                          active: str = Form("")):
        d.db.update("legend", item_id, claim=claim.strip(), note=note.strip() or None, active=1 if active else 0)
        return back(request)

    @app.post("/legend/company/delete")
    async def legend_company_delete(request: Request, company: str = Form(...)):
        d.db.execute("DELETE FROM legend WHERE lower(trim(company)) = ?", [company.strip().lower()])
        return back(request)

    @app.post("/legend/{item_id}/delete")
    async def legend_delete(request: Request, item_id: int):
        d.db.execute("DELETE FROM legend WHERE id = ?", [item_id])
        return back(request)

    # ---------------- профиль и промпты ----------------

    @app.get("/profile", response_class=HTMLResponse)
    async def profile_page(request: Request):
        return page(request, "profile.html", profile=prompts.profile())

    @app.post("/profile")
    async def profile_save(request: Request, text: str = Form("")):
        paths.PROFILE_FILE.parent.mkdir(parents=True, exist_ok=True)
        paths.PROFILE_FILE.write_text(text.strip() + "\n", encoding="utf-8")
        return back(request)

    @app.post("/profile/structure")
    async def profile_structure(request: Request, notes: str = Form(...)):
        prompt = ("Из заметок кандидата ниже составь профиль для ИИ-суфлёра собеседований в Markdown: "
                  "целевая роль и уровень; стек (по уровню владения); опыт по местам работы (компания, годы, "
                  "роль, проекты, масштаб, конкретные достижения с цифрами); образование; языки (уровень "
                  "английского); что важно в работе; ограничения (зарплата, формат, релокация). Ничего не "
                  "выдумывай: чего нет в заметках — оставь пункт с пометкой «не указано».\n\nЗаметки:\n" + notes)
        try:
            text, _ = await d.agents.run(prompt, task="профиль кандидата")
            existing = prompts.profile()
            paths.PROFILE_FILE.write_text((existing + "\n\n" if existing else "") + text.strip() + "\n", encoding="utf-8")
        except AgentError as e:
            d.bus.publish("notice", {"text": str(e)[:200]})
        return back(request)

    @app.get("/prompts", response_class=HTMLResponse)
    async def prompts_page(request: Request):
        groups: dict[str, list[dict[str, Any]]] = {}
        for k, (group, title) in prompts.CATALOG.items():
            groups.setdefault(group, []).append({"kind": k, "title": title, "text": prompts.load(k),
                                                 "custom": prompts.is_customized(k)})
        return page(request, "prompts.html", groups=groups, prompts_dir=str(paths.PROMPTS_DIR))

    @app.post("/prompts/{kind}")
    async def prompt_save(request: Request, kind: str, text: str = Form(...)):
        if kind in prompts.CATALOG:
            if " ".join(text.split()) == " ".join(prompts.builtin(kind).split()):
                prompts.reset(kind)  # совпадает со встроенным — своя копия не нужна
            else:
                prompts.save(kind, text)
        return back(request)

    @app.post("/prompts/{kind}/reset")
    async def prompt_reset(request: Request, kind: str):
        if kind in prompts.CATALOG:
            prompts.reset(kind)
        return back(request)

    @app.post("/prompts/{kind}/rewrite")
    async def prompt_rewrite(request: Request, kind: str, wishes: str = Form("")):
        if kind not in prompts.CATALOG:
            return back(request)
        prompt = prompts.render(prompts.builtin("prompt_author"), kind=prompts.KINDS.get(kind, kind),
                                current=prompts.load(kind), profile=prompts.profile() or "(не заполнен)",
                                wishes=wishes or "(нет)")
        try:
            text, agent = await d.agents.run(prompt, task=f"промпт {kind}")
            prompts.save(kind, text.strip().strip("`"))
            d.bus.publish("notice", {"text": f"Промпт «{kind}» переписан ({agent})"})
        except AgentError as e:
            d.bus.publish("notice", {"text": str(e)[:200]})
        return back(request)

    # ---------------- настройки ----------------

    @app.get("/settings", response_class=HTMLResponse)
    async def settings_page(request: Request):
        return page(request, "settings.html", c=d.cfg, agents=d.agents.status(),
                    devices=audio_devices(), config_path=paths.short(paths.CONFIG_FILE),
                    autostart=desktop.autostart_enabled(),
                    providers=config_mod.provider_options(), preset=d.cfg.provider,
                    provider_name=d.cfg.provider_name,
                    provider_bases={key: value.base_url for key, value in config_mod.PROVIDERS.items()})

    @app.post("/settings")
    async def settings_save(request: Request):
        form = await request.form()
        # Сначала провайдер: он подставляет адрес, модели и режим распознавания речи.
        # Дальше явно заполненные поля (адрес, модели, ключ) имеют приоритет над пресетом.
        chosen = str(form.get("openrouter.provider", "") or "").strip()
        if chosen and chosen != d.cfg.openrouter.provider:
            config_mod.apply_provider(d.cfg, chosen)
        for key, value in form.items():
            if key.startswith("_") or key.split(".")[0] not in EDITABLE_SECTIONS or key == "openrouter.provider":
                continue
            if not value and (key.endswith("api_key") or key.endswith("password")
                              or key == "openrouter.base_url"):
                continue  # пустое поле секрета или адреса = «не менять» (адрес задаёт пресет)
            try:
                config_mod.set_value(d.cfg, key, value)
            except (AttributeError, ValueError):
                pass
        # Чекбоксы: не пришли — значит выключены.
        for key in str(form.get("_checkboxes", "")).split(","):
            if key and key not in form:
                config_mod.set_value(d.cfg, key, False)
        config_mod.save(d.cfg)
        d.reload_config()
        return back(request)

    @app.post("/api/provider/test")
    async def provider_test():
        """Короткий запрос «работает ли ключ и адрес» прямо из панели."""
        model = d.cfg.models.utility or d.cfg.models.answer
        if not d.cfg.api_key and d.cfg.requires_key:
            return JSONResponse({"ok": False, "error": f"не задан ключ {d.cfg.provider_name}"})
        if not model:
            return JSONResponse({"ok": False, "error": "в разделе «Модели» не указана ни одна модель"})
        try:
            res = await d.llm.chat(model, [{"role": "user", "content": "Ответь одним словом: ок"}],
                                   max_tokens=8)
        except OpenRouterError as e:
            return JSONResponse({"ok": False, "error": str(e)[:300]})
        except Exception as e:  # noqa: BLE001
            return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"[:300]})
        return JSONResponse({"ok": True, "model": res.model, "text": res.text[:120],
                             "address": d.llm.base_url})

    # ---------------- подключаемые разделы ----------------
    from ..trainer.web import register as register_trainer
    from ..jobs.web import register as register_jobs
    from .chat import register as register_chat
    from .interviews import register as register_interviews
    register_trainer(app, d, page)
    register_interviews(app, d, page)
    register_chat(app, d, page)
    register_jobs(app, d, page)

    return app
