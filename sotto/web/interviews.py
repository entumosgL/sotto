"""Раздел «Собеседования»: план, исходы, разбор отказов."""

from __future__ import annotations

import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .. import interviews as ivs
from ..trainer.topics import refresh_soon

if TYPE_CHECKING:
    from ..daemon import Daemon


def _local(ts: float | None) -> str:
    """Время → значение поля datetime-local."""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%dT%H:%M") if ts else ""


def register(app: FastAPI, d: "Daemon", page: Callable[..., HTMLResponse]) -> None:
    db = d.db

    @app.get("/interviews", response_class=HTMLResponse)
    async def interviews_page(request: Request, invite: str = ""):
        upcoming = ivs.upcoming(db, 50)
        up_ids = {r["id"] for r in upcoming}
        past = [r for r in db.all("""SELECT i.*, (SELECT COUNT(*) FROM calls c WHERE c.interview_id = i.id) AS calls
                                     FROM interviews i ORDER BY COALESCE(scheduled_at, created_at) DESC LIMIT 200""")
                if r["id"] not in up_ids]
        funnel = {s["status"]: s["n"] for s in db.all("SELECT status, COUNT(*) AS n FROM interviews GROUP BY status")}
        companies = db.one("SELECT COUNT(DISTINCT lower(trim(company))) AS n FROM interviews")["n"]
        draft = d.interview_draft if invite else {}
        return page(request, "interviews.html", upcoming=upcoming, past=past, funnel=funnel, companies=companies,
                    statuses=ivs.STATUSES, summary=db.get_meta("rejections_summary") or "",
                    summary_at=float(db.get_meta("rejections_summary_at") or 0), draft=draft, now=time.time())

    @app.post("/interviews/parse")
    async def interviews_parse(text: str = Form(...)):
        """Вставили приглашение — модель заполнит форму, ты проверишь и сохранишь."""
        data = await ivs.parse_invite(d, text)
        data["notes"] = (str(data.get("notes") or "") + "\n\n" + text.strip()).strip()
        d.interview_draft = data
        return RedirectResponse("/interviews?invite=1#new", status_code=303)

    @app.post("/interviews/new")
    async def interviews_new(company: str = Form(...), role: str = Form(""), stage: str = Form(""),
                             when: str = Form(""), duration_min: int = Form(60), link: str = Form(""),
                             vacancy: str = Form(""), notes: str = Form(""), vacancy_id: str = Form("")):
        iid = db.insert("interviews", company=company.strip(), role=role.strip() or None, stage=stage.strip() or None,
                        scheduled_at=ivs.parse_when(when), duration_min=duration_min, link=link.strip() or None,
                        vacancy=vacancy.strip() or None, notes=notes.strip() or None, created_at=time.time())
        d.interview_draft = {}
        if vacancy.strip():
            d._spawn(ivs.make_brief(d, iid))
            refresh_soon(d, force=True)  # темы тренера — под новую вакансию
        if vacancy_id.isdigit():
            db.execute("UPDATE vacancies SET status = 'replied' WHERE id = ?", [int(vacancy_id)])
        return RedirectResponse(f"/interviews/{iid}", status_code=303)

    @app.get("/interviews/{iid}", response_class=HTMLResponse)
    async def interview_page(request: Request, iid: int):
        iv = db.one("SELECT * FROM interviews WHERE id = ?", [iid])
        if not iv:
            return RedirectResponse("/interviews", status_code=303)
        calls = ivs.company_calls(db, iv)
        others = db.all("""SELECT * FROM interviews WHERE lower(trim(company)) = ? AND id != ?
                           ORDER BY COALESCE(scheduled_at, created_at)""", [iv["company"].strip().lower(), iid])
        legend = db.all("SELECT * FROM legend WHERE lower(trim(company)) = ? ORDER BY created_at",
                        [iv["company"].strip().lower()])
        return page(request, "interview.html", iv=iv, when=_local(iv["scheduled_at"]), calls=calls, others=others,
                    legend=legend, statuses=ivs.STATUSES)

    @app.post("/interviews/{iid}")
    async def interview_edit(iid: int, company: str = Form(...), role: str = Form(""), stage: str = Form(""),
                             when: str = Form(""), duration_min: int = Form(60), link: str = Form(""),
                             vacancy: str = Form(""), brief: str = Form(""), notes: str = Form("")):
        old = db.one("SELECT * FROM interviews WHERE id = ?", [iid])
        if not old:
            return RedirectResponse("/interviews", status_code=303)
        at = ivs.parse_when(when)
        db.update("interviews", iid, company=company.strip(), role=role.strip() or None, stage=stage.strip() or None,
                  scheduled_at=at, duration_min=duration_min, link=link.strip() or None,
                  vacancy=vacancy.strip() or None, brief=brief.strip() or None, notes=notes.strip() or None,
                  reminded=0 if at != old["scheduled_at"] else old["reminded"], updated_at=time.time())
        if company.strip() != old["company"]:  # переименовали компанию — звонки и легенда за ней
            for c in db.all("SELECT id FROM calls WHERE interview_id = ?", [iid]):
                db.set_call_company(c["id"], company.strip(), iid)
        if vacancy.strip() and not brief.strip():
            d._spawn(ivs.make_brief(d, iid))
        return RedirectResponse(f"/interviews/{iid}", status_code=303)

    @app.post("/interviews/{iid}/status")
    async def interview_status(iid: int, status: str = Form(...), outcome: str = Form("")):
        ivs.set_status(d, iid, status, outcome)
        return RedirectResponse(f"/interviews/{iid}", status_code=303)

    @app.post("/interviews/{iid}/next")
    async def interview_next(iid: int):
        """Позвали дальше: этот этап — «позвали дальше», следующий — черновик у той же компании."""
        iv = db.one("SELECT * FROM interviews WHERE id = ?", [iid])
        if not iv:
            return RedirectResponse("/interviews", status_code=303)
        ivs.set_status(d, iid, "passed")
        nid = db.insert("interviews", company=iv["company"], role=iv["role"], vacancy=iv["vacancy"], brief=iv["brief"],
                        link=iv["link"], created_at=time.time())
        return RedirectResponse(f"/interviews/{nid}", status_code=303)

    @app.post("/interviews/{iid}/analyze")
    async def interview_analyze(iid: int):
        d._spawn(ivs.analyze_rejection(d, iid))
        return RedirectResponse(f"/interviews/{iid}", status_code=303)

    @app.post("/interviews/{iid}/brief")
    async def interview_brief(iid: int):
        d._spawn(ivs.make_brief(d, iid))
        return RedirectResponse(f"/interviews/{iid}", status_code=303)

    @app.post("/interviews/{iid}/train")
    async def interview_train(iid: int, lang: str = Form("en")):
        """Текстовая тренировка с интервьюером этой компании."""
        iv = db.one("SELECT * FROM interviews WHERE id = ?", [iid])
        if not iv or not d.cfg.api_key:
            d.bus.publish("notice", {"text": "Нужен ключ OpenRouter: интервьюер — языковая модель"})
            return RedirectResponse(f"/interviews/{iid}", status_code=303)
        vacancy = iv["vacancy"] or f"{iv['company']}: {iv['role'] or ''}\n{iv['brief'] or ''}"
        s = d.chats.start("interviewer", lang, iv["role"] or "", vacancy)
        return RedirectResponse(f"/chat/{s.id}", status_code=303)

    @app.post("/interviews/{iid}/delete")
    async def interview_delete(iid: int):
        db.execute("UPDATE calls SET interview_id = NULL WHERE interview_id = ?", [iid])
        db.execute("DELETE FROM interviews WHERE id = ?", [iid])
        db.cleanup_legend()
        return RedirectResponse("/interviews", status_code=303)

    @app.post("/interviews-summary")
    async def interviews_summary():
        d._spawn(ivs.analyze_all(d))
        return RedirectResponse("/interviews#summary", status_code=303)

    @app.post("/calls/{call_id}/interview")
    async def call_link(call_id: int, interview_id: str = Form(""), company: str = Form("")):
        """Привязать звонок к собеседованию (или просто указать компанию) — легенда переедет туда же."""
        if interview_id.isdigit():
            iv: Any = db.one("SELECT * FROM interviews WHERE id = ?", [int(interview_id)])
            if iv:
                db.set_call_company(call_id, iv["company"], iv["id"])
                if iv["status"] == "planned":
                    db.update("interviews", iv["id"], status="done", updated_at=time.time())
        else:
            db.set_call_company(call_id, company)
            db.execute("UPDATE calls SET interview_id = NULL WHERE id = ?", [call_id])
        return RedirectResponse(f"/calls/{call_id}", status_code=303)
