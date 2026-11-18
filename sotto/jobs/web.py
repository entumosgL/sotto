"""Разделы «Вакансии» и «Резюме» в панели. Любая отправка — только по нажатию кнопки."""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Callable

from fastapi import FastAPI, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse

from .. import config as config_mod
from ..agents import AgentError
from . import emailer, linkedin, resume, telegram
from .pipeline import fetch_all, suggest_queries
from .telegram import SendLimit

if TYPE_CHECKING:
    from ..daemon import Daemon


def register(app: FastAPI, d: "Daemon", page: Callable) -> None:
    db = d.db

    def back(request: Request, default: str = "/jobs") -> RedirectResponse:
        return RedirectResponse(request.headers.get("referer") or default, status_code=303)

    def notify(text: str) -> None:
        d.bus.publish("notice", {"text": text[:300]})

    # ---------------- список ----------------

    @app.get("/jobs", response_class=HTMLResponse)
    async def jobs_page(request: Request, cluster: int | None = None, status: str = "active", q: str = ""):
        where, params = ["1=1"], []
        if cluster is not None:
            where.append("v.cluster_id = ?")
            params.append(cluster)
        if status == "active":
            where.append("v.status NOT IN ('hidden')")
        elif status:
            where.append("v.status = ?")
            params.append(status)
        if q:
            hits = await asyncio.to_thread(d.search.query, q, ("vacancy",), 60)
            ids = [h["ref_id"] for h in hits]
            where.append(f"v.id IN ({','.join('?' * len(ids)) or 'NULL'})")
            params += ids
        vacancies = db.all(f"""SELECT v.*, c.name AS cluster_name FROM vacancies v
                               LEFT JOIN clusters c ON c.id = v.cluster_id WHERE {' AND '.join(where)}
                               ORDER BY COALESCE(v.fit, 0.5) DESC, v.found_at DESC LIMIT 300""", params)
        for v in vacancies:
            v["contacts_list"] = json.loads(v["contacts"] or "[]")
        clusters = db.all("""SELECT c.*, (SELECT COUNT(*) FROM vacancies v WHERE v.cluster_id = c.id) AS n
                             FROM clusters c ORDER BY n DESC""")
        queries = db.all("SELECT * FROM search_queries ORDER BY active DESC, hits DESC, id")
        return page(request, "jobs.html", vacancies=vacancies, clusters=clusters, queries=queries,
                    cluster=cluster, status_filter=status, q=q, tg_logged=telegram.logged_in(),
                    li_logged=linkedin.logged_in(), j=d.cfg.jobs)

    @app.post("/jobs/fetch")
    async def jobs_fetch(request: Request):
        d._spawn(fetch_all(d.cfg, db, d.agents, d.search, d.bus))
        notify("Сбор вакансий запущен — это займёт несколько минут")
        return back(request)

    @app.post("/jobs/queries/add")
    async def query_add(request: Request, query: str = Form(...)):
        if not query.strip():
            return back(request)
        db.execute("INSERT OR IGNORE INTO search_queries(query, source, created_at) VALUES(?, 'manual', ?)",
                   [query.strip(), time.time()])
        return back(request)

    @app.post("/jobs/queries/suggest")
    async def query_suggest(request: Request):
        n = await suggest_queries(db, d.agents)
        notify(f"Добавлено запросов: {n}")
        return back(request)

    @app.post("/jobs/queries/{qid}/toggle")
    async def query_toggle(request: Request, qid: int):
        db.execute("UPDATE search_queries SET active = 1 - active WHERE id = ?", [qid])
        return back(request)

    @app.post("/jobs/settings")
    async def jobs_settings(request: Request):
        form = await request.form()
        for key, value in form.items():
            if not key.startswith("jobs.") or (key.endswith("api_hash") or key.endswith("password")) and not value:
                continue
            try:
                config_mod.set_value(d.cfg, key, value)
            except (AttributeError, ValueError):
                pass
        if "jobs.linkedin.enabled" not in form:
            d.cfg.jobs.linkedin.enabled = False
        config_mod.save(d.cfg)
        return back(request)

    # ---------------- вакансия ----------------

    @app.get("/jobs/{vid}", response_class=HTMLResponse)
    async def vacancy_page(request: Request, vid: int):
        v = db.one("SELECT * FROM vacancies WHERE id = ?", [vid])
        if not v:
            return RedirectResponse("/jobs", status_code=303)
        cluster = db.one("SELECT * FROM clusters WHERE id = ?", [v["cluster_id"]]) if v["cluster_id"] else None
        v["contacts_list"] = json.loads(v["contacts"] or "[]")
        sent = db.all("SELECT * FROM outreach WHERE vacancy_id = ? ORDER BY sent_at DESC", [vid])
        clusters = db.all("SELECT id, name FROM clusters ORDER BY name")
        return page(request, "vacancy.html", v=v, cluster=cluster, clusters=clusters, sent=sent,
                    letter=resume.letter_for(v, cluster), resume_md=resume.resume_for(v, cluster, db))

    @app.post("/jobs/{vid}/status")
    async def vacancy_status(request: Request, vid: int, status: str = Form(...)):
        db.update("vacancies", vid, status=status)
        return back(request)

    @app.post("/jobs/{vid}/cluster")
    async def vacancy_cluster(request: Request, vid: int, cluster_id: str = Form("")):
        db.update("vacancies", vid, cluster_id=int(cluster_id) if cluster_id else None)
        return back(request)

    @app.post("/jobs/{vid}/override")
    async def vacancy_override(request: Request, vid: int, letter: str = Form(""), resume_md: str = Form("")):
        v = db.one("SELECT * FROM vacancies WHERE id = ?", [vid])
        cluster = db.one("SELECT * FROM clusters WHERE id = ?", [v["cluster_id"]]) if v and v["cluster_id"] else None
        # Сохраняем только то, что действительно отличается от шаблона кластера.
        base_letter = resume.letter_for({**v, "letter_override": None}, cluster)
        base_resume = resume.resume_for({**v, "resume_override": None}, cluster, db)
        db.update("vacancies", vid,
                  letter_override=letter.strip() if letter.strip() and letter.strip() != base_letter.strip() else None,
                  resume_override=resume_md.strip() if resume_md.strip() and resume_md.strip() != base_resume.strip() else None)
        return back(request)

    @app.get("/jobs/{vid}/resume.pdf")
    async def vacancy_pdf(vid: int):
        v = db.one("SELECT * FROM vacancies WHERE id = ?", [vid])
        cluster = db.one("SELECT * FROM clusters WHERE id = ?", [v["cluster_id"]]) if v["cluster_id"] else None
        pdf = await asyncio.to_thread(resume.render_pdf, resume.resume_for(v, cluster, db),
                                      f"resume_{cluster['name'] if cluster else 'base'}_{vid}")
        return FileResponse(pdf, media_type="application/pdf", filename=pdf.name)

    @app.post("/jobs/{vid}/send")
    async def vacancy_send(request: Request, vid: int, channel: str = Form(...), target: str = Form(...)):
        """Единственная точка отправки. Вызывается только кнопкой пользователя."""
        v = db.one("SELECT * FROM vacancies WHERE id = ?", [vid])
        cluster = db.one("SELECT * FROM clusters WHERE id = ?", [v["cluster_id"]]) if v["cluster_id"] else None
        letter = resume.letter_for(v, cluster)
        resume_md = resume.resume_for(v, cluster, db)
        if not letter or not resume_md:
            notify("Нет письма или резюме: сначала сгенерируй их для кластера (раздел «Резюме»)")
            return back(request)
        name = (prompts_name() or "resume").replace(" ", "_")
        pdf = await asyncio.to_thread(resume.render_pdf, resume_md, f"{name}_{(cluster or {}).get('name', 'CV')}")
        status = "sent"
        try:
            if channel == "telegram":
                await telegram.send(d.cfg.jobs.telegram, db, target, letter, pdf)
            elif channel == "email":
                subject = (f"Отклик: {v['title']}" if v.get("language") != "en" else f"Application: {v['title']}") \
                    if v.get("title") else "Отклик на вакансию"
                await asyncio.to_thread(emailer.send, d.cfg.jobs.email, db, target, subject, letter, pdf)
            elif channel in ("linkedin", "url"):
                linkedin.open_for_reply(target if target.startswith("http") else (v.get("url") or target), letter)
                status = "opened_in_browser"
                notify("Открыл страницу, текст письма в буфере обмена. PDF: " + str(pdf))
            else:
                raise RuntimeError(f"неизвестный канал {channel}")
        except SendLimit as e:
            notify(str(e))
            return back(request)
        except Exception as e:  # noqa: BLE001
            db.insert("outreach", vacancy_id=vid, channel=channel, target=target, text=letter,
                      sent_at=time.time(), status="failed")
            notify(f"Не отправлено: {e}")
            return back(request)
        db.insert("outreach", vacancy_id=vid, channel=channel, target=target, text=letter, sent_at=time.time(), status=status)
        if status == "sent":
            db.update("vacancies", vid, status="sent")
            notify(f"Отправлено: {target}")
        return back(request)

    # ---------------- резюме ----------------

    @app.get("/resume", response_class=HTMLResponse)
    async def resume_page(request: Request):
        clusters = db.all("""SELECT c.*, (SELECT COUNT(*) FROM vacancies v WHERE v.cluster_id = c.id) AS n
                             FROM clusters c ORDER BY n DESC""")
        return page(request, "resume.html", base_ru=db.get_meta("resume_base_ru") or "",
                    base_en=db.get_meta("resume_base_en") or "", clusters=clusters,
                    use_legend=db.get_meta("resume_use_legend") == "1")

    @app.post("/resume/base/{lang}")
    async def resume_base_save(request: Request, lang: str, text: str = Form("")):
        db.set_meta(f"resume_base_{lang}", text.strip())
        return back(request, "/resume")

    @app.post("/resume/base/{lang}/generate")
    async def resume_base_generate(request: Request, lang: str, use_legend: str = Form("")):
        db.set_meta("resume_use_legend", "1" if use_legend else "0")
        try:
            await resume.generate_base(db, d.agents, bool(use_legend), lang)
        except AgentError as e:
            notify(str(e))
        return back(request, "/resume")

    @app.get("/resume/base/{lang}.pdf")
    async def resume_base_pdf(lang: str):
        pdf = await asyncio.to_thread(resume.render_pdf, db.get_meta(f"resume_base_{lang}") or "",
                                      f"{prompts_name() or 'resume'}_{lang}")
        return FileResponse(pdf, media_type="application/pdf", filename=pdf.name)

    @app.post("/resume/cluster/{cid}")
    async def resume_cluster_save(request: Request, cid: int, resume_md: str = Form(""), letter_md: str = Form(""),
                                  name: str = Form("")):
        db.update("clusters", cid, resume_md=resume_md.strip(), letter_md=letter_md.strip(), updated_at=time.time(),
                  **({"name": name.strip()} if name.strip() else {}))
        return back(request, "/resume")

    @app.post("/resume/cluster/{cid}/generate")
    async def resume_cluster_generate(request: Request, cid: int):
        try:
            await resume.generate_for_cluster(db, d.agents, cid, db.get_meta("resume_use_legend") == "1")
        except AgentError as e:
            notify(str(e))
        return back(request, "/resume")

    @app.get("/resume/cluster/{cid}.pdf")
    async def resume_cluster_pdf(cid: int):
        c = db.one("SELECT * FROM clusters WHERE id = ?", [cid])
        pdf = await asyncio.to_thread(resume.render_pdf, c["resume_md"] or "", f"resume_{c['name']}")
        return FileResponse(pdf, media_type="application/pdf", filename=pdf.name)


def prompts_name() -> str:
    """Имя для файла резюме — из первой строки «# Имя» базового резюме или профиля."""
    from ..copilot import prompts
    for line in prompts.profile().splitlines():
        if line.startswith("# ") and "профиль" not in line.lower():
            return line[2:].strip()
    return ""
