"""Резюме и сопроводительные письма.

* Базовое резюме собирается агентом из профиля (+ принятой тобой легенды, если включено).
* Для каждого кластера вакансий — свой вариант резюме и шаблон письма с подстановками
  {position}, {company}, {greeting}. Под конкретную вакансию можно поправить руками (override).
* PDF рендерится локально из Markdown: WeasyPrint (Linux), а в Windows, где у WeasyPrint нет
  системных библиотек GTK, — печать в PDF через Edge/Chrome в headless-режиме.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import markdown as md

from .. import paths
from ..agents import Agents
from ..copilot import prompts
from ..db import DB

log = logging.getLogger(__name__)

CSS = """
@page { size: A4; margin: 16mm 16mm; }
body { font-family: "DejaVu Sans", "Noto Sans", sans-serif; font-size: 10.5pt; line-height: 1.38; color: #1a1a1a; }
h1 { font-size: 20pt; margin: 0 0 2mm; }
h2 { font-size: 12pt; margin: 6mm 0 2mm; padding-bottom: 1mm; border-bottom: 0.4pt solid #999; text-transform: uppercase; letter-spacing: .04em; }
h3 { font-size: 11pt; margin: 4mm 0 1mm; }
p { margin: 1mm 0; }
ul { margin: 1mm 0 2mm; padding-left: 5mm; }
li { margin: .6mm 0; }
a { color: #1a4fd6; text-decoration: none; }
"""

RESUME_RULES = """Правила резюме:
- Markdown: # Имя, строка контактов, ## Разделы (О себе — 2–3 строки; Навыки; Опыт; Образование; Языки).
- Опыт: компания, роль, годы; 3–5 пунктов-достижений на место: глагол + что сделал + результат в цифрах.
- Ключевые слова из требований кластера — естественно, без переспама (ATS это ценит).
- 1–2 страницы. Без фото, без «коммуникабельный, стрессоустойчивый».
- Опирайся только на профиль и легенду ниже. Не выдумывай новых мест работы, дат и цифр сверх них."""


def _legend_text(db: DB) -> str:
    items = db.active_legend()
    return "\n".join(f"- {i['claim']}" + (f" ({i['note']})" if i.get("note") else "") for i in items) or "(нет)"


async def generate_base(db: DB, agents: Agents, use_legend: bool, language: str = "ru") -> str:
    prompt = (f"Составь резюме кандидата на языке: {'русский' if language == 'ru' else 'английский'}.\n{RESUME_RULES}\n\n"
              f"Профиль:\n{prompts.profile()}\n\nЛегенда (рассказанный на собеседованиях опыт):\n"
              f"{_legend_text(db) if use_legend else '(не использовать)'}\n\nВерни ТОЛЬКО Markdown резюме.")
    text, _ = await agents.run(prompt, task="базовое резюме")
    text = _strip_fence(text)
    db.set_meta(f"resume_base_{language}", text)
    return text


async def generate_for_cluster(db: DB, agents: Agents, cluster_id: int, use_legend: bool) -> dict[str, Any]:
    cluster = db.one("SELECT * FROM clusters WHERE id = ?", [cluster_id])
    vacancies = db.all("SELECT title, company, text, language FROM vacancies WHERE cluster_id = ? LIMIT 8", [cluster_id])
    langs = [v["language"] for v in vacancies if v["language"]]
    lang = "en" if langs and langs.count("en") > len(langs) / 2 else "ru"
    base = db.get_meta(f"resume_base_{lang}") or db.get_meta("resume_base_ru") or "(базового резюме нет)"
    sample = "\n\n".join(f"- {v['title']} @ {v['company']}: {v['text'][:700]}" for v in vacancies)
    prompt = (
        f"Адаптируй резюме под группу похожих вакансий «{cluster['name']}» и напиши шаблон сопроводительного письма.\n"
        f"Язык обоих текстов: {'английский' if lang == 'en' else 'русский'}.\n{RESUME_RULES}\n\n"
        "Письмо: 5–8 предложений, живым языком, без канцелярита; почему подходишь — 2–3 конкретных факта из опыта, "
        "релевантных этой группе; в конце — предложение созвониться. Используй подстановки ровно в таком виде: "
        "{greeting} (приветствие), {position} (должность), {company} (компания или пусто). Не ставь подпись с "
        "контактами — они в резюме.\n\n"
        f"Описание группы: {cluster['description'] or ''}\n\nПримеры вакансий:\n{sample}\n\n"
        f"Базовое резюме:\n{base}\n\nПрофиль:\n{prompts.profile()}\n\n"
        f"Легенда:\n{_legend_text(db) if use_legend else '(не использовать)'}\n\n"
        'Верни ТОЛЬКО JSON {"resume_md": "...", "letter_md": "..."}')
    data, _ = await agents.run_json(prompt, task=f"резюме для кластера {cluster['name']}")
    resume_md, letter_md = _strip_fence(str(data.get("resume_md") or "")), str(data.get("letter_md") or "")
    db.update("clusters", cluster_id, resume_md=resume_md, letter_md=letter_md)
    return {"resume_md": resume_md, "letter_md": letter_md}


def letter_for(vacancy: dict[str, Any], cluster: dict[str, Any] | None) -> str:
    """Письмо под вакансию: ручная правка, иначе шаблон кластера с подстановками."""
    if vacancy.get("letter_override"):
        return vacancy["letter_override"]
    template = (cluster or {}).get("letter_md") or ""
    lang_en = vacancy.get("language") == "en"
    greeting = "Hi" if lang_en else "Здравствуйте"
    return (template.replace("{greeting}", greeting)
            .replace("{position}", vacancy.get("title") or ("the position" if lang_en else "вакансию"))
            .replace("{company}", vacancy.get("company") or "")
            .replace(" ,", ",").replace("  ", " ").strip())


def resume_for(vacancy: dict[str, Any], cluster: dict[str, Any] | None, db: DB) -> str:
    if vacancy.get("resume_override"):
        return vacancy["resume_override"]
    if cluster and cluster.get("resume_md"):
        return cluster["resume_md"]
    return db.get_meta("resume_base_ru") or ""


def render_pdf(markdown_text: str, name: str) -> Path:
    """PDF резюме. Возвращает .pdf, а если напечатать нечем — .html рядом (его можно распечатать)."""
    out_dir = paths.JOBS_DIR / "pdf"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w\-]+", "_", name).strip("_")[:60] or "resume"
    out = out_dir / f"{safe}.pdf"
    html = f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{md.markdown(markdown_text, extensions=['sane_lists'])}</body></html>"
    try:
        from weasyprint import HTML
        HTML(string=html).write_pdf(str(out))
        return out
    except Exception as e:  # noqa: BLE001 — в Windows обычно нет GTK-библиотек
        log.info("WeasyPrint недоступен (%s) — печатаю резюме через браузер", str(e)[:120])
    if _print_to_pdf(html, out):
        return out
    fallback = out.with_suffix(".html")
    fallback.write_text(html, encoding="utf-8")
    log.warning("PDF не получился: нет ни WeasyPrint, ни Edge/Chrome. Резюме сохранено как %s", fallback)
    return fallback


def _print_to_pdf(html_text: str, out: Path) -> bool:
    """Печать HTML в PDF браузером на движке Chromium (Edge есть в Windows 11 всегда)."""
    source = out.with_suffix(".source.html")
    source.write_text(html_text, encoding="utf-8")
    candidates = [shutil.which(name) for name in ("msedge", "chrome", "chromium", "brave")]
    candidates += [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                   r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
                   r"C:\Program Files\Google\Chrome\Application\chrome.exe"]
    for exe in candidates:
        if not exe or not Path(exe).exists():
            continue
        try:
            subprocess.run([exe, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                            f"--print-to-pdf={out}", source.as_uri()],
                           timeout=120, capture_output=True)
        except (OSError, subprocess.SubprocessError):
            continue
        if out.exists() and out.stat().st_size > 1000:
            source.unlink(missing_ok=True)
            log.info("Резюме напечатано в PDF через %s", Path(exe).name)
            return True
    source.unlink(missing_ok=True)
    return False


def _strip_fence(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:markdown|md)?\s*\n(.*)\n```$", text, re.S)
    return m.group(1).strip() if m else text
