"""Сторона чтения (CQRS): готовые данные для страниц сайта.

Страницам нужны объединения и сводки (звонок с репликами, записями и собеседованиями; воронка
собеседований; расходы за месяц), а не сущности с поведением. Поэтому чтение идёт отдельными запросами
в обход доменной модели, а всё, что меняет данные, — через сценарии (application/) и сущности (domain/).
"""

from __future__ import annotations

import time
from typing import Any

from ...domain import interviews as iv_rules
from ...domain.calls import TRAINING_APP
from ...domain.companies import norm_company
from ...domain.interviews import Interview
from .db import DB
from .repositories import CURATED


class Queries:
    def __init__(self, db: DB):
        self.db = db

    # --- главная ---

    def recent_calls(self, limit: int = 8) -> list[dict[str, Any]]:
        return self.db.all("SELECT * FROM calls ORDER BY started_at DESC LIMIT ?", [limit])

    def month_spend(self) -> dict[str, Any]:
        month = time.time() - 30 * 86400
        spent = self.db.one("SELECT COALESCE(SUM(cost), 0) AS s, COUNT(*) AS n FROM answers WHERE created_at > ?",
                            [month]) or {"s": 0, "n": 0}
        spent["stt"] = (self.db.one("SELECT COALESCE(SUM(cost), 0) AS s FROM utterances WHERE started_at > ?",
                                    [month]) or {"s": 0})["s"]
        return spent

    def next_call_brief(self) -> str:
        return self.db.get_meta("next_call_brief") or ""

    # --- звонки ---

    def calls(self, limit: int = 200) -> list[dict[str, Any]]:
        return self.db.all("""SELECT c.*, (SELECT COUNT(*) FROM utterances u WHERE u.call_id = c.id) AS n
                              FROM calls c ORDER BY started_at DESC LIMIT ?""", [limit])

    def call_page(self, call_id: int) -> dict[str, Any] | None:
        call = self.db.one("SELECT * FROM calls WHERE id = ?", [call_id])
        if not call:
            return None
        now = time.time()
        transcript = self.db.all(
            """SELECT u.*, COALESCE(s.name, s.label) AS speaker, s.role AS speaker_role
               FROM utterances u LEFT JOIN speakers s ON s.id = u.speaker_id
               WHERE u.call_id = ? ORDER BY u.started_at""", [call_id])
        recs = self.db.all("""SELECT * FROM recordings WHERE started_at < ? AND COALESCE(ended_at, ?) > ?
                              ORDER BY started_at""", [call["ended_at"] or now, now, call["started_at"]])
        for u in transcript:  # в какой файл записи попала реплика — чтобы открыть видео с этого места
            u["rec_id"] = next((r["id"] for r in recs if r["started_at"] <= u["started_at"] < (r["ended_at"] or now)),
                               None)
        interview = self.db.one("SELECT * FROM interviews WHERE id = ?", [call["interview_id"]]) \
            if call.get("interview_id") else None
        return {
            "call": call,
            "speakers": self.db.all("SELECT * FROM speakers WHERE call_id = ? ORDER BY is_me DESC, id", [call_id]),
            "transcript": transcript,
            "answers": self.db.all("SELECT * FROM answers WHERE call_id = ? ORDER BY created_at", [call_id]),
            "legend": self.db.all("SELECT * FROM legend WHERE call_id = ?", [call_id]),
            "recordings": recs,
            "interview": interview,
            "interviews": self.db.all("""SELECT id, company, role, stage, scheduled_at FROM interviews
                                         ORDER BY COALESCE(scheduled_at, created_at) DESC LIMIT 50"""),
        }

    # --- записи ---

    def recordings(self, limit: int = 500) -> tuple[list[dict[str, Any]], int]:
        recs = self.db.all("SELECT * FROM recordings ORDER BY started_at DESC LIMIT ?", [limit])
        total = (self.db.one("SELECT COALESCE(SUM(size_bytes), 0) AS s FROM recordings") or {"s": 0})["s"]
        return recs, total

    # --- легенда ---

    def legend_groups(self) -> dict[str, list[dict[str, Any]]]:
        items = self.db.all("""SELECT l.*, c.title AS call_title FROM legend l
                               LEFT JOIN calls c ON c.id = l.call_id
                               ORDER BY COALESCE(l.company, '') = '', lower(l.company), l.created_at DESC""")
        groups: dict[str, list[dict[str, Any]]] = {}
        for i in items:
            key = (i["company"] or "").strip() or ("Без компании: только этот звонок" if i["call_id"]
                                                   else "Общее (добавлено вручную)")
            groups.setdefault(key, []).append(i)
        return groups

    # --- собеседования ---

    def _interviews(self) -> list[Interview]:
        return [Interview.from_row(r) for r in self.db.all("SELECT * FROM interviews")]

    def upcoming(self, limit: int = 5) -> list[Interview]:
        return iv_rules.upcoming(self._interviews(), time.time(), limit)

    def nearest_vacancy(self) -> str:
        """Вакансия ближайшего собеседования — для тренировок «как будто это оно»."""
        return next((iv.vacancy for iv in self.upcoming() if (iv.vacancy or "").strip()), "")

    def interviews_page(self) -> dict[str, Any]:
        upcoming = self.upcoming(50)
        up_ids = {iv.id for iv in upcoming}
        calls = {r["interview_id"]: r["n"] for r in self.db.all(
            "SELECT interview_id, COUNT(*) AS n FROM calls WHERE interview_id IS NOT NULL GROUP BY interview_id")}
        past = sorted((iv for iv in self._interviews() if iv.id not in up_ids), key=lambda iv: -iv.when)[:200]
        for iv in past:
            iv.calls = calls.get(iv.id, 0)  # type: ignore[attr-defined]
        funnel: dict[str, int] = {}
        for iv in self._interviews():
            funnel[iv.status] = funnel.get(iv.status, 0) + 1
        return {
            "upcoming": upcoming, "past": past, "funnel": funnel,
            "companies": len({iv.company_key for iv in self._interviews()}),
            "summary": self.db.get_meta("rejections_summary") or "",
            "summary_at": float(self.db.get_meta("rejections_summary_at") or 0),
        }

    def interview(self, iid: int) -> Interview | None:
        row = self.db.one("SELECT * FROM interviews WHERE id = ?", [iid])
        return Interview.from_row(row) if row else None

    def interview_page(self, iid: int) -> dict[str, Any] | None:
        row = self.db.one("SELECT * FROM interviews WHERE id = ?", [iid])
        if not row:
            return None
        iv = Interview.from_row(row)
        key = iv.company_key
        return {
            "iv": iv,
            "calls": self.db.all("""SELECT * FROM calls WHERE app IS NOT ? AND (interview_id = ? OR lower(trim(company)) = ?)
                                    ORDER BY started_at""", [TRAINING_APP, iid, key]),
            "others": [Interview.from_row(r) for r in self.db.all(
                """SELECT * FROM interviews WHERE lower(trim(company)) = ? AND id != ?
                   ORDER BY COALESCE(scheduled_at, created_at)""", [key, iid])],
            "legend": self.db.all("SELECT * FROM legend WHERE lower(trim(company)) = ? ORDER BY created_at",
                                  [norm_company(iv.company)]),
        }

    # --- тренер ---

    def trainer_page(self, session_id: int | None) -> dict[str, Any]:
        return {
            "items": {k: self.db.all(f"""SELECT * FROM trainer_items WHERE kind = ? AND {CURATED}
                                         ORDER BY due_at ASC, score ASC LIMIT 60""", [k])
                      for k in ("knowledge", "english")},
            "sessions": self.db.all("SELECT * FROM trainer_sessions ORDER BY started_at DESC LIMIT 20"),
            "turns": self.db.all("SELECT * FROM trainer_turns WHERE session_id = ? ORDER BY id", [session_id])
            if session_id else [],
        }

    def trainer_session(self, sid: int) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
        return (self.db.one("SELECT * FROM trainer_sessions WHERE id = ?", [sid]),
                self.db.all("SELECT * FROM trainer_turns WHERE session_id = ? ORDER BY id", [sid]))

    # --- текстовый режим ---

    def past_text_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.db.all("""SELECT id, title, started_at FROM calls WHERE app = ? AND title LIKE 'Текст%'
                              ORDER BY started_at DESC LIMIT ?""", [TRAINING_APP, limit])

    # --- данные о рынке для подбора тем ---

    def clusters(self) -> list[dict[str, Any]]:
        return self.db.all("SELECT name, description FROM clusters ORDER BY id")

    def recent_vacancies(self, limit: int) -> list[dict[str, Any]]:
        return self.db.all("SELECT title, text FROM vacancies WHERE status != 'hidden' ORDER BY found_at DESC LIMIT ?",
                           [limit])
