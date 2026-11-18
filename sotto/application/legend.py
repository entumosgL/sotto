"""Легенда в звонках: что суфлёр помнит, куда переезжает придуманное и когда оно удаляется.
Правила — в domain/legend.py; здесь — только загрузка и сохранение."""

from __future__ import annotations

import logging
import time

from ..domain import interviews as iv_rules
from ..domain import legend as rules
from ..domain.legend import LegendItem
from .ports import CallRepository, InterviewRepository, LegendRepository

log = logging.getLogger(__name__)


class LegendService:
    def __init__(self, calls: CallRepository, legend: LegendRepository, interviews: InterviewRepository):
        self.calls = calls
        self.legend = legend
        self.interviews = interviews

    def for_call(self, call_id: int | None) -> list[LegendItem]:
        call = self.calls.get(call_id) if call_id is not None else None
        return rules.visible_in(call, self.legend.all())

    def remember(self, call_id: int | None, answer_id: int, hint: str) -> list[str]:
        """Новые придуманные факты из подсказки. На тренировках не сохраняются."""
        claims = rules.claims(hint)
        call = self.calls.get(call_id) if call_id is not None else None
        if call is None or call.is_training:
            return claims
        for claim in rules.new_claims(hint, self.for_call(call_id)):
            self.legend.add(claim, call_id=call.id, answer_id=answer_id, company=call.company)
        return claims

    # --- правки с сайта ---

    def add(self, claim: str, note: str = "", company: str = "") -> None:
        claim = " ".join(claim.split())
        if claim:
            self.legend.add(claim, note=note.strip() or None, company=company.strip() or None)

    def edit(self, item_id: int, claim: str, note: str, active: bool) -> None:
        item = self.legend.get(item_id)
        if item is not None:
            item.edit(claim, note, active)
            self.legend.save(item)

    def delete(self, item_id: int) -> None:
        self.legend.delete([item_id])

    def forget_company(self, company: str) -> None:
        self.legend.delete_company(company)

    def assign_company(self, call_id: int, company: str | None, interview_id: int | None = None) -> None:
        """Компания звонка; придуманное в нём переезжает к этой компании."""
        self.calls.set_company(call_id, company, interview_id)
        self.legend.move_call(call_id, company)

    def cleanup(self, now: float | None = None) -> int:
        items = self.legend.all()
        calls = self.calls.by_ids(it.call_id for it in items if it.call_id is not None)
        closed = iv_rules.closed_companies(self.interviews.all())
        ids = rules.expired(items, calls, closed, now or time.time())
        self.legend.delete(ids)
        if ids:
            log.info("Удалено устаревшей легенды: %s", len(ids))
        return len(ids)
