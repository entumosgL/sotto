"""Легенда — опыт, который суфлёр придумал (в подсказке он обёрнут в [[…]]) или который ты добавил сам.

Правила:
  • придуманное держится в пределах одной компании: в звонке видна легенда этого звонка, этой компании
    и общая (добавленная вручную без компании). В другие компании не переносится;
  • на тренировках легенды нет вовсе — ни старой, ни новой;
  • удаляется сама: с тренировок; из звонков, где компания так и не определилась (через сутки);
    у компаний, процесс с которыми закрыт (отказ, ты отказался, отменено).
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .calls import Call
from .companies import norm_company
from .errors import DomainError

MARK = re.compile(r"\[\[(.+?)\]\]", re.S)
UNKNOWN_COMPANY_TTL = 86400.0


@dataclass
class LegendItem:
    id: int
    claim: str
    created_at: float
    note: str | None = None
    call_id: int | None = None
    answer_id: int | None = None
    company: str | None = None
    active: int = 1

    @property
    def is_manual(self) -> bool:
        return self.call_id is None

    @property
    def company_key(self) -> str:
        return norm_company(self.company)

    def edit(self, claim: str, note: str | None, active: bool) -> None:
        claim = " ".join(claim.split())
        if not claim:
            raise DomainError("Пустой пункт легенды")
        self.claim, self.note, self.active = claim, (note or "").strip() or None, int(active)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> LegendItem:
        return cls(**{k: row.get(k) for k in cls.__dataclass_fields__ if k in row})  # type: ignore[arg-type]


def visible_in(call: Call | None, items: Iterable[LegendItem]) -> list[LegendItem]:
    """Легенда, которой суфлёр держится в звонке."""
    if call is not None and call.is_training:
        return []
    company = norm_company(call.company) if call else ""
    call_id = call.id if call else None
    out = []
    for it in items:
        if not it.active:
            continue
        if (call_id is not None and it.call_id == call_id) \
                or (it.is_manual and not it.company_key) \
                or (company and it.company_key == company):
            out.append(it)
    return sorted(out, key=lambda it: it.created_at)


def expired(items: Iterable[LegendItem], calls: dict[int, Call], closed_companies: set[str],
            now: float) -> list[int]:
    """Пункты, которые больше не пригодятся (см. правила в начале модуля)."""
    out = []
    for it in items:
        call = calls.get(it.call_id) if it.call_id is not None else None
        if call is not None and call.is_training:
            out.append(it.id)
        elif it.call_id is not None and not it.company_key and now - it.created_at > UNKNOWN_COMPANY_TTL:
            out.append(it.id)
        elif it.company_key and it.company_key in closed_companies:
            out.append(it.id)
    return out


def claims(text: str) -> list[str]:
    """Придуманные факты из подсказки — всё, что в [[двойных скобках]], без повторов."""
    seen, out = set(), []
    for m in MARK.finditer(text):
        claim = " ".join(m.group(1).split())
        if claim and claim.lower() not in seen:
            seen.add(claim.lower())
            out.append(claim)
    return out


def new_claims(text: str, known: Iterable[LegendItem]) -> list[str]:
    """Факты из подсказки, которых в легенде ещё нет."""
    have = {it.claim.lower() for it in known}
    return [c for c in claims(text) if c.lower() not in have]


def strip_marks(text: str) -> str:
    """Подсказка так, как её произносят: без [[ ]]."""
    return text.replace("[[", "").replace("]]", "")
