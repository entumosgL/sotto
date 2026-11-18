"""Собеседование — один этап у одной компании (HR, техническое, финальное…).

Звонок, начавшийся в окне собеседования (за 30 минут до и до полутора часов после начала), относится
к нему: суфлёр получает бриф, легенда — компанию. Процесс с компанией закрыт, когда её последнее
собеседование закончилось отказом, твоим отказом или отменено.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .companies import norm_company, same_company
from .errors import DomainError


def _clean(value: str | None) -> str | None:
    return (value or "").strip() or None


class Status(StrEnum):
    PLANNED = "planned"
    DONE = "done"
    PASSED = "passed"
    OFFER = "offer"
    REJECTED = "rejected"
    DECLINED = "declined"
    CANCELLED = "cancelled"


LABELS = {
    Status.PLANNED: "запланировано",
    Status.DONE: "прошло, жду ответа",
    Status.PASSED: "позвали дальше",
    Status.OFFER: "оффер",
    Status.REJECTED: "отказ",
    Status.DECLINED: "я отказался",
    Status.CANCELLED: "отменено",
}
STATUSES = {s.value: label for s, label in LABELS.items()}
CLOSED = frozenset({Status.REJECTED, Status.DECLINED, Status.CANCELLED})
OPEN = frozenset({Status.PLANNED, Status.DONE, Status.PASSED})

BEFORE = 30 * 60      # звонок за полчаса до собеседования — уже он
AFTER = 90 * 60       # и до полутора часов после начала
REMIND = 15 * 60      # напоминание за 15 минут


@dataclass
class Interview:
    id: int
    company: str
    created_at: float
    role: str | None = None
    stage: str | None = None
    scheduled_at: float | None = None
    duration_min: int = 60
    link: str | None = None
    vacancy: str | None = None
    brief: str | None = None
    notes: str | None = None
    status: str = Status.PLANNED
    outcome: str | None = None
    analysis: str | None = None
    reminded: int = 0
    updated_at: float | None = None

    @property
    def company_key(self) -> str:
        return norm_company(self.company)

    @property
    def is_planned(self) -> bool:
        return self.status == Status.PLANNED

    @property
    def is_closed(self) -> bool:
        return self.status in CLOSED

    @property
    def when(self) -> float:
        return self.scheduled_at or self.created_at

    @property
    def status_label(self) -> str:
        return STATUSES.get(self.status, self.status)

    # --- поведение ---

    def edit(self, *, company: str, role: str | None, stage: str | None, scheduled_at: float | None,
             duration_min: int, link: str | None, vacancy: str | None, brief: str | None, notes: str | None) -> bool:
        """Правка формы. Возвращает True, если сменилась компания (звонки и легенда переезжают за ней)."""
        company = company.strip()
        if not company:
            raise DomainError("Нужна компания")
        renamed = norm_company(company) != self.company_key
        if scheduled_at != self.scheduled_at:
            self.reminded = 0  # новое время — новое напоминание
        self.company, self.role, self.stage = company, _clean(role), _clean(stage)
        self.scheduled_at, self.duration_min, self.link = scheduled_at, duration_min, _clean(link)
        self.vacancy, self.brief, self.notes = _clean(vacancy), _clean(brief), _clean(notes)
        return renamed

    def change_status(self, status: str, outcome: str | None = None) -> None:
        if not is_status(status):
            raise DomainError(f"Неизвестный статус: {status}")
        self.status = Status(status)
        if outcome is not None:
            self.outcome = _clean(outcome)

    def call_happened(self) -> None:
        """Звонок по этому собеседованию состоялся — ждём ответа."""
        if self.is_planned:
            self.status = Status.DONE

    def next_stage(self, now: float) -> Interview:
        """Позвали дальше: этот этап закрыт успехом, следующий — черновик у той же компании."""
        self.status = Status.PASSED
        return Interview(id=0, company=self.company, created_at=now, role=self.role, vacancy=self.vacancy,
                         brief=self.brief, link=self.link)

    def mark_reminded(self) -> None:
        self.reminded = 1

    @property
    def needs_brief(self) -> bool:
        return bool((self.vacancy or "").strip()) and not (self.brief or "").strip()

    def call_brief(self) -> str:
        """Что суфлёр знает о собеседовании в начале звонка."""
        head = f"Компания: {self.company}" + (f", роль: {self.role}" if self.role else "") + \
               (f", этап: {self.stage}" if self.stage else "")
        body = self.brief or (self.vacancy or "")[:3000]
        return head + ("\n\n" + body if body else "")

    def get(self, key: str, default: Any = None) -> Any:  # шаблоны и старый код обращаются как к словарю
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    @classmethod
    def new(cls, company: str, now: float, **fields: Any) -> Interview:
        iv = cls(id=0, company="?", created_at=now)
        iv.edit(company=company, role=fields.get("role"), stage=fields.get("stage"),
                scheduled_at=fields.get("scheduled_at"), duration_min=fields.get("duration_min") or 60,
                link=fields.get("link"), vacancy=fields.get("vacancy"), brief=fields.get("brief"),
                notes=fields.get("notes"))
        return iv

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Interview:
        data = {k: row[k] for k in cls.__dataclass_fields__ if k in row and row[k] is not None}
        return cls(**data)

    def to_row(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "id"}


def is_status(value: str) -> bool:
    return value in STATUSES


def for_call(interviews: Iterable[Interview], at: float) -> Interview | None:
    """Запланированное собеседование, в окно которого попадает начало звонка; ближайшее по времени."""
    fits = [iv for iv in interviews if iv.is_planned and iv.scheduled_at is not None
            and at - AFTER <= iv.scheduled_at <= at + BEFORE]
    return min(fits, key=lambda iv: abs(iv.scheduled_at - at)) if fits else None  # type: ignore[operator]


def by_company(interviews: Iterable[Interview], company: str | None, at: float) -> Interview | None:
    """Открытое собеседование с этой компанией, ближайшее ко времени звонка (компания — из итогов звонка)."""
    fits = [iv for iv in interviews if iv.status in OPEN and same_company(iv.company, company)]
    return min(fits, key=lambda iv: abs(iv.when - at)) if fits else None


def closed_companies(interviews: Iterable[Interview]) -> set[str]:
    """Компании, где последнее по времени собеседование — отказ, твой отказ или отмена."""
    latest: dict[str, Interview] = {}
    for iv in sorted(interviews, key=lambda iv: (iv.when, iv.id)):
        latest[iv.company_key] = iv
    return {c for c, iv in latest.items() if c and iv.is_closed}


def is_upcoming(iv: Interview, now: float) -> bool:
    return iv.is_planned and (iv.scheduled_at is None or iv.scheduled_at > now - AFTER)


def upcoming(interviews: Iterable[Interview], now: float, limit: int = 5) -> list[Interview]:
    """Запланированные: сначала с датой по порядку, потом без даты."""
    rows = [iv for iv in interviews if is_upcoming(iv, now)]
    rows.sort(key=lambda iv: (iv.scheduled_at is None, iv.scheduled_at or 0))
    return rows[:limit]


def due_reminders(interviews: Iterable[Interview], now: float) -> list[Interview]:
    return [iv for iv in interviews if iv.is_planned and not iv.reminded and iv.scheduled_at is not None
            and now - 60 <= iv.scheduled_at <= now + REMIND]


def reminder_text(iv: Interview, now: float) -> str:
    mins = max(0, round(((iv.scheduled_at or now) - now) / 60))
    return f"Через {mins} мин собеседование: {iv.company}" + (f" ({iv.stage})" if iv.stage else "")
