"""Компания — по названию, введённому руками или найденному агентом: «Acme», « acme », «ACME Inc»."""

from __future__ import annotations


def norm_company(name: str | None) -> str:
    return (name or "").strip().lower()


def same_company(a: str | None, b: str | None) -> bool:
    """Одна ли это компания: совпадение или одно название содержит другое («Acme» и «Acme Inc»)."""
    a, b = norm_company(a), norm_company(b)
    return bool(a and b) and (a == b or a in b or b in a)
