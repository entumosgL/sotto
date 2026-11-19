"""HTTP к своей же панели (127.0.0.1) — всегда мимо прокси.

Зачем отдельный модуль: httpx по умолчанию читает HTTP_PROXY/HTTPS_PROXY из окружения.
В корпоративной сети (и в системах, где весь трафик идёт через прокси) локальные запросы
уходили бы на прокси: демон решал бы, что «Sotto уже запущен», оверлей не видел бы панель,
а `sotto ctl` из хоткея не достучался бы до демона. Для 127.0.0.1 прокси не нужен никогда.
"""

from __future__ import annotations

import httpx

LOCAL = {"trust_env": False}  # идём напрямую, без прокси и netrc


def _opts(kwargs: dict) -> dict:
    options = dict(LOCAL)
    options.update(kwargs)
    return options


def get(url: str, **kwargs) -> httpx.Response:
    return httpx.get(url, **_opts(kwargs))


def post(url: str, **kwargs) -> httpx.Response:
    return httpx.post(url, **_opts(kwargs))


def stream(method: str, url: str, **kwargs):
    return httpx.stream(method, url, **_opts(kwargs))


def client(**kwargs) -> httpx.Client:
    return httpx.Client(**_opts(kwargs))


def async_client(**kwargs) -> httpx.AsyncClient:
    return httpx.AsyncClient(**_opts(kwargs))
