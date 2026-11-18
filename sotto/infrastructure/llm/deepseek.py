"""DeepSeek напрямую (platform.deepseek.com): те же модели, что через OpenRouter, но без прослойки
и дешевле за счёт собственного кэша контекста (prompt_cache_hit_tokens).

Только текст: распознавания речи и картинок у DeepSeek нет — STT и vision остаются на OpenRouter
или локально. Баланса API не отдаёт (только /user/balance без истории) — стоимость подсказок считаем
по прайс-таблице ниже.
"""

from __future__ import annotations

from typing import Any

from .chatapi import ChatAPI, ChatAPIError

__all__ = ["DeepSeek", "DeepSeekError"]


class DeepSeekError(ChatAPIError):
    pass


class DeepSeek(ChatAPI):
    name = "DeepSeek"

    # $/1M токенов: (ввод без кэша, ввод из кэша, вывод). Цены platform.deepseek.com, ноябрь 2025.
    PRICES = {
        "deepseek-chat": (0.28, 0.028, 0.42),
        "deepseek-reasoner": (0.28, 0.028, 0.42),
    }

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1"):
        super().__init__(api_key, base_url)

    def _cost(self, model: str, usage: dict[str, Any]) -> float | None:
        prices = self.PRICES.get(model)
        if not prices or not usage:
            return None
        price_in, price_cached, price_out = prices
        hit = usage.get("prompt_cache_hit_tokens") or 0
        miss = (usage.get("prompt_tokens") or 0) - hit
        out = usage.get("completion_tokens") or 0
        return (miss * price_in + hit * price_cached + out * price_out) / 1_000_000
