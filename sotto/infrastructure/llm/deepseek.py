"""DeepSeek напрямую (platform.deepseek.com): те же модели, что через OpenRouter, но без прослойки
и дешевле за счёт собственного кэша контекста (prompt_cache_hit_tokens).

Актуальные модели: `deepseek-flash` (умеет картинки — стандартные блоки image_url, тем же форматом
Sotto шлёт скриншоты) и `deepseek-v4-pro` (только текст). Распознавания речи в OpenAI-совместимом
виде нет — STT остаётся на OpenRouter или локально (свой ASR у DeepSeek есть, но с другим API).

По умолчанию у моделей включён thinking-режим (high) — для подсказок это лишняя задержка, поэтому
при reasoning=False шлём thinking: disabled. Баланса API не отдаёт — стоимость подсказок считаем
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

    # $/1M токенов в пик: (ввод без кэша, ввод из кэша, вывод). Вне пика — вдвое дешевле.
    # api-docs.deepseek.com/quick_start/pricing, ноябрь 2026.
    PRICES = {
        "deepseek-flash": (0.30, 0.006, 1.20),
        "deepseek-v4-pro": (1.32, 0.044, 3.96),
    }

    # Старые имена сняты с обслуживания — запросы обслуживает Flash и биллит по его цене.
    ALIASES = {
        "deepseek-chat": "deepseek-flash",
        "deepseek-reasoner": "deepseek-flash",
        "deepseek-v4-flash": "deepseek-flash",
        "deepseek-v4.1-flash": "deepseek-flash",
        "deepseek-v4-flash-vision-exp": "deepseek-flash",
    }

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com/v1"):
        super().__init__(api_key, base_url)

    def _model(self, model: str) -> str:
        return self.ALIASES.get(model, model)

    def _body(self, model: str, messages: list[dict], *, max_tokens: int, temperature: float,
              session_id: str | None, reasoning: bool, stream: bool,
              json_mode: bool = False) -> dict[str, Any]:
        body = super()._body(self._model(model), messages, max_tokens=max_tokens, temperature=temperature,
                             session_id=session_id, reasoning=reasoning, stream=stream, json_mode=json_mode)
        # thinking включён по умолчанию (усилие high) — подсказкам нужна скорость, а не размышления
        body["thinking"] = {"type": "enabled" if reasoning else "disabled"}
        return body

    def _cost(self, model: str, usage: dict[str, Any]) -> float | None:
        prices = self.PRICES.get(self._model(model))
        if not prices or not usage:
            return None
        price_in, price_cached, price_out = prices
        hit = usage.get("prompt_cache_hit_tokens") or 0
        miss = (usage.get("prompt_tokens") or 0) - hit
        out = usage.get("completion_tokens") or 0
        return (miss * price_in + hit * price_cached + out * price_out) / 1_000_000
