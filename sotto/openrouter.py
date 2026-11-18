"""Клиент OpenRouter: потоковые ответы моделей и распознавание речи — один ключ, один баланс.

Задержка до первого токена — главное для подсказок, поэтому:
* рассуждение («thinking») по умолчанию выключено — оно и даёт «огромную latency» в рейтингах;
* провайдер внутри модели выбирается по задержке (`provider.sort`);
* неизменная часть промпта идёт первой и одинаковой байт-в-байт — её кэшируют провайдеры;
* `session_id` держит запросы одного звонка на одном провайдере (тёплый кэш).
"""

from __future__ import annotations

import base64
import io
import json
import time
import wave
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import httpx
import numpy as np


class OpenRouterError(RuntimeError):
    pass


@dataclass
class Completion:
    text: str = ""
    first_token_ms: int | None = None
    total_ms: int = 0
    cost: float | None = None
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)


def wav_bytes(samples: np.ndarray, sample_rate: int = 16000) -> bytes:
    """float32 [-1, 1] → WAV (16 bit mono)."""
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


class OpenRouter:
    def __init__(self, api_key: str, base_url: str = "https://openrouter.ai/api/v1",
                 provider_sort: str = "latency"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.provider_sort = provider_sort
        # Модели, у которых «размышления» нельзя выключить (например, Sonnet 5.5): им — минимальные.
        self.reasoning_required: set[str] = set()
        self.http = httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/sotto-copilot",
                "X-Title": "Sotto",
            },
            http2=False,
        )

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    async def close(self) -> None:
        await self.http.aclose()

    def _body(self, model: str, messages: list[dict], *, max_tokens: int, temperature: float,
              session_id: str | None, reasoning: bool, stream: bool,
              json_mode: bool = False) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            "provider": {"sort": self.provider_sort},
            "usage": {"include": True},
        }
        if not reasoning:
            if model in self.reasoning_required:
                body["reasoning"] = {"effort": "minimal"}
            else:
                body["reasoning"] = {"enabled": False}
        if session_id:
            body["session_id"] = session_id
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        return body

    async def stream_chat(self, model: str, messages: list[dict], *, max_tokens: int = 700,
                          temperature: float = 0.4, session_id: str | None = None,
                          reasoning: bool = False) -> AsyncIterator[tuple[str, Completion]]:
        """Отдаёт (кусок_текста, накопленный Completion). Последний кусок — пустая строка с итогом."""
        if not self.configured:
            raise OpenRouterError("Не задан ключ OpenRouter (config.toml → [openrouter].api_key или OPENROUTER_API_KEY)")
        for attempt in range(2):
            body = self._body(model, messages, max_tokens=max_tokens, temperature=temperature,
                              session_id=session_id, reasoning=reasoning, stream=True)
            result = Completion(model=model)
            start = time.perf_counter()
            try:
                async for item in self._stream(body, result, start):
                    yield item
                return
            except httpx.HTTPError as e:
                raise OpenRouterError(f"сеть: {e!r}") from e
            except OpenRouterError as e:
                if attempt == 0 and not result.text and self._learn_reasoning(model, str(e)):
                    continue
                raise

    async def _stream(self, body: dict[str, Any], result: Completion,
                      start: float) -> AsyncIterator[tuple[str, Completion]]:
        async with self.http.stream("POST", f"{self.base_url}/chat/completions", json=body) as resp:
            if resp.status_code != 200:
                raise OpenRouterError(f"{resp.status_code}: {(await resp.aread()).decode(errors='ignore')[:500]}")
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue  # ": OPENROUTER PROCESSING" и пустые строки
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                if "error" in chunk:
                    raise OpenRouterError(str(chunk["error"]))
                if chunk.get("usage"):
                    result.usage = chunk["usage"]
                    result.cost = chunk["usage"].get("cost")
                for choice in chunk.get("choices", []):
                    delta = (choice.get("delta") or {}).get("content") or ""
                    if delta:
                        if result.first_token_ms is None:
                            result.first_token_ms = int((time.perf_counter() - start) * 1000)
                        result.text += delta
                        yield delta, result
        result.total_ms = int((time.perf_counter() - start) * 1000)
        yield "", result

    async def chat(self, model: str, messages: list[dict], *, max_tokens: int = 700,
                   temperature: float = 0.2, json_mode: bool = False,
                   reasoning: bool = False) -> Completion:
        """Обычный (не потоковый) запрос — для служебных задач."""
        if not self.configured:
            raise OpenRouterError("Не задан ключ OpenRouter")
        for attempt in range(2):
            body = self._body(model, messages, max_tokens=max_tokens, temperature=temperature,
                              session_id=None, reasoning=reasoning, stream=False, json_mode=json_mode)
            start = time.perf_counter()
            try:
                resp = await self.http.post(f"{self.base_url}/chat/completions", json=body)
            except httpx.HTTPError as e:
                raise OpenRouterError(f"сеть: {e!r}") from e
            if attempt == 0 and resp.status_code == 400 and self._learn_reasoning(model, resp.text):
                continue
            break
        if resp.status_code != 200:
            raise OpenRouterError(f"{resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        text = data["choices"][0]["message"].get("content") or ""
        usage = data.get("usage") or {}
        return Completion(text=text, total_ms=int((time.perf_counter() - start) * 1000),
                          cost=usage.get("cost"), model=model, usage=usage)

    def _learn_reasoning(self, model: str, error: str) -> bool:
        """Модель отказалась работать без размышлений — запомнить и повторить с минимальными."""
        if "Reasoning is mandatory" in error and model not in self.reasoning_required:
            self.reasoning_required.add(model)
            return True
        return False

    async def transcribe(self, model: str, samples: np.ndarray, sample_rate: int = 16000,
                         language: str | None = None) -> tuple[str, str | None, float | None]:
        """Распознаёт одну реплику. Возвращает (текст, язык, стоимость)."""
        if not self.configured:
            raise OpenRouterError("Не задан ключ OpenRouter")
        body: dict[str, Any] = {
            "model": model,
            "input_audio": {"data": base64.b64encode(wav_bytes(samples, sample_rate)).decode(), "format": "wav"},
            "response_format": "verbose_json",
        }
        if language:
            body["language"] = language
        try:
            resp = await self.http.post(f"{self.base_url}/audio/transcriptions", json=body)
            if resp.status_code == 400 and "verbose_json" in resp.text:
                body["response_format"] = "json"  # не все провайдеры умеют verbose_json
                resp = await self.http.post(f"{self.base_url}/audio/transcriptions", json=body)
        except httpx.HTTPError as e:
            raise OpenRouterError(f"сеть: {e!r}") from e
        if resp.status_code != 200:
            raise OpenRouterError(f"STT {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        usage = data.get("usage") or {}
        return (data.get("text") or "").strip(), data.get("language"), usage.get("cost")

    async def models(self) -> list[dict[str, Any]]:
        resp = await self.http.get(f"{self.base_url}/models")
        resp.raise_for_status()
        return resp.json()["data"]

    async def credits(self, max_age: float = 300) -> dict[str, Any] | None:
        """Баланс. Кэш на 5 минут: главная страница не ждёт сеть при каждом открытии."""
        cached = getattr(self, "_credits", None)
        if cached and time.monotonic() - cached[0] < max_age:
            return cached[1]
        try:
            resp = await self.http.get(f"{self.base_url}/credits", timeout=3)
            data = resp.json().get("data") if resp.status_code == 200 else None
        except (httpx.HTTPError, ValueError):
            data = None
        self._credits = (time.monotonic(), data)
        return data
