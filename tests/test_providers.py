"""Провайдеры моделей: маршрутизация по префиксу, тело запроса DeepSeek, стоимость по прайсу."""

import pytest

from sotto.config import Config
from sotto.infrastructure.llm.chatapi import ChatAPIError
from sotto.infrastructure.llm.deepseek import DeepSeek
from sotto.infrastructure.llm.openrouter import OpenRouter
from sotto.infrastructure.llm.router import RoutedLLM


def cfg_with_deepseek(key: str = "sk-test") -> Config:
    cfg = Config()
    cfg.providers.deepseek.api_key = key
    return cfg


def test_route_to_deepseek_when_key_set():
    llm = RoutedLLM(cfg_with_deepseek())
    client, model = llm._route("deepseek/deepseek-chat")
    assert isinstance(client, DeepSeek) and model == "deepseek-chat"


def test_route_falls_back_to_openrouter_without_key():
    llm = RoutedLLM(Config())
    client, model = llm._route("deepseek/deepseek-chat")
    assert isinstance(client, OpenRouter) and model == "deepseek/deepseek-chat"


def test_route_openrouter_models_untouched():
    llm = RoutedLLM(cfg_with_deepseek())
    client, model = llm._route("openai/whisper-large-v3-turbo")
    assert isinstance(client, OpenRouter) and model == "openai/whisper-large-v3-turbo"


def test_configured():
    assert not RoutedLLM(Config()).configured
    assert RoutedLLM(cfg_with_deepseek()).configured


def test_reconfigure_adds_and_drops_provider():
    cfg = Config()
    llm = RoutedLLM(cfg)
    assert "deepseek" not in llm.providers
    cfg.providers.deepseek.api_key = "sk-test"
    llm.reconfigure(cfg)
    assert isinstance(llm._route("deepseek/deepseek-chat")[0], DeepSeek)
    cfg.providers.deepseek.api_key = ""
    llm.reconfigure(cfg)
    assert isinstance(llm._route("deepseek/deepseek-chat")[0], OpenRouter)


def test_deepseek_body_has_no_openrouter_fields():
    ds = DeepSeek("sk-test")
    body = ds._body("deepseek-flash", [{"role": "user", "content": "hi"}], max_tokens=100, temperature=0.4,
                    session_id="s1", reasoning=False, stream=True)
    assert body["model"] == "deepseek-flash"
    for field in ("provider", "session_id", "reasoning", "usage"):
        assert field not in body
    assert body["stream_options"] == {"include_usage": True}


def test_deepseek_thinking_disabled_for_hints():
    ds = DeepSeek("sk-test")
    kw = dict(max_tokens=100, temperature=0.4, session_id=None, stream=False)
    off = ds._body("deepseek-flash", [], reasoning=False, **kw)
    on = ds._body("deepseek-flash", [], reasoning=True, **kw)
    assert off["thinking"] == {"type": "disabled"}   # thinking по умолчанию включён — гасим ради задержки
    assert on["thinking"] == {"type": "enabled"}


def test_deepseek_legacy_names_map_to_flash():
    ds = DeepSeek("sk-test")
    body = ds._body("deepseek-v4.1-flash", [], max_tokens=100, temperature=0.4,
                    session_id=None, reasoning=False, stream=False)
    assert body["model"] == "deepseek-flash"
    assert ds._model("deepseek-chat") == "deepseek-flash"
    assert ds._model("deepseek-v4-pro") == "deepseek-v4-pro"


def test_deepseek_cost_uses_cache_prices():
    ds = DeepSeek("sk-test")
    usage = {"prompt_tokens": 10_000, "prompt_cache_hit_tokens": 8_000, "completion_tokens": 1_000}
    cost = ds._cost("deepseek-flash", usage)
    expected = (2_000 * 0.30 + 8_000 * 0.006 + 1_000 * 1.20) / 1_000_000
    assert cost == pytest.approx(expected)
    assert ds._cost("deepseek-chat", usage) == pytest.approx(expected)  # старое имя биллится как Flash
    assert ds._cost("unknown-model", usage) is None
    assert ds._cost("deepseek-flash", {}) is None


async def test_deepseek_cannot_transcribe():
    import numpy as np
    with pytest.raises(ChatAPIError, match="не умеет распознавать"):
        await DeepSeek("sk-test").transcribe("deepseek-chat", np.zeros(16000, dtype=np.float32))


async def test_unconfigured_provider_raises():
    with pytest.raises(ChatAPIError, match="Не задан ключ"):
        await DeepSeek("").chat("deepseek-chat", [{"role": "user", "content": "hi"}])
