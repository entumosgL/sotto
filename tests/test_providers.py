"""Провайдеры моделей: OpenRouter, DeepSeek, свой OpenAI-совместимый сервер.

Главное, что проверяем: у не-OpenRouter серверов из тела запроса уходят OpenRouter-специфичные
поля (иначе DeepSeek отвечает 400), и что пресет правильно меняет адрес, модели и распознавание.
"""

import pytest

from sotto import config as config_mod
from sotto.openrouter import OpenRouter, OpenRouterError


def test_default_provider_is_openrouter():
    cfg = config_mod.Config()
    assert cfg.openrouter.provider == "openrouter"
    assert cfg.openrouter.extras is True
    assert cfg.provider.cloud_stt is True
    assert cfg.provider_name == "OpenRouter"


def test_apply_deepseek_preset():
    cfg = config_mod.Config()
    assert config_mod.apply_provider(cfg, "deepseek") is True
    assert cfg.openrouter.provider == "deepseek"
    assert cfg.openrouter.base_url == "https://api.deepseek.com/v1"
    assert cfg.openrouter.extras is False
    assert cfg.models.answer == "deepseek-chat"
    assert cfg.models.answer_detailed == "deepseek-chat"
    assert cfg.models.vision == ""            # зрения у DeepSeek нет — подсказки по экрану выключены
    assert cfg.audio.stt == "local"           # облачного распознавания у DeepSeek тоже нет
    assert cfg.provider.cloud_stt is False
    assert cfg.provider_name == "DeepSeek"


def test_apply_custom_keeps_models():
    cfg = config_mod.Config()
    cfg.models.answer = "qwen2.5:14b"
    assert config_mod.apply_provider(cfg, "custom") is True
    assert cfg.openrouter.base_url.startswith("http://127.0.0.1")
    assert cfg.openrouter.extras is False
    assert cfg.models.answer == "qwen2.5:14b"  # свой сервер: имена моделей задаёт человек
    assert cfg.audio.stt == "local"


def test_switching_back_restores_openrouter_defaults():
    cfg = config_mod.Config()
    config_mod.apply_provider(cfg, "deepseek")
    config_mod.apply_provider(cfg, "openrouter")
    assert cfg.models.answer == "openai/gpt-6.1-terra"
    assert cfg.models.vision == "anthropic/claude-sonnet-5.5"
    assert cfg.audio.stt == "auto"
    assert cfg.openrouter.extras is True


def test_unknown_provider_is_ignored():
    cfg = config_mod.Config()
    assert config_mod.apply_provider(cfg, "нет-такого") is False
    assert cfg.openrouter.provider == "openrouter"


def test_openrouter_body_has_extras():
    client = OpenRouter("key")
    body = client._body("openai/gpt-6.1-terra", [], max_tokens=10, temperature=0.4,
                        session_id="sotto-call-1", reasoning=False, stream=True)
    assert body["provider"] == {"sort": "latency"}
    assert body["usage"] == {"include": True}
    assert body["session_id"] == "sotto-call-1"
    assert "reasoning" in body


def test_plain_body_has_no_openrouter_extras():
    """DeepSeek и локальные серверы: только стандартные поля OpenAI."""
    client = OpenRouter("key", "https://api.deepseek.com/v1", extras=False, label="DeepSeek",
                        cloud_stt=False)
    body = client._body("deepseek-chat", [], max_tokens=10, temperature=0.4,
                        session_id="sotto-call-1", reasoning=False, stream=True, json_mode=True)
    assert set(body) == {"model", "messages", "max_tokens", "temperature", "stream", "response_format"}


def test_json_mode_added():
    client = OpenRouter("key")
    body = client._body("m", [], max_tokens=5, temperature=0.1, session_id=None,
                        reasoning=True, stream=False, json_mode=True)
    assert body["response_format"] == {"type": "json_object"}


async def test_cloud_stt_refused_without_support():
    client = OpenRouter("key", "https://api.deepseek.com/v1", extras=False,
                        label="DeepSeek", cloud_stt=False)
    with pytest.raises(OpenRouterError, match="облачного распознавания"):
        await client.transcribe("whisper", __import__("numpy").zeros(1600, dtype="float32"))


async def test_no_credits_for_non_openrouter():
    client = OpenRouter("key", extras=False)
    assert await client.credits() is None


class _FakeLLM:
    configured = True

    def __init__(self, cloud_stt: bool):
        self.cloud_stt = cloud_stt


def test_listener_prefers_local_stt_when_cloud_is_missing():
    from sotto.copilot.listener import Listener
    from sotto.db import DB
    from sotto.events import EventBus

    cfg = config_mod.Config()
    cfg.audio.stt = "auto"  # обычно «auto» без видеокарты уходит в облако
    listener = Listener(cfg, DB(), EventBus(), _FakeLLM(cloud_stt=False), on_utterance=lambda _ch: None)
    assert listener.use_local_stt() is True  # но у DeepSeek облака нет — распознаём локально


def test_setup_menu_maps_numbers_to_providers(monkeypatch):
    import sotto.setup as setup

    cfg = config_mod.Config()
    monkeypatch.setattr(setup, "ask", lambda _q, _default="": "2")
    assert setup.choose_provider(cfg) == "deepseek"
    monkeypatch.setattr(setup, "ask", lambda _q, _default="": "3")
    assert setup.choose_provider(cfg) == "custom"
    # Enter (значение по умолчанию) и мусор — остаёмся на текущем провайдере
    monkeypatch.setattr(setup, "ask", lambda _q, default="": default)
    assert setup.choose_provider(cfg) == "openrouter"
    monkeypatch.setattr(setup, "ask", lambda _q, _default="": "99")
    assert setup.choose_provider(cfg) == "openrouter"
