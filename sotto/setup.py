"""`sotto setup` — первичная настройка в терминале: ключ, модели, профиль, промпты, автозапуск."""

from __future__ import annotations

import getpass
import shutil
import subprocess
import textwrap

from . import config as config_mod
from . import desktop, paths
from .agents import AGENTS, AgentError, Agents
from .audio import models
from .copilot import prompts
from .db import DB

PROFILE_TEMPLATE = """\
# Профиль кандидата

## Цель
Роль: …  Уровень: …  Формат: удалёнка/офис, город, зарплатная вилка: …

## Стек
Уверенно: …
Работал: …

## Опыт
### Компания — роль (годы)
- Проект: что делали, масштаб (RPS, данные, команда), мой вклад, результат в цифрах

## Образование

## Языки
Английский: уровень …

## Что важно в работе
"""


def ask(question: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    answer = input(f"{question}{suffix}: ").strip()
    return answer or default


def choose_provider(cfg: config_mod.Config) -> str:
    """Спросить провайдера моделей в терминале. Возвращает ключ пресета (openrouter/deepseek/custom)."""
    options = list(config_mod.PROVIDERS)
    current = cfg.openrouter.provider if cfg.openrouter.provider in options else "openrouter"
    current_option = options.index(current) + 1
    print("Кто отвечает на подсказки:")
    for index, key in enumerate(options, 1):
        mark = " (сейчас)" if key == current else ""
        print(f"  {index}) {config_mod.PROVIDERS[key].title}{mark}")
    chosen = ask("Номер провайдера", str(current_option))
    if chosen.isdigit() and 1 <= int(chosen) <= len(options):
        return options[int(chosen) - 1]
    return current


async def run_setup() -> int:
    paths.ensure_dirs()
    cfg = config_mod.load()
    print("Настройка Sotto. Enter — оставить как есть.\n")

    # 1. Провайдер и ключ
    provider = choose_provider(cfg)
    if provider != cfg.openrouter.provider:
        config_mod.apply_provider(cfg, provider)
        config_mod.save(cfg)
    preset = config_mod.provider_preset(provider)
    print(f"\n{preset.title}")
    print(f"  адрес: {preset.base_url}")
    print(f"  {preset.note}\n")

    if provider == "openrouter":
        if cfg.api_key:
            print("✔ Ключ OpenRouter уже задан.")
        else:
            print("Ключ нужен для подсказок и расшифровки (один баланс на всё).")
            print("Получить: https://openrouter.ai/keys — пополнение от $5, хватит надолго.")
            key = getpass.getpass("Ключ (sk-or-…, ввод скрыт; Enter — позже): ").strip()
            if key:
                cfg.openrouter.api_key = key
                config_mod.save(cfg)
                print(f"✔ Ключ сохранён в {paths.short(paths.CONFIG_FILE)} (доступен только тебе)")
    elif provider == "deepseek":
        print("Ключ: platform.deepseek.com → API keys. Подсказки идут через api.deepseek.com,")
        print("речь распознаётся локально на этой машине, скриншоты пока недоступны.")
        if cfg.api_key:
            print("✔ Ключ уже задан (если он от другого провайдера — замени его здесь).")
        key = getpass.getpass("Ключ DeepSeek (sk-…, ввод скрыт; Enter — позже): ").strip()
        if key:
            cfg.openrouter.api_key = key
            config_mod.save(cfg)
            print(f"✔ Ключ сохранён в {paths.short(paths.CONFIG_FILE)}")
    else:
        url = ask("Адрес сервера (OpenAI-совместимый, например http://127.0.0.1:1234/v1)", preset.base_url)
        if url:
            cfg.openrouter.base_url = url.rstrip("/")
        key = ask("Ключ (если сервер его не требует — Enter)", cfg.openrouter.api_key)
        cfg.openrouter.api_key = key
        model = ask("Имя модели для подсказок (например llama-3.3-70b или qwen2.5:14b)", cfg.models.answer)
        if model:
            cfg.models.answer = cfg.models.answer_fallback = cfg.models.utility = model
        config_mod.save(cfg)
        print(f"✔ Сохранено в {paths.short(paths.CONFIG_FILE)}")
    if provider != "openrouter":
        print("  распознавание речи: локальное (модель скачается при первом звонке)")
        print("  подсказки по скриншоту: выключены — включить можно, вписав модель со зрением в «Модели»")

    # 2. Локальные модели
    print("\nСкачиваю локальные модели (детектор речи, голосовые отпечатки, голоса для тренеров)…")
    models.file("silero_vad.onnx")
    models.file("wespeaker_en_voxceleb_resnet34_LM.onnx")
    for voice in (cfg.trainer.voice_ru, cfg.trainer.voice_en, "ru_RU-irina-medium"):
        models.piper_voice(voice)
    print("✔ Готово")
    try:  # Piper-голоса молчат на sherpa-onnx 2.x: там нет espeak-фронтенда
        from .audio import tts
        tts.synthesize("Проверка озвучки.", cfg.trainer.voice_ru)
        print("✔ Озвучка тренеров работает")
    except Exception as e:  # noqa: BLE001
        print(f"✘ Озвучка не работает: {e}")

    # 3. Агенты
    agents = Agents(DB(), cfg.agents)
    found = agents.installed()
    print(f"\nАгенты для фоновых задач: {', '.join(found) or 'не найдены'}")
    missing = [n for n in cfg.agents.order if n not in found]
    if missing:
        print(f"  не найдены: {', '.join(missing)} (можно поставить позже; известны: {', '.join(AGENTS)})")

    # 4. Профиль
    if not prompts.profile():
        paths.PROFILE_FILE.write_text(PROFILE_TEMPLATE, encoding="utf-8")
        print(f"\nСоздан шаблон профиля: {paths.PROFILE_FILE}")
        print("Заполни его (или в панели: Профиль → «Структурировать» из свободных заметок).")
    else:
        print("\n✔ Профиль заполнен")

    # 5. Промпты под профиль
    if found and prompts.profile().strip() != PROFILE_TEMPLATE.strip():
        if ask("\nПереписать основной промпт подсказок под твой профиль агентом? (y/n)", "y").lower().startswith("y"):
            prompt = prompts.render(prompts.builtin("prompt_author"), kind=prompts.KINDS["answer"],
                                    current=prompts.load("answer"), profile=prompts.profile(), wishes="(нет)")
            try:
                text, agent = await agents.run(prompt, task="промпт answer")
                prompts.save("answer", text.strip().strip("`"))
                print(f"✔ Промпт переписан ({agent}). Посмотреть/поправить: панель → Промпты")
            except AgentError as e:
                print(f"✘ {e}")

    # 6. Автозапуск
    if desktop.WINDOWS:
        hints = """\
    Автозапуск:
      • sotto autostart on   — Sotto будет стартовать при входе в систему (без консольного окна)
      • или просто запускай: sotto run   (панель: http://{url})
    Хоткеи регистрируются сами при запуске демона (SUPER+A и др., раздел [hotkeys] в config.toml).
    Проверка и обучение: sotto sim tutorial;  подбор моделей: sotto bench --apply
    """
    else:
        hints = """\
    Автозапуск:
      • home-manager: imports = [ sotto.homeManagerModules.default ]; services.sotto.enable = true;
      • или вручную:  sotto run   (панель: http://{url})
    Хоткеи регистрируются в Hyprland сами при запуске демона (SUPER+A и др.).
    Проверка и обучение: sotto sim tutorial;  подбор моделей: sotto bench --apply
    """
    print(textwrap.dedent(hints.format(url=f"{cfg.web.host}:{cfg.web.port}")))
    if not shutil.which("ffmpeg"):
        print("⚠ ffmpeg не найден: без него не будет записи экрана и звуковых дорожек. "
              "Windows: winget install Gyan.FFmpeg;  Linux: пакет ffmpeg")
    if desktop.WINDOWS:
        if not desktop.autostart_enabled() and \
                ask("Запускать Sotto при входе в систему? (y/n)", "y").lower().startswith("y"):
            desktop.set_autostart(True)
            print("✔ Автозапуск включён (реестр HKCU, без прав администратора)")
    elif shutil.which("systemctl") and ask("Создать systemd user-сервис sotto.service без home-manager? (y/n)", "n").lower().startswith("y"):
        unit = paths.CONFIG_DIR.parent / "systemd" / "user" / "sotto.service"
        unit.parent.mkdir(parents=True, exist_ok=True)
        exe = shutil.which("sotto") or "sotto"
        unit.write_text(textwrap.dedent(f"""\
            [Unit]
            Description=Sotto interview copilot
            PartOf=graphical-session.target
            After=graphical-session.target pipewire.service

            [Service]
            ExecStart={exe} run
            Restart=on-failure
            RestartSec=3

            [Install]
            WantedBy=graphical-session.target
            """), encoding="utf-8")
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)
        subprocess.run(["systemctl", "--user", "enable", "--now", "sotto.service"], check=False)
        print("✔ sotto.service включён")
    return 0
