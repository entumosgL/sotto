"""`sotto setup` — первичная настройка в терминале: ключ, модели, профиль, промпты, автозапуск."""

from __future__ import annotations

import getpass
import shutil
import subprocess
import textwrap

from . import config as config_mod
from . import paths
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


async def run_setup() -> int:
    paths.ensure_dirs()
    cfg = config_mod.load()
    print("Настройка Sotto. Enter — оставить как есть.\n")

    # 1. Ключ
    if cfg.api_key:
        print("✔ Ключ OpenRouter уже задан.")
    else:
        print("Ключ OpenRouter нужен для подсказок и расшифровки (один баланс на всё).")
        print("Получить: https://openrouter.ai/keys — пополнение от $5, хватит надолго.")
        key = getpass.getpass("Ключ (sk-or-…, ввод скрыт; Enter — позже): ").strip()
        if key:
            cfg.openrouter.api_key = key
            config_mod.save(cfg)
            print("✔ Ключ сохранён в ~/.config/sotto/config.toml (права 600)")

    # 2. Локальные модели
    print("\nСкачиваю локальные модели (детектор речи, голосовые отпечатки, голоса для тренеров)…")
    models.file("silero_vad.onnx")
    models.file("wespeaker_en_voxceleb_resnet34_LM.onnx")
    for voice in (cfg.trainer.voice_ru, cfg.trainer.voice_en, "ru_RU-irina-medium"):
        models.piper_voice(voice)
    print("✔ Готово")

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
    print(textwrap.dedent(f"""
    Автозапуск:
      • home-manager: imports = [ sotto.homeManagerModules.default ]; services.sotto.enable = true;
      • или вручную:  sotto run   (панель: http://{cfg.web.host}:{cfg.web.port})
    Хоткеи регистрируются в Hyprland сами при запуске демона (SUPER+A и др.).
    Проверка и обучение: sotto sim tutorial;  подбор моделей: sotto bench --apply
    """))
    if shutil.which("systemctl") and ask("Создать systemd user-сервис sotto.service без home-manager? (y/n)", "n").lower().startswith("y"):
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
