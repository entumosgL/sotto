"""Где Sotto хранит файлы (по XDG; на Windows — %APPDATA% и %LOCALAPPDATA%)."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _xdg(var: str, default: str) -> Path:
    if os.environ.get(var):
        return Path(os.environ[var])
    if sys.platform == "win32":
        appdata = "APPDATA" if var == "XDG_CONFIG_HOME" else "LOCALAPPDATA"
        return Path(os.environ.get(appdata) or Path.home() / "AppData/Roaming")
    return Path.home() / default


CONFIG_DIR = _xdg("XDG_CONFIG_HOME", ".config") / "sotto"
DATA_DIR = _xdg("XDG_DATA_HOME", ".local/share") / "sotto"
STATE_DIR = _xdg("XDG_STATE_HOME", ".local/state") / "sotto"

CONFIG_FILE = CONFIG_DIR / "config.toml"
PROFILE_FILE = CONFIG_DIR / "profile.md"     # кто ты: опыт, стек, цели — основа для ответов
PROMPTS_DIR = CONFIG_DIR / "prompts"         # промпты, написанные агентами (перекрывают встроенные)
DB_FILE = DATA_DIR / "sotto.db"
MODELS_DIR = DATA_DIR / "models"
RECORDINGS_DIR = DATA_DIR / "recordings"
SCREENSHOTS_DIR = DATA_DIR / "screenshots"
JOBS_DIR = DATA_DIR / "jobs"                 # сессии Telegram/браузера, PDF резюме
LOG_FILE = STATE_DIR / "sotto.log"

PACKAGE_DIR = Path(__file__).resolve().parent
BUILTIN_PROMPTS_DIR = PACKAGE_DIR / "prompts"


def ensure_dirs() -> None:
    for d in (CONFIG_DIR, DATA_DIR, STATE_DIR, PROMPTS_DIR, MODELS_DIR,
              RECORDINGS_DIR, SCREENSHOTS_DIR, JOBS_DIR):
        d.mkdir(parents=True, exist_ok=True)
