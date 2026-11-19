"""Где Sotto хранит файлы: XDG на Linux, стандартные каталоги Windows на Windows.

Linux:   ~/.config/sotto, ~/.local/share/sotto, ~/.local/state/sotto
Windows: %APPDATA%\\sotto (настройки), %LOCALAPPDATA%\\sotto (данные), %LOCALAPPDATA%\\sotto\\state (лог)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

WINDOWS = sys.platform == "win32"


def _xdg(var: str, default: str) -> Path:
    return Path(os.environ.get(var) or Path.home() / default)


def _windows_dirs() -> tuple[Path, Path, Path]:
    roaming = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    local = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    config = Path(os.environ.get("XDG_CONFIG_HOME") or roaming) / "sotto"
    data = Path(os.environ.get("XDG_DATA_HOME") or local) / "sotto"
    state = Path(os.environ.get("XDG_STATE_HOME") or local / "state") / "sotto"
    return config, data, state


if WINDOWS:
    CONFIG_DIR, DATA_DIR, STATE_DIR = _windows_dirs()
else:
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


def short(path: Path) -> str:
    """Путь для показа человеку: «~/.config/sotto/config.toml» или «%APPDATA%\\sotto\\config.toml»."""
    if WINDOWS:
        for var, folder in (("APPDATA", Path(os.environ.get("APPDATA") or "")),
                            ("LOCALAPPDATA", Path(os.environ.get("LOCALAPPDATA") or ""))):
            try:
                rel = Path(path).relative_to(folder)
            except (ValueError, OSError):
                continue
            return f"%{var}%\\" + str(rel).replace("/", "\\")
        return str(path)
    try:
        return "~/" + str(Path(path).relative_to(Path.home()))
    except (ValueError, OSError):
        return str(path)
