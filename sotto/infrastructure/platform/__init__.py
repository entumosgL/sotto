"""Платформы — всё, что зависит от ОС (порт Platform в application/ports.py).

  linux/   — Linux + PipeWire + Hyprland (Wayland)
  windows/ — заготовка для Windows: невидимый оверлей есть, звука/записи/хоткеев пока нет
  null     — без рабочего стола: тесты и пока не поддержанные ОС

Как добавить ОС: создать каталог с классом, у которого те же методы, что у NullPlatform, и выбрать
его в detect(). Остальной код не меняется. Начать удобно с копии windows/ или null.py.
"""

from __future__ import annotations

import os
import shutil
import sys
from typing import Any

_current: Any = None


def detect(stage_name: str = "SOTTO-STAGE", mirror_on: str = "") -> Any:
    """Платформа этой машины. SOTTO_PLATFORM=null — принудительно без рабочего стола."""
    forced = os.environ.get("SOTTO_PLATFORM", "")
    if forced != "null":
        if sys.platform.startswith("linux") and shutil.which("pw-record"):
            from .linux import LinuxPlatform
            return LinuxPlatform(stage_name, mirror_on)
        if sys.platform == "win32":
            from .windows import WindowsPlatform
            return WindowsPlatform()
    from .null import NullPlatform
    return NullPlatform()


def current() -> Any:
    global _current
    if _current is None:
        _current = detect()
    return _current


def use(platform: Any) -> None:
    """Задать платформу процесса (демон — из настроек сцены; тесты — заглушку)."""
    global _current
    _current = platform
