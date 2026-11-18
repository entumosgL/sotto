"""Платформы — всё, что зависит от ОС (порт Platform в application/ports.py).

  linux/  — Linux + PipeWire + Hyprland (Wayland)
  null    — без рабочего стола: тесты и пока не поддержанные ОС

Как добавить ОС (например, Windows): создать windows/ с классом, у которого те же методы, что у
NullPlatform, и выбрать его в detect(). Остальной код не меняется. Подсказки, что использовать:
  звук          — WASAPI loopback (пакет soundcard или pyaudiowpatch)
  кто держит мик — Core Audio sessions (pycaw)
  невидимость   — SetWindowDisplayAffinity(hwnd, WDA_EXCLUDEFROMCAPTURE)
  хоткеи        — RegisterHotKey
  запись экрана — ffmpeg -f ddagrab (+ h264_nvenc / hevc_amf)
  уведомления   — win10toast / Windows.UI.Notifications
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
    if forced != "null" and sys.platform.startswith("linux") and shutil.which("pw-record"):
        from .linux import LinuxPlatform
        return LinuxPlatform(stage_name, mirror_on)
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
