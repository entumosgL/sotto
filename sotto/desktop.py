"""Рабочий стол: окна, хоткеи, скриншоты, «сцена», блокировка экрана, буфер обмена.

Одна точка входа для всего, что зависит от операционной системы:

* Linux → `hypr.py` (Hyprland, PipeWire, wf-recorder, grim/slurp, wl-clipboard);
* Windows 11 → `winsys.py` (Win32, WASAPI, ffmpeg+ddagrab, mss, RegisterHotKey).

Демон, движок подсказок и оверлей знают только этот модуль, поэтому «как именно» их не касается.
Оверлей на обеих системах не попадает в демонстрацию экрана: в Hyprland — правило
`no_screen_share`, в Windows — `SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)`.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

WINDOWS = sys.platform == "win32"
MACOS = sys.platform == "darwin"

OVERLAY_CLASS = "sotto-overlay"

if WINDOWS:  # pragma: no cover - выбирается на Windows
    from . import winsys as _os
else:  # pragma: no cover - выбирается на Linux
    from . import hypr as _os


# --- доступность окружения ---

def available() -> bool:
    """Можно ли управлять окнами (Hyprland запущен / мы вообще на Windows)."""
    if WINDOWS:
        return True
    return bool(_os.available())


def missing_requirement() -> str:
    """Чего не хватает для оверлея и хоткеев — текстом для человека."""
    if WINDOWS:
        if not _os.supports_capture_exclusion():
            return ("Windows старше 10 2004: система не умеет исключать окно из захвата экрана. "
                    "Оверлей показывать нельзя — обнови Windows")
        return ""
    return "Hyprland не найден (нет HYPRLAND_INSTANCE_SIGNATURE?)"


# --- окно оверлея ---

def apply_overlay_rules(x: int, y: int, w: int, h: int, opacity: float) -> bool:
    """Задать оверлею размер/положение и исключить его из демонстрации экрана.
    True — правило принято системой; иначе оверлей показывать нельзя."""
    if WINDOWS:
        return _os.supports_capture_exclusion()
    return _os.apply_overlay_rules(x, y, w, h, opacity)


def find_overlay(pid: int | None = None) -> Any:
    """Окно оверлея: адрес в Hyprland или HWND в Windows."""
    if WINDOWS:
        windows = _os.windows_of_process(pid) if pid else []
        if not windows:  # venv-питон запускает настоящий интерпретатор отдельным процессом
            marked = [h for h in _os.windows_marked_as_overlay()]
            windows = marked
        return windows[0] if windows else None
    return _os.overlay_address()


def mark_overlay(handle: Any) -> None:
    """Windows: пометить своё окно, чтобы демон нашёл его и скрыл из демонстрации."""
    if WINDOWS and handle:
        _os.mark_overlay(handle)


def overlay_pids() -> list[int]:
    """pid процессов оверлея (в Linux окно живёт в нашем процессе-ребёнке — там не нужно)."""
    return _os.overlay_pids() if WINDOWS else []


def close_overlay_windows() -> None:
    """Закрыть окна оверлея по-хорошему (WM_CLOSE) перед тем, как гасить процесс."""
    if WINDOWS:
        _os.close_overlay_windows()


def overlay_hidden(handle: Any) -> bool:
    if not handle:
        return False
    return _os.overlay_hidden(handle)


def hide_overlay(handle: Any) -> None:
    if not handle:
        return
    if WINDOWS:
        _os.hide_overlay(handle)
    else:
        _os.hide_window(handle)


def overlay_geometry(handle: Any) -> tuple[int, int, int, int] | None:
    """(x, y, ширина, высота) окна оверлея."""
    if not handle:
        return None
    return _os.overlay_geometry(handle)


def no_activate(handle: Any) -> None:
    """Windows: окно не берёт фокус и не висит в Alt+Tab. В Linux это делает правило Hyprland."""
    if WINDOWS and handle:
        _os.make_no_activate(handle)


def prepare_process() -> None:
    """Windows: DPI-awareness и своё имя процесса до создания окон."""
    if WINDOWS:
        _os.set_dpi_awareness()
        _os.set_app_user_model_id(OVERLAY_CLASS)


def default_overlay_position(width: int, stage_name: str = "") -> tuple[int, int]:
    """Сверху по центру основного монитора — там камера ноутбука."""
    if WINDOWS:
        screen_w = _os.primary_monitor_size()[0]
        return max(0, (screen_w - width) // 2), 44
    return _os.default_overlay_position(width, stage_name)


def focus_window(handle: Any) -> None:
    """Вернуть фокус окну звонка после перетаскивания оверлея (Linux)."""
    if not WINDOWS and handle:
        _os.focus_window(handle)


# --- информация об окнах ---

def active_window() -> dict[str, Any]:
    try:
        return _os.active_window()
    except Exception:  # noqa: BLE001
        return {}


def focused_monitor(stage_name: str = "") -> str:
    if WINDOWS:
        return _os.focused_monitor()
    return _os.focused_monitor()


def screenshot(dest: Path, output: str | None = None, region: bool = False, scale: float = 0.75) -> Path | None:
    """Снимок экрана (или выделенной области) в JPEG. None — выделение отменили."""
    return _os.screenshot(dest, output, region, scale)


# --- хоткеи ---

_hotkeys: Any = None  # winsys.Hotkeys, живёт между перерегистрациями


def apply_binds(binds: dict[str, str], action_cmd: str,
                on_action: Callable[[str], None] | None = None) -> list[str]:
    """binds: {"SUPER + A": "ask"}. На Linux повторяются в Hyprland как `sotto ctl <action>`,
    на Windows регистрируются в процессе демона и сразу вызывают on_action."""
    if WINDOWS:
        global _hotkeys
        if _hotkeys is None:
            _hotkeys = _os.Hotkeys()
        return _hotkeys.bind(binds, on_action or (lambda _name: None))
    _os.apply_binds(binds, action_cmd)
    return [k for k in binds if k]


def remove_binds(keys: list[str]) -> None:
    if WINDOWS:
        global _hotkeys
        if _hotkeys is not None:
            _hotkeys.close()
            _hotkeys = None
        return
    _os.remove_binds(keys)


def listen_events(on_event: Callable[[str, str], None], stop: threading.Event) -> threading.Thread:
    """События оконного менеджера. В Windows слушать нечего: хоткеи приходят напрямую."""
    if WINDOWS:
        thread = threading.Thread(target=lambda: stop.wait(), name="win-events", daemon=True)
        thread.start()
        return thread
    return _os.listen_events(on_event, stop)


# --- «сцена»: виртуальный монитор для демонстрации ---

class Stage:
    """Виртуальный монитор, который показываешь в звонке (только Linux: headless-выход Hyprland).
    В Windows виртуального монитора без стороннего драйвера не создать — есть второй физический
    монитор или показ окна вместо экрана."""

    WORKSPACE = "name:stage"

    def __init__(self, name: str = "SOTTO-STAGE", mirror_on: str = ""):
        self.name = name
        self.mirror_on = mirror_on
        self._impl = None if WINDOWS else _os.Stage(name, mirror_on)

    def active(self) -> bool:
        return False if self._impl is None else self._impl.active()

    def on(self) -> str:
        if self._impl is None:
            raise RuntimeError(
                "Виртуальный монитор «сцена» есть только в Linux/Hyprland. "
                "В Windows показывай в звонке окно (а не весь экран) или заведи второй монитор.")
        return self._impl.on()

    def off(self) -> str:
        if self._impl is None:
            return "Сцена доступна только в Linux/Hyprland"
        return self._impl.off()


# --- система ---

def screen_locked() -> bool:
    return _os.screen_locked()


def microphone_users(ignore: list[str]) -> list[str]:
    """Какие приложения сейчас держат микрофон (кроме нас и ignore)."""
    return _os.microphone_users(ignore)


def copy_text(text: str) -> None:
    _os.copy_text(text)


def paste_text() -> str:
    return _os.paste_text()


def open_url(url: str) -> None:
    _os.open_url(url)


def open_path(path: str | Path, at: float = 0) -> None:
    _os.open_path(path, at)


def notify(title: str, body: str) -> None:
    """Напоминание вне панели: notify-send в Linux, всплывающее уведомление в Windows."""
    if WINDOWS:
        _os.notify(title, body)
        return
    if shutil.which("notify-send"):
        subprocess.Popen(["notify-send", "-a", "Sotto", title, body],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def autostart_enabled() -> bool:
    if WINDOWS:
        return _os.autostart_enabled()
    unit = Path.home() / ".config" / "systemd" / "user" / "sotto.service"
    return unit.exists()


def set_autostart(enabled: bool) -> bool:
    if WINDOWS:
        return _os.set_autostart(enabled)
    raise RuntimeError("автозапуск в Linux настраивается home-manager или systemd")
