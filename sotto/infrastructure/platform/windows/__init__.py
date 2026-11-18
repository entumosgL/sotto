"""Windows: заготовка платформы (порт Platform — application/ports.py).

Работает сейчас: сайт, текстовый режим, собеседования, легенда, поиск — и невидимый оверлей
(SetWindowDisplayAffinity вырезает окно из демонстрации экрана и скриншотов, Windows 10 2004+).
Звука, записи экрана, хоткеев и «сцены» пока нет: заглушки унаследованы от null-платформы,
у каждой ниже — чем её реализовать. Карта замен Linux → Windows:

  звук (PipeWire)      — WASAPI loopback: пакет soundcard (get_microphone(..., include_loopback=True))
                         или pyaudiowpatch; отдавать те же куски float32 16 кГц, что ждёт listener
  кто держит микрофон  — сессии Core Audio: pycaw (AudioUtilities.GetAllSessions, session.Process.name())
  скриншот             — mss (экран, монитор, регион) или QScreen.grabWindow
  запись экрана        — ffmpeg -f ddagrab -framerate FPS (-c:v h264_nvenc / hevc_amf / libx265)
  хоткеи               — RegisterHotKey + цикл GetMessageW в своём потоке (заготовка в WindowsWM.bind_hotkeys)
  уведомления          — win10toast или winrt (Windows.UI.Notifications)
  «сцена»              — виртуального монитора из коробки нет (нужен IddCx-драйвер); проще: показывать
                         в звонке окно, а не экран — оверлей и так вырезан из захвата

Модуль импортируется только на Windows (platform.detect), поэтому ctypes.windll трогаем только внутри
функций: так файл можно читать и импортировать на любой ОС.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import logging
from typing import Any

from ..null import NullPlatform, NullWM

log = logging.getLogger(__name__)

_WDA_EXCLUDEFROMCAPTURE = 0x00000011   # Windows 10 2004+: окна нет ни в демонстрации, ни в скриншотах
_OVERLAY_TITLE = "Sotto"               # заголовок окна оверлея (interfaces/overlay.py)


def _user32() -> Any:
    return ctypes.windll.user32


def _find_window(title: str) -> int | None:
    """Первый видимый hwnd с точным заголовком."""
    user32 = _user32()
    found: list[int] = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def each(hwnd: int, _: int) -> bool:
        if user32.IsWindowVisible(hwnd) and user32.GetWindowTextLengthW(hwnd):
            buf = ctypes.create_unicode_buffer(user32.GetWindowTextLengthW(hwnd) + 1)
            user32.GetWindowTextW(hwnd, buf, len(buf))
            if buf.value == title:
                found.append(hwnd)
        return True

    user32.EnumWindows(each, 0)
    return found[0] if found else None


class WindowsWM(NullWM):
    """Окна: невидимость оверлея и его геометрия — настоящие; хоткеи и «сцена» — TODO."""

    def available(self) -> bool:
        return True  # скрыть оверлей из демонстрации умеем — показывать его можно

    def overlay_window(self) -> str | None:
        hwnd = _find_window(_OVERLAY_TITLE)
        return str(hwnd) if hwnd else None

    def protect(self, window: str) -> None:
        if not _user32().SetWindowDisplayAffinity(int(window), _WDA_EXCLUDEFROMCAPTURE):
            log.warning("SetWindowDisplayAffinity не сработал (нужна Windows 10 2004+)")

    def is_protected(self, window: str) -> bool:
        value = wt.DWORD(0)
        user32 = _user32()
        return bool(user32.GetWindowDisplayAffinity(int(window), ctypes.byref(value))) \
            and value.value == _WDA_EXCLUDEFROMCAPTURE

    def protect_overlay(self, x: int, y: int, w: int, h: int) -> bool:
        hwnd = _find_window(_OVERLAY_TITLE)
        if not hwnd:
            return False
        self.protect(str(hwnd))
        return self.is_protected(str(hwnd))

    def overlay_geometry(self) -> tuple[int, int, int, int] | None:
        hwnd = _find_window(_OVERLAY_TITLE)
        if not hwnd:
            return None
        rect = wt.RECT()
        if not _user32().GetWindowRect(hwnd, ctypes.byref(rect)):
            return None
        return rect.left, rect.top, rect.right - rect.left, rect.bottom - rect.top

    def default_overlay_position(self, width: int) -> tuple[int, int]:
        screen_w = _user32().GetSystemMetrics(0)  # SM_CXSCREEN основного монитора
        return max(0, (screen_w - width) // 2), 24

    def focus(self, window: str) -> None:
        _user32().SetForegroundWindow(int(window))

    def bind_hotkeys(self, binds: dict[str, str], command: str) -> list[str]:
        # TODO: RegisterHotKey + цикл GetMessageW в выделенном потоке:
        #   поток: for key, action in binds: RegisterHotKey(None, id, MOD_WIN|MOD_SHIFT…, vk)
        #          while GetMessageW(msg): WM_HOTKEY → subprocess command + action
        #   модификаторы: MOD_WIN=0x8, MOD_SHIFT=0x4, MOD_CONTROL=0x2, MOD_ALT=0x1, MOD_NOREPEAT=0x4000
        #   vk буквы/цифры — ord(); парсить строки вида "SUPER + SHIFT + O" (SUPER = Win)
        # Пока возвращаем пусто — «конфликтов нет», но и клавиши не работают: управление — с сайта
        # и через `sotto ctl <действие>`.
        log.warning("Хоткеи на Windows ещё не реализованы; управляй суфлёром с сайта или через sotto ctl")
        return []


class WindowsPlatform(NullPlatform):
    name = "windows"

    def __init__(self, *_: Any, **__: Any):
        self.wm = WindowsWM()

    def clipboard(self) -> str:
        user32 = _user32()
        kernel32 = ctypes.windll.kernel32
        kernel32.GlobalLock.restype = wt.LPVOID
        cf_unicodetext = 13
        if not user32.OpenClipboard(None):
            return ""
        try:
            handle = user32.GetClipboardData(cf_unicodetext)
            if not handle:
                return ""
            ptr = kernel32.GlobalLock(handle)
            if not ptr:
                return ""
            try:
                return ctypes.wstring_at(ptr)
            finally:
                kernel32.GlobalUnlock(handle)
        finally:
            user32.CloseClipboard()

    def screen_locked(self) -> bool:
        user32 = _user32()
        desk = user32.OpenInputDesktop(0, False, 0)   # None — экран блокировки / заставка
        if desk:
            ctypes.windll.user32.CloseDesktop(desk)
        return not desk

    def active_window(self) -> dict[str, Any]:
        user32 = _user32()
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return {}
        title = ctypes.create_unicode_buffer(user32.GetWindowTextLengthW(hwnd) + 1)
        user32.GetWindowTextW(hwnd, title, len(title))
        cls = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, cls, len(cls))
        return {"class": cls.value, "title": title.value}

    def notify(self, title: str, text: str) -> None:
        # TODO: win10toast или winrt (Windows.UI.Notifications); MessageBox блокирует — не подходит
        pass

    # TODO: audio_input — WASAPI loopback (soundcard / pyaudiowpatch), куски float32 16 кГц моно
    # TODO: microphone_users — pycaw (сессии Core Audio → имена процессов)
    # TODO: play — воспроизведение numpy через sounddevice; target — устройство вывода WASAPI
    # TODO: screenshot — mss.grab монитора/региона в dest; output — номер монитора
    # TODO: video_available / record_screen — ffmpeg -f ddagrab (+ h264_nvenc / hevc_amf), сегменты как в linux/screen.py
