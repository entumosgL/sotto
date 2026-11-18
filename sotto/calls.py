"""Определение звонка: какое-то приложение (браузер, Zoom, Telegram…) захватывает микрофон.

Опрашиваем `pw-dump` раз в 2 секунды и ищем чужие потоки Stream/Input/Audio в состоянии running.
Начало звонка — поток держится `start_after_s`, конец — потока нет `end_after_s`.
Ручной режим (хоткей) имеет приоритет над автоматикой.
"""

from __future__ import annotations

import json
import logging
import subprocess
import threading
import time
from typing import Callable

from .config import Calls as CallsCfg

log = logging.getLogger(__name__)


def microphone_users(ignore: list[str]) -> list[str]:
    """Имена приложений, которые прямо сейчас пишут звук с микрофона (кроме нас и игнор-листа)."""
    try:
        out = subprocess.run(["pw-dump"], capture_output=True, text=True, timeout=5).stdout
        objects = json.loads(out or "[]")
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return []
    ignore_l = [i.lower() for i in ignore]
    users = []
    for o in objects:
        info = o.get("info") or {}
        props = info.get("props") or {}
        if props.get("media.class") != "Stream/Input/Audio" or info.get("state") != "running":
            continue
        if props.get("stream.capture.sink"):
            continue  # запись системного звука (не микрофона) звонком не считаем
        node = str(props.get("node.name") or "")
        app = str(props.get("application.process.binary") or props.get("application.name") or node)
        if node.startswith("sotto-") or any(i in app.lower() or i in node.lower() for i in ignore_l):
            continue
        users.append(app)
    return users


class CallDetector:
    def __init__(self, cfg: CallsCfg, on_start: Callable[[str], None], on_end: Callable[[], None]):
        self.cfg = cfg  # живая ссылка: изменения в панели применяются без перезапуска
        self.on_start = on_start
        self.on_end = on_end
        self.in_call = False
        self.manual = False          # звонок включён вручную — автоматика его не закрывает
        # Звонок закончен вручную, а приложение ещё держит микрофон — не начинаем его заново,
        # пока микрофон не освободится.
        self._suppressed = False
        self._seen_since: float | None = None
        self._absent_since: float | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        threading.Thread(target=self._loop, name="call-detector", daemon=True).start()

    def stop(self) -> None:
        self._stop.set()

    def toggle_manual(self, app: str = "manual") -> bool:
        """Ручное включение/выключение звонка. Возвращает новое состояние."""
        if self.in_call:
            self.in_call, self.manual = False, False
            self._suppressed, self._seen_since, self._absent_since = True, None, None
            self.on_end()
        else:
            self.in_call, self.manual = True, True
            self.on_start(app)
        return self.in_call

    def _loop(self) -> None:
        while not self._stop.wait(2.0):
            if not self.cfg.auto_detect or self.manual:
                continue
            users = microphone_users(self.cfg.ignore_apps)
            now = time.time()
            if not users:
                self._suppressed = False
            if users and not self._suppressed:
                self._absent_since = None
                self._seen_since = self._seen_since or now
                if not self.in_call and now - self._seen_since >= self.cfg.start_after_s:
                    self.in_call = True
                    log.info("Звонок начался: %s", users)
                    self.on_start(users[0])
            elif not users:
                self._seen_since = None
                if self.in_call:
                    self._absent_since = self._absent_since or now
                    if now - self._absent_since >= self.cfg.end_after_s:
                        self.in_call = False
                        self._absent_since = None
                        log.info("Звонок закончился")
                        self.on_end()
