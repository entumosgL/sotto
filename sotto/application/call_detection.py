"""Определение звонка: какое-то приложение (браузер, Zoom, Telegram…) захватывает микрофон.

Кто держит микрофон, спрашиваем у платформы раз в 2 секунды. Начало звонка — поток держится `start_after_s`, конец — потока нет `end_after_s`.
Ручной режим (хоткей) имеет приоритет над автоматикой.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable

from ..config import Calls as CallsCfg

log = logging.getLogger(__name__)


class CallDetector:
    def __init__(self, cfg: CallsCfg, mic_users: Callable[[list[str]], list[str]],
                 on_start: Callable[[str], None], on_end: Callable[[], None]):
        self.cfg = cfg  # живая ссылка: изменения в панели применяются без перезапуска
        self.mic_users = mic_users  # платформа: кто сейчас пишет с микрофона
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
            users = self.mic_users(self.cfg.ignore_apps)
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
