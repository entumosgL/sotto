"""Постоянная запись экрана и звука (сценарий; как именно пишется — у платформы и ffmpeg).

* Видео и звук режутся на куски: по умолчанию 30 минут и на границах звонков (во время звонка — больше
  кадров в секунду). Кусок, где процесс записи умер, заменяется новым.
* Звук демон уже захватывает (микрофон и собеседники) — он пишется в тот же кусок.
* Когда экран заблокирован — пауза.
* Удаляются записи только вручную (сайт → Записи).
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .. import paths
from ..config import Recording
from .ports import Platform, RecordingRepository

log = logging.getLogger(__name__)


class Piece(Protocol):
    started_at: float

    def write_audio(self, channel: str, samples: np.ndarray) -> None: ...
    def alive(self) -> bool: ...
    def close(self) -> Path | None: ...


class RecordingBackend(Protocol):
    def available(self) -> bool: ...
    def open(self, base: Path, fps: int, settings: Any) -> Piece: ...
    def recover(self, base: Path) -> Path | None: ...


class ScreenRecorder:
    def __init__(self, cfg: Recording, rows: RecordingRepository, backend: RecordingBackend, platform: Platform):
        self.cfg = cfg
        self.rows = rows
        self.backend = backend
        self.platform = platform
        self.segment: Piece | None = None
        self.row_id: int | None = None
        self.in_call = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._finishing: list[threading.Thread] = []

    @property
    def available(self) -> bool:
        return self.backend.available()

    def current(self) -> int | None:
        """Строка записи, которая пишется сейчас (её нельзя удалять)."""
        return self.row_id if self.segment is not None else None

    def start(self) -> None:
        if not self.available:
            log.warning("Запись экрана недоступна на этой системе")
            return
        threading.Thread(target=self._loop, name="recorder", daemon=True).start()

    def stop(self) -> None:
        """Останавливает и склеивает текущий кусок синхронно — иначе процесс выйдет раньше.
        Склейки, начатые раньше в фоне, тоже дожидаемся: иначе от куска остаются обрывки."""
        self._stop.set()
        self._rotate(open_new=False, wait=True)
        for t in list(self._finishing):
            t.join(timeout=60)

    def recover(self) -> None:
        """Куски, оставшиеся после сбоя или выключения питания: склеить и закрыть строки в базе."""
        for row in self.rows.unfinished():
            if self.segment is not None and row["id"] == self.row_id:
                continue
            final = self.backend.recover(Path(row["path"]).with_suffix(""))
            if final is None or not final.exists():
                self.rows.delete(row["id"])
                continue
            self.rows.close(row["id"], str(final), final.stat().st_mtime, final.stat().st_size)

    def set_call(self, in_call: bool) -> None:
        """Граница звонка — новый кусок с другим fps."""
        self.in_call = in_call
        if self.segment is not None and not self._stop.is_set():
            self._rotate(open_new=True)

    def write_audio(self, channel: str, samples: np.ndarray) -> None:
        seg = self.segment
        if seg is not None:
            seg.write_audio(channel, samples)

    def _open(self) -> None:
        now = datetime.now()
        folder = paths.RECORDINGS_DIR / now.strftime("%Y-%m-%d")
        folder.mkdir(parents=True, exist_ok=True)
        base = folder / now.strftime("%H%M%S")
        fps = self.cfg.fps_call if self.in_call else self.cfg.fps_idle
        self.segment = self.backend.open(base, fps, self.cfg)
        self.row_id = self.rows.open(str(base.with_suffix(".mkv")), self.segment.started_at, fps)

    def _rotate(self, open_new: bool, wait: bool = False) -> None:
        with self._lock:
            old, row_id = self.segment, self.row_id
            self.segment = None
            if open_new:
                self._open()
        if old is not None and row_id is not None:
            if wait:
                self._finish(old, row_id)
            else:  # склейка может занять секунды — не держим ею поток записи
                t = threading.Thread(target=self._finish, args=(old, row_id), daemon=True)
                self._finishing = [x for x in self._finishing if x.is_alive()] + [t]
                t.start()

    def _finish(self, seg: Piece, row_id: int) -> None:
        final = seg.close()
        if final is None:
            self.rows.delete(row_id)
            return
        self.rows.close(row_id, str(final), time.time(), final.stat().st_size)

    def _loop(self) -> None:
        self.recover()
        while not self._stop.is_set():
            locked = self.cfg.pause_when_locked and self.platform.screen_locked()
            seg = self.segment
            if not self.cfg.enabled or locked:  # выключено в настройках или экран заблокирован
                if seg is not None:
                    self._rotate(open_new=False)
                self._stop.wait(5.0)
                continue
            too_old = seg is not None and time.time() - seg.started_at > self.cfg.segment_minutes * 60
            died = seg is not None and not seg.alive()
            if seg is None or too_old or died:
                self._rotate(open_new=True)
            self._stop.wait(5.0)
