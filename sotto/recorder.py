"""Постоянная запись экрана и звука.

* Видео: wf-recorder, аппаратный HEVC через VAAPI (встройка почти не тратит батарею),
  2 кадра/с вне звонка и 5 во время звонка — экран в основном статичен, этого хватает.
  ~100–150 МБ в час. Если VAAPI нет — libx265 ultrafast на CPU.
* Звук: демон уже захватывает микрофон и системный звук — пишем их в Opus (≈30 МБ/час на оба).
* Файлы режутся на куски (по умолчанию 30 мин и на границах звонков), в конце куска
  видео и звук склеиваются в один .mkv: дорожка 1 — общий звук, 2 — микрофон, 3 — собеседники.
* Когда экран заблокирован (hyprlock) — пауза.
Удаляются записи только вручную (панель → Записи).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from . import proc

from . import paths
from .config import Recording
from .db import DB

log = logging.getLogger(__name__)


def detect_codec(cfg: Recording) -> list[str]:
    """Аргументы кодека для wf-recorder."""
    if cfg.codec in ("auto", "hevc_vaapi") and Path(cfg.vaapi_device).exists():
        probe = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-vaapi_device", cfg.vaapi_device,
             "-f", "lavfi", "-i", "testsrc=size=640x360:rate=2", "-t", "0.5",
             "-vf", "format=nv12,hwupload", "-c:v", "hevc_vaapi", "-f", "null", "-"],
            capture_output=True)
        if probe.returncode == 0:
            return ["-c", "hevc_vaapi", "-d", cfg.vaapi_device, "-p", "qp=34"]
    return ["-c", "libx265", "-p", "crf=30", "-p", "preset=ultrafast"]


def screen_locked() -> bool:
    return subprocess.run(["pgrep", "-x", "hyprlock"], capture_output=True).returncode == 0


class AudioArchive:
    """Пишет поток float32 одного канала в Opus через ffmpeg."""

    def __init__(self, path: Path):
        self.path = path
        self.proc = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "s16le", "-ar", "16000", "-ac", "1",
             "-i", "-", "-c:a", "libopus", "-b:a", "24k", "-application", "voip", "-y", str(path)],
            stdin=subprocess.PIPE, **proc.CHILD)

    def write(self, samples: np.ndarray) -> None:
        if self.proc.stdin and self.proc.poll() is None:
            try:
                self.proc.stdin.write((samples * 32767).astype(np.int16).tobytes())
            except BrokenPipeError:
                pass

    def close(self) -> None:
        if self.proc.stdin:
            try:
                self.proc.stdin.close()
            except BrokenPipeError:
                pass
        self.proc.wait(timeout=30)


class Segment:
    def __init__(self, base: Path, fps: int, codec_args: list[str], output: str):
        self.base = base
        self.started_at = time.time()
        self.fps = fps
        self.video = base.with_suffix(".video.mkv")
        self.audio = {ch: AudioArchive(base.with_suffix(f".{ch}.opus")) for ch in ("me", "them")}
        cmd = ["wf-recorder", "-r", str(fps), "-D", *codec_args, "-f", str(self.video), "-y"]
        if output:
            cmd += ["-o", output]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                     stdin=subprocess.DEVNULL, **proc.CHILD)

    def close(self) -> Path | None:
        if self.proc.poll() is None:
            self.proc.send_signal(2)  # SIGINT — wf-recorder корректно закрывает файл
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        for a in self.audio.values():
            a.close()
        return self._mux()

    def _mux(self) -> Path | None:
        return mux_parts(self.base)


MIN_PART_BYTES = 2048


def mux_parts(base: Path) -> Path | None:
    """Склеивает base.video.mkv + base.me.opus + base.them.opus в base.mkv и удаляет куски."""
    final = base.with_suffix(".mkv")
    video, me, them = base.with_suffix(".video.mkv"), base.with_suffix(".me.opus"), base.with_suffix(".them.opus")
    # Кусок, открытый на долю секунды (например, при выходе), — пустой: только заголовки файлов.
    inputs = [p for p in (video, me, them) if p.exists() and p.stat().st_size > MIN_PART_BYTES]
    if not inputs:
        for p in (video, me, them):
            p.unlink(missing_ok=True)
        return final if final.exists() else None
    has_video = video in inputs
    audio_inputs = [p for p in inputs if p != video]
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    for p in inputs:
        cmd += ["-i", str(p)]
    maps: list[str] = []
    if has_video:
        maps += ["-map", "0:v", "-c:v", "copy"]
    first_audio = 1 if has_video else 0
    if len(audio_inputs) == 2:
        cmd += ["-filter_complex", f"[{first_audio}:a][{first_audio + 1}:a]amix=inputs=2:normalize=0[mix]"]
        maps += ["-map", "[mix]", "-map", f"{first_audio}:a", "-map", f"{first_audio + 1}:a",
                 "-c:a", "libopus", "-b:a", "24k",
                 "-metadata:s:a:0", "title=Все", "-metadata:s:a:1", "title=Микрофон",
                 "-metadata:s:a:2", "title=Собеседники"]
    elif audio_inputs:
        maps += ["-map", f"{first_audio}:a", "-c:a", "copy"]
    res = subprocess.run([*cmd, *maps, "-y", str(final)], capture_output=True)
    if res.returncode != 0:
        log.error("Не удалось склеить запись %s: %s", final, res.stderr[-400:])
        return video if has_video else None
    for p in inputs:
        p.unlink(missing_ok=True)
    return final


class ScreenRecorder:
    def __init__(self, cfg: Recording, db: DB):
        self.cfg = cfg
        self.db = db
        self.segment: Segment | None = None
        self.in_call = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._codec: list[str] | None = None
        self._row_id: int | None = None
        self._finishing: list[threading.Thread] = []

    @property
    def available(self) -> bool:
        return shutil.which("wf-recorder") is not None and shutil.which("ffmpeg") is not None

    def start(self) -> None:
        if not self.available:
            log.warning("Запись экрана недоступна: нет wf-recorder/ffmpeg")
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
        """Куски, оставшиеся после сбоя/выключения питания: склеить и закрыть строки в базе."""
        for row in self.db.all("SELECT * FROM recordings WHERE ended_at IS NULL"):
            if self.segment is not None and row["id"] == self._row_id:
                continue
            base = Path(row["path"]).with_suffix("")
            final = mux_parts(base)
            if final is None or not final.exists():
                self.db.execute("DELETE FROM recordings WHERE id = ?", [row["id"]])
                continue
            self.db.update("recordings", row["id"], path=str(final), ended_at=final.stat().st_mtime,
                           size_bytes=final.stat().st_size)

    def set_call(self, in_call: bool) -> None:
        """Граница звонка — новый кусок с другим fps."""
        self.in_call = in_call
        if self.segment is not None and not self._stop.is_set():
            self._rotate(open_new=True)

    def write_audio(self, channel: str, samples: np.ndarray) -> None:
        seg = self.segment
        if seg is not None:
            seg.audio[channel].write(samples)

    def _open(self) -> None:
        day = datetime.now().strftime("%Y-%m-%d")
        folder = paths.RECORDINGS_DIR / day
        folder.mkdir(parents=True, exist_ok=True)
        base = folder / datetime.now().strftime("%H%M%S")
        fps = self.cfg.fps_call if self.in_call else self.cfg.fps_idle
        self.segment = Segment(base, fps, self._codec or [], self.cfg.output)
        self._row_id = self.db.insert("recordings", path=str(base.with_suffix(".mkv")),
                                      started_at=self.segment.started_at, fps=fps)

    def _rotate(self, open_new: bool, wait: bool = False) -> None:
        with self._lock:
            old, row_id = self.segment, self._row_id
            self.segment = None
            if open_new:
                if self._codec is None:
                    self._codec = detect_codec(self.cfg)
                    log.info("Запись экрана: %s", " ".join(self._codec))
                self._open()
        if old is not None and row_id is not None:
            if wait:
                self._finish(old, row_id)
            else:  # склейка может занять секунды — не держим ею поток записи
                t = threading.Thread(target=self._finish, args=(old, row_id), daemon=True)
                self._finishing = [x for x in self._finishing if x.is_alive()] + [t]
                t.start()

    def _finish(self, seg: Segment, row_id: int) -> None:
        final = seg.close()
        if final is None:
            self.db.execute("DELETE FROM recordings WHERE id = ?", [row_id])
            return
        self.db.update("recordings", row_id, path=str(final), ended_at=time.time(),
                       size_bytes=final.stat().st_size)

    def _loop(self) -> None:
        self.recover()
        while not self._stop.is_set():
            locked = self.cfg.pause_when_locked and screen_locked()
            seg = self.segment
            if not self.cfg.enabled or locked:  # выключено в настройках или экран заблокирован
                if seg is not None:
                    self._rotate(open_new=False)
                self._stop.wait(5.0)
                continue
            too_old = seg is not None and time.time() - seg.started_at > self.cfg.segment_minutes * 60
            died = seg is not None and seg.proc.poll() is not None
            if seg is None or too_old or died:
                self._rotate(open_new=True)
            self._stop.wait(5.0)
