"""Файлы записи через ffmpeg (есть на любой ОС): звук каналов в Opus и склейка куска в один .mkv.

Кусок записи = видео экрана (его пишет платформа) + звук микрофона + звук собеседников. В конце куска
всё склеивается в base.mkv: дорожка 1 — общий звук, 2 — микрофон, 3 — собеседники.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import process as proc

log = logging.getLogger(__name__)


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



def remove_parts(final: Path) -> None:
    """Файлы записи: итоговый .mkv и куски, если склейка не успела."""
    base = final.with_suffix("")
    for suffix in (".mkv", ".video.mkv", ".me.opus", ".them.opus"):
        base.with_suffix(suffix).unlink(missing_ok=True)


class Piece:
    """Один кусок записи: экран (платформа) + два канала звука."""

    def __init__(self, base: Path, fps: int, platform: Any, settings: Any):
        self.base = base
        self.started_at = time.time()
        self.fps = fps
        self.audio = {ch: AudioArchive(base.with_suffix(f".{ch}.opus")) for ch in ("me", "them")}
        self.video = platform.record_screen(base.with_suffix(".video.mkv"), fps, settings.output, settings)

    def write_audio(self, channel: str, samples: np.ndarray) -> None:
        self.audio[channel].write(samples)

    def alive(self) -> bool:
        return self.video.alive()

    def close(self) -> Path | None:
        self.video.close()
        for a in self.audio.values():
            a.close()
        return mux_parts(self.base)


class FfmpegRecording:
    """Реализация порта RecordingBackend."""

    def __init__(self, platform: Any):
        self.platform = platform

    def available(self) -> bool:
        return self.platform.video_available() and shutil.which("ffmpeg") is not None

    def open(self, base: Path, fps: int, settings: Any) -> Piece:
        return Piece(base, fps, self.platform, settings)

    def recover(self, base: Path) -> Path | None:
        return mux_parts(base)
