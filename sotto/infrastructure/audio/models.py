"""Локальные модели (скачиваются один раз в ~/.local/share/sotto/models)."""

from __future__ import annotations

import logging
import tarfile
import tempfile
import urllib.request
from pathlib import Path

from ... import paths

log = logging.getLogger(__name__)

SHERPA = "https://github.com/k2-fsa/sherpa-onnx/releases/download"

FILES = {
    "silero_vad.onnx": f"{SHERPA}/asr-models/silero_vad.onnx",
    # Голосовой отпечаток не зависит от языка, поэтому англоязычная модель подходит и для русского.
    "wespeaker_en_voxceleb_resnet34_LM.onnx":
        f"{SHERPA}/speaker-recongition-models/wespeaker_en_voxceleb_resnet34_LM.onnx",
}


def _download(url: str, dest: Path) -> None:
    log.info("Скачиваю %s", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        with urllib.request.urlopen(url, timeout=60) as resp:
            while chunk := resp.read(1 << 20):
                tmp.write(chunk)
    Path(tmp.name).rename(dest)


def file(name: str) -> Path:
    path = paths.MODELS_DIR / name
    if not path.exists():
        _download(FILES[name], path)
    return path


def piper_voice(voice: str) -> Path:
    """Голос Piper для синтеза речи, например "ru_RU-dmitri-medium"."""
    directory = paths.MODELS_DIR / f"vits-piper-{voice}"
    if not directory.exists():
        archive = paths.MODELS_DIR / f"vits-piper-{voice}.tar.bz2"
        _download(f"{SHERPA}/tts-models/vits-piper-{voice}.tar.bz2", archive)
        with tarfile.open(archive) as tar:
            tar.extractall(paths.MODELS_DIR, filter="data")
        archive.unlink()
    return directory
