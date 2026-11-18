"""Голоса переводчика.

* Твой голос: клон по образцу (30 с чтения текста) через OpenRouter, модель с `input_references`
  (по умолчанию Fish Audio S2.1 Pro — хорошо переносит голос между языками: образец русский, речь английская).
  Без образца или ключа — локальный голос Piper.
* Голоса собеседников: локальный Piper (бесплатно, ~0.1 с на фразу). Каждому собеседнику — свой:
  по высоте голоса оригинала выбираем мужской/женский, плюс небольшой сдвиг тона, чтобы различать.
"""

from __future__ import annotations

import base64
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

from ... import paths
from ...infrastructure.audio import SAMPLE_RATE, tts

log = logging.getLogger(__name__)

VOICE_DIR = paths.DATA_DIR / "translator"

# Текст для образца: ~15 секунд, разные звуки, обычная разговорная интонация.
# Образец отправляется с каждой фразой, поэтому он короткий (24 кГц, ~15 с ≈ 1 МБ).
ENROLL_TEXT = (
    "Привет! Последние несколько лет я занимаюсь бэкендом: проектирую сервисы, разбираюсь с базами данных, "
    "очередями и производительностью. Люблю, когда задача сформулирована чётко, а код читается легко."
)


def reference_paths() -> tuple[Path, Path]:
    return VOICE_DIR / "my_voice.wav", VOICE_DIR / "my_voice.txt"


def has_reference() -> bool:
    wav, _ = reference_paths()
    return wav.exists() and wav.stat().st_size > 24000 * 2 * 5  # хотя бы 5 секунд


def record_reference(seconds: int = 20) -> Path:
    """Пишет образец голоса с микрофона по умолчанию (24 кГц моно WAV), обрезает тишину по краям."""
    import wave
    VOICE_DIR.mkdir(parents=True, exist_ok=True)
    wav, txt = reference_paths()
    raw = subprocess.run(["timeout", str(seconds), "pw-record", "--raw", "--rate", "24000", "--channels", "1",
                          "--format", "s16", "-"], capture_output=True).stdout
    x = np.frombuffer(raw, dtype=np.int16)
    loud = np.where(np.abs(x) > 1500)[0]
    if len(loud):
        x = x[max(0, loud[0] - 4800):loud[-1] + 4800]
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(x.tobytes())
    txt.write_text(ENROLL_TEXT, encoding="utf-8")
    return wav


# --- воспроизведение ---

class Player:
    """Проигрывает PCM (s16 моно) в заданное устройство PipeWire; можно оборвать."""

    def __init__(self, target: str | None):
        self.target = target
        self.current: subprocess.Popen | None = None

    def play_pcm(self, pcm: bytes, rate: int) -> None:
        cmd = ["pw-play", "--raw", "--rate", str(rate), "--channels", "1", "--format", "s16"]
        if self.target:
            cmd += ["--target", self.target]
        self.current = subprocess.Popen([*cmd, "-"], stdin=subprocess.PIPE)
        try:
            assert self.current.stdin is not None
            self.current.stdin.write(pcm)
            self.current.stdin.close()
        except BrokenPipeError:
            pass
        self.current.wait()
        self.current = None

    def play(self, samples: np.ndarray, rate: int) -> None:
        self.play_pcm((np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes(), rate)

    def stop(self) -> None:
        if self.current and self.current.poll() is None:
            self.current.terminate()


# --- твой голос ---

class MyVoice:
    def __init__(self, api_key: str, base_url: str, model: str, fallback_voice: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.fallback_voice = fallback_voice
        self._ref_b64: str | None = None
        self._ref_text = ""
        if has_reference():
            wav, txt = reference_paths()
            self._ref_b64 = base64.b64encode(wav.read_bytes()).decode()
            self._ref_text = txt.read_text(encoding="utf-8") if txt.exists() else ""

    @property
    def cloned(self) -> bool:
        return bool(self.api_key and self._ref_b64)

    def synthesize_to(self, text: str, player: Player) -> float:
        """Синтезирует фразу и проигрывает в player. Возвращает время до начала звука (с)."""
        start = time.perf_counter()
        if not self.cloned:
            samples, sr = tts.synthesize(text, self.fallback_voice)
            first = time.perf_counter() - start
            player.play(samples, sr)
            return first
        body = {
            "model": self.model,
            "input": text,
            "response_format": "mp3",
            "input_references": [
                {"type": "input_audio", "input_audio": {"data": f"data:audio/wav;base64,{self._ref_b64}"}},
                *([{"type": "text", "text": self._ref_text}] if self._ref_text else []),  # что сказано в образце
            ],
        }
        # mp3 декодируем ffmpeg'ом на лету и сразу отдаём в pw-play: звук начинается до конца загрузки.
        decoder = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "pipe:0",
                                    "-f", "s16le", "-ac", "1", "-ar", "24000", "pipe:1"],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        cmd = ["pw-play", "--raw", "--rate", "24000", "--channels", "1", "--format", "s16"]
        if player.target:
            cmd += ["--target", player.target]
        sink = subprocess.Popen([*cmd, "-"], stdin=decoder.stdout)
        player.current = sink
        first = None
        try:
            with httpx.stream("POST", f"{self.base_url}/audio/speech", json=body, timeout=30,
                              headers={"Authorization": f"Bearer {self.api_key}"}) as resp:
                if resp.status_code != 200:
                    raise RuntimeError(f"TTS {resp.status_code}: {resp.read()[:300]!r}")
                for chunk in resp.iter_bytes():
                    if first is None:
                        first = time.perf_counter() - start
                    assert decoder.stdin is not None
                    decoder.stdin.write(chunk)
        finally:
            if decoder.stdin:
                decoder.stdin.close()
            decoder.wait()
            sink.wait()
            player.current = None
        return first or (time.perf_counter() - start)


# --- голоса собеседников ---

def pitch_hz(samples: np.ndarray, sr: int = SAMPLE_RATE) -> float | None:
    """Грубая оценка основного тона (автокорреляция по громким окнам). Хватает, чтобы отличить М/Ж."""
    frame = int(sr * 0.04)
    estimates = []
    for i in range(0, len(samples) - frame, frame):
        x = samples[i:i + frame]
        if np.sqrt(np.mean(x ** 2)) < 0.02:
            continue
        x = x - x.mean()
        corr = np.correlate(x, x, mode="full")[frame - 1:]
        lo, hi = sr // 400, sr // 70  # 70–400 Гц
        lag = lo + int(np.argmax(corr[lo:hi]))
        if corr[lag] > 0.3 * corr[0]:
            estimates.append(sr / lag)
    return float(np.median(estimates)) if estimates else None


@dataclass
class SpeakerVoice:
    voice: str
    pitch: float  # множитель частоты воспроизведения: 1.06 — чуть выше и быстрее


PITCH_STEPS = [1.0, 0.94, 1.07, 0.9, 1.12]


class TheirVoices:
    """Каждому собеседнику — свой узнаваемый русский голос."""

    def __init__(self, male: list[str], female: list[str]):
        self.male, self.female = male, female
        self.assigned: dict[int, SpeakerVoice] = {}

    def for_speaker(self, speaker: int, sample: np.ndarray) -> SpeakerVoice:
        if speaker not in self.assigned:
            f0 = pitch_hz(sample)
            pool = self.female if (f0 or 0) > 165 else self.male
            n_same = sum(1 for v in self.assigned.values() if v.voice in pool)
            voice = pool[n_same % len(pool)]
            pitch = PITCH_STEPS[(n_same // len(pool)) % len(PITCH_STEPS)]
            self.assigned[speaker] = SpeakerVoice(voice, pitch)
            log.info("Собеседник %s: F0≈%s Гц → %s ×%.2f", speaker, f0 and round(f0), voice, pitch)
        return self.assigned[speaker]

    def speak(self, text: str, voice: SpeakerVoice, player: Player, speed: float = 1.0) -> None:
        samples, sr = tts.synthesize(text, voice.voice, speed=speed)
        player.play(samples, int(sr * voice.pitch))
