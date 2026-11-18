"""Экспериментальный переводчик: оценка тона и выбор голосов (без звука и сети)."""

import numpy as np

from sotto.experimental.translator.voices import TheirVoices, pitch_hz


def tone(freq: float, seconds: float = 1.0, sr: int = 16000) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (0.3 * np.sin(2 * np.pi * freq * t) + 0.1 * np.sin(2 * np.pi * 2 * freq * t)).astype(np.float32)


def test_pitch_estimate():
    assert abs(pitch_hz(tone(120)) - 120) < 8
    assert abs(pitch_hz(tone(220)) - 220) < 12
    assert pitch_hz(np.zeros(16000, np.float32)) is None


def test_voices_distinct_per_speaker():
    v = TheirVoices(male=["m1", "m2"], female=["f1"])
    a = v.for_speaker(0, tone(110))
    b = v.for_speaker(1, tone(120))
    c = v.for_speaker(2, tone(230))
    d = v.for_speaker(3, tone(240))
    assert (a.voice, b.voice, c.voice) == ("m1", "m2", "f1")
    assert d.voice == "f1" and d.pitch != c.pitch  # второй женский голос — другой тон
    assert v.for_speaker(0, tone(300)) is a       # голос закрепляется за человеком
