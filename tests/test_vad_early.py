"""Упреждающий старт: короткая пауза даёт предварительную реплику, продолжение речи её отменяет."""

import numpy as np
import pytest

from sotto.infrastructure.audio import CHUNK, SAMPLE_RATE


def speech(text):
    from sotto.infrastructure.audio import tts
    s, sr = tts.synthesize(text, "en_US-lessac-medium")
    n = int(len(s) * SAMPLE_RATE / sr)
    x = np.interp(np.linspace(0, len(s), n, endpoint=False), np.arange(len(s)), s).astype(np.float32)
    loud = np.flatnonzero(np.abs(x) > 0.02)  # синтез добавляет по краям тишину разной длины — убираем
    return x[loud[0]:loud[-1] + 1]


def run(audio):
    from sotto.infrastructure.audio.vad import ChannelVAD
    vad = ChannelVAD("them", early_silence=0.2)
    log = []
    for i in range(0, len(audio) - CHUNK + 1, CHUNK):
        t = i / SAMPLE_RATE
        for seg in vad.feed(audio[i:i + CHUNK], 1000.0 + t):
            log.append(("final", t, seg))
        for kind, seg in vad.take_events():
            log.append((kind, t, seg))
    return log


def silence(sec):
    return np.zeros(int(sec * SAMPLE_RATE), dtype=np.float32)


@pytest.fixture(scope="module")
def phrases():
    try:
        return speech("Tell me about your experience"), speech("with Kafka in production.")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"нет голоса для синтеза: {e}")


def test_early_fires_before_final_and_covers_phrase(phrases):
    a, _ = phrases
    log = run(np.concatenate([silence(0.5), a, silence(1.5)]))
    kinds = [k for k, _, _ in log]
    assert kinds == ["early", "final"], kinds
    early_t, final_t = log[0][1], log[1][1]
    assert final_t - early_t >= 0.2           # предварительная реплика раньше окончательной
    assert log[0][2].duration >= len(a) / SAMPLE_RATE  # и содержит всю фразу


def test_pause_mid_sentence_resumes(phrases):
    a, b = phrases
    log = run(np.concatenate([silence(0.5), a, silence(0.35), b, silence(1.5)]))
    kinds = [k for k, _, _ in log]
    # пауза 0.3 с на грани порога: упреждение посреди фразы может быть, а может и нет —
    # но если было, речь продолжилась и оно обязано отмениться
    assert kinds in (["early", "resume", "early", "final"], ["early", "final"]), kinds
    assert log[-2][2].duration >= (len(a) + len(b)) / SAMPLE_RATE  # последнее упреждение — с начала фразы
