"""Выбор источника системного звука среди нескольких устройств вывода.

Сценарий из жизни: Discord играет в наушники, а устройство по умолчанию — колонки
(или виртуальный кабель). Раньше «собеседники» в таком случае молчали.
"""

import numpy as np
import pytest

from sotto.audio.select import LoudestPicker


def block(value: float = 0.2) -> np.ndarray:
    return np.full(512, value, dtype=np.float32)


def test_silence_returns_nothing():
    picker = LoudestPicker()
    assert picker.pick(now=100.0) is None


def test_picks_the_loud_device():
    picker = LoudestPicker()
    picker.observe("quiet", rms=0.0, block=block(0.0), ts=100.0)
    picker.observe("discord", rms=0.2, block=block(0.2), ts=100.0)
    device, picked, ts = picker.pick(now=100.0)
    assert device == "discord"
    assert float(picked[0]) == pytest.approx(0.2)
    assert ts == 100.0


def test_holds_device_through_pauses_between_words():
    picker = LoudestPicker(hold_s=1.0, loud=1e-4)
    picker.observe("discord", rms=0.2, block=block(0.2), ts=100.0)
    picker.observe("music", rms=0.0, block=block(0.0), ts=100.0)
    assert picker.pick(now=100.0)[0] == "discord"
    # в Discord пауза, во втором устройстве тишина — канал не переключаем
    for step in range(1, 9):  # 0.03 с на шаг, ещё внутри hold_s
        now = 100.0 + step * 0.03
        picker.observe("discord", rms=0.0, block=block(0.0), ts=now)
        picker.observe("music", rms=0.0, block=block(0.0), ts=now)
        assert picker.pick(now=now)[0] == "discord"


def test_switches_when_other_device_starts_playing():
    picker = LoudestPicker(hold_s=1.0)
    picker.observe("discord", rms=0.0, block=block(0.0), ts=100.0)
    picker.observe("browser", rms=0.0, block=block(0.0), ts=100.0)
    picker.pick(now=100.0)
    picker.observe("discord", rms=0.0, block=block(0.0), ts=101.0)
    picker.observe("browser", rms=0.3, block=block(0.3), ts=101.0)
    assert picker.pick(now=101.0)[0] == "browser"


def test_stale_device_is_ignored():
    picker = LoudestPicker(fresh_s=0.5)
    picker.observe("gone", rms=0.5, block=block(0.5), ts=100.0)   # устройство отвалилось
    picker.observe("live", rms=0.01, block=block(0.01), ts=101.0)
    device, picked, _ts = picker.pick(now=101.0)
    assert device == "live"
    assert float(picked[0]) == pytest.approx(0.01)


def test_forget_drops_device():
    picker = LoudestPicker()
    picker.observe("usb", rms=0.3, block=block(0.3), ts=100.0)
    assert picker.pick(now=100.0)[0] == "usb"
    picker.forget("usb")
    assert picker.pick(now=100.0) is None


def test_does_not_flap_between_similar_devices():
    """Два устройства играют почти одинаково (например, зеркало) — канал не должен дёргаться."""
    picker = LoudestPicker(switch_gain=1.8)
    picker.observe("a", rms=0.10, block=block(0.10), ts=100.0)
    picker.observe("b", rms=0.11, block=block(0.11), ts=100.0)
    chosen = picker.pick(now=100.0)[0]
    for step in range(1, 20):
        now = 100.0 + step * 0.032
        picker.observe("a", rms=0.10 + (step % 2) * 0.01, block=block(0.1), ts=now)
        picker.observe("b", rms=0.11 - (step % 2) * 0.01, block=block(0.1), ts=now)
        assert picker.pick(now=now)[0] == chosen


def test_switches_when_other_is_clearly_louder():
    picker = LoudestPicker(switch_gain=1.8)
    picker.observe("discord", rms=0.05, block=block(0.05), ts=100.0)
    picker.observe("browser", rms=0.001, block=block(0.001), ts=100.0)
    assert picker.pick(now=100.0)[0] == "discord"
    picker.observe("discord", rms=0.05, block=block(0.05), ts=100.1)
    picker.observe("browser", rms=0.30, block=block(0.30), ts=100.1)
    assert picker.pick(now=100.1)[0] == "browser"
