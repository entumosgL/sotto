"""Виртуальные устройства PipeWire для переводчика.

* «Sotto: звонок» (sink) — выбери его динамиком в приложении звонка. Его звук мы распознаём и тихо
  (original_volume) пропускаем в твои наушники, чтобы слышать интонации оригинала.
* «Sotto: микрофон» (source) — выбери его микрофоном в приложении звонка. Туда звучит английская
  речь твоим голосом (или твой настоящий микрофон, если перевод исходящей речи выключен).

Устройства создаются `pw-loopback` с нулевым приоритетом — они не становятся устройствами по умолчанию
и исчезают, когда переводчик закрывается.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time

log = logging.getLogger(__name__)

CALL_SINK = "sotto_call"          # сюда приложение звонка выводит звук собеседников
CALL_OUT = "sotto_call_out"       # петля из CALL_SINK в твои наушники (громкость = «оригинал»)
MIC_FEED = "sotto_mic_feed"       # сюда мы играем английскую речь
MIC_SOURCE = "sotto_mic"          # это видит приложение звонка как микрофон

QUIET = "priority.session=0 priority.driver=0"


def node_id(name: str) -> int | None:
    out = subprocess.run(["pw-dump"], capture_output=True, text=True, timeout=5).stdout
    for o in json.loads(out or "[]"):
        props = (o.get("info") or {}).get("props") or {}
        if props.get("node.name") == name:
            return int(o["id"])
    return None


class VirtualDevices:
    def __init__(self, original_volume: float):
        self.original_volume = original_volume
        self._procs: list[subprocess.Popen] = []

    def __enter__(self) -> VirtualDevices:
        self._procs.append(subprocess.Popen(
            ["pw-loopback", "-n", "sotto-call",
             f"--capture-props=media.class=Audio/Sink node.name={CALL_SINK} "
             f"node.description=\"Sotto: звонок (перевод)\" audio.channels=1 {QUIET}",
             f"--playback-props=node.name={CALL_OUT} node.description=\"Sotto: оригинал звонка\" node.passive=true"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        self._procs.append(subprocess.Popen(
            ["pw-loopback", "-n", "sotto-mic",
             f"--capture-props=media.class=Audio/Sink node.name={MIC_FEED} node.description=\"Sotto: вход микрофона\" "
             f"audio.channels=1 node.passive=true {QUIET}",
             f"--playback-props=media.class=Audio/Source node.name={MIC_SOURCE} "
             f"node.description=\"Sotto: микрофон (перевод)\" audio.channels=1 {QUIET}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        for _ in range(20):
            if node_id(CALL_OUT) and node_id(MIC_SOURCE):
                break
            time.sleep(0.2)
        self.set_original_volume(self.original_volume)
        return self

    def set_original_volume(self, volume: float) -> None:
        self.original_volume = max(0.0, min(1.0, volume))
        nid = node_id(CALL_OUT)
        if nid is not None:
            subprocess.run(["wpctl", "set-volume", str(nid), f"{self.original_volume:.2f}"], check=False)

    def __exit__(self, *_exc) -> None:
        for p in self._procs:
            if p.poll() is None:
                p.terminate()
        for p in self._procs:
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()
