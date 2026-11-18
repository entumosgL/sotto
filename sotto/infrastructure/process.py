"""Запуск вспомогательных процессов (захват звука, запись экрана, ffmpeg, CLI-агенты).

* Своя сессия / группа: Ctrl+C в терминале получает только демон, а он сам корректно гасит детей.
* Linux: PR_SET_PDEATHSIG — если демон упал или убит (kill -9), дети получают SIGINT и завершаются,
  а не пишут экран вечно. На других ОС — только отдельная группа процессов.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from typing import Any


def _linux_child_setup() -> None:
    import ctypes
    os.setsid()
    ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGINT)  # PR_SET_PDEATHSIG


def child_kwargs() -> dict[str, Any]:
    if sys.platform.startswith("linux"):
        return {"preexec_fn": _linux_child_setup}
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}  # type: ignore[attr-defined]
    return {"start_new_session": True}


CHILD = child_kwargs()
