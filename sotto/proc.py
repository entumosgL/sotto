"""Запуск вспомогательных процессов (pw-record, wf-recorder, ffmpeg).

* Своя сессия: Ctrl+C в терминале получает только демон, а он сам корректно гасит детей
  (wf-recorder при этом успевает дописать файл).
* PR_SET_PDEATHSIG: если демон упал или убит (kill -9), дети получают SIGINT и завершаются,
  а не пишут экран вечно.
"""

from __future__ import annotations

import ctypes
import os
import signal

PR_SET_PDEATHSIG = 1
_libc = ctypes.CDLL(None, use_errno=True)


def _child_setup() -> None:
    os.setsid()
    _libc.prctl(PR_SET_PDEATHSIG, signal.SIGINT)


CHILD = {"preexec_fn": _child_setup}
