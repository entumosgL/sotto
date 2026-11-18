"""Проверки перед коммитом и пушем, которых нет в готовых хуках.

  secrets  — ключи API и пароли в файлах (OpenRouter sk-or-…, заполненные api_key/password в .toml)
  private  — личные слова (имя, компании, город) из .git/info/private-words: этот файл не в репозитории,
             его ведёт владелец; без него проверка молча проходит

  python scripts/guard.py secrets FILE…
  python scripts/guard.py private FILE…    (без файлов — весь репозиторий)
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

KEY = re.compile(r"sk-or-v1-[0-9a-f]{20,}|sk-ant-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{30,}")
FILLED = re.compile(r"""^\s*(api_key|api_hash|password)\s*=\s*(?:"[^"]+"|'[^']+')""", re.M)


def _text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return ""


def secrets(files: list[str]) -> int:
    bad = []
    for f in files:
        text = _text(f)
        if KEY.search(text):
            bad.append(f"{f}: похоже на ключ API")
        if f.endswith(".toml") and FILLED.search(text):
            bad.append(f"{f}: заполненный секрет в .toml")
    for b in bad:
        print(b)
    return 1 if bad else 0


def private(files: list[str]) -> int:
    root = Path(subprocess.run(["git", "rev-parse", "--git-common-dir"], capture_output=True, text=True).stdout.strip() or ".git")
    words_file = root / "info" / "private-words"
    if not words_file.exists():
        return 0
    words = [w.strip() for w in words_file.read_text(encoding="utf-8").splitlines() if w.strip()]
    if not files:
        files = [f for f in subprocess.run(["git", "ls-files", "-z"], capture_output=True, text=True).stdout.split("\0") if f]
    bad = []
    for f in files:
        text = _text(f)
        for w in words:
            # без учёта регистра и с любым окончанием: «Иванов» ловит и «Иванова»
            if re.search(rf"(?<!\w){re.escape(w)}", text, re.I):
                bad.append(f"{f}: личное слово из private-words")
                break
    for b in bad:
        print(b)
    return 1 if bad else 0


if __name__ == "__main__":
    cmd, *args = sys.argv[1:] or ["secrets"]
    sys.exit({"secrets": secrets, "private": private}[cmd](args))
