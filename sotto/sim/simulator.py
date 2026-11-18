"""Тестовый интервьюер: проверка суфлёра и тренажёр работы с ним.

Сценарий (TOML) — последовательность шагов. Синтезированный голос «интервьюера» играет в системный
выход, и демон слышит его так же, как настоящий звонок (через монитор выхода). Симулятор следит
за событиями демона и меряет:
  • сколько от конца фразы до появления реплики в расшифровке (распознавание);
  • сколько до первого слова подсказки и до конца ответа;
  • сработал ли авто-режим там, где должен, и промолчал ли там, где не должен;
  • есть ли в ответе ожидаемые ключевые слова; узнал ли суфлёр имя из представления.

В режиме обучения (сценарий tutorial) шаги подсказывают, какой хоткей нажать, и ждут действия.

Формат шага:
  say = "фраза интервьюера"          # озвучить
  voice = "ru_RU-irina-medium"       # голос (по умолчанию — по языку текста)
  hint = "что сделать пользователю"   # строка в терминале
  wait = "answer" | "skip" | "utterance" | "none"   # чего ждать после шага
  expect_kind = "auto" | "manual" | "screen" | "questions"
  keywords = ["идемпотент", "ключ"]  # хотя бы одно должно быть в ответе
  expect_name = "Анна"               # суфлёр должен узнать имя
  ctl = "toggle_mode"                # выполнить действие демона
  timeout = 40
"""

from __future__ import annotations

import json
import queue
import threading
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .. import config as config_mod
from .. import paths
from ..audio import tts

SCENARIOS_DIR = Path(__file__).parent / "scenarios"


def list_scenarios() -> list[tuple[str, str]]:
    out = []
    for f in sorted(SCENARIOS_DIR.glob("*.toml")):
        out.append((f.stem, tomllib.loads(f.read_text(encoding="utf-8")).get("title", "")))
    return out


@dataclass
class StepResult:
    say: str = ""
    stt_ms: int | None = None          # конец фразы → реплика в расшифровке
    first_token_ms: int | None = None  # конец фразы → первый кусок подсказки
    answer_ms: int | None = None       # конец фразы → ответ целиком
    kind: str | None = None
    answer: str = ""
    passed: bool = True
    notes: list[str] = field(default_factory=list)


class EventStream:
    """Подписка на события демона в отдельном потоке."""

    def __init__(self, base: str):
        self.q: queue.Queue = queue.Queue()
        self.base = base
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        with httpx.stream("GET", f"{self.base}/api/events", timeout=None) as r:
            for line in r.iter_lines():
                if line.startswith("data: "):
                    self.q.put(json.loads(line[6:]))

    def drain(self) -> None:
        while not self.q.empty():
            self.q.get_nowait()

    def next(self, timeout: float) -> dict[str, Any] | None:
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None


def run_scenario(name: str) -> int:
    path = SCENARIOS_DIR / f"{name}.toml"
    if not path.exists():
        print(f"Нет сценария {name}. Доступные: {', '.join(n for n, _ in list_scenarios())}")
        return 1
    scenario = tomllib.loads(path.read_text(encoding="utf-8"))
    cfg = config_mod.load()
    base = f"http://{cfg.web.host}:{cfg.web.port}"
    try:
        status = httpx.get(f"{base}/api/status", timeout=3).json()
    except httpx.HTTPError:
        status = {"panel_only": True}
    if status.get("panel_only"):
        print("Сначала запусти голосовой суфлёр: кнопкой на сайте (sotto) или командой sotto run")
        return 1

    print(f"\n=== {scenario.get('title', name)} ===\n{scenario.get('intro', '').strip()}\n")
    events = EventStream(base)

    time.sleep(0.5)
    if not status["in_call"]:
        httpx.post(f"{base}/api/ctl/toggle_training_call", timeout=10)
        time.sleep(1.0)
    if scenario.get("mode") and status["mode"] != scenario["mode"]:
        httpx.post(f"{base}/api/ctl/toggle_mode", timeout=10)

    results: list[StepResult] = []
    try:
        for i, step in enumerate(scenario.get("step", []), 1):
            res = _run_step(i, step, base, events, cfg)
            results.append(res)
    except KeyboardInterrupt:
        print("\nПрервано.")
    finally:
        if not status["in_call"]:
            httpx.post(f"{base}/api/ctl/toggle_training_call", timeout=10)
        if scenario.get("mode") and status["mode"] != scenario["mode"]:
            httpx.post(f"{base}/api/ctl/toggle_mode", timeout=10)

    _report(name, results)
    return 0 if all(r.passed for r in results) else 2


def _run_step(i: int, step: dict[str, Any], base: str, events: EventStream, cfg: config_mod.Config) -> StepResult:
    res = StepResult(say=step.get("say", ""))
    if step.get("hint"):
        print(f"\n{step['hint']}")
    if step.get("hint") and res.say:
        time.sleep(float(step.get("hint_s", 4)))  # успеть прочитать плашку до того, как заговорят
    if step.get("ctl"):
        httpx.post(f"{base}/api/ctl/{step['ctl']}", timeout=10)
    events.drain()
    spoken_at = time.time()
    if res.say:
        voice = step.get("voice") or tts.pick_voice(res.say, "ru_RU-irina-medium", cfg.trainer.voice_en)
        print(f"[{i}] {res.say}")
        tts.say(res.say, voice)
        spoken_at = time.time()  # отсчёт — от конца фразы, как в живом разговоре

    wait = step.get("wait", "answer" if res.say else "none")
    timeout = float(step.get("timeout", 45))
    if wait == "none" and "timeout" in step:
        time.sleep(timeout)  # пауза, чтобы пользователь успел нажать хоткей
    deadline = time.time() + timeout
    name_seen = None
    while wait != "none" and time.time() < deadline:
        e = events.next(timeout=0.5)
        if e is None:
            if wait == "skip" and time.time() - spoken_at > float(step.get("quiet_s", 8)):
                break  # авто-режим промолчал — так и надо
            continue
        t, ms = e["type"], int((time.time() - spoken_at) * 1000)
        if t == "utterance" and e["channel"] == "them" and res.stt_ms is None:
            res.stt_ms = ms
            print(f"    распознано за {ms} мс: {e['text']}")
            if wait == "utterance":
                break
        elif t == "speaker" and e.get("name"):
            name_seen = e["name"]
        elif t == "answer.start":
            res.kind = e["kind"]
        elif t == "answer.delta" and res.first_token_ms is None:
            res.first_token_ms = ms
        elif t == "answer.done":
            res.answer_ms, res.answer = ms, e["text"]
            if wait == "answer":
                break
        elif t == "answer.skip" and wait == "skip":
            break
        elif t == "answer.error":
            res.notes.append("ошибка: " + e["error"])
            break

    # --- проверки ---
    if wait == "answer" and not res.answer:
        res.passed = False
        res.notes.append("ответа не было")
    if wait == "skip" and res.answer:
        res.passed = False
        res.notes.append("ответил там, где стоило промолчать")
    if step.get("expect_kind") and res.kind and res.kind != step["expect_kind"]:
        res.passed = False
        res.notes.append(f"тип ответа {res.kind}, ожидался {step['expect_kind']}")
    kws = [k.lower() for k in step.get("keywords", [])]
    if kws and res.answer and not any(k in res.answer.lower() for k in kws):
        res.passed = False
        res.notes.append("нет ключевых слов: " + ", ".join(kws))
    if step.get("expect_name"):
        if name_seen is None:
            time.sleep(2.0)
            for _ in range(20):
                e = events.next(timeout=0.2)
                if e and e["type"] == "speaker" and e.get("name"):
                    name_seen = e["name"]
        if not name_seen or step["expect_name"].lower() not in name_seen.lower():
            res.passed = False
            res.notes.append(f"имя не распознано (ждали {step['expect_name']}, получили {name_seen})")

    if res.answer:
        preview = res.answer.replace("\n", " ")
        print(f"    подсказка ({res.kind}): первый токен {res.first_token_ms} мс, весь ответ {res.answer_ms} мс")
        print(f"        {preview[:220]}{'…' if len(preview) > 220 else ''}")
    print(f"    {'ok' if res.passed else 'FAIL'} {'; '.join(res.notes)}")
    return res


def _report(name: str, results: list[StepResult]) -> None:
    def med(vals: list[int]) -> str:
        vals = sorted(v for v in vals if v is not None)
        return f"{vals[len(vals) // 2]} мс" if vals else "—"

    ok = sum(r.passed for r in results)
    print("\n=== Итог ===")
    print(f"Шагов пройдено: {ok}/{len(results)}")
    print(f"Медиана: распознавание {med([r.stt_ms for r in results])}, "
          f"первый токен {med([r.first_token_ms for r in results])}, "
          f"ответ целиком {med([r.answer_ms for r in results])} (от конца фразы)")
    report_dir = paths.STATE_DIR / "sim"
    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-{name}.json"
    out.write_text(json.dumps([r.__dict__ for r in results], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Отчёт: {out}")
