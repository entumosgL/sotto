"""`sotto bench` — замер моделей OpenRouter на твоём ключе и из твоей сети.

Публичные рейтинги задержки мало что говорят: они усреднены по миру и включают «размышления»
моделей. Здесь меряем ровно то, что важно суфлёру:
  • ответ: время до первого токена (без размышлений), время всего ответа, цена;
  • распознавание: задержка и точность (CER) на синтезированных фразах RU / EN / смешанных;
  • скриншоты: понимает ли модель, что на экране (тестовая картинка с задачей без подсветки).
С флагом --apply лучшие модели записываются в config.toml.
"""

from __future__ import annotations

import asyncio
import base64
import statistics
import time
from dataclasses import dataclass, field

import numpy as np

from .. import config as config_mod
from ..audio import tts
from ..copilot import context
from ..openrouter import OpenRouter, OpenRouterError

# Кандидаты: семейства, которые хорошо говорят по-русски и быстро отвечают. Берутся те, что есть в каталоге.
ANSWER_CANDIDATES = [
    "anthropic/claude-haiku-5.5", "openai/gpt-6-luna", "google/gemini-3.8-flash",
    "deepseek/deepseek-v4.1-flash-1017", "qwen/qwen3.8-flash", "x-ai/grok-4.7",
    "anthropic/claude-sonnet-5.5", "openai/gpt-6.1-terra", "minimax/minimax-m3.1-flash",
    "z-ai/glm-5.3-flash", "moonshotai/kimi-k3.1-pro",
]
VISION_CANDIDATES = ["anthropic/claude-sonnet-5.5", "openai/gpt-6.1-terra", "google/gemini-3.8-flash",
                     "anthropic/claude-haiku-5.5", "openai/gpt-6-luna"]
STT_CANDIDATES = ["openai/whisper-large-v3-turbo", "qwen/qwen3-asr-1.7b", "openai/whisper-large-v3",
                  "x-ai/grok-stt-2.0", "elevenlabs/scribe-v2", "mistralai/voxtral-mini-transcribe-3-0",
                  "openai/gpt-transcribe", "nvidia/nemotron-3.5-transcribe"]

TRANSCRIPT = [
    {"channel": "them", "speaker": "Дмитрий", "speaker_role": "техлид", "started_at": time.time() - 90,
     "text": "Расскажите, с какими брокерами сообщений работали."},
    {"channel": "me", "started_at": time.time() - 80,
     "text": "В основном с Kafka, немного с RabbitMQ на прошлом месте."},
    {"channel": "them", "speaker": "Дмитрий", "speaker_role": "техлид", "started_at": time.time() - 30,
     "text": "Хорошо. Как в Kafka гарантировать, что сообщение обработается ровно один раз?"},
]
PROFILE = "Backend-разработчик, 5 лет, Go и Python, PostgreSQL, Kafka, Kubernetes. Финтех и e-commerce."

STT_PHRASES = [
    ("ru", "Расскажите, как устроены индексы в базе данных и когда они не помогают."),
    ("ru", "Какой у вас опыт с микросервисами и как вы решали проблему распределённых транзакций?"),
    ("en", "Tell me about a time you had to deal with a production incident under pressure."),
    ("mixed", "Как вы настраивали consumer groups в Kafka и что делали с rebalance?"),
]


@dataclass
class Row:
    model: str
    ttft: list[float] = field(default_factory=list)
    total: list[float] = field(default_factory=list)
    cost: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    sample: str = ""
    score: float | None = None  # CER для STT, 0/1 для зрения

    def med(self, xs: list[float]) -> float | None:
        return statistics.median(xs) if xs else None


def cer(ref: str, hyp: str) -> float:
    """Доля ошибок по символам (расстояние Левенштейна / длина эталона)."""
    norm = lambda s: " ".join("".join(ch for ch in s.lower() if ch.isalnum() or ch == " ").split())
    a, b = norm(ref), norm(hyp)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(len(a), 1)


def _to16k(x: np.ndarray, sr: int) -> np.ndarray:
    n = int(len(x) * 16000 / sr)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


async def bench_answers(llm: OpenRouter, models: list[str], runs: int) -> list[Row]:
    system = context.system_prompt("answer", allow_invention=True, profile=PROFILE, legend=[], brief=None)
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": context.transcript_message(TRANSCRIPT, "Подскажи ответ на последний вопрос.")}]
    rows = []
    for model in models:
        row = Row(model)
        for i in range(runs):
            try:
                async for delta, res in llm.stream_chat(model, messages, max_tokens=500, session_id=f"bench-{model}"):
                    if not delta:
                        row.ttft.append(res.first_token_ms or res.total_ms)
                        row.total.append(res.total_ms)
                        if res.cost is not None:
                            row.cost.append(res.cost)
                        row.sample = res.text
            except (OpenRouterError, Exception) as e:  # noqa: BLE001
                row.errors.append(str(e)[:120])
                break
        rows.append(row)
        _print_row("ответ", row)
    return rows


async def bench_stt(llm: OpenRouter, models: list[str], cfg: config_mod.Config) -> list[Row]:
    clips = []
    for lang, text in STT_PHRASES:
        voice = cfg.trainer.voice_en if lang == "en" else "ru_RU-irina-medium"
        samples, sr = tts.synthesize(text, voice)
        clips.append((text, _to16k(samples, sr)))
    rows = []
    for model in models:
        row = Row(model)
        errs = []
        for ref, clip in clips:
            t = time.perf_counter()
            try:
                text, _lang, cost = await llm.transcribe(model, clip)
            except (OpenRouterError, Exception) as e:  # noqa: BLE001
                row.errors.append(str(e)[:120])
                break
            row.total.append((time.perf_counter() - t) * 1000)
            if cost is not None:
                row.cost.append(cost)
            errs.append(cer(ref, text))
            row.sample = text
        row.score = statistics.mean(errs) if errs else None
        rows.append(row)
        _print_row("распознавание", row)
    return rows


def understood_task(text: str) -> bool:
    """Понял задачу про скобки (стек) и не полез «чинить отсутствие подсветки»."""
    low = text.lower()
    body = low.split("\n", 1)[1] if "\n" in low else ""
    solved = ("стек" in low or "stack" in low) and ("скоб" in low or "bracket" in low or "paren" in low)
    return solved and "подсвет" not in body and "highlight" not in body


def _task_image() -> bytes:
    """Картинка «онлайн-редактор без подсветки с задачей» — рисуем Qt без окна."""
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QBuffer, QIODevice, Qt
    from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QPainter

    _app = QGuiApplication.instance() or QGuiApplication([])
    img = QImage(1280, 720, QImage.Format.Format_RGB32)
    img.fill(QColor("white"))
    p = QPainter(img)
    p.setPen(QColor("#222"))
    p.setFont(QFont("Sans", 14))
    p.drawText(20, 30, "Яндекс Контест — Задача B. Скобочная последовательность")
    p.setFont(QFont("Monospace", 13))
    lines = ["Дана строка из символов ( ) [ ] { }. Определите, является ли она",
             "правильной скобочной последовательностью. Длина строки до 10^5.",
             "", "def is_valid(s: str) -> bool:", "    # ваш код", "    pass"]
    for i, line in enumerate(lines):
        p.drawText(20, 90 + i * 28, line)
    p.end()
    buf = QBuffer()
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "JPEG", 85)
    return bytes(buf.data())


async def bench_vision(llm: OpenRouter, models: list[str]) -> list[Row]:
    image = base64.b64encode(_task_image()).decode()
    system = context.system_prompt("screen", allow_invention=False, profile=PROFILE, legend=[], brief=None)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": [
        {"type": "text", "text": "Активное окно: chromium — «Яндекс Контест». Помоги с тем, что на экране."},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}}]}]
    rows = []
    for model in models:
        row = Row(model)
        try:
            async for delta, res in llm.stream_chat(model, messages, max_tokens=900):
                if not delta:
                    row.ttft.append(res.first_token_ms or res.total_ms)
                    row.total.append(res.total_ms)
                    row.cost += [res.cost] if res.cost is not None else []
                    row.sample = res.text
                    row.score = float(understood_task(res.text))
        except (OpenRouterError, Exception) as e:  # noqa: BLE001
            row.errors.append(str(e)[:120])
        rows.append(row)
        _print_row("скриншот", row)
    return rows


def _fmt(x: float | None, unit: str = "мс") -> str:
    return f"{x:.0f} {unit}" if x is not None else "—"


def _print_row(kind: str, r: Row) -> None:
    if r.errors:
        print(f"  {kind:14} {r.model:45} ошибка: {r.errors[0]}")
        return
    cost = f"${statistics.mean(r.cost):.5f}" if r.cost else "—"
    extra = f"  CER {r.score:.1%}" if kind == "распознавание" and r.score is not None else ""
    extra += f"  понял задачу: {'да' if r.score else 'нет'}" if kind == "скриншот" else ""
    print(f"  {kind:14} {r.model:45} первый токен {_fmt(r.med(r.ttft)):>8}  всего {_fmt(r.med(r.total)):>8}  {cost}{extra}")


def _pick_answer(rows: list[Row]) -> Row | None:
    ok = [r for r in rows if not r.errors and r.ttft and len(r.sample) > 80]
    if not ok:
        return None
    # Самый быстрый до первого токена среди тех, кто не дороже 3× самого дешёвого.
    cheapest = min((statistics.mean(r.cost) for r in ok if r.cost), default=None)
    pool = [r for r in ok if cheapest is None or not r.cost or statistics.mean(r.cost) <= cheapest * 3 + 0.0005]
    return min(pool, key=lambda r: r.med(r.ttft))


def _pick_stt(rows: list[Row]) -> Row | None:
    ok = [r for r in rows if not r.errors and r.score is not None]
    if not ok:
        return None
    best_cer = min(r.score for r in ok)
    good = [r for r in ok if r.score <= best_cer + 0.03]  # почти так же точно, как лучший…
    return min(good, key=lambda r: (statistics.mean(r.cost) if r.cost else 0, r.med(r.total)))  # …и дешевле


async def run_bench(apply: bool = False, quick: bool = False) -> int:
    cfg = config_mod.load()
    if not cfg.api_key:
        print("Нужен ключ OpenRouter: sotto setup или OPENROUTER_API_KEY")
        return 1
    llm = OpenRouter(cfg.api_key, cfg.openrouter.base_url, cfg.openrouter.provider_sort)
    catalog = {m["id"] for m in await llm.models()}
    stt_catalog = set()
    try:
        r = await llm.http.get(f"{cfg.openrouter.base_url}/models", params={"output_modalities": "transcription"})
        stt_catalog = {m["id"] for m in r.json()["data"]}
    except Exception:  # noqa: BLE001
        pass

    def present(cands: list[str], current: str, cat: set[str]) -> list[str]:
        out = [current] + [c for c in cands if c != current]
        out = [c for c in out if c in cat or not cat]
        return out[:4] if quick else out

    print("\n— Ответы (3 прогона, медиана) —")
    answers = await bench_answers(llm, present(ANSWER_CANDIDATES, cfg.models.answer, catalog), 2 if quick else 3)
    print("\n— Распознавание речи —")
    stt = await bench_stt(llm, present(STT_CANDIDATES, cfg.models.stt, stt_catalog), cfg)
    print("\n— Скриншоты —")
    vision = await bench_vision(llm, present(VISION_CANDIDATES, cfg.models.vision, catalog))
    await llm.close()

    best_answer, best_stt = _pick_answer(answers), _pick_stt(stt)
    understood = [r for r in vision if r.score and not r.errors]
    best_vision = min(understood, key=lambda r: r.med(r.ttft)) if understood else None
    print("\n— Рекомендация —")
    for role, row in (("answer", best_answer), ("stt", best_stt), ("vision", best_vision)):
        print(f"  {role:8} {row.model if row else '(без изменений)'}")
    if apply:
        if best_answer:
            cfg.models.answer = cfg.models.utility = cfg.models.answer_short = best_answer.model
        if best_stt:
            cfg.models.stt = best_stt.model
        if best_vision:
            cfg.models.vision = best_vision.model
        config_mod.save(cfg)
        print("Записано в config.toml. Демон подхватит после «sotto ctl reload» или перезапуска.")
    else:
        print("Чтобы применить: sotto bench --apply")
    return 0
