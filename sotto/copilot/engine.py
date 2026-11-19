"""Движок подсказок: ручной и автоматический режимы, скриншоты, вопросы собеседнику, легенда.

События для оверлея:
  answer.start  {id, kind, slot, model}
  answer.delta  {id, slot, text}            — кусок текста
  answer.done   {id, slot, text, first_token_ms, total_ms, cost, legend}
  answer.skip   {id, slot}                   — модель решила, что отвечать не нужно (авто-режим)
  answer.error  {id, slot, error}
"""

from __future__ import annotations

import asyncio
import re
import base64
import difflib
import logging
import time
from pathlib import Path
from typing import Any, Callable

from .. import desktop, paths
from ..config import Config
from ..db import DB
from ..events import EventBus
from ..openrouter import OpenRouter, OpenRouterError
from . import context, prompts

log = logging.getLogger(__name__)

SKIP = "∅"


BACKCHANNEL_WORDS = {
    "угу", "ага", "ну", "да", "так", "понятно", "ясно", "хорошо", "окей", "ок", "отлично", "супер", "ладно",
    "спасибо", "понял", "поняла", "принято", "здорово", "класс", "хм", "ммм", "мм", "аха", "ок", "интересно",
    "вот", "это", "уже", "большое", "отличный", "хороший", "ответ", "круто",
    "ok", "okay", "mhm", "uh", "huh", "yeah", "yes", "yep", "right", "sure", "great", "got", "it", "i", "see",
    "cool", "nice", "thanks", "thank", "you", "good", "perfect", "alright", "understood", "makes", "sense",
    "interesting", "awesome",
}


def is_backchannel(text: str) -> bool:
    """«Угу, понятно», «Okay, got it» — поддакивание без вопроса."""
    if "?" in text:
        return False
    words = re.findall(r"[\w']+", text.lower())
    return 0 < len(words) <= 5 and all(w in BACKCHANNEL_WORDS for w in words)


def is_interjection(text: str) -> bool:
    """Короткая реплика собеседника посреди ответа кандидата — кашель, «ага», обрывок. Не вопрос:
    перегенерировать подсказку, которую кандидат сейчас читает, нельзя."""
    return "?" not in text and len(re.findall(r"[\w']+", text)) <= 3


AUTO_ASK = "Подскажи ответ на последний вопрос/реплику собеседника."
_HANGING = re.compile(r"(,|\b(и|а|но|или|что|чтобы|потому|если|когда|как|в|на|с|по|для|and|or|but|so|the|a|to|of|with|for|that|if|when)|\.\.\.|…|-)$", re.I)


def looks_unfinished(text: str) -> bool:
    """«Расскажите, как вы…», «and the» — собеседник явно не договорил, упреждать нечего."""
    return bool(_HANGING.search(text.strip().rstrip(" ")))


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.lower()).split())


def _same_text(a: str, b: str) -> bool:
    """Whisper на том же звуке плюс 0.3 с тишины пишет почти то же самое — сравниваем без пунктуации."""
    a, b = _norm(a), _norm(b)
    return a == b or difflib.SequenceMatcher(None, a, b).ratio() >= 0.9


class _Speculation:
    def __init__(self, text: str, rows: list[dict[str, Any]]):
        self.text = text
        self.rows = rows
        self.task: asyncio.Task | None = None
        self.visible = False
        self.failed = False
        self.revealed = asyncio.Event()
        self.on_reveal: Callable[[], None] | None = None

    def reveal(self) -> None:
        self.visible = True
        self.revealed.set()
        if self.on_reveal:
            self.on_reveal()


class Copilot:
    def __init__(self, cfg: Config, db: DB, bus: EventBus, llm: OpenRouter,
                 them_speaking: Callable[[], bool], stage_name: Callable[[], str | None],
                 me_speaking: Callable[[], bool] = lambda: False):
        self.cfg = cfg
        self.db = db
        self.bus = bus
        self.llm = llm
        self.them_speaking = them_speaking
        self.me_speaking = me_speaking
        self.stage_name = stage_name
        self.call_id: int | None = None
        self.record_only = False
        self._tasks: dict[str, asyncio.Task] = {}   # slot → текущая генерация
        self._auto_timer: asyncio.Task | None = None
        self._spec: _Speculation | None = None
        self.last_hint = ""  # последняя показанная подсказка — по ней распознаётся речь кандидата     # подсказка, начатая до конца фразы собеседника

    # --- жизненный цикл звонка ---

    def set_call(self, call_id: int | None) -> None:
        self.call_id = call_id
        self.last_hint = ""
        self.cancel_all()

    @property
    def mode(self) -> str:
        return self.cfg.answer.mode

    def toggle_mode(self) -> str:
        self.cfg.answer.mode = "auto" if self.cfg.answer.mode == "manual" else "manual"
        return self.cfg.answer.mode

    def cancel_all(self) -> None:
        for t in self._tasks.values():
            t.cancel()
        self._tasks.clear()
        self.drop_speculation()
        if self._auto_timer:
            self._auto_timer.cancel()

    # --- триггеры ---

    def speculate(self, text: str) -> None:
        """Собеседник замолчал на 0.2 с. Начинаем подсказку сразу, но не показываем: если через 0.3 с
        реплика закончится тем же текстом — показываем то, что уже пишется (экономит ~1 с), если
        собеседник продолжил — выбрасываем."""
        if (self.mode != "auto" or self.record_only or self.call_id is None or self.cfg.answer.style == "dual"
                or is_backchannel(text) or len(text.split()) < 3 or looks_unfinished(text)
                or (self.me_speaking() and is_interjection(text))):
            self.drop_speculation()
            return
        if self._spec and _same_text(self._spec.text, text):
            return  # уже пишется по тому же тексту
        self.drop_speculation()
        rows = self._recent()
        last = next((r for r in reversed(rows) if r["channel"] == "them"), {})
        rows = rows + [{"channel": "them", "text": text, "started_at": time.time(),
                        "speaker": last.get("speaker"), "speaker_role": last.get("speaker_role")}]
        messages = [{"role": "system", "content": self._system("answer")},
                    {"role": "user", "content": context.transcript_message(rows, AUTO_ASK)}]
        log.debug("Упреждение: «%s»", text[-60:])
        spec = _Speculation(text, rows)
        spec.task = asyncio.create_task(self._run("main", "auto", self.cfg.models.answer, messages,
                                                  self.cfg.answer.max_tokens, rows, spec=spec))
        self._spec = spec

    def drop_speculation(self) -> None:
        spec, self._spec = self._spec, None
        if spec and not spec.visible and spec.task and not spec.task.done():
            spec.task.cancel()

    def _promote(self, final_text: str) -> bool:
        """Окончательная реплика совпала с предварительной — показываем уже начатую подсказку."""
        spec, self._spec = self._spec, None
        if spec is None or spec.task is None:
            return False
        if spec.task.done() and spec.failed or not _same_text(spec.text, final_text):
            log.debug("Упреждение не подошло: «%s» ≠ «%s»", spec.text[-50:], final_text[-50:])
            if not spec.task.done():
                spec.task.cancel()
            return False
        old = self._tasks.get("main")
        if old and not old.done() and old is not spec.task:
            old.cancel()
        self._tasks["main"] = spec.task
        spec.reveal()
        log.debug("Упреждение подошло, подсказка уже пишется")
        return True

    def on_utterance(self, channel: str) -> None:
        """Вызывается после сохранения новой реплики. В авто-режиме ждём паузу и отвечаем."""
        if channel != "them" or self.mode != "auto" or self.record_only or self.call_id is None:
            return
        if self._auto_timer and not self._auto_timer.done():
            self._auto_timer.cancel()
        self._auto_timer = asyncio.create_task(self._auto_after_pause())

    async def _auto_after_pause(self) -> None:
        pause = self.cfg.answer.auto_pause_ms / 1000
        if pause > 0:
            await asyncio.sleep(pause)
        # Собеседник продолжил говорить — дождёмся следующей реплики (она снова запустит таймер).
        if self.them_speaking():
            return
        rows = self._recent()
        last = next((r for r in reversed(rows) if r["channel"] == "them"), None)
        if last and is_backchannel(last["text"]):
            self.drop_speculation()
            return  # «угу», «понятно, спасибо» — отвечать не на что, модель не спрашиваем
        if last and self.me_speaking() and is_interjection(last["text"]):
            self.drop_speculation()
            log.debug("Реплика посреди ответа кандидата, подсказку не трогаю: «%s»", last["text"])
            return
        if last and self._promote(last["text"]):
            return
        await self.answer(kind="auto")

    async def answer(self, kind: str = "manual") -> None:
        if self.record_only:
            self.bus.publish("notice", {"text": "Режим «только запись»: подсказки выключены"})
            return
        rows = self._recent()
        if self.cfg.answer.style == "dual":
            jobs = [("short", "answer_short", self.cfg.models.answer_short, 250),
                    ("detailed", "answer_detailed", self.cfg.models.answer_detailed, self.cfg.answer.max_tokens * 2)]
        else:
            jobs = [("main", "answer", self.cfg.models.answer, self.cfg.answer.max_tokens)]
        for slot, prompt_kind, model, max_tokens in jobs:
            ask = AUTO_ASK
            if kind != "auto":  # нажали хоткей — человеку нужна подсказка, «∅» здесь недопустим
                ask += (" Ответ обязателен: если последняя реплика — не вопрос, подскажи, что сказать "
                        "или на какой вопрос из разговора ответить. Символ ∅ не используй.")
            messages = [
                {"role": "system", "content": self._system(prompt_kind)},
                {"role": "user", "content": context.transcript_message(rows, ask)},
            ]
            self._launch(slot, kind, model, messages, max_tokens, rows)

    async def screen(self, region: bool = False) -> None:
        stage = self.stage_name()
        output = stage or await asyncio.to_thread(desktop.focused_monitor)
        dest = paths.SCREENSHOTS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.jpg"
        shot = await asyncio.to_thread(desktop.screenshot, dest, output, region)
        if shot is None:
            return
        win = await asyncio.to_thread(desktop.active_window)
        window_info = f"Активное окно: {win.get('class', '?')} — «{win.get('title', '?')}»" if win else ""
        rows = self._recent()
        messages = [
            {"role": "system", "content": self._system("screen")},
            {"role": "user", "content": []},
        ]
        text = f"{window_info}\nСкриншот экрана кандидата ниже. Помоги с тем, что на нём."
        if not self.cfg.models.vision:
            # Провайдер без моделей со зрением (DeepSeek, локальный текстовый сервер): картинку не шлём.
            self.bus.publish("notice", {"text": "Скриншоты: у выбранных моделей нет зрения — "
                                                "подсказываю по разговору и названию окна"})
            text = (f"{window_info}\nСкриншот посмотреть нечем (у модели нет зрения). "
                    "Помоги с тем, что происходит у кандидата на экране, по разговору и названию окна.")
        content: list[dict] = [{"type": "text", "text": context.transcript_message(rows, text)}]
        if self.cfg.models.vision:
            image = base64.b64encode(Path(shot).read_bytes()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image}"}})
        messages[1]["content"] = content
        model = self.cfg.models.vision or self.cfg.models.answer
        self._launch("main", "screen", model, messages, 1500, rows)

    async def questions(self) -> None:
        rows = self._recent()
        messages = [
            {"role": "system", "content": self._system("questions")},
            {"role": "user", "content": context.transcript_message(rows, "Какие вопросы задать собеседнику?")},
        ]
        self._launch("main", "questions", self.cfg.models.answer, messages, 300, rows)

    # --- внутреннее ---

    def _recent(self) -> list[dict[str, Any]]:
        if self.call_id is None:
            return []
        return self.db.call_transcript(self.call_id, since=context.since(self.cfg.answer.context_minutes))

    def _system(self, kind: str) -> str:
        brief = None
        if self.call_id is not None:
            row = self.db.one("SELECT brief FROM calls WHERE id = ?", [self.call_id])
            brief = row["brief"] if row else None
        return context.system_prompt(
            kind, allow_invention=self.cfg.answer.allow_invention, profile=prompts.profile(),
            legend=self.db.active_legend(self.call_id), brief=brief)

    def reading_hint(self) -> str:
        return self.last_hint.replace("[[", "").replace("]]", "") if self.call_id is not None else ""

    def _launch(self, slot: str, kind: str, model: str, messages: list[dict], max_tokens: int,
                rows: list[dict[str, Any]]) -> None:
        old = self._tasks.get(slot)
        if old and not old.done():
            old.cancel()
        self._tasks[slot] = asyncio.create_task(self._run(slot, kind, model, messages, max_tokens, rows))

    async def _run(self, slot: str, kind: str, model: str, messages: list[dict], max_tokens: int,
                   rows: list[dict[str, Any]], spec: "_Speculation | None" = None) -> None:
        """spec — упреждающая генерация: пока её не раскрыли (spec.reveal), ничего не публикуем,
        текст копится; при раскрытии публикуется всё накопленное и дальше — как обычно."""
        answer_id = f"{int(time.time() * 1000)}-{slot}"
        question = next((r["text"] for r in reversed(rows) if r["channel"] == "them"), "")
        start = {"id": answer_id, "kind": kind, "slot": slot, "model": model,
                 "question": question if kind in ("auto", "manual") else ""}
        state = {"sent": 0, "text": ""}  # сколько символов уже показали; накопленный текст

        def show() -> None:
            text = state["text"]
            # Пока непонятно, не «∅» ли это, ничего не показываем: иначе затрём прошлый ответ.
            if text.startswith(SKIP) or len(text) < 2 or (spec and not spec.visible):
                return
            if len(text) > state["sent"]:
                self.bus.publish("answer.delta", {"id": answer_id, "slot": slot, "text": text[state["sent"]:]})
                state["sent"] = len(text)
                if slot == "main":
                    self.last_hint = text

        finished: list[Any] = []

        async def finish(result: Any) -> None:
            if spec and not spec.visible:
                finished.append(result)  # дописали раньше, чем реплика подтвердилась
                await spec.revealed.wait()
            show()
            await self._finish(answer_id, slot, kind, result, rows)

        if spec:
            def on_reveal() -> None:
                self.bus.publish("answer.start", start)
                show()
            spec.on_reveal = on_reveal
        else:
            self.bus.publish("answer.start", start)
        session = f"sotto-call-{self.call_id}" if self.call_id else None
        fallback = self.cfg.models.answer_fallback
        models = [model] + ([fallback] if fallback and fallback != model and slot == "main" else [])
        for attempt, current in enumerate(models):
            stream = self.llm.stream_chat(current, messages, max_tokens=max_tokens, session_id=session)
            try:
                async for delta, result in stream:
                    if delta:
                        state["text"] = result.text.lstrip()
                        show()
                    else:
                        await finish(result)
                return
            except asyncio.CancelledError:
                raise
            except (OpenRouterError, Exception) as e:  # noqa: BLE001 — показываем любую ошибку в оверлее
                if state["sent"] == 0 and not state["text"] and attempt + 1 < len(models):
                    log.warning("Модель %s не ответила (%s) — подсказку пишет %s", current, str(e)[:120],
                                models[attempt + 1])
                    continue
                if spec and not spec.visible:
                    spec.failed = True  # окончательная реплика запустит обычную подсказку
                    return
                log.exception("Ошибка генерации")
                self.bus.publish("answer.error", {"id": answer_id, "slot": slot, "error": str(e)[:300]})
                return
            finally:
                await stream.aclose()  # оборванная генерация закрывает HTTP-поток сразу

    async def _finish(self, answer_id: str, slot: str, kind: str, result: Any,
                      rows: list[dict[str, Any]]) -> None:
        text = result.text.strip()
        if not text or text.startswith(SKIP):
            self.bus.publish("answer.skip", {"id": answer_id, "slot": slot})
            return
        last_them = next((r["text"] for r in reversed(rows) if r["channel"] == "them"), "")
        row_id = self.db.insert(
            "answers", call_id=self.call_id, kind=kind if slot == "main" else slot, model=result.model,
            prompt_tail=last_them, text=text, created_at=time.time(),
            first_token_ms=result.first_token_ms, total_ms=result.total_ms, cost=result.cost)
        legend = context.extract_legend(text) if kind != "questions" else []
        call = self.db.one("SELECT app, company FROM calls WHERE id = ?", [self.call_id]) if self.call_id else None
        if call and call["app"] != "training":  # тренировки начинаются с чистого листа — легенду не копим
            known = {x["claim"].lower() for x in self.db.active_legend(self.call_id)}
            for claim in legend:
                if claim.lower() not in known:
                    self.db.insert("legend", claim=claim, call_id=self.call_id, answer_id=row_id,
                                   company=call["company"], created_at=time.time())
        self.bus.publish("answer.done", {
            "id": answer_id, "slot": slot, "text": text, "first_token_ms": result.first_token_ms,
            "total_ms": result.total_ms, "cost": result.cost, "legend": legend})
