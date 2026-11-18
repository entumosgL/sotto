"""Тренировочное собеседование голосом: ИИ-интервьюер ведёт разговор, суфлёр подсказывает.

Как устроен разговор (чтобы было похоже на живого человека):
  • роль и язык выбираешь в начале; вопросы интервьюер придумывает сам под роль и задаёт
    уточняющие по твоим ответам («а почему так?», «а цифры какие?»);
  • ждёт, пока ты договоришь: после законченной фразы — ~2 с тишины, после оборванной
    («Я бизнес…», «…и», «, ну») — до 6 с;
  • если заговоришь, пока он говорит, — замолкает и слушает (нужны наушники: без них микрофон
    слышит его самого — тогда перебивание выключается само);
  • реплику произносит по предложениям: первое звучит, пока пишется следующее.

Интервьюера пишет та же модель, что и подсказки ([models].answer), через OpenRouter.
Звонок помечается тренировочным: придуманная здесь легенда не попадёт в настоящие звонки.
"""

from __future__ import annotations

import asyncio
import json
import queue
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from .... import config as config_mod
from .... import paths
from ....application import interviewer
from ....application.interviewer import CONTEXT, END_MARK, LANGS, PLAN_RULES
from ....infrastructure.audio import tts
from ....infrastructure.llm.openrouter import OpenRouter
from ....infrastructure.prompts import prompts

DONE_SILENCE = 2.0          # тишина после законченной фразы — ты договорил
UNFINISHED_SILENCE = 6.0    # после оборванной фразы ждём дольше
ANSWER_START_TIMEOUT = 25.0
MAX_ANSWER = 240.0
BARGE_IN_SPEECH = 0.6       # столько твоей речи поверх интервьюера — он замолкает
BRIEFING = """
Собеседование голосом, 10–15 минут. Интервьюер говорит из колонок — лучше в наушниках. Ты отвечаешь
вслух, как на настоящем звонке. Он ждёт, пока ты договоришь, задаёт уточняющие вопросы, а если
заговоришь поверх него — замолкает.

Суфлёр:
  - подсказка появляется в оверлее сама через 1–2 с после вопроса; начинай говорить своими словами
    и подхватывай мысли из подсказки;
  - другим оттенком — придуманный опыт («легенда»), суфлёр будет его держаться;
  - оверлей прозрачен для мыши; передвинуть или растянуть — SUPER+SHIFT+O, поставь его под камеру;
  - SUPER+A — подсказать сейчас, SUPER+H — спрятать оверлей.

Ctrl+C — закончить в любой момент.
"""


def read_vacancy_source(answer: str) -> str | None:
    """Текст вакансии из буфера обмена («б»/«b») или из файла. None — источник не похож ни на то, ни на другое."""
    import os

    from ....infrastructure.platform import current
    a = answer.strip().strip("'\"")
    if a.lower() in ("б", "b", "буфер", "clipboard"):
        return current().clipboard()
    path = os.path.expanduser(a)
    if a and os.path.isfile(path):
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    return None


async def digest_vacancy(raw: str, cfg: config_mod.Config) -> dict[str, str] | None:
    llm = OpenRouter(cfg.api_key, cfg.openrouter.base_url, cfg.openrouter.provider_sort)
    try:
        return await interviewer.digest_vacancy(llm, cfg.models.answer, prompts, raw)
    finally:
        await llm.close()


@dataclass
class Stats:
    names: dict[int, str] = field(default_factory=dict)
    hint_ms: list[int] = field(default_factory=list)   # конец вопроса → первые слова подсказки
    hints: int = 0
    turns: int = 0
    interrupted: int = 0


def looks_unfinished(text: str) -> bool:
    """«Я бизнес… блин…», «…и», «…, ну» — человек запнулся, а не закончил."""
    t = text.strip().lower()
    if not t:
        return False
    if t.endswith(("...", "…", ",", "-", "—")):
        return True
    if len(re.findall(r"[\w']+", t)) < 4:  # «Я бизнес.» — Whisper ставит точку и там, где человек запнулся
        return True
    last = re.findall(r"[\w']+", t)[-1:]
    return bool(last) and last[0] in {"и", "а", "но", "что", "чтобы", "потому", "ну", "в", "на", "с", "по", "для",
                                      "это", "как", "когда", "and", "but", "so", "the", "a", "to", "because", "of"}


class Speaker:
    """Очередь фраз интервьюера: синтез и воспроизведение по предложениям, можно оборвать."""

    def __init__(self, voice: str, target: str | None):
        self.voice, self.target = voice, target
        self.q: queue.Queue[str] = queue.Queue()
        self.proc = None
        self.busy = False
        self._cancel = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def say(self, sentence: str) -> None:
        self.busy = True
        self.q.put(sentence)

    def cancel(self) -> None:
        self._cancel.set()
        while not self.q.empty():
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()

    def idle(self) -> bool:
        return not self.busy and self.q.empty()

    def _run(self) -> None:
        while True:
            text = self.q.get()
            self._cancel.clear()
            self.busy = True
            try:
                # страховка: английский голос, читающий кириллицу, — нечленораздельная тарабарщина, и наоборот
                voice = tts.pick_voice(text, LANGS["ru"]["voice"], LANGS["en"]["voice"])
                samples, sr = tts.synthesize(text, voice)
                if not self._cancel.is_set():
                    self.proc = tts.start_playback(samples, sr, self.target)
                    while self.proc.poll() is None:
                        if self._cancel.is_set():
                            self.proc.terminate()
                            break
                        time.sleep(0.05)
            except Exception:  # noqa: BLE001 — одна фраза не должна ронять разговор
                pass
            finally:
                self.busy = not self.q.empty()


class Interview:
    def __init__(self, role: str, lang: str, base: str, cfg: config_mod.Config, tts_target: str | None = None,
                 vacancy: str = "", resume: str = ""):
        self.role, self.lang, self.base, self.cfg = role, lang, base, cfg
        ctx = CONTEXT[lang]
        self.context = ((ctx["vacancy"].format(text=vacancy) if vacancy else "")
                        + (ctx["resume"].format(text=resume) if resume else ""))
        self.plan = ""
        self._plan_task: asyncio.Task | None = None
        self.info = LANGS[lang]
        self.name = self.info["name"]
        self.llm = OpenRouter(cfg.api_key, cfg.openrouter.base_url, cfg.openrouter.provider_sort)
        self.speaker = Speaker(self.info["voice"], tts_target)
        self.stats = Stats()
        self.history: list[dict[str, str]] = []
        self.me_speaking = False
        self.me_speech_since: float | None = None   # с какого момента ты говоришь без паузы
        self.last_me_voice = 0.0
        self.me_utterances: list[str] = []
        self.question_end = 0.0
        self.waiting_hint = False
        self.barge_in = True
        self._echo_samples: list[bool] = []

    # --- события демона ---

    async def _events(self) -> None:
        async with httpx.AsyncClient(timeout=None) as http:
            async with http.stream("GET", f"{self.base}/api/events") as r:
                async for line in r.aiter_lines():
                    if line.startswith("data: "):
                        self._on_event(json.loads(line[6:]))

    def _on_event(self, e: dict[str, Any]) -> None:
        t = e["type"]
        if t == "audio.level":
            speaking = bool(e.get("me_speaking"))
            now = time.time()
            if speaking and not self.me_speaking:
                self.me_speech_since = now
            if not speaking:
                self.me_speech_since = None
            self.me_speaking = speaking
            if speaking:
                self.last_me_voice = now
            if not self.speaker.idle() and self.stats.turns == 0:
                self._echo_samples.append(speaking)  # пока звучит приветствие — слышит ли микрофон колонки
        elif t == "utterance" and e["channel"] == "me":
            self.me_utterances.append(e["text"])
            self.last_me_voice = max(self.last_me_voice, time.time() - 0.3)
        elif t == "speaker" and e.get("name"):
            self.stats.names[e["id"]] = e["name"]
        elif t == "answer.delta" and self.waiting_hint:
            self.waiting_hint = False
            self.stats.hints += 1
            self.stats.hint_ms.append(int((time.time() - self.question_end) * 1000))

    # --- реплика интервьюера ---

    async def speak_turn(self) -> str:
        """Генерирует реплику и произносит её по предложениям. Возвращает текст с маркером конца, если он был."""
        if self.history and self._plan_task is not None:
            # план пишется, пока звучит приветствие и кандидат представляется — к первому вопросу он готов
            try:
                self.plan = await asyncio.wait_for(self._plan_task, timeout=15)
            except Exception:  # noqa: BLE001 — без плана интервьюер всё равно справится
                self.plan = ""
            self._plan_task = None
        plan = PLAN_RULES[self.lang].format(plan=self.plan) if self.plan else ""
        system = interviewer.system_prompt(prompts, self.lang, name=self.name, role=self.role, end=END_MARK, context=self.context + plan)
        messages = [{"role": "system", "content": system}, *self.history]
        if not self.history:
            messages.append({"role": "user", "content": self.info["joined"]})
        text, pending = "", ""
        interrupted = False
        self.waiting_hint = False
        stream = self.llm.stream_chat(self.cfg.models.answer, messages, max_tokens=300, temperature=0.7)
        try:
            async for delta, _res in stream:
                if self._user_barged_in():
                    interrupted = True
                    break
                text += delta
                pending += delta
                parts = re.split(r"(?<=[.!?…])\s+", pending)
                for sentence in parts[:-1]:
                    self._say(sentence)
                pending = parts[-1]
        finally:
            await stream.aclose()
        if not interrupted:
            self._say(pending)
        spoken = text.replace(END_MARK, "").strip()
        print(f"\n{self.name}: {spoken}")
        while not interrupted and not self.speaker.idle():
            if self._user_barged_in():
                interrupted = True
                break
            await asyncio.sleep(0.05)
        if interrupted:
            self.speaker.cancel()
            self.stats.interrupted += 1
            print("  (ты перебил — слушаю)" if self.lang == "ru" else "  (interrupted — listening)")
        self.history.append({"role": "assistant", "content": spoken})
        self.question_end = time.time()
        self.waiting_hint = True
        self._check_echo()
        return text

    def _say(self, sentence: str) -> None:
        sentence = sentence.replace(END_MARK, "").strip()
        if sentence:
            self.speaker.say(sentence)

    def _user_barged_in(self) -> bool:
        return (self.barge_in and self.me_speech_since is not None
                and time.time() - self.me_speech_since >= BARGE_IN_SPEECH)

    def _check_echo(self) -> None:
        if self._echo_samples and self.stats.turns == 0:
            share = sum(self._echo_samples) / len(self._echo_samples)
            if share > 0.4:
                self.barge_in = False
                print("  Микрофон слышит интервьюера из колонок — перебивать его не получится. Лучше в наушниках."
                      if self.lang == "ru" else "  The mic hears the speakers — interrupting is off. Use headphones.")
            self._echo_samples.clear()

    # --- ответ кандидата ---

    async def listen(self) -> str:
        """Ждём, пока ты ответишь и договоришь."""
        start = time.time()
        while time.time() - start < ANSWER_START_TIMEOUT:
            if self.me_speaking or self.me_utterances:
                break
            await asyncio.sleep(0.1)
        else:
            return ""
        while time.time() - start < MAX_ANSWER:
            quiet = time.time() - self.last_me_voice
            if not self.me_speaking:
                text = " ".join(self.me_utterances)
                need = UNFINISHED_SILENCE if (not self.me_utterances or looks_unfinished(text)) else DONE_SILENCE
                if quiet >= need:
                    break
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.3)  # последняя реплика ещё распознаётся
        answer = " ".join(self.me_utterances).strip()
        self.me_utterances.clear()
        if answer:
            print(f"Ты: {answer}")
        return answer

    async def _make_plan(self) -> str:
        prompt = interviewer.plan_prompt(prompts, self.lang, name=self.name, role=self.role, context=self.context)
        res = await self.llm.chat(self.cfg.models.answer, [{"role": "user", "content": prompt}],
                                  max_tokens=1200, temperature=0.7)
        return res.text.strip()

    async def run(self) -> Stats:
        self._plan_task = asyncio.create_task(self._make_plan())
        events = asyncio.create_task(self._events())
        await asyncio.sleep(0.5)
        try:
            for _ in range(30):
                self.me_utterances.clear()
                said = await self.speak_turn()
                self.stats.turns += 1
                if END_MARK in said:
                    break
                answer = await self.listen()
                self.history.append({"role": "user", "content": answer or self.info["silence"]})
            while not self.speaker.idle():
                await asyncio.sleep(0.1)
        finally:
            events.cancel()
            if self._plan_task is not None:
                self._plan_task.cancel()
            self.speaker.cancel()
            await self.llm.close()
        return self.stats


def ask(prompt: str, default: str) -> str:
    answer = input(f"{prompt} [{default}]: ").strip()
    return answer or default


def run_interview(role: str | None = None, lang: str | None = None, tts_target: str | None = None) -> int:
    cfg = config_mod.load()
    base = f"http://{cfg.web.host}:{cfg.web.port}"
    try:
        status = httpx.get(f"{base}/api/status", timeout=3).json()
    except httpx.HTTPError:
        status = {"panel_only": True}
    if status.get("panel_only"):
        print("Сначала запусти голосовой суфлёр: кнопкой на сайте (sotto) или командой sotto run")
        return 1
    if not cfg.api_key:
        print("Нужен ключ OpenRouter: интервьюер — языковая модель.")
        return 1

    print("\nТренировочное собеседование")
    print(BRIEFING)
    try:
        if lang is None:
            lang = ask("Язык собеседования: ru или en", "ru").lower()
            lang = "en" if lang.startswith(("en", "анг")) else "ru"
        vacancy = ""
        if role is None:
            print("\nВакансия: открой её на сайте, выдели всё (Ctrl+A), скопируй (Ctrl+C) и введи «б».")
            print("Или путь к файлу с текстом вакансии, или просто название роли.")
            saved = paths.CONFIG_DIR / "vacancy.txt"
            role = ask("Вакансия или роль", str(saved) if saved.exists() else LANGS[lang]["default_role"])
        raw = read_vacancy_source(role)
        if raw is not None:
            print("Читаю вакансию…")
            v = asyncio.run(digest_vacancy(raw, cfg)) if raw.strip() else None
            if v is None:
                print("Не похоже на вакансию (пустой буфер или не тот текст).")
                role = ask("Роль", LANGS[lang]["default_role"])
            else:
                role = v["role"] + (f" в {v['company']}" if v["company"] else "")
                vacancy = v["brief"]
                print(f"Вакансия: {role}\n{vacancy}\n")
        if not tts_target:
            input("Enter — начать. ")
    except (KeyboardInterrupt, EOFError):
        print()
        return 0

    if vacancy:  # суфлёр тоже знает вакансию: подсказки под её требования
        httpx.post(f"{base}/brief", data={"brief": f"{role}\n{vacancy}"}, timeout=10)
    resume = prompts.profile()
    if not resume:
        print("Профиль пуст: интервьюер не видит резюме. Заполни ~/.config/sotto/profile.md или в панели.")

    started_call = not status["in_call"]
    if started_call:
        httpx.post(f"{base}/api/ctl/toggle_training_call", timeout=10)
    switched_mode = status["mode"] != "auto"
    if switched_mode:
        httpx.post(f"{base}/api/ctl/toggle_mode", timeout=10)
    call_id = httpx.get(f"{base}/api/status", timeout=3).json().get("call_id")

    interview = Interview(role, lang, base, cfg, tts_target, vacancy=vacancy, resume=resume)
    try:
        asyncio.run(interview.run())
    except KeyboardInterrupt:
        interview.speaker.cancel()
        print("\nЗакончили.")
    finally:
        try:
            if started_call:
                httpx.post(f"{base}/api/ctl/toggle_training_call", timeout=10)
            if switched_mode:
                httpx.post(f"{base}/api/ctl/toggle_mode", timeout=10)
        except httpx.HTTPError:
            pass  # суфлёр уже остановлен

    stats = interview.stats
    print("\nИтоги")
    names = ", ".join(sorted(set(stats.names.values()))) or "не узнал"
    print(f"  Имя интервьюера суфлёр услышал как: {names}")
    if stats.hint_ms:
        ms = sorted(stats.hint_ms)
        print(f"  Подсказок: {stats.hints}, первые слова через {ms[len(ms) // 2] / 1000:.1f} с после конца вопроса "
              f"(от {ms[0] / 1000:.1f} до {ms[-1] / 1000:.1f} с)")
    if call_id:
        print(f"  Запись и подсказки: {base}/calls/{call_id} (сайт: sotto)")
    return 0
