"""Демон Sotto: связывает звук, звонки, запись, подсказки, Hyprland, оверлей и веб-панель.

Запуск: `sotto run`. Всё управление — через локальный HTTP (127.0.0.1:8765):
хоткеи вызывают `sotto ctl <действие>`, оверлей и веб-панель слушают /api/events.
"""

from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np
import uvicorn

from . import config as config_mod
from . import hypr, paths
from . import interviews
from .agents import Agents
from .audio.capture import AudioCapture
from .calls import CallDetector
from .config import Config
from .copilot.engine import Copilot
from .copilot.listener import Listener
from .db import DB
from .events import EventBus
from .openrouter import OpenRouter
from .postcall import process_call
from .recorder import ScreenRecorder
from .search import Search
from .trainer.topics import refresh_soon

log = logging.getLogger(__name__)

def sotto_command() -> str:
    exe = shutil.which("sotto")
    if exe:
        return exe
    # Запуск из исходников (nix develop): хоткеи Hyprland не видят окружение оболочки.
    return f"env PYTHONPATH={paths.PACKAGE_DIR.parent} {sys.executable} -m sotto"


class Daemon:
    def __init__(self, cfg: Config, with_overlay: bool = True, with_hyprland: bool = True):
        self.cfg = cfg
        self.with_overlay = with_overlay
        self.with_hyprland = with_hyprland
        paths.ensure_dirs()
        self.db = DB()
        self.bus = EventBus()
        self.llm = OpenRouter(cfg.api_key, cfg.openrouter.base_url, cfg.openrouter.provider_sort)
        self.agents = Agents(self.db, cfg.agents)
        self.search = Search(self.db)
        self.stage = hypr.Stage(cfg.stage.name, cfg.stage.mirror_on)
        self.listener = Listener(cfg, self.db, self.bus, self.llm, on_utterance=self._on_utterance)
        self.copilot = Copilot(cfg, self.db, self.bus, self.llm, them_speaking=self.listener.them_speaking,
                               me_speaking=self.listener.me_speaking,
                               stage_name=lambda: self.stage.name if self.stage_active() else None)
        self.listener.on_early = self.copilot.speculate
        self.listener.on_resume = self.copilot.drop_speculation
        self.listener.reading_hint = self.copilot.reading_hint
        self.recorder = ScreenRecorder(cfg.recording, self.db)
        self.capture = AudioCapture(cfg.audio.mic, cfg.audio.system, self._on_audio)
        self.detector = CallDetector(cfg.calls, on_start=self._call_started_threadsafe,
                                     on_end=self._call_ended_threadsafe)
        self.loop: asyncio.AbstractEventLoop | None = None
        self.call_id: int | None = None
        self.overlay_visible = True
        self._overlay_proc: subprocess.Popen | None = None
        self._stop = threading.Event()
        self._background: set[asyncio.Task] = set()
        self.trainer = None  # TrainerSession, пока идёт голосовая тренировка
        self.capture_used = False
        self.voice = False   # голосовой суфлёр (звук, звонки, запись, оверлей, хоткеи) запущен
        self._voice_stop = threading.Event()
        from .textchat import TextSessions
        self.chats = TextSessions(self)
        self.interview_draft: dict[str, Any] = {}  # разобранное приглашение, пока его не сохранили
        self.overlay_allowed = False      # правило no_screen_share принято Hyprland
        self._overlay_auto_pos: tuple[int, int] | None = None
        self._overlay_address: str | None = None
        self._prev_focus: str | None = None
        self._bound_keys: list[str] = []
        self._stage_cache: tuple[float, bool] = (0.0, False)

    # --- состояние для оверлея и панели ---

    def status(self) -> dict[str, Any]:
        return {
            "mode": self.cfg.answer.mode,
            "style": self.cfg.answer.style,
            "in_call": self.call_id is not None,
            "call_id": self.call_id,
            "record_only": self.copilot.record_only,
            "recording": self.recorder.segment is not None,
            "stage": self.stage_active(),
            "overlay_visible": self.overlay_visible,
            "api_key": bool(self.cfg.api_key),
            "voice": self.voice,
            "panel_only": not self.voice,
        }

    def publish_status(self) -> None:
        self.bus.publish("status", self.status())

    # --- звук ---

    def _on_audio(self, channel: str, samples: np.ndarray, ts: float) -> None:
        self.recorder.write_audio(channel, samples)
        self.listener.on_chunk(channel, samples, ts)
        trainer = self.trainer
        if trainer is not None and channel == "me" and not trainer.stopped:
            trainer.on_mic(samples, ts)

    def _on_utterance(self, channel: str) -> None:
        self.copilot.on_utterance(channel)

    # --- звонки ---

    def _call_started_threadsafe(self, app: str) -> None:
        assert self.loop is not None
        self.loop.call_soon_threadsafe(self.call_started, app)

    def _call_ended_threadsafe(self) -> None:
        assert self.loop is not None
        self.loop.call_soon_threadsafe(self.call_ended)

    def call_started(self, app: str) -> None:
        if self.call_id is not None:
            return
        if self.trainer is not None and not self.trainer.stopped:
            self.trainer.stop()  # начался настоящий звонок — тренировку прекращаем
        self.call_id = self.db.insert("calls", started_at=time.time(), app=app,
                                      record_only=int(self.copilot.record_only))
        pending_brief = self.db.get_meta("next_call_brief")
        if pending_brief:
            self.db.update("calls", self.call_id, brief=pending_brief)
            self.db.set_meta("next_call_brief", "")
        planned = interviews.find_for_call(self.db, time.time())
        if planned:  # звонок в окне запланированного собеседования — бриф и компания оттуда
            interviews.attach_call(self.db, self.call_id, planned)
            log.info("Звонок привязан к собеседованию: %s", planned["company"])
        self.listener.start_call(self.call_id)
        self.copilot.set_call(self.call_id)
        self.recorder.set_call(True)
        log.info("Звонок #%s (%s)", self.call_id, app)
        self.publish_status()

    def call_ended(self) -> None:
        if self.call_id is None:
            return
        call_id = self.call_id
        self.db.update("calls", call_id, ended_at=time.time())
        self.call_id = None
        self.listener.end_call()
        self.copilot.set_call(None)
        self.recorder.set_call(False)
        self.publish_status()
        self._spawn(self.process_call(call_id))

    async def process_call(self, call_id: int) -> None:
        """Итоги звонка; после собеседования темы тренера обновляются под то, что спрашивали."""
        data = await process_call(call_id, self.db, self.agents, self.search, self.bus)
        if data and data.get("kind") == "interview":
            refresh_soon(self, force=True)

    def _finish_interrupted_calls(self) -> None:
        """Итоги звонков, которые не успели посчитаться (демон остановили сразу после звонка)."""
        week = time.time() - 7 * 86400
        for row in self.db.all(
                """SELECT c.id FROM calls c WHERE c.ended_at IS NOT NULL AND c.ended_at > ?
                   AND (c.title IS NULL OR c.title = '')
                   AND (SELECT COUNT(*) FROM utterances u WHERE u.call_id = c.id) >= 4""", [week]):
            log.info("Досчитываю итоги звонка #%s", row["id"])
            self._spawn(self.process_call(row["id"]))

    def _spawn(self, coro) -> None:
        task = asyncio.ensure_future(coro)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    # --- действия (хоткеи / API) ---

    async def action(self, name: str) -> dict[str, Any]:
        if name == "ask":
            await self.copilot.answer("manual")
        elif name in ("screen", "screen_region"):
            self._spawn(self.copilot.screen(region=name == "screen_region"))
        elif name == "questions":
            await self.copilot.questions()
        elif name == "toggle_mode":
            mode = self.copilot.toggle_mode()
            self.bus.publish("notice", {"text": "Авто-ответы" if mode == "auto" else "Ответы по хоткею"})
        elif name == "toggle_style":
            self.cfg.answer.style = "dual" if self.cfg.answer.style == "single" else "single"
            self.bus.publish("notice", {"text": f"Стиль ответа: {self.cfg.answer.style}"})
        elif name == "toggle_record_only":
            self.copilot.record_only = not self.copilot.record_only
            self.bus.publish("notice", {"text": "Только запись" if self.copilot.record_only else "Подсказки включены"})
        elif name == "toggle_call":
            await asyncio.to_thread(self.detector.toggle_manual)
        elif name == "voice_start":
            await self.start_voice()
        elif name == "voice_stop":
            await self.stop_voice()
        elif name == "toggle_training_call":  # тренажёр: звонок, чья легенда не идёт в настоящие
            await asyncio.to_thread(self.detector.toggle_manual, "training")
        elif name == "reload":
            self.reload_config()
        elif name == "toggle_overlay":
            self.overlay_visible = not self.overlay_visible
        elif name in ("scroll_up", "scroll_down", "clear"):
            self.bus.publish("overlay." + name, {})
        else:
            return {"ok": False, "error": f"неизвестное действие {name}"}
        self.publish_status()
        return {"ok": True}

    # --- Hyprland ---

    def setup_hyprland(self) -> None:
        """Правила окна оверлея и хоткеи. Без подтверждённого no_screen_share оверлей не показываем."""
        if not self.with_hyprland:
            return
        if not hypr.available():
            self.overlay_allowed = False
            log.warning("Hyprland не найден (нет HYPRLAND_INSTANCE_SIGNATURE?) — оверлей не запускаю, хоткеи не зарегистрированы")
            self.bus.publish("notice", {"text": "Hyprland не найден: оверлей выключен, чтобы не попасть в демонстрацию"})
            return
        o = self.cfg.overlay
        x, y = o.x, o.y
        if x < 0 or y < 0:
            x, y = hypr.default_overlay_position(o.width, self.cfg.stage.name)
        # Прозрачность — только у фона (её рисует сам оверлей), текст остаётся чётким.
        self.overlay_allowed = hypr.apply_overlay_rules(x, y, o.width, o.height, 1.0)
        if not self.overlay_allowed:
            self.bus.publish("notice", {"text": "Hyprland не принял правило no_screen_share — оверлей выключен"})
        address = hypr.overlay_address()
        if address:  # окно уже открыто (например, после hyprctl reload) — скрываем его сразу
            hypr.hide_window(address)
        hypr.remove_binds(self._bound_keys)  # старые сочетания — после смены [hotkeys] не должны работать
        hk = self.cfg.hotkeys
        binds = {getattr(hk, f): f for f in vars(hk)}
        hypr.apply_binds(binds, sotto_command())
        self._bound_keys = [k for k in binds if k]

    def teardown_hyprland(self) -> None:
        if self.with_hyprland and hypr.available():
            hypr.remove_binds(self._bound_keys)

    def _on_hypr_event(self, name: str, payload: str) -> None:
        if name == "activewindowv2":
            # Оверлей взял фокус, потому что его тащили мышью, — возвращаем фокус окну звонка/редактора.
            addr = payload if payload.startswith("0x") else "0x" + payload
            if addr == self._overlay_address:
                prev = self._prev_focus
                if prev:
                    threading.Timer(0.8, hypr.focus_window, args=(prev,)).start()
            elif payload and payload != ",":
                self._prev_focus = addr
        if name == "configreloaded":
            log.info("Hyprland перечитал конфиг — заново регистрирую правила и хоткеи")
            self.setup_hyprland()

    # --- оверлей ---

    def _overlay_watchdog(self) -> None:
        """Держит оверлей запущенным и проверяет, что его окно действительно скрыто из демонстрации."""
        stop = self._voice_stop
        while not stop.wait(3.0):
            running = self._overlay_proc is not None and self._overlay_proc.poll() is None
            if not self.overlay_allowed:
                if running:
                    self._overlay_proc.terminate()
                continue
            if not running:
                cmd = sotto_command().split() + ["overlay"]
                self._overlay_proc = subprocess.Popen(cmd, start_new_session=True)
                continue
            address = hypr.overlay_address()
            self._overlay_address = address
            if address:
                self._remember_overlay_geometry()
            if address and not hypr.overlay_hidden(address):
                hypr.hide_window(address)
                if not hypr.overlay_hidden(address):
                    log.error("Окно оверлея не скрыто из демонстрации — закрываю его")
                    self.bus.publish("notice", {"text": "Не удалось скрыть оверлей из демонстрации — он закрыт"})
                    self.overlay_allowed = False
                    self._overlay_proc.terminate()

    def _warm_up_stt(self) -> None:
        """Загрузить Whisper заранее, чтобы первая реплика звонка не ждала ~5 с загрузки модели."""
        from .audio import local_stt
        try:
            local_stt.warm_up(self.cfg.trainer.stt_model)
            log.info("Распознавание: локальный Whisper (%s)", "GPU" if local_stt.has_cuda() else "CPU")
        except Exception:  # noqa: BLE001
            log.exception("Локальный Whisper не загрузился")

    def _remember_overlay_geometry(self) -> None:
        """Окно передвинули или растянули мышью — запоминаем, чтобы в следующий раз оно открылось там же."""
        geo = hypr.overlay_geometry()
        o = self.cfg.overlay
        if not geo or geo == (o.x, o.y, o.width, o.height):
            return
        if o.x < 0 and geo[2:] == (o.width, o.height) and self._overlay_auto_pos == geo[:2]:
            return  # стоит на позиции по умолчанию — не записываем её как выбранную
        if o.x < 0 and self._overlay_auto_pos is None:
            self._overlay_auto_pos = geo[:2]  # первое появление окна на позиции по умолчанию
            return
        o.x, o.y, o.width, o.height = geo
        try:
            fresh = config_mod.load()
            fresh.overlay.x, fresh.overlay.y, fresh.overlay.width, fresh.overlay.height = geo
            config_mod.save(fresh)
        except Exception:  # noqa: BLE001 — не критично
            log.exception("Не удалось сохранить положение оверлея")

    # --- запуск ---

    async def start_voice(self) -> None:
        """Голосовой суфлёр: звук, определение звонков, запись, Hyprland, оверлей. Запускается и
        останавливается с сайта; сайт при этом работает всегда, закрытие вкладки суфлёр не трогает."""
        if self.voice:
            return
        if self.capture_used:  # потоки захвата одноразовые — после остановки создаём заново
            self.recorder = ScreenRecorder(self.cfg.recording, self.db)
            self.capture = AudioCapture(self.cfg.audio.mic, self.cfg.audio.system, self._on_audio)
            self.detector = CallDetector(self.cfg.calls, on_start=self._call_started_threadsafe,
                                         on_end=self._call_ended_threadsafe)
        self.capture_used = True
        self._voice_stop = threading.Event()
        self.voice = True
        self.capture.start()
        self.recorder.start()
        self.detector.start()
        await asyncio.to_thread(self.setup_hyprland)
        hypr.listen_events(self._on_hypr_event, self._voice_stop)
        if self.with_overlay:
            threading.Thread(target=self._overlay_watchdog, name="overlay", daemon=True).start()
        if self.listener.use_local_stt():
            threading.Thread(target=self._warm_up_stt, name="stt-warmup", daemon=True).start()
        log.info("Голосовой суфлёр запущен")
        self.publish_status()

    async def stop_voice(self) -> None:
        if not self.voice:
            return
        self.voice = False
        self._voice_stop.set()
        self.capture.stop()
        self.detector.stop()
        self.recorder.stop()  # до завершения звонка: иначе граница звонка откроет новый пустой кусок
        if self.call_id is not None:
            self.call_ended()
        if self.trainer is not None and not self.trainer.stopped:
            self.trainer.stop()
        await asyncio.to_thread(self.teardown_hyprland)
        if self._overlay_proc and self._overlay_proc.poll() is None:
            self._overlay_proc.terminate()
        log.info("Голосовой суфлёр остановлен")
        self.publish_status()

    async def serve(self, voice: bool = True) -> None:
        """Сайт (история, поиск, легенда, вакансии, текстовая проверка) работает всегда; голосовой
        суфлёр — если voice или когда его запустят с сайта."""
        from .web.app import build_app

        self.loop = asyncio.get_running_loop()
        self.bus.bind(self.loop)
        self.listener.bind(self.loop)
        self._finish_interrupted_calls()
        removed = self.db.cleanup_legend()
        if removed:
            log.info("Удалено устаревшей легенды: %s", removed)
        self._spawn(interviews.reminders(self))
        refresh_soon(self)
        if voice:
            await self.start_voice()
        self.publish_status()
        app = build_app(self)
        # Оверлей держит SSE-подключение вечно — без таймаута остановка ждала бы его бесконечно.
        server = uvicorn.Server(uvicorn.Config(app, host=self.cfg.web.host, port=self.cfg.web.port,
                                               log_level="warning", access_log=False,
                                               timeout_graceful_shutdown=2))
        log.info("Sotto: %s%s. Ctrl+C — остановить всё", self.url,
                 "" if voice else " (голосовой суфлёр запускается на сайте)")
        try:
            await server.serve()
        finally:
            await self.shutdown()

    @property
    def url(self) -> str:
        return f"http://{self.cfg.web.host}:{self.cfg.web.port}"

    async def shutdown(self) -> None:
        self._stop.set()
        await self.chats.close_all()
        await self.stop_voice()
        await self.llm.close()

    def reload_config(self) -> None:
        """Перечитать config.toml без перезапуска. Объекты настроек обновляются на месте, поэтому
        запись, определение звонков, агенты и подсказки сразу видят новые значения."""
        config_mod.update_in_place(self.cfg, config_mod.load())
        self.llm.api_key = self.cfg.api_key
        self.llm.base_url = self.cfg.openrouter.base_url.rstrip("/")
        self.llm.provider_sort = self.cfg.openrouter.provider_sort
        self.llm.http.headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        self.setup_hyprland()
        self.publish_status()

    def stage_active(self) -> bool:
        """Сцену включают из другого процесса (sotto stage on), поэтому спрашиваем Hyprland — но не чаще раза в 5 с."""
        checked_at, active = self._stage_cache
        if time.time() - checked_at > 5:
            active = self.with_hyprland and hypr.available() and self.stage.active()
            self._stage_cache = (time.time(), active)
        return active


def run(with_overlay: bool = True, with_hyprland: bool = True, voice: bool = True,
        open_browser: bool = False) -> None:
    import webbrowser

    cfg = config_mod.load()
    daemon = Daemon(cfg, with_overlay=with_overlay, with_hyprland=with_hyprland)
    if _already_running(daemon.url):
        print(f"Sotto уже запущен: {daemon.url}")
        if open_browser:
            webbrowser.open(daemon.url)
        return
    if open_browser:
        threading.Timer(1.5, webbrowser.open, args=(daemon.url,)).start()
    try:
        asyncio.run(daemon.serve(voice=voice))
    except KeyboardInterrupt:
        pass  # остановка по Ctrl+C — штатная, без трассировки


def _already_running(url: str) -> bool:
    import httpx

    try:
        httpx.get(f"{url}/api/status", timeout=2)
        return True
    except httpx.HTTPError:
        return False
