"""Интеграция с Hyprland (0.55+, Lua-конфиг).

Всё регистрируется на лету через `hyprctl eval`, поэтому руками в конфиг ничего прописывать
не надо (и он может быть read-only из home-manager). После `hyprctl reload` правила
пропадают — демон слушает событие `configreloaded` и применяет их снова.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

OVERLAY_CLASS = "sotto-overlay"
MIRROR_CLASS = "at.yrlf.wl_mirror"


def available() -> bool:
    return bool(os.environ.get("HYPRLAND_INSTANCE_SIGNATURE")) and shutil.which("hyprctl") is not None


def hyprctl(*args: str, json_out: bool = False) -> Any:
    cmd = ["hyprctl", *(["-j"] if json_out else []), *args]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
    out = res.stdout.strip()
    if json_out:
        return json.loads(out or "null")
    return out


def lua(code: str) -> str:
    out = hyprctl("eval", code)
    if out and out != "ok":
        log.warning("hyprctl eval: %s → %s", code[:120], out)
    return out


def lua_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


# --- окно оверлея ---

def apply_overlay_rules(x: int, y: int, w: int, h: int, opacity: float) -> bool:
    """Оверлей: плавает, на всех рабочих столах, НЕ попадает в демонстрацию экрана.
    Фокус не берёт при появлении и при наведении мыши (no_follow_mouse). Правило no_focus ставить
    нельзя: Hyprland тогда вообще не видит окно под курсором — ни перетащить, ни растянуть.
    Возвращает True, только если Hyprland принял правило — иначе оверлей показывать нельзя."""
    ok = lua(
        "hl.window_rule({ name = \"sotto-overlay\", match = { class = \"^" + OVERLAY_CLASS + "$\" }, "
        "float = true, pin = true, no_focus = false, no_follow_mouse = true, no_initial_focus = true, no_screen_share = true, "
        "no_shadow = true, no_blur = false, no_anim = true, border_size = 0, decorate = false, "
        f"size = {{{w}, {h}}}, move = {{{x}, {y}}}, opacity = \"{opacity} override\" }})"
    ) == "ok"
    # Зеркало «сцены» тоже прячем от захвата и не даём ему фокус.
    lua(
        "hl.window_rule({ name = \"sotto-mirror\", match = { class = \"^" + MIRROR_CLASS.replace(".", "\\\\.") + "$\" }, "
        "no_screen_share = true, no_focus = true, no_anim = true })"
    )
    return ok


def overlay_address() -> str | None:
    for c in hyprctl("clients", json_out=True) or []:
        if c.get("class") == OVERLAY_CLASS:
            return c["address"]
    return None


def overlay_geometry() -> tuple[int, int, int, int] | None:
    """(x, y, ширина, высота) окна оверлея относительно его монитора."""
    clients = hyprctl("clients", json_out=True) or []
    win = next((c for c in clients if c.get("class") == OVERLAY_CLASS), None)
    if not win:
        return None
    mon = next((m for m in monitors() if m.get("id") == win.get("monitor")), {"x": 0, "y": 0})
    (x, y), (w, h) = win["at"], win["size"]
    return x - mon["x"], y - mon["y"], w, h


def default_overlay_position(width: int, stage_name: str) -> tuple[int, int]:
    """Сверху по центру основного монитора — там камера ноутбука."""
    m = primary_monitor(stage_name)
    if not m:
        return 24, 24
    logical_w = int(m["width"] / (m.get("scale") or 1))
    return max(0, (logical_w - width) // 2), 44


def focus_window(address: str) -> None:
    hyprctl("dispatch", f"hl.dsp.focus({{window = {lua_str('address:' + address)}}})")


def overlay_hidden(address: str) -> bool:
    """Скрыто ли уже открытое окно оверлея из демонстрации экрана."""
    return hyprctl("getprop", f"address:{address}", "no_screen_share").strip().lower() in ("true", "1")


def hide_window(address: str) -> None:
    hyprctl("setprop", f"address:{address}", "no_screen_share", "1")


def apply_binds(binds: dict[str, str], sotto_cmd: str) -> None:
    """binds: {"SUPER + A": "ask", ...} → вызывает `sotto ctl <action>`."""
    for keys, action in binds.items():
        if not keys:
            continue
        lua(f"hl.unbind({lua_str(keys)})")
        lua(f"hl.bind({lua_str(keys)}, hl.dsp.exec_cmd({lua_str(sotto_cmd + ' ctl ' + action)}))")


def remove_binds(keys: list[str]) -> None:
    for k in keys:
        if k:
            lua(f"hl.unbind({lua_str(k)})")


# --- информация об окнах и мониторах ---

def active_window() -> dict[str, Any]:
    try:
        return hyprctl("activewindow", json_out=True) or {}
    except Exception:
        return {}


def monitors() -> list[dict[str, Any]]:
    try:
        return hyprctl("monitors", json_out=True) or []
    except Exception:
        return []


def focused_monitor() -> str:
    for m in monitors():
        if m.get("focused"):
            return m["name"]
    mons = monitors()
    return mons[0]["name"] if mons else ""


def primary_monitor(stage_name: str) -> dict[str, Any] | None:
    real = [m for m in monitors() if m["name"] != stage_name]
    return next((m for m in real if m.get("focused")), real[0] if real else None)


# --- скриншоты ---

def screenshot(dest: Path, output: str | None = None, region: bool = False, scale: float = 0.75) -> Path | None:
    """Снимок монитора (или выделенной области) в JPEG. None — если пользователь отменил выделение."""
    cmd = ["grim", "-t", "jpeg", "-q", "85", "-s", str(scale)]
    if region:
        sel = subprocess.run(["slurp"], capture_output=True, text=True)
        if sel.returncode != 0 or not sel.stdout.strip():
            return None
        cmd += ["-g", sel.stdout.strip()]
    elif output:
        cmd += ["-o", output]
    cmd.append(str(dest))
    subprocess.run(cmd, check=True, timeout=10)
    return dest


# --- «сцена»: виртуальный монитор для демонстрации ---

class Stage:
    """Виртуальный (headless) монитор, который ты показываешь в звонке.

    Сцена ставится справа от основного монитора, а на основном мониторе на весь экран
    открывается её зеркало (wl-mirror). Ты видишь сцену и оверлей поверх, собеседник —
    только сцену. Окна, которые надо показать, перекладываются на рабочий стол сцены.
    """

    WORKSPACE = "name:stage"

    def __init__(self, name: str = "SOTTO-STAGE", mirror_on: str = ""):
        self.name = name
        self.mirror_on = mirror_on  # где показывать зеркало сцены; пусто — основной монитор
        self._mirror: subprocess.Popen | None = None

    def active(self) -> bool:
        return any(m["name"] == self.name for m in monitors())

    def on(self) -> str:
        main = next((m for m in monitors() if m["name"] == self.mirror_on), None) or primary_monitor(self.name)
        if main is None:
            raise RuntimeError("Не нашёл ни одного монитора")
        if not self.active():
            hyprctl("output", "create", "headless", self.name)
            time.sleep(0.5)
        w, h = main["width"], main["height"]
        if main.get("transform", 0) in (1, 3, 5, 7):  # повёрнутый монитор
            w, h = h, w
        # Координаты в Hyprland логические: ширина в пикселях режима делится на масштаб.
        right_edge = main["x"] + round(w / (main.get("scale") or 1))
        lua("hl.monitor({ output = " + lua_str(self.name) + f", mode = \"{w}x{h}@30\", "
            f"position = \"{right_edge}x{main['y']}\", scale = {main['scale']} }})")
        lua("hl.workspace_rule({ workspace = " + lua_str(self.WORKSPACE) + ", monitor = " + lua_str(self.name) + ", persistent = true })")
        hyprctl("dispatch", f"hl.dsp.focus({{workspace = {lua_str(self.WORKSPACE)}}})")
        if shutil.which("wl-mirror"):
            self._mirror = subprocess.Popen(
                ["wl-mirror", "--fullscreen-output", main["name"], "--title", "sotto-stage-mirror", self.name],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        return (f"Сцена {self.name} включена справа от {main['name']}. "
                f"В звонке показывай монитор «{self.name}». Окна для показа переноси на рабочий стол «stage».")

    def off(self) -> str:
        subprocess.run(["pkill", "-f", f"wl-mirror.*{self.name}"], check=False)
        if self.active():
            # Окна со сцены возвращаем на основной монитор, иначе Hyprland раскидает их сам.
            main = primary_monitor(self.name)
            for c in hyprctl("clients", json_out=True) or []:
                if c.get("workspace", {}).get("name") == "stage" and main:
                    hyprctl("dispatch", f"hl.dsp.window.move({{workspace = {main['activeWorkspace']['id']}, "
                                        f"window = \"address:{c['address']}\", follow = false}})")
            hyprctl("output", "remove", self.name)
        return "Сцена выключена"


# --- система ---

def supports_capture_exclusion() -> bool:
    return available()


def screen_locked() -> bool:
    return subprocess.run(["pgrep", "-x", "hyprlock"], capture_output=True).returncode == 0


def microphone_users(ignore: list[str]) -> list[str]:
    """Имена приложений, которые прямо сейчас пишут звук с микрофона (кроме нас и игнор-листа).
    Опрашиваем `pw-dump` и ищем чужие потоки Stream/Input/Audio в состоянии running."""
    try:
        out = subprocess.run(["pw-dump"], capture_output=True, text=True, timeout=5).stdout
        objects = json.loads(out or "[]")
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
        return []
    ignore_l = [i.lower() for i in ignore]
    users = []
    for o in objects:
        info = o.get("info") or {}
        props = info.get("props") or {}
        if props.get("media.class") != "Stream/Input/Audio" or info.get("state") != "running":
            continue
        if props.get("stream.capture.sink"):
            continue  # запись системного звука (не микрофона) звонком не считаем
        node = str(props.get("node.name") or "")
        app = str(props.get("application.process.binary") or props.get("application.name") or node)
        if node.startswith("sotto-") or any(i in app.lower() or i in node.lower() for i in ignore_l):
            continue
        users.append(app)
    return users


def copy_text(text: str) -> None:
    subprocess.run(["wl-copy"], input=text.encode(), check=False)


def paste_text() -> str:
    try:
        return subprocess.run(["wl-paste", "--no-newline"], capture_output=True, text=True, timeout=3).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def open_url(url: str) -> None:
    subprocess.Popen(["xdg-open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     start_new_session=True)


def open_path(path: Path | str, at: float = 0) -> None:
    """Открыть файл: mpv/vlc с нужной секунды, иначе программа по умолчанию."""
    for player in (["mpv", f"--start={at:.0f}"], ["vlc", f"--start-time={at:.0f}"], ["xdg-open"]):
        try:
            subprocess.Popen([*player, str(path)], start_new_session=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except FileNotFoundError:
            continue


# --- события Hyprland (socket2) ---

def listen_events(on_event: Callable[[str, str], None], stop: threading.Event) -> threading.Thread:
    """Читает события Hyprland ("configreloaded", "activewindow>>..."), вызывает on_event(имя, данные)."""
    def run() -> None:
        sig = os.environ.get("HYPRLAND_INSTANCE_SIGNATURE", "")
        runtime = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
        path = Path(runtime) / "hypr" / sig / ".socket2.sock"
        while not stop.is_set():
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                    s.connect(str(path))
                    buf = b""
                    while not stop.is_set():
                        data = s.recv(4096)
                        if not data:
                            break
                        buf += data
                        while b"\n" in buf:
                            line, buf = buf.split(b"\n", 1)
                            name, _, payload = line.decode(errors="ignore").partition(">>")
                            try:
                                on_event(name, payload)
                            except Exception:  # noqa: BLE001
                                log.exception("Обработчик события Hyprland %s", name)
            except OSError:
                time.sleep(2)
    t = threading.Thread(target=run, name="hypr-events", daemon=True)
    t.start()
    return t
