"""Оверлей с подсказками (Qt). Отдельный процесс: слушает события демона и рисует.

Задача окна — не отвлекать. Спокойные цвета, ничего не мигает и не меняет размер:
  • строка состояния — мелко и тускло: звонок, кто собеседник (имя из представления);
  • подсказка — главное, обычным текстом; над ней тускло вопрос, на который она отвечает.
    Шрифт сам уменьшается, чтобы подсказка влезла целиком, — прокручивать не нужно;
  • последние реплики разговора — внизу, тускло.

Окно двигается мышью за любое место и тянется за края (или SUPER + кнопки мыши, как любое
плавающее окно). Положение запоминается: в Linux его снимает демон через Hyprland, в Windows окно
сохраняет себя само (координаты Qt, чтобы не путаться с масштабом монитора). Фокус окно не берёт,
из демонстрации экрана скрыто: в Hyprland — правило `no_screen_share`, в Windows —
`SetWindowDisplayAffinity(WDA_EXCLUDEFROMCAPTURE)`.
"""

from __future__ import annotations

import html
import json
import logging
import re
import sys
import threading
import time

import httpx
import markdown as md
from PySide6.QtCore import QObject, QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QCursor, QFont, QGuiApplication
from PySide6.QtWidgets import QApplication, QLabel, QTextBrowser, QVBoxLayout, QWidget

from . import config as config_mod
from . import desktop, local
from .desktop import OVERLAY_CLASS

log = logging.getLogger(__name__)

TEXT = "#e6e4df"
DIM = "#8d8b85"
CSS = f"""
body {{ color: {TEXT}; }}
p {{ margin: 0 0 6px 0; line-height: 135%; }}
ul {{ margin: 0 0 6px 0; -qt-list-indent: 1; }}
li {{ margin-bottom: 3px; line-height: 135%; }}
b, strong {{ font-weight: normal; }}
code, pre {{ font-family: monospace; background: rgba(255,255,255,0.06); }}
pre {{ padding: 6px; }}
.legend {{ color: #cbbfe0; }}
.q, .slot {{ color: {DIM}; }}
.who {{ color: #b5b2ab; }}
"""
EDGE = 10  # px от края, за которые окно тянется


class Events(QObject):
    event = Signal(dict)


def listen(base_url: str, events: Events) -> None:
    """Поток: держит SSE-подключение к демону и переподключается при обрыве."""
    while True:
        try:
            with local.stream("GET", f"{base_url}/api/events", timeout=None) as resp:
                for line in resp.iter_lines():
                    if line.startswith("data: "):
                        events.event.emit(json.loads(line[6:]))
        except Exception:
            events.event.emit({"type": "disconnected"})
            time.sleep(1.5)


def render_answer(text: str) -> str:
    """Markdown → HTML; придуманные факты [[...]] чуть другим оттенком."""
    text = re.sub(r"\[\[(.+?)\]\]", r'<span class="legend">\1</span>', text, flags=re.S)
    text = text.replace("[[", "")  # незакрытая скобка во время стриминга
    return md.markdown(text, extensions=["fenced_code", "nl2br"])


class Overlay(QWidget):
    def __init__(self, cfg: config_mod.Config):
        super().__init__()
        self.cfg = cfg
        o = cfg.overlay
        self.setWindowTitle("Sotto")
        flags = (Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
                 | Qt.WindowType.WindowDoesNotAcceptFocus)
        if desktop.WINDOWS:
            # Tool: окна нет в Alt+Tab и на панели задач — суфлёр не должен ничем себя выдавать.
            flags |= Qt.WindowType.Tool
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setMouseTracking(True)
        self.resize(o.width, o.height)
        self._geometry_saved: tuple[int, int, int, int] | None = None
        self._auto_position: tuple[int, int] | None = None
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(1200)
        self._save_timer.timeout.connect(self._save_geometry)
        self.place_initial()

        self.setStyleSheet(f"""
            QWidget#root {{ background: rgba(24,24,26,{int(o.opacity * 255)}); border-radius: 10px;
                            border: 1px solid rgba(255,255,255,0.06); }}
            QLabel#status {{ color: {DIM}; }}
            QTextBrowser {{ background: transparent; border: none; color: {TEXT}; }}
            QTextBrowser#transcript {{ border-top: 1px solid rgba(255,255,255,0.06); padding-top: 4px; }}
        """)
        root = QWidget(self)
        root.setObjectName("root")
        root.setMouseTracking(True)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(root)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(18, 10, 22, 10)
        lay.setSpacing(6)

        small = max(o.font_size - 4, 10)
        self.status = QLabel("Sotto: подключение…")
        self.status.setObjectName("status")
        self.status.setFont(QFont(QFont().family(), small))

        self.answer = QTextBrowser()
        self.answer.document().setDefaultStyleSheet(CSS)
        self.answer.document().setDefaultFont(QFont(QFont().family(), o.font_size))
        self.answer.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.answer.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.transcript = QTextBrowser()
        self.transcript.setObjectName("transcript")
        self.transcript.document().setDefaultStyleSheet(CSS)
        self.transcript.setFont(QFont(QFont().family(), small))
        self.transcript.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.transcript.setFixedHeight(int(small * 2.1 * o.transcript_lines) + 14)

        # Мышь проходит сквозь текст к окну: окно двигается за любое место.
        for w in (self.status, self.answer, self.answer.viewport(), self.transcript, self.transcript.viewport()):
            w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        lay.addWidget(self.status)
        lay.addWidget(self.answer, 1)
        lay.addWidget(self.transcript)

        self.lines: list[tuple[str, int | None, str]] = []  # (канал, id собеседника, текст)
        self.speakers: dict[int, dict] = {}                 # id → {"name", "role"}
        self.slots: dict[str, dict] = {}   # slot → {"id", "text"}
        self.pending: dict[str, str] = {}  # slot → id ответа, который начался, но ещё без текста
        self.question = ""                 # вопрос, на который отвечает подсказка
        self._next_question = ""
        self.thinking = ""                 # «думаю…» пока ждём первые слова (только по хоткею)
        self.state: dict = {}
        self.notice = ""
        self._notice_until = 0.0
        self._dirty = True
        self.overflow = False
        timer = QTimer(self)
        timer.timeout.connect(self._refresh)  # перерисовка не чаще 20 раз в секунду
        timer.start(50)

    # --- положение и невидимость ---

    def place_initial(self) -> None:
        """Своё положение из настроек; иначе сверху по центру — под камерой ноутбука."""
        o = self.cfg.overlay
        if o.x >= 0 and o.y >= 0:
            self.move(o.x, o.y)
            return
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        area = screen.availableGeometry()
        x = area.x() + max(0, (area.width() - o.width) // 2)
        y = area.y() + 44
        self.move(x, y)
        self._auto_position = (x, y)

    def apply_invisibility(self) -> None:
        """Windows: исключить окно из захвата экрана (Zoom, Meet, Teams, OBS, браузер его не увидят).
        В Linux то же делает правило Hyprland `no_screen_share`. Не вышло — окно не показываем."""
        if not desktop.WINDOWS:
            return
        hwnd = int(self.winId())
        desktop.mark_overlay(hwnd)  # демон найдёт окно по метке, даже если запуск шёл через посредник
        desktop.no_activate(hwnd)
        desktop.hide_overlay(hwnd)
        if not desktop.overlay_hidden(hwnd):
            log.error("Не удалось исключить оверлей из захвата экрана — закрываю его")
            self.setVisible(False)

    def _save_geometry(self) -> None:
        """Окно передвинули или растянули мышью — запоминаем, чтобы в следующий раз открылось там же.
        Работает только в Windows: в Linux положение снимает демон через Hyprland."""
        if not desktop.WINDOWS:
            return
        o = self.cfg.overlay
        geo = (self.x(), self.y(), self.width(), self.height())
        if geo == self._geometry_saved:
            return
        if o.x < 0 and self._auto_position == geo[:2]:
            return  # стоит на позиции по умолчанию — не записываем её как выбранную
        self._geometry_saved = geo
        o.x, o.y, o.width, o.height = geo
        try:
            fresh = config_mod.load()
            fresh.overlay.x, fresh.overlay.y, fresh.overlay.width, fresh.overlay.height = geo
            config_mod.save(fresh)
        except Exception:  # noqa: BLE001 — не критично
            log.exception("Не удалось сохранить положение оверлея")

    # --- мышь: перетаскивание, размер, колесо ---

    def _edges(self, pos: QPoint) -> Qt.Edge:
        edges = Qt.Edge(0)
        if pos.x() < EDGE:
            edges |= Qt.Edge.LeftEdge
        if pos.x() > self.width() - EDGE:
            edges |= Qt.Edge.RightEdge
        if pos.y() < EDGE:
            edges |= Qt.Edge.TopEdge
        if pos.y() > self.height() - EDGE:
            edges |= Qt.Edge.BottomEdge
        return edges

    def mousePressEvent(self, e) -> None:  # noqa: N802 — имя метода Qt
        if e.button() != Qt.MouseButton.LeftButton or self.windowHandle() is None:
            return
        edges = self._edges(e.position().toPoint())
        if edges:
            self.windowHandle().startSystemResize(edges)
        else:
            self.windowHandle().startSystemMove()

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        edges = self._edges(e.position().toPoint())
        horiz = bool(edges & (Qt.Edge.LeftEdge | Qt.Edge.RightEdge))
        vert = bool(edges & (Qt.Edge.TopEdge | Qt.Edge.BottomEdge))
        if horiz and vert:
            diag = edges in (Qt.Edge.LeftEdge | Qt.Edge.TopEdge, Qt.Edge.RightEdge | Qt.Edge.BottomEdge)
            shape = Qt.CursorShape.SizeFDiagCursor if diag else Qt.CursorShape.SizeBDiagCursor
        elif horiz:
            shape = Qt.CursorShape.SizeHorCursor
        elif vert:
            shape = Qt.CursorShape.SizeVerCursor
        else:
            shape = Qt.CursorShape.SizeAllCursor
        self.setCursor(QCursor(shape))

    def wheelEvent(self, e) -> None:  # noqa: N802 — на случай, если подсказка всё же не влезла
        bar = self.answer.verticalScrollBar()
        bar.setValue(bar.value() - e.angleDelta().y())

    def resizeEvent(self, e) -> None:  # noqa: N802
        super().resizeEvent(e)
        self._dirty = True
        if desktop.WINDOWS and self.isVisible():
            self._save_timer.start()

    def moveEvent(self, e) -> None:  # noqa: N802
        super().moveEvent(e)
        if desktop.WINDOWS and self.isVisible():
            self._save_timer.start()

    # --- события ---

    def on_event(self, e: dict) -> None:
        t = e.get("type")
        if t == "status":
            if e.get("call_id") != self.state.get("call_id"):
                self.speakers.clear()  # новый звонок — новые собеседники
            self.state = e
            self.setVisible(bool(e.get("overlay_visible", True)))
        elif t == "disconnected":
            self.state = {}
        elif t == "speaker":
            sp = self.speakers.setdefault(e["id"], {})
            if e.get("name") is not None:
                sp["name"] = e["name"] or None
                sp["role"] = e.get("role")
            elif e.get("label") and not sp.get("name"):
                sp["label"] = e["label"]
            self._render_transcript()
        elif t == "utterance":
            sid = e.get("speaker_id") if e["channel"] == "them" else None
            if sid is not None and e.get("speaker"):
                self.speakers.setdefault(sid, {}).setdefault("name", e["speaker"])
            self.lines.append((e["channel"], sid, e["text"]))
            self.lines = self.lines[-20:]
            self._render_transcript()
        elif t == "answer.start":
            self.pending[e["slot"]] = e["id"]
            self._next_question = e.get("question", "")
            if e["kind"] != "auto":  # авто-режим молчит, пока не ясно, нужен ли ответ
                self.thinking = {"screen": "смотрю на экран…", "questions": "подбираю вопросы…"}.get(e["kind"], "думаю…")
                self._dirty = True
        elif t == "answer.delta":
            slot = e["slot"]
            if self.pending.get(slot) == e["id"]:
                # Первый кусок нового ответа — только сейчас заменяем старую подсказку.
                self.pending.pop(slot)
                if slot == "main":
                    self.slots = {}
                else:  # режим dual: короткий и подробный живут рядом, обычный ответ убираем
                    self.slots.pop("main", None)
                self.slots[slot] = {"id": e["id"], "text": ""}
                self.question = self._next_question
                self.thinking = ""
            if slot in self.slots and self.slots[slot]["id"] == e["id"]:
                self.slots[slot]["text"] += e["text"]
                self._dirty = True
        elif t == "answer.done":
            if e["slot"] in self.slots and self.slots[e["slot"]]["id"] == e["id"]:
                self.slots[e["slot"]]["text"] = e["text"]
            self.thinking = ""
            self._dirty = True
        elif t in ("answer.skip", "answer.error"):
            self.pending.pop(e["slot"], None)
            if self.thinking:
                self.thinking = ""
                self._dirty = True
            if t == "answer.error":
                self._show_notice("ошибка: " + e["error"], 8)
        elif t == "notice":
            self._show_notice(e["text"], 4)
        elif t == "overlay.scroll_up":
            bar = self.answer.verticalScrollBar()
            bar.setValue(bar.value() - self.answer.height() // 2)
        elif t == "overlay.scroll_down":
            bar = self.answer.verticalScrollBar()
            bar.setValue(bar.value() + self.answer.height() // 2)
        elif t == "overlay.clear":
            self.slots.clear()
            self.lines.clear()
            self.question = ""
            self._render_transcript()
            self._dirty = True

    def _show_notice(self, text: str, seconds: float) -> None:
        self.notice = text
        self._notice_until = time.time() + seconds

    # --- отрисовка ---

    def _who(self, channel: str, sid: int | None) -> str:
        if channel == "me":
            return "Я"
        sp = self.speakers.get(sid, {}) if sid is not None else {}
        return sp.get("name") or sp.get("label") or "Собеседник"

    def _render_transcript(self) -> None:
        n = self.cfg.overlay.transcript_lines
        rows = [f'<span class="who">{html.escape(self._who(ch, sid))}:</span> {html.escape(text)}'
                for ch, sid, text in self.lines[-n:]]
        self.transcript.setHtml(f'<div style="color:{DIM}">' + "<br>".join(rows) + "</div>")
        self.transcript.verticalScrollBar().setValue(self.transcript.verticalScrollBar().maximum())

    def answer_html(self) -> str:
        parts = []
        if self.question:
            q = self.question if len(self.question) < 140 else self.question[:137] + "…"
            parts.append(f'<p class="q"><small>{html.escape(q)}</small></p>')
        if len(self.slots) > 1:
            for k in ("short", "detailed"):
                if k in self.slots:
                    parts.append(f'<p class="slot"><small>{"кратко" if k == "short" else "подробно"}</small></p>')
                    parts.append(render_answer(self.slots[k]["text"]))
        else:
            parts += [render_answer(v["text"]) for v in self.slots.values()]
        if self.thinking:
            parts.append(f'<p class="q">{self.thinking}</p>')
        if not self.slots and not self.thinking:
            parts.append('<p class="q">Подсказка появится здесь, когда собеседник задаст вопрос.</p>')
        return "".join(parts)

    def _show_answer(self, body: str) -> None:
        """Шрифт один и тот же: подгонка размера под длину заставляла текст прыгать во время чтения.
        Если подсказка не влезла — в строке состояния появляется «колесо мыши — дальше»."""
        self.answer.setHtml(body)
        doc = self.answer.document()
        doc.setTextWidth(self.answer.viewport().width())
        self.overflow = doc.size().height() > self.answer.viewport().height() + 2

    def _status_text(self) -> str:
        s = self.state
        if not s:
            return "Sotto не запущен"
        parts = ["звонок" if s.get("in_call") else "нет звонка"]
        if s.get("mode") != "auto":
            parts.append("только по хоткею")
        if s.get("record_only"):
            parts.append("только запись")
        if not s.get("api_key"):
            parts.append("нет ключа OpenRouter")
        people = []
        for sp in self.speakers.values():
            if sp.get("name"):
                people.append(sp["name"] + (f", {sp['role']}" if sp.get("role") else ""))
        if people:
            parts.append(" · ".join(people))
        text = "  ·  ".join(parts)
        if self.overflow:
            text += "  ·  не влезло — колесо мыши"
        if self.notice and time.time() < self._notice_until:
            text += "  ·  " + self.notice
        return text

    def _refresh(self) -> None:
        status = self._status_text()
        if status != self.status.text():
            self.status.setText(status)
        if self._dirty:
            self._dirty = False
            self._show_answer(self.answer_html())


def main() -> None:
    cfg = config_mod.load()
    if desktop.WINDOWS:
        desktop.prepare_process()
    # Класс окна для правил Hyprland задаётся через app id (Wayland).
    QGuiApplication.setDesktopFileName(OVERLAY_CLASS)
    app = QApplication(sys.argv)
    app.setApplicationName(OVERLAY_CLASS)
    overlay = Overlay(cfg)
    events = Events()
    events.event.connect(overlay.on_event)
    base = f"http://{cfg.web.host}:{cfg.web.port}"
    threading.Thread(target=listen, args=(base, events), daemon=True).start()
    overlay.show()
    overlay.apply_invisibility()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
