"""Текстовый режим: сам задаёшь вопросы / вопросы задаёт интервьюер. Поддельная модель, без звука."""

import asyncio

from fastapi.testclient import TestClient

from sotto import config as config_mod
from sotto.daemon import Daemon
from sotto.openrouter import Completion
from sotto.web.app import build_app

from test_engine import FakeLLM


class FakeChatLLM(FakeLLM):
    def __init__(self, replies, interviewer):
        super().__init__(replies)
        self.interviewer = list(interviewer)

    async def chat(self, model, messages, **_):
        res = Completion(model=model)
        res.text = self.interviewer.pop(0) if self.interviewer else "PLAN"
        return res


def make(replies, interviewer=()):
    cfg = config_mod.Config()
    cfg.recording.enabled = False
    d = Daemon(cfg, with_overlay=False, with_hyprland=False)
    d.bus.bind(asyncio.get_running_loop())
    d.llm = FakeChatLLM(replies, interviewer)
    return d


async def wait_idle(s):
    for _ in range(100):
        await asyncio.sleep(0.02)
        if not s.busy and not s._tasks:
            return


async def test_self_mode_hint_then_candidate_said_it(isolated_dirs):
    d = make(["I used [[Redis locks]] for that.", "∅", "Second answer"])
    s = d.chats.start("self", "en", "", "")
    await s.ask("How did you avoid double processing?")
    assert s.hint == "I used Redis locks for that."
    await s.ask("Okay, thanks.")            # кандидат не написал ответ — значит, сказал подсказку
    assert s.hint == "∅"
    await s.candidate("My own words")
    await s.ask("And then?")
    rows = [(r["channel"], r["text"]) for r in d.db.call_transcript(s.call_id)]
    assert rows == [("them", "How did you avoid double processing?"), ("me", "I used Redis locks for that."),
                    ("them", "Okay, thanks."), ("me", "My own words"), ("them", "And then?")]
    assert d.db.active_legend(call_id=s.call_id) == []  # на тренировках легенды нет
    assert d.db.one("SELECT COUNT(*) AS n FROM legend")["n"] == 0
    await s.end()
    assert d.db.one("SELECT ended_at FROM calls WHERE id = ?", [s.call_id])["ended_at"]


async def test_interviewer_mode(isolated_dirs):
    d = make(["Hi Emily, I'm Alex.", "I built it with FastAPI."],
             interviewer=["Hi, I'm Emily. Tell me about yourself.", "PLAN", "What did you build? "])
    s = d.chats.start("interviewer", "en", "backend developer", "")
    await wait_idle(s)
    assert s.hint == "Hi Emily, I'm Alex."
    await s.candidate("")                      # пустой ответ = подсказка
    rows = [(r["channel"], r["text"]) for r in d.db.call_transcript(s.call_id)]
    assert rows[1] == ("me", "Hi Emily, I'm Alex.") and rows[2] == ("them", "What did you build?")
    assert s.hint == "I built it with FastAPI."
    assert s.plan == "PLAN"


def test_pages_and_voice_off(isolated_dirs):
    cfg = config_mod.Config()
    d = Daemon(cfg, with_overlay=False, with_hyprland=False)
    c = TestClient(build_app(d), base_url=f"http://127.0.0.1:{cfg.web.port}")
    assert c.get("/api/status").json()["voice"] is False
    assert "Запустить суфлёр" in c.get("/").text
    assert c.get("/chat").status_code == 200
