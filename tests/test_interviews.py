"""Собеседования: план, привязка звонка, легенда по компании, отказ."""

import time
from datetime import datetime

from test_web import make_client

from sotto.application.interviews import parse_when


def test_plan_attach_and_reject(isolated_dirs, monkeypatch):
    d, c = make_client()
    when = datetime.fromtimestamp(time.time() + 600).strftime("%Y-%m-%dT%H:%M")
    r = c.post("/interviews/new", data={"company": "Acme", "role": "Python dev", "stage": "HR", "when": when},
               follow_redirects=False)
    iid = int(r.headers["location"].rsplit("/", 1)[1])
    for path in ["/interviews", f"/interviews/{iid}", "/", "/prompts", "/legend", "/trainer", "/chat"]:
        assert c.get(path).status_code == 200, path

    # Звонок за 10 минут до собеседования — его звонок: компания и бриф оттуда.
    d.voice.call_started("chrome")
    call = d.voice.call_id
    row = d.db.one("SELECT company, interview_id, brief FROM calls WHERE id = ?", [call])
    assert row["company"] == "Acme" and row["interview_id"] == iid and "Acme" in row["brief"]
    d.db.insert("legend", claim="вёл миграцию", call_id=call, company="Acme", created_at=time.time())
    other = d.db.insert("calls", started_at=time.time(), company="Globex")
    assert d.legend.for_call(other) == []
    assert c.get(f"/calls/{call}").status_code == 200

    spawned = []
    monkeypatch.setattr(d.background, "spawn", lambda coro: (spawned.append(coro), coro.close()))
    c.post(f"/interviews/{iid}/status", data={"status": "rejected", "outcome": "Мало опыта с Kafka"})
    assert d.db.one("SELECT status, outcome FROM interviews WHERE id = ?", [iid]) == \
        {"status": "rejected", "outcome": "Мало опыта с Kafka"}
    assert d.db.one("SELECT COUNT(*) AS n FROM legend")["n"] == 0   # процесс закрыт — легенда удалена
    assert spawned  # разбор отказа запущен
    assert "Мало опыта" in c.get(f"/interviews/{iid}").text


def test_next_stage_keeps_legend(isolated_dirs, monkeypatch):
    d, c = make_client()
    monkeypatch.setattr(d.background, "spawn", lambda coro: coro.close())
    iid = d.db.insert("interviews", company="Acme", stage="HR", created_at=time.time())
    d.db.insert("legend", claim="X", company="Acme", call_id=None, created_at=time.time())
    r = c.post(f"/interviews/{iid}/next", follow_redirects=False)
    nid = int(r.headers["location"].rsplit("/", 1)[1])
    assert nid != iid and d.db.one("SELECT status FROM interviews WHERE id = ?", [iid])["status"] == "passed"
    assert d.legend.cleanup() == 0


def test_link_after_call_by_company(isolated_dirs):
    d, _ = make_client()
    iid = d.db.insert("interviews", company="Acme Corp", created_at=time.time(), scheduled_at=time.time() - 86400)
    call = d.db.insert("calls", started_at=time.time())
    d.interviews.link_after_call(call, "acme corp")
    row = d.db.one("SELECT company, interview_id FROM calls WHERE id = ?", [call])
    assert row == {"company": "Acme Corp", "interview_id": iid}
    assert d.db.one("SELECT status FROM interviews WHERE id = ?", [iid])["status"] == "done"


def test_parse_when():
    assert parse_when("2026-11-20T15:00") == datetime(2026, 11, 20, 15, 0).timestamp()
    assert parse_when("") is None and parse_when("завтра") is None
