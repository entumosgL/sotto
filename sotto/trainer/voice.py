"""`sotto trainer knowledge|english [тема]` — запуск тренировки из терминала.

Сама тренировка идёт в демоне (он держит микрофон), здесь — только старт и вывод разбора.
"""

from __future__ import annotations

import json

import httpx

from .. import config as config_mod


async def run_trainer(kind: str, topic: str, mode: str) -> int:
    cfg = config_mod.load()
    base = f"http://{cfg.web.host}:{cfg.web.port}"
    try:
        res = local.post(f"{base}/api/trainer/start", params={"kind": kind, "topic": topic, "mode": mode}, timeout=10).json()
    except httpx.HTTPError:
        print("Демон не запущен: sotto run")
        return 1
    if "error" in res:
        print(res["error"])
        return 1
    print(f"Тема: {res['topic']}. Говори голосом; «стоп» или Ctrl+C — закончить. Разбор появится здесь и в панели.\n")
    try:
        with local.stream("GET", f"{base}/api/events", timeout=None) as r:
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                e = json.loads(line[6:])
                if e["type"] == "trainer.turn":
                    who = "🎓" if e["role"] == "coach" else "🙂"
                    print(f"{who} {e['text']}")
                elif e["type"] == "trainer.feedback" and e.get("feedback"):
                    print(f"\n   📋 {e['feedback']}\n")
                elif e["type"] == "trainer.state":
                    if e["phase"] == "listening":
                        print("   … слушаю")
                    if e["phase"] == "finished":
                        print("Тренировка закончена.")
                        return 0
                elif e["type"] == "notice":
                    print(f"   ⚠ {e['text']}")
    except KeyboardInterrupt:
        local.post(f"{base}/api/trainer/stop", timeout=5)
    return 0
