"""`sotto translate …` — экспериментальный голосовой переводчик звонка.

  sotto translate enroll        записать образец голоса (15 с чтения текста)
  sotto translate say "Hello"   проверить клон голоса — прозвучит в твоих наушниках
  sotto translate run           переводить звонок (Ctrl+C — выход)
"""

from __future__ import annotations

import asyncio
import sys
import threading

from ... import config as config_mod
from . import devices
from .engine import Phrase, TranslatorEngine
from .voices import ENROLL_TEXT, MyVoice, Player, has_reference, record_reference

WINDOWS = sys.platform == "win32"
UNSUPPORTED = ("Эксперимент «переводчик звонка» работает только в Linux: он создаёт виртуальные "
               "устройства PipeWire, в которые звонок смотрит как в микрофон и колонки. В Windows "
               "таких устройств без отдельного драйвера (VB-CABLE и подобных) не сделать.")


def _fmt(p: Phrase) -> str:
    if p.direction == "in":
        head = f"\033[33m◀ собеседник {p.speaker + 1}\033[0m"
    else:
        head = "\033[36m▶ ты\033[0m"
    times = f"\033[2m(распознано {p.t_stt:.1f} с · перевод {p.t_translate:.1f} с · звук {p.t_audio:.1f} с от конца фразы)\033[0m"
    return f"{head}: {p.text}\n   → {p.translation}\n   {times}"


def enroll() -> int:
    print("Запишем образец твоего голоса. Прочитай вслух спокойно, как говоришь обычно:\n")
    print(f"   «{ENROLL_TEXT}»\n")
    input("Нажми Enter и начинай читать (запись 20 секунд)… ")
    print("● Запись…")
    wav = record_reference(20)
    seconds = (wav.stat().st_size - 44) / (24000 * 2)
    print(f"Готово: {seconds:.1f} с → {wav}")
    if seconds < 8:
        print("Слишком коротко — запиши ещё раз (sotto translate enroll).")
        return 1
    print('Проверь, как звучит клон: sotto translate say "Hi! I have five years of backend experience."')
    return 0


def say(text: str) -> int:
    cfg = config_mod.load()
    voice = MyVoice(cfg.api_key, cfg.openrouter.base_url, cfg.translator.clone_model, cfg.translator.fallback_my_voice)
    if not voice.cloned:
        print("Клона нет (нужны ключ OpenRouter и образец: sotto translate enroll) — говорю запасным голосом.")
    first = voice.synthesize_to(text, Player(None))
    print(f"Звук пошёл через {first:.2f} с")
    return 0


def _commands(engine: TranslatorEngine, devs: devices.VirtualDevices, loop: asyncio.AbstractEventLoop,
              stop: asyncio.Event) -> None:
    help_text = ("Команды (Enter после буквы): m — твой перевод вкл/выкл (выкл = говоришь сам), "
                 "i — перевод собеседников вкл/выкл, o 0.3 — громкость оригинала, q — выход")
    print(help_text)
    for line in sys.stdin:
        cmd = line.strip().split()
        if not cmd:
            continue
        if cmd[0] == "m":
            engine.translate_out = not engine.translate_out
            if engine.translate_out:
                engine.stop_passthrough()
            print("Твоя речь:", "переводится" if engine.translate_out else "идёт как есть (без перевода)")
        elif cmd[0] == "i":
            engine.translate_in = not engine.translate_in
            print("Собеседники:", "переводятся" if engine.translate_in else "без перевода")
        elif cmd[0] == "o" and len(cmd) > 1:
            devs.set_original_volume(float(cmd[1]))
            print(f"Громкость оригинала: {devs.original_volume:.2f}")
        elif cmd[0] == "q":
            loop.call_soon_threadsafe(stop.set)
            return
        else:
            print(help_text)


async def run() -> int:
    cfg = config_mod.load()
    if not cfg.api_key:
        print("Нужен ключ OpenRouter (перевод и клон голоса идут через него): sotto setup")
        return 1
    print("ЭКСПЕРИМЕНТ: голосовой переводчик. Только в наушниках — иначе микрофон услышит перевод.\n")
    if not has_reference():
        print("⚠ Нет образца голоса — собеседники услышат стандартный голос. Записать: sotto translate enroll\n")
    with devices.VirtualDevices(cfg.translator.original_volume) as devs:
        print("В приложении звонка выбери:")
        print("   динамик  → «Sotto: звонок (перевод)»")
        print("   микрофон → «Sotto: микрофон (перевод)»\n")
        engine = TranslatorEngine(cfg, on_phrase=lambda p: print(_fmt(p), flush=True))
        loop = asyncio.get_running_loop()
        stop = asyncio.Event()
        threading.Thread(target=_commands, args=(engine, devs, loop, stop), daemon=True).start()
        main = asyncio.create_task(engine.run())
        waiter = asyncio.create_task(stop.wait())
        try:
            await asyncio.wait({main, waiter}, return_when=asyncio.FIRST_COMPLETED)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            main.cancel()
            try:
                await main
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
    print("Переводчик остановлен, виртуальные устройства удалены.")
    return 0


def main(action: str, text: str = "") -> int:
    if WINDOWS:
        print(UNSUPPORTED, file=sys.stderr)
        return 1
    if action == "enroll":
        return enroll()
    if action == "say":
        return say(text or "Hi! I have five years of backend experience, mostly Go and PostgreSQL.")
    if action == "run":
        try:
            return asyncio.run(run())
        except KeyboardInterrupt:
            return 0
    return 1

