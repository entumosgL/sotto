"""Правки звонков и записей с сайта (команды): название и описание, участники, бриф, удаление."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from .ports import CallRepository, Events, Meta, Platform, RecordingRepository, SearchIndex


class CallService:
    def __init__(self, calls: CallRepository, recordings: RecordingRepository, meta: Meta, search: SearchIndex,
                 events: Events, platform: Platform, current_call: Callable[[], int | None],
                 recording_now: Callable[[], int | None], remove_files: Callable[[Path], None]):
        self.calls = calls
        self.recordings = recordings
        self.meta = meta
        self.search = search
        self.events = events
        self.platform = platform
        self.current_call = current_call
        self.recording_now = recording_now
        self.remove_files = remove_files

    async def edit(self, call_id: int, title: str, summary: str) -> None:
        self.calls.update(call_id, title=title.strip(), summary=summary.strip())
        await asyncio.to_thread(self.search.index_call, call_id)

    def delete(self, call_id: int) -> None:
        """Звонок с расшифровкой и подсказками; видео остаётся в «Записях»."""
        self.search.remove("utterance", self.calls.delete(call_id))
        self.search.remove("call", [call_id])

    def rename_speaker(self, speaker_id: int, name: str, role: str) -> None:
        self.calls.rename_speaker(speaker_id, name.strip() or None, role.strip() or None)
        self.events.publish("speaker", {"id": speaker_id, "name": name.strip(), "role": role.strip()})

    def set_next_brief(self, brief: str) -> None:
        """Бриф к звонку вне плана; если звонок уже идёт — и ему."""
        self.meta.set_meta("next_call_brief", brief.strip())
        call_id = self.current_call()
        if call_id is not None:
            self.calls.update(call_id, brief=brief.strip())

    def open_recording(self, rec_id: int, at: float = 0) -> None:
        rec = self.recordings.get(rec_id)
        if rec and Path(rec["path"]).exists():
            self.platform.open_media(Path(rec["path"]), max(0, at - rec["started_at"]) if at else 0)

    def delete_recording(self, rec_id: int) -> bool:
        """Удаляет файлы записи. Ту, что пишется сейчас, удалить нельзя."""
        rec = self.recordings.get(rec_id)
        if not rec or self.recording_now() == rec_id:
            return False
        self.remove_files(Path(rec["path"]))
        self.recordings.delete(rec_id)
        return True
