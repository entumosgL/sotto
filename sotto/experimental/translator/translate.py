"""Перевод фраз через OpenRouter с контекстом разговора.

Две разные задачи:
* им → тебе: точный, короткий разговорный русский; термины оставляем как говорят в индустрии;
* тебе → им: не дословно, а так, как сказал бы уверенный англоговорящий инженер — без «ээ», повторов
  и кальки; смысл и технические детали сохраняем полностью, ничего не добавляем от себя.
"""

from __future__ import annotations

from collections import deque

from ...application.prompting import render
from ...infrastructure.llm.openrouter import OpenRouter
from ...infrastructure.prompts import prompts

LANG = {"ru": "русский", "en": "английский", "de": "немецкий", "es": "испанский", "fr": "французский"}

TO_ME = """Ты синхронный переводчик созвона. Переводи реплику собеседника с языка «{src}» на язык «{dst}».
Коротко и естественно, разговорным языком, сохраняя смысл и тон. Технические термины, названия технологий,
компаний и продуктов оставляй как принято у разработчиков (Kafka, rate limiter, pull request).
Реплика может быть оборванной на полуслове — переводи то, что есть, не дописывай.
Выведи ТОЛЬКО перевод, без кавычек и пояснений. Если в реплике нет слов (шум, «мм») — выведи пустую строку."""

FROM_ME = """Ты переводишь речь кандидата на собеседовании с языка «{src}» на язык «{dst}» — его голосом услышат
собеседники. Сделай так, как это сказал бы уверенный в языке инженер: естественные фразы, без слов-паразитов,
повторов, оговорок и кальки. Смысл, факты, цифры и технические детали сохраняй полностью и ничего не добавляй
от себя. Длина — примерно как у оригинала. Реплика может быть частью фразы — переводи её как есть, так чтобы
её можно было произнести отдельно.
Профиль кандидата (для терминов и имён собственных):
{profile}
Выведи ТОЛЬКО текст для произнесения, без кавычек и пояснений. Если слов нет — выведи пустую строку."""


class Translator:
    def __init__(self, llm: OpenRouter, model: str, their: str, mine: str):
        self.llm = llm
        self.model = model
        self.their, self.mine = their, mine
        self.history: deque[str] = deque(maxlen=8)  # последние реплики обеих сторон — для связности

    def _system(self, direction: str) -> str:
        if direction == "in":
            return render(TO_ME, src=LANG.get(self.their, self.their), dst=LANG.get(self.mine, self.mine))
        return render(FROM_ME, src=LANG.get(self.mine, self.mine), dst=LANG.get(self.their, self.their),
                              profile=prompts.profile()[:3000] or "(нет)")

    async def translate(self, direction: str, text: str, who: str) -> tuple[str, int | None]:
        """direction: "in" — им→тебе, "out" — тебе→им. Возвращает (перевод, мс до первого токена)."""
        context = "\n".join(self.history) or "(начало разговора)"
        messages = [
            {"role": "system", "content": self._system(direction)},
            {"role": "user", "content": f"Предыдущие реплики (для контекста, не переводить):\n{context}\n\n"
                                        f"Переведи реплику ({who}):\n{text}"},
        ]
        out, first = "", None
        async for delta, res in self.llm.stream_chat(self.model, messages, max_tokens=400, temperature=0.2,
                                                     session_id="sotto-translator"):
            if delta:
                out = res.text
            else:
                first = res.first_token_ms
        out = out.strip().strip('"«»').strip()
        self.history.append(f"{who}: {text}" + (f" → {out}" if out else ""))
        return out, first
