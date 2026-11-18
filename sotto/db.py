"""SQLite-хранилище: звонки, реплики, ответы, легенда, тренеры, вакансии.

Одно соединение на процесс под замком: запросы короткие, SQLite справляется.
Векторный поиск — расширение sqlite-vec (таблица `vectors`).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from . import paths

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

-- Звонок = отрезок времени, когда какое-то приложение держало микрофон (или включён вручную).
CREATE TABLE IF NOT EXISTS calls (
    id          INTEGER PRIMARY KEY,
    started_at  REAL NOT NULL,
    ended_at    REAL,
    app         TEXT,            -- кто держал микрофон: chrome, zoom, telegram...
    title       TEXT,            -- семантическое название (после звонка)
    summary     TEXT,            -- короткое описание (после звонка)
    language    TEXT,
    record_only INTEGER DEFAULT 0,
    brief       TEXT             -- бриф о компании/вакансии, если задан до звонка
);

CREATE TABLE IF NOT EXISTS speakers (
    id        INTEGER PRIMARY KEY,
    call_id   INTEGER NOT NULL REFERENCES calls(id) ON DELETE CASCADE,
    label     TEXT NOT NULL,     -- "Я", "Собеседник 1"...
    name      TEXT,              -- имя из представления или введённое руками
    role      TEXT,
    is_me     INTEGER DEFAULT 0,
    embedding BLOB               -- средний голосовой отпечаток
);

CREATE TABLE IF NOT EXISTS utterances (
    id         INTEGER PRIMARY KEY,
    call_id    INTEGER NOT NULL REFERENCES calls(id) ON DELETE CASCADE,
    speaker_id INTEGER REFERENCES speakers(id) ON DELETE SET NULL,
    channel    TEXT NOT NULL,    -- "me" (микрофон) | "them" (звук системы)
    started_at REAL NOT NULL,
    ended_at   REAL NOT NULL,
    text       TEXT NOT NULL,
    language   TEXT,
    cost       REAL              -- стоимость распознавания реплики
);
CREATE INDEX IF NOT EXISTS utterances_call ON utterances(call_id, started_at);

CREATE TABLE IF NOT EXISTS answers (
    id           INTEGER PRIMARY KEY,
    call_id      INTEGER REFERENCES calls(id) ON DELETE CASCADE,
    kind         TEXT NOT NULL,  -- manual | auto | screen | questions | short | detailed
    model        TEXT,
    prompt_tail  TEXT,           -- последняя реплика, на которую отвечали
    text         TEXT NOT NULL,
    created_at   REAL NOT NULL,
    first_token_ms INTEGER,
    total_ms     INTEGER,
    cost         REAL
);

-- Легенда: всё, что модель придумала о твоём опыте. Используется в следующих ответах.
CREATE TABLE IF NOT EXISTS legend (
    id         INTEGER PRIMARY KEY,
    claim      TEXT NOT NULL,
    note       TEXT,             -- твои правки/уточнения
    call_id    INTEGER REFERENCES calls(id) ON DELETE SET NULL,
    answer_id  INTEGER REFERENCES answers(id) ON DELETE SET NULL,
    created_at REAL NOT NULL,
    active     INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS recordings (
    id         INTEGER PRIMARY KEY,
    path       TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at   REAL,
    fps        INTEGER,
    size_bytes INTEGER
);

-- Семантический индекс: что проиндексировано (kind+ref_id) и его вектор (по rowid).
CREATE TABLE IF NOT EXISTS search_items (
    id      INTEGER PRIMARY KEY,
    kind    TEXT NOT NULL,       -- call | utterance | vacancy
    ref_id  INTEGER NOT NULL,
    text    TEXT NOT NULL,
    UNIQUE(kind, ref_id)
);

-- Тренеры.
CREATE TABLE IF NOT EXISTS trainer_sessions (
    id         INTEGER PRIMARY KEY,
    kind       TEXT NOT NULL,    -- knowledge | english
    topic      TEXT,
    started_at REAL NOT NULL,
    ended_at   REAL,
    feedback   TEXT
);
CREATE TABLE IF NOT EXISTS trainer_turns (
    id         INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES trainer_sessions(id) ON DELETE CASCADE,
    role       TEXT NOT NULL,    -- coach | user
    text       TEXT NOT NULL,
    feedback   TEXT,             -- разбор ответа пользователя
    created_at REAL NOT NULL
);
-- Темы/ошибки для интервального повторения.
CREATE TABLE IF NOT EXISTS trainer_items (
    id         INTEGER PRIMARY KEY,
    kind       TEXT NOT NULL,    -- knowledge | english
    item       TEXT NOT NULL,    -- тема или повторяющаяся ошибка
    source     TEXT,             -- calls | vacancies | manual | session
    score      REAL DEFAULT 0,   -- 0..1, насколько уверенно владеешь
    due_at     REAL DEFAULT 0,
    seen       INTEGER DEFAULT 0,
    UNIQUE(kind, item)
);

-- Вакансии и отклики.
CREATE TABLE IF NOT EXISTS vacancies (
    id          INTEGER PRIMARY KEY,
    source      TEXT NOT NULL,   -- telegram | linkedin_jobs | linkedin_posts
    source_ref  TEXT NOT NULL,   -- ссылка/идентификатор сообщения
    url         TEXT,
    title       TEXT,
    company     TEXT,
    location    TEXT,
    salary      TEXT,
    remote      TEXT,
    language    TEXT,
    text        TEXT NOT NULL,
    summary     TEXT,            -- 2 строки от агента: стек, уровень, требования
    fit         REAL,            -- 0..1, насколько подходит под профиль (оценка агента)
    contacts    TEXT,            -- JSON: [{"type": "telegram"|"email"|"linkedin"|"url", "value": "..."}]
    found_at    REAL NOT NULL,
    posted_at   REAL,
    cluster_id  INTEGER REFERENCES clusters(id) ON DELETE SET NULL,
    status      TEXT DEFAULT 'new',  -- new | hidden | ready | sent | replied
    letter_override TEXT,        -- правки письма под конкретную вакансию
    resume_override TEXT,
    UNIQUE(source, source_ref)
);
CREATE TABLE IF NOT EXISTS clusters (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT,
    resume_md   TEXT,            -- вариант резюме под кластер (Markdown)
    letter_md   TEXT,            -- шаблон сопроводительного письма
    centroid    BLOB,
    updated_at  REAL
);
CREATE TABLE IF NOT EXISTS outreach (
    id          INTEGER PRIMARY KEY,
    vacancy_id  INTEGER REFERENCES vacancies(id) ON DELETE CASCADE,
    channel     TEXT NOT NULL,   -- telegram | email | linkedin
    target      TEXT NOT NULL,
    text        TEXT NOT NULL,
    sent_at     REAL NOT NULL,
    status      TEXT NOT NULL    -- sent | opened_in_browser | failed
);
CREATE TABLE IF NOT EXISTS search_queries (
    id        INTEGER PRIMARY KEY,
    query     TEXT NOT NULL UNIQUE,
    source    TEXT NOT NULL,     -- manual | agent
    active    INTEGER DEFAULT 1,
    hits      INTEGER DEFAULT 0,
    created_at REAL NOT NULL
);

-- Собеседования: план (дата, компания, этап) и исход. Звонок в окне собеседования привязывается к нему,
-- а легенда — к компании: придуманное держится в пределах одной компании и удаляется, когда процесс закрыт.
CREATE TABLE IF NOT EXISTS interviews (
    id           INTEGER PRIMARY KEY,
    company      TEXT NOT NULL,
    role         TEXT,
    stage        TEXT,            -- HR, техническое, финальное...
    scheduled_at REAL,
    duration_min INTEGER DEFAULT 60,
    link         TEXT,            -- ссылка на звонок
    vacancy      TEXT,            -- текст вакансии
    brief        TEXT,            -- бриф для суфлёра
    notes        TEXT,
    status       TEXT DEFAULT 'planned',  -- planned | done | passed | offer | rejected | declined | cancelled
    outcome      TEXT,            -- текст отказа / обратная связь
    analysis     TEXT,            -- разбор отказа (Markdown)
    reminded     INTEGER DEFAULT 0,
    created_at   REAL NOT NULL,
    updated_at   REAL
);

-- Агенты: когда агент упёрся в лимит — не трогаем до cooldown_until.
CREATE TABLE IF NOT EXISTS agent_state (
    name           TEXT PRIMARY KEY,
    cooldown_until REAL DEFAULT 0,
    last_error     TEXT,
    calls          INTEGER DEFAULT 0
);
"""


class DB:
    def __init__(self, path: Path | None = None):
        path = path or paths.DB_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.vec_enabled = self._load_vec()

    def _migrate(self) -> None:
        """Новые столбцы в старых базах (CREATE TABLE IF NOT EXISTS их не добавит)."""
        add = {"calls": [("company", "TEXT"), ("interview_id", "INTEGER")],
               "legend": [("company", "TEXT")]}
        for table, cols in add.items():
            have = {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")}
            for name, kind in cols:
                if name not in have:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {kind}")

    def _load_vec(self) -> bool:
        try:
            import sqlite_vec
            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            return True
        except Exception:
            return False

    # --- базовые операции ---

    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self.conn.execute(sql, tuple(params))

    def insert(self, table: str, **values: Any) -> int:
        cols = ", ".join(values)
        marks = ", ".join("?" for _ in values)
        cur = self.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", values.values())
        return int(cur.lastrowid)

    def update(self, table: str, row_id: int, **values: Any) -> None:
        if not values:
            return
        sets = ", ".join(f"{k} = ?" for k in values)
        self.execute(f"UPDATE {table} SET {sets} WHERE id = ?", [*values.values(), row_id])

    def one(self, sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
        row = self.execute(sql, params).fetchone()
        return dict(row) if row else None

    def all(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        return [dict(r) for r in self.execute(sql, params).fetchall()]

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.one("SELECT value FROM meta WHERE key = ?", [key])
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        self.execute("INSERT INTO meta(key, value) VALUES(?, ?) "
                     "ON CONFLICT(key) DO UPDATE SET value = excluded.value", [key, value])

    # --- векторный индекс ---

    def ensure_vectors(self, dim: int, model: str) -> None:
        """Создаёт таблицу векторов; при смене модели эмбеддингов — пересоздаёт."""
        if not self.vec_enabled:
            return
        signature = f"{model}|{dim}|v2"  # v2: вид записи — partition key, чтобы KNN искал внутри вида
        if self.get_meta("embed_signature") != signature:
            self.execute("DROP TABLE IF EXISTS vectors")
            self.execute("DELETE FROM search_items")
            self.set_meta("embed_signature", signature)
        self.execute("CREATE VIRTUAL TABLE IF NOT EXISTS vectors USING "
                     f"vec0(kind text partition key, embedding float[{dim}])")

    # --- удобные запросы, которые нужны в нескольких местах ---

    def active_legend(self, call_id: int | None = None) -> list[dict[str, Any]]:
        """Легенда, которой суфлёр держится в этом звонке: придуманное в этом же звонке и раньше — с той же
        компанией, плюс то, что ты добавил вручную без компании. В другие компании придуманное не переносится.
        На тренировках легенды нет вовсе: каждое тренировочное собеседование — с чистого листа."""
        call = self.one("SELECT app, company FROM calls WHERE id = ?", [call_id]) if call_id is not None else None
        if call and call["app"] == "training":
            return []
        company = norm_company(call["company"]) if call else ""
        return self.all(
            """SELECT * FROM legend WHERE active = 1
               AND (call_id = ? OR (call_id IS NULL AND (company IS NULL OR company = ''))
                    OR (? != '' AND lower(trim(company)) = ?))
               ORDER BY created_at""", [call_id if call_id is not None else -1, company, company])

    def set_call_company(self, call_id: int, company: str | None, interview_id: int | None = None) -> None:
        """Компания звонка; придуманное в нём переезжает к этой компании."""
        company = (company or "").strip() or None
        values: dict[str, Any] = {"company": company}
        if interview_id is not None:
            values["interview_id"] = interview_id
        self.update("calls", call_id, **values)
        self.execute("UPDATE legend SET company = ? WHERE call_id = ?", [company, call_id])

    def closed_companies(self) -> list[str]:
        """Компании, где последнее собеседование закончилось отказом (их или твоим) или отменено."""
        latest: dict[str, str] = {}
        for r in self.all("""SELECT lower(trim(company)) AS c, status FROM interviews
                             ORDER BY COALESCE(scheduled_at, created_at), id"""):
            latest[r["c"]] = r["status"]
        return [c for c, st in latest.items() if c and st in ("rejected", "declined", "cancelled")]

    def cleanup_legend(self, keep_days: float = 1.0) -> int:
        """Убирает легенду, которая больше не пригодится:
        - с тренировок (их легенда не используется);
        - из звонков, где компания так и не определилась (старше keep_days);
        - компаний, у которых процесс закрыт (отказ, отказался, отменено) и нет запланированных собеседований."""
        before = self.one("SELECT COUNT(*) AS n FROM legend")["n"]
        self.execute("DELETE FROM legend WHERE call_id IN (SELECT id FROM calls WHERE app = 'training')")
        self.execute("""DELETE FROM legend WHERE call_id IS NOT NULL AND (company IS NULL OR company = '')
                        AND created_at < ?""", [time.time() - keep_days * 86400])
        for c in self.closed_companies():
            self.execute("DELETE FROM legend WHERE lower(trim(company)) = ?", [c])
        return before - self.one("SELECT COUNT(*) AS n FROM legend")["n"]

    def call_transcript(self, call_id: int, since: float = 0.0) -> list[dict[str, Any]]:
        return self.all(
            """SELECT u.*, COALESCE(s.name, s.label) AS speaker, s.role AS speaker_role
               FROM utterances u LEFT JOIN speakers s ON s.id = u.speaker_id
               WHERE u.call_id = ? AND u.started_at >= ? ORDER BY u.started_at""",
            [call_id, since],
        )


def norm_company(name: str | None) -> str:
    return (name or "").strip().lower()


def now() -> float:
    return time.time()


def to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)
