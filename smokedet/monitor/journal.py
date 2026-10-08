"""Журнал эпизодов и заданий проверки видео (sqlite) и срок хранения фрагментов.

Эпизод всегда отмечает система (`detection = 'auto'`), а решение по нему — отдельное поле `review`
(pending — не проверен, confirmed — подтверждён оператором, rejected — отклонён оператором).
Фрагмент записи хранится `retention` секунд от сохранения; по истечении срока файл удаляется, запись журнала
остаётся с датой удаления.

Время фрагмента (`clip_start`/`clip_end`) — в шкале источника: у камеры — unix-время, у файла — секунды от
начала записи. В той же шкале начало эпизода: у камеры `started_at`, у файла `offset_sec`.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

log = logging.getLogger(__name__)

DAY = 86400.0
REVIEWS = ("pending", "confirmed", "rejected")
JOB_STATES = ("queued", "analyzing", "done", "error")

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY,
    source_type TEXT NOT NULL,              -- camera | file
    source_id TEXT NOT NULL,                -- id камеры или задания
    source_name TEXT NOT NULL,              -- название камеры или имя файла
    track INTEGER,
    started_at REAL NOT NULL,               -- unix-время: начало эпизода (у файла — время загрузки)
    offset_sec REAL,                        -- у файла: начало эпизода от начала записи, с
    duration REAL NOT NULL DEFAULT 0,
    cycles INTEGER NOT NULL DEFAULT 0,
    confidence REAL NOT NULL DEFAULT 0,
    detection TEXT NOT NULL DEFAULT 'auto',
    active INTEGER NOT NULL DEFAULT 0,      -- эпизод ещё идёт
    review TEXT NOT NULL DEFAULT 'pending',
    reviewed_at REAL,
    demo INTEGER NOT NULL DEFAULT 0,        -- найден демо-анализом, а не пайплайном
    clip_kind TEXT,                         -- sim | video | range; NULL — фрагмент ещё не сохранён
    clip_path TEXT,
    clip_start REAL,
    clip_end REAL,
    clip_saved_at REAL,
    clip_expires_at REAL,
    clip_deleted_at REAL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS events_started ON events (started_at);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    path TEXT,
    size INTEGER NOT NULL DEFAULT 0,
    duration REAL,
    state TEXT NOT NULL DEFAULT 'queued',
    progress REAL NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    analyzer TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    expires_at REAL,
    deleted_at REAL
);
"""

EVENT_FIELDS = (
    "source_type", "source_id", "source_name", "track", "started_at", "offset_sec", "duration", "cycles",
    "confidence", "active", "demo",
)  # fmt: skip


class Journal:
    """Потокобезопасная обёртка над sqlite: пишут поток камер, очередь анализа и HTTP-обработчики."""

    def __init__(self, path: Path, retention: float = 7 * DAY, clock: Callable[[], float] = time.time):
        self.path = Path(path)
        self.retention = retention
        self.clock = clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self.version = 0  # растёт при любом изменении: клиенты перечитывают журнал
        with self._lock:
            self._db.executescript(SCHEMA)
            # задания, прерванные остановкой сервера, не продолжатся сами
            self._db.execute(
                "UPDATE jobs SET state='error', error='анализ прерван остановкой сервера' "
                "WHERE state IN ('queued', 'analyzing')"
            )
            self._db.execute("UPDATE events SET active=0 WHERE active=1")
            self._db.commit()

    def _write(self, sql: str, args=()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._db.execute(sql, args)
            self._db.commit()
            self.version += 1
            return cur

    def _read(self, sql: str, args=()) -> List[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute(sql, args).fetchall()]

    # ---------- эпизоды ----------

    def add_event(self, **fields) -> int:
        unknown = set(fields) - set(EVENT_FIELDS)
        if unknown:
            raise ValueError(f"неизвестные поля эпизода: {sorted(unknown)}")
        cols = list(fields) + ["created_at"]
        vals = list(fields.values()) + [self.clock()]
        sql = f"INSERT INTO events ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
        return self._write(sql, vals).lastrowid

    def update_event(self, event_id: int, **fields) -> None:
        allowed = set(EVENT_FIELDS) - {"source_type", "source_id"}
        if set(fields) - allowed:
            raise ValueError(f"нельзя менять поля: {sorted(set(fields) - allowed)}")
        sets = ", ".join(f"{k}=?" for k in fields)
        self._write(f"UPDATE events SET {sets} WHERE id=?", [*fields.values(), event_id])

    def attach_clip(self, event_id: int, kind: str, path: Optional[str], start: float, end: float,
                    saved_at: Optional[float] = None) -> None:  # fmt: skip
        saved = self.clock() if saved_at is None else saved_at
        self._write(
            "UPDATE events SET clip_kind=?, clip_path=?, clip_start=?, clip_end=?, clip_saved_at=?, clip_expires_at=? "
            "WHERE id=?",
            (kind, path, start, end, saved, saved + self.retention, event_id),
        )

    def review(self, event_id: int, decision: str) -> bool:
        """Решение оператора. pending снимает решение. False — эпизода нет."""
        if decision not in REVIEWS:
            raise ValueError(decision)
        at = None if decision == "pending" else self.clock()
        return self._write("UPDATE events SET review=?, reviewed_at=? WHERE id=?", (decision, at, event_id)).rowcount > 0

    def event(self, event_id: int) -> Optional[dict]:
        rows = self._read("SELECT * FROM events WHERE id=?", (event_id,))
        return rows[0] if rows else None

    def events(self, source_type: Optional[str] = None, source_id: Optional[str] = None, review: Optional[str] = None,
               since: Optional[float] = None, until: Optional[float] = None, limit: int = 200) -> List[dict]:  # fmt: skip
        where, args = [], []
        for col, val in (("source_type", source_type), ("source_id", source_id), ("review", review)):
            if val:
                where.append(f"{col}=?")
                args.append(val)
        if since is not None:
            where.append("started_at + duration >= ?")
            args.append(since)
        if until is not None:
            where.append("started_at <= ?")
            args.append(until)
        sql = "SELECT * FROM events" + (" WHERE " + " AND ".join(where) if where else "")
        return self._read(sql + " ORDER BY started_at DESC, id DESC LIMIT ?", (*args, limit))

    # ---------- задания ----------

    def add_job(self, job_id: str, filename: str, path: str, size: int, duration: Optional[float], analyzer: str) -> None:
        self._write(
            "INSERT INTO jobs (id, filename, path, size, duration, analyzer, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (job_id, filename, path, size, duration, analyzer, self.clock()),
        )

    def update_job(self, job_id: str, **fields) -> None:
        allowed = {"state", "progress", "error", "duration", "started_at", "finished_at", "expires_at"}
        if set(fields) - allowed:
            raise ValueError(f"нельзя менять поля задания: {sorted(set(fields) - allowed)}")
        if "state" in fields and fields["state"] not in JOB_STATES:
            raise ValueError(fields["state"])
        sets = ", ".join(f"{k}=?" for k in fields)
        self._write(f"UPDATE jobs SET {sets} WHERE id=?", [*fields.values(), job_id])

    def set_progress(self, job_id: str, progress: float) -> None:
        """Прогресс без увеличения version: иначе клиенты перечитывали бы журнал на каждом кадре."""
        with self._lock:
            self._db.execute("UPDATE jobs SET progress=? WHERE id=?", (progress, job_id))
            self._db.commit()

    def job(self, job_id: str) -> Optional[dict]:
        rows = self._read("SELECT * FROM jobs WHERE id=?", (job_id,))
        return rows[0] if rows else None

    def jobs(self, limit: int = 100) -> List[dict]:
        return self._read("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))

    # ---------- срок хранения ----------

    def purge(self, now: Optional[float] = None) -> Dict[str, int]:
        """Удаляет фрагменты и загруженные файлы с истёкшим сроком. Записи журнала остаются."""
        now = self.clock() if now is None else now
        removed = {"clips": 0, "uploads": 0}
        for ev in self._read(
            "SELECT id, clip_kind, clip_path FROM events WHERE clip_expires_at <= ? AND clip_deleted_at IS NULL", (now,)
        ):
            if ev["clip_kind"] in ("sim", "video") and ev["clip_path"]:
                _unlink(ev["clip_path"])
            self._write("UPDATE events SET clip_deleted_at=? WHERE id=?", (now, ev["id"]))
            removed["clips"] += 1
        for job in self._read(
            "SELECT id, path FROM jobs WHERE expires_at <= ? AND deleted_at IS NULL", (now,)
        ):
            if job["path"]:
                _unlink(job["path"])
            self._write("UPDATE jobs SET deleted_at=? WHERE id=?", (now, job["id"]))
            removed["uploads"] += 1
        if removed["clips"] or removed["uploads"]:
            log.info("срок хранения истёк: удалено фрагментов %d, загруженных файлов %d", removed["clips"],
                     removed["uploads"])  # fmt: skip
        return removed

    def close(self) -> None:
        with self._lock:
            self._db.close()


def _unlink(path: str) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError as e:
        log.warning("не удалось удалить %s: %s", path, e)


def start_purger(journal: Journal, interval: float = 600.0) -> threading.Thread:
    """Фоновая очистка: сразу при запуске и затем раз в `interval` секунд."""

    def loop() -> None:
        while True:
            try:
                journal.purge()
            except Exception:
                log.exception("ошибка очистки хранилища")
            time.sleep(interval)

    thread = threading.Thread(target=loop, name="monitor-purge", daemon=True)
    thread.start()
    return thread
