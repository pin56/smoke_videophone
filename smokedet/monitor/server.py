"""HTTP-сервер мониторинга: только 127.0.0.1, стандартная библиотека.

Маршруты:
- `/` — страница (`ui/static/wall.html`), `/static/*` — её файлы;
- `/api/live` — поток SSE: `frame` (кадры камер, 10 к/с) и `state` (состояние камер, раз в секунду);
- `/api/state`, `/api/config`, `/api/archive?from=&to=&step=`, `/api/archive/range`;
- `/api/events[?source_type=&source_id=&review=&since=&until=&limit=]`, `/api/events/<id>`,
  `POST /api/events/<id>/review` ({"decision": "confirmed" | "rejected" | "pending"});
- `/api/jobs`, `/api/jobs/<id>`, `POST /api/jobs` (тело — файл, имя в заголовке `X-Filename`, длительность,
  измеренная браузером, — в `X-Duration`);
- `/clips/<id>[?download=1]` — фрагмент эпизода, `/files/<job>` — загруженный файл; оба с HTTP Range.
"""
from __future__ import annotations

import json
import logging
import re
import sys
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import parse_qs, quote, unquote, urlparse

from .analysis import MAX_UPLOAD, VIDEO_EXT, AnalysisQueue, new_job_id
from .cameras import FPS, POST_ROLL, PRE_ROLL, CameraWall
from .journal import DAY, REVIEWS, Journal

log = logging.getLogger(__name__)

STATIC = Path(__file__).resolve().parent.parent / "ui" / "static"
CONTENT_TYPES = {  # на Windows mimetypes берёт типы из реестра и может отдать .js как text/plain
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
    ".webm": "video/webm",
    ".mkv": "video/x-matroska",
    ".avi": "video/x-msvideo",
    ".ts": "video/mp2t",
}
CHUNK = 1 << 16
ARCHIVE_MAX = 180.0  # самое длинное окно архива за один запрос, с


def parse_range(header: Optional[str], size: int) -> Optional[Tuple[int, int]]:
    """Range "bytes=a-b" → (начало, конец включительно); None — весь файл; ValueError — вне файла (416).
    Та же логика, что в `ui.server.parse_range` (тот модуль требует cv2)."""
    m = re.fullmatch(r"\s*bytes=(\d*)-(\d*)\s*", header or "")
    if not m or m.group(1) == m.group(2) == "":
        return None
    a, b = m.groups()
    if a == "":
        start, end = max(0, size - int(b)), size - 1
    else:
        start, end = int(a), min(int(b), size - 1) if b else size - 1
    if start >= size or start > end:
        raise ValueError(header)
    return start, end


def event_view(ev: dict) -> dict:
    """Эпизод для клиента: поля журнала + состояние и адреса фрагмента."""
    out = dict(ev)
    if ev["clip_deleted_at"]:
        status = "deleted"
    elif ev["clip_kind"]:
        status = "saved"
    else:
        status = "recording" if ev["active"] or ev["source_type"] == "camera" else "none"
    out["clip_status"] = status
    if status == "saved":
        out["clip_url"] = f"/clips/{ev['id']}"
        out["download_url"] = f"/clips/{ev['id']}?download=1"
    return out


class MonitorServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = sys.platform != "win32"  # на Windows порт молча заняли бы двое

    def __init__(self, address, journal: Journal, wall: CameraWall, analysis: AnalysisQueue, static_dir: Path = STATIC):
        self.journal, self.wall, self.analysis = journal, wall, analysis
        self.static_dir = static_dir.resolve()
        super().__init__(address, Handler)

    def config(self) -> dict:
        analyzer = self.analysis.analyzer
        return {
            "analyzer": analyzer.name,
            "analyzer_problem": analyzer.available(),
            "retention_days": round(self.journal.retention / DAY, 3),
            "ffmpeg": bool(self.analysis.ffmpeg),
            "archive_sec": self.wall.archive_sec,
            "pre_roll": PRE_ROLL,
            "post_roll": POST_ROLL,
            "fps": FPS,
            "video_ext": sorted(VIDEO_EXT),
        }


class Handler(BaseHTTPRequestHandler):
    server: MonitorServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        log.debug("%s %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:
        self._safe(self._get, head=False)

    def do_HEAD(self) -> None:
        self._safe(self._get, head=True)

    def do_POST(self) -> None:
        self._safe(self._post)

    def _safe(self, fn, **kw) -> None:
        try:
            fn(**kw)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # браузер обрывает запросы видео при перемотке и поток при закрытии вкладки

    # ---------- GET ----------

    def _get(self, head: bool) -> None:
        url = urlparse(self.path)
        path, q = unquote(url.path), {k: v[-1] for k, v in parse_qs(url.query).items()}
        s = self.server
        if path in ("/", "/index.html"):
            return self._file(s.static_dir / "wall.html", head)
        if path.startswith("/static/"):
            target = (s.static_dir / path[len("/static/"):]).resolve()
            if s.static_dir in target.parents and target.is_file():
                return self._file(target, head)
            return self._error(HTTPStatus.NOT_FOUND, head)
        if path == "/api/live":
            return self._live()
        if path == "/api/config":
            return self._json(s.config(), head)
        if path == "/api/state":
            return self._json({**s.wall.state(), "journal": s.journal.version}, head)
        if path == "/api/archive/range":
            a, b = s.wall.archive_range()
            return self._json({"from": a, "to": b}, head)
        if path == "/api/archive":
            try:
                a, b = float(q["from"]), float(q["to"])
                step = int(q.get("step", 1))
            except (KeyError, ValueError):
                return self._error(HTTPStatus.BAD_REQUEST, head, "нужны from и to")
            b = min(b, a + ARCHIVE_MAX)
            return self._send(s.wall.archive(a, b, step).encode(), "application/json; charset=utf-8", head)
        if path == "/api/events":
            try:
                kw = {k: q.get(k) for k in ("source_type", "source_id", "review")}
                for k in ("since", "until"):
                    kw[k] = float(q[k]) if k in q else None
                kw["limit"] = max(1, min(int(q.get("limit", 200)), 1000))  # LIMIT < 0 в sqlite — без ограничения
            except ValueError:
                return self._error(HTTPStatus.BAD_REQUEST, head)
            return self._json([event_view(e) for e in s.journal.events(**kw)], head)
        m = re.fullmatch(r"/api/events/(\d+)", path)
        if m:
            ev = s.journal.event(int(m.group(1)))
            return self._json(event_view(ev), head) if ev else self._error(HTTPStatus.NOT_FOUND, head)
        if path == "/api/jobs":
            return self._json([self._job_view(j) for j in s.journal.jobs()], head)
        m = re.fullmatch(r"/api/jobs/([\w-]+)", path)
        if m:
            job = s.journal.job(m.group(1))
            if not job:
                return self._error(HTTPStatus.NOT_FOUND, head)
            events = s.journal.events(source_type="file", source_id=job["id"], limit=1000)
            return self._json({**self._job_view(job), "events": [event_view(e) for e in events]}, head)
        m = re.fullmatch(r"/clips/(\d+)", path)
        if m:
            return self._clip(int(m.group(1)), "download" in q, head)
        m = re.fullmatch(r"/files/([\w-]+)", path)
        if m:
            job = s.journal.job(m.group(1))
            if not job or job["deleted_at"] or not job["path"] or not Path(job["path"]).is_file():
                return self._error(HTTPStatus.GONE if job else HTTPStatus.NOT_FOUND, head)
            return self._file(Path(job["path"]), head)
        return self._error(HTTPStatus.NOT_FOUND, head)

    def _job_view(self, job: dict) -> dict:
        out = {k: v for k, v in job.items() if k != "path"}
        out["file_url"] = None if job["deleted_at"] else f"/files/{job['id']}"
        out["episodes"] = len(self.server.journal.events(source_type="file", source_id=job["id"], limit=10000))
        return out

    def _clip(self, event_id: int, download: bool, head: bool) -> None:
        ev = self.server.journal.event(event_id)
        if not ev:
            return self._error(HTTPStatus.NOT_FOUND, head)
        if ev["clip_deleted_at"]:
            return self._error(HTTPStatus.GONE, head, "срок хранения фрагмента истёк")
        if ev["clip_kind"] == "range":
            job = self.server.journal.job(ev["source_id"])
            path = Path(job["path"]) if job and job["path"] and not job["deleted_at"] else None
        else:
            path = Path(ev["clip_path"]) if ev["clip_path"] else None
        if path is None or not path.is_file():
            return self._error(HTTPStatus.NOT_FOUND, head, "фрагмент ещё не сохранён")
        name = None
        if download:
            base = re.sub(r"[^\w.-]+", "_", f"episode_{event_id}_{ev['source_id']}")
            name = base + (path.suffix if ev["clip_kind"] != "range" else "_" + ev["source_name"])
        self._file(path, head, download_name=name)

    def _live(self) -> None:
        """SSE: кадр после каждого шага камер и состояние раз в секунду. Соединение закрывается клиентом."""
        wall, journal = self.server.wall, self.server.journal
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        seq, last_state = -1, 0.0
        while True:
            with wall.cond:
                wall.cond.wait_for(lambda: wall.seq != seq, timeout=1.0)
                seq = wall.seq
            chunks = [f"event: frame\ndata: {wall.live_frame()}\n\n"]
            now = time.monotonic()
            if now - last_state >= 1.0:
                last_state = now
                state = {**wall.state(), "journal": journal.version}
                chunks.append(f"event: state\ndata: {json.dumps(state, ensure_ascii=False)}\n\n")
            self.wfile.write("".join(chunks).encode("utf-8"))
            self.wfile.flush()

    # ---------- POST ----------

    def _post(self) -> None:
        path = unquote(urlparse(self.path).path)
        m = re.fullmatch(r"/api/events/(\d+)/review", path)
        if m:
            body = self._body_json()
            decision = body.get("decision") if isinstance(body, dict) else None
            if decision not in REVIEWS:
                return self._error(HTTPStatus.BAD_REQUEST, False, "decision: confirmed | rejected | pending")
            if not self.server.journal.review(int(m.group(1)), decision):
                return self._error(HTTPStatus.NOT_FOUND, False)
            return self._json(event_view(self.server.journal.event(int(m.group(1)))), False)
        if path == "/api/jobs":
            return self._upload()
        return self._error(HTTPStatus.NOT_FOUND, False)

    def _body_json(self):
        try:
            n = max(0, min(int(self.headers.get("Content-Length") or 0), 1 << 20))
            return json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return None

    def _upload(self) -> None:
        s = self.server
        name = Path(unquote(self.headers.get("X-Filename") or "")).name
        ext = Path(name).suffix.lower()
        try:
            size = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            size = 0
        if not name or ext not in VIDEO_EXT:
            self._drain(size)
            return self._error(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, False, f"нужен видеофайл: {', '.join(sorted(VIDEO_EXT))}")
        if size <= 0 or size > MAX_UPLOAD:
            self._drain(size)
            return self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, False, "файл пустой или больше 8 ГБ")
        try:
            duration = float(self.headers.get("X-Duration") or "nan")
            duration = duration if duration > 0 and duration < 1e7 else None
        except ValueError:
            duration = None
        job_id = new_job_id()
        target = s.analysis.uploads_dir / f"{job_id}{ext}"
        left = size
        with open(target, "wb") as f:
            while left > 0:
                chunk = self.rfile.read(min(CHUNK * 16, left))
                if not chunk:
                    break
                f.write(chunk)
                left -= len(chunk)
        if left:
            target.unlink(missing_ok=True)
            return self._error(HTTPStatus.BAD_REQUEST, False, "загрузка прервана")
        s.journal.add_job(job_id, name, str(target), size, duration, s.analysis.analyzer.name)
        s.analysis.submit(job_id)
        log.info("[%s] загружен %s (%.1f МБ)", job_id, name, size / 2**20)
        self._json(self._job_view(s.journal.job(job_id)), False, HTTPStatus.CREATED)

    def _drain(self, n: int) -> None:
        while n > 0:
            chunk = self.rfile.read(min(CHUNK * 16, n))
            if not chunk:
                break
            n -= len(chunk)

    # ---------- ответы ----------

    def _send(self, data: bytes, ctype: str, head: bool, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head:
            self.wfile.write(data)

    def _json(self, obj, head: bool, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(obj, ensure_ascii=False, allow_nan=False, default=_nan_safe).encode("utf-8")
        self._send(data, "application/json; charset=utf-8", head, status)

    def _error(self, status: HTTPStatus, head: bool, message: str = "") -> None:
        self._json({"error": message or status.phrase}, head, status)

    def _file(self, path: Path, head: bool, download_name: Optional[str] = None) -> None:
        size = path.stat().st_size
        try:
            rng = parse_range(self.headers.get("Range"), size)
        except ValueError:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        start, end = rng if rng else (0, size - 1)
        self.send_response(HTTPStatus.PARTIAL_CONTENT if rng else HTTPStatus.OK)
        self.send_header("Content-Type", CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-store")
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download_name:
            self.send_header("Content-Disposition", f"attachment; filename*=UTF-8''{quote(download_name)}")
        self.end_headers()
        if head:
            return
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(CHUNK, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)


def _nan_safe(o):
    raise TypeError(type(o).__name__)
