"""Веб-интерфейс повтора анализа: клипы обрабатываются в фоне по очереди, HTTP-сервер слушает только 127.0.0.1.

Видео объекта нельзя отправлять наружу, поэтому сервер не открывается в сеть, а статика не тянет ничего с CDN.
Маршруты: `/` и `/static/*` — страница, `/api/clips` — список клипов и их состояние, `/api/clips/<id>` — данные
анализа (payload.py), `/video/<id>` — видео с поддержкой HTTP Range (без неё браузер не перематывает).
"""
from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple
from urllib.parse import quote, unquote, urlparse

import cv2

from .payload import build_payload

log = logging.getLogger(__name__)

STATIC = Path(__file__).resolve().parent / "static"

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".mp4": "video/mp4",
    ".m4v": "video/mp4",
    ".mov": "video/quicktime",
}
BROWSER_CODECS = {"avc1", "h264", "x264"}
BROWSER_CONTAINERS = {".mp4", ".m4v", ".mov"}
CHUNK = 1 << 16


@dataclass
class ClipJob:
    clip: object
    state: str = "queued"
    progress: float = 0.0
    error: str = ""
    payload: Optional[bytes] = None
    video: Optional[Path] = None
    events: int = 0

    def summary(self) -> dict:
        return {
            "id": self.clip.clip_id,
            "camera_id": self.clip.camera_id,
            "video_name": Path(self.clip.video_path).name,
            "state": self.state,
            "progress": round(self.progress, 3),
            "error": self.error,
            "events": self.events,
        }


def video_codec(path: Path) -> str:
    cap = cv2.VideoCapture(str(path))
    try:
        code = int(cap.get(cv2.CAP_PROP_FOURCC))
    finally:
        cap.release()
    return "".join(chr((code >> 8 * i) & 0xFF) for i in range(4)).strip("\x00 ").lower()


def browser_video(src: Path, proxy_dir: Path, clip_id: str) -> Path:
    """Файл, который проиграет браузер: исходный H.264-mp4 или прокси, перекодированный ffmpeg.

    `-fps_mode passthrough` сохраняет pts исходника: время анализа — pts кадра, и разметка совпадает с видео.
    """
    if src.suffix.lower() in BROWSER_CONTAINERS and video_codec(src) in BROWSER_CODECS:
        return src
    proxy = proxy_dir / f"{clip_id}.mp4"
    if proxy.exists() and proxy.stat().st_mtime >= src.stat().st_mtime:
        return proxy
    if not shutil.which("ffmpeg"):
        log.warning("[%s] кодек не H.264, а ffmpeg не найден: браузер может не проиграть %s", clip_id, src)
        return src
    proxy_dir.mkdir(parents=True, exist_ok=True)
    tmp = proxy.with_suffix(".tmp.mp4")
    log.info("[%s] перекодирую видео для браузера → %s", clip_id, proxy)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-map", "0:v:0", "-c:v", "libx264", "-preset",
           "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-fps_mode", "passthrough", "-movflags", "+faststart",
           str(tmp)]
    if subprocess.run(cmd).returncode == 0:
        tmp.replace(proxy)
        return proxy
    log.warning("[%s] ffmpeg не смог перекодировать видео, отдаю исходник", clip_id)
    return src


class ClipStore:
    """Очередь клипов: трекинг (с кэшем) → анализ → данные для интерфейса. Работает в одном фоновом потоке."""

    def __init__(self, clips: Iterable, out_dir: Path, pipeline_factory: Callable, use_cache: bool = True,
                 lazy: bool = False):
        """lazy — клипы ждут (idle), пока их не откроют в интерфейсе: папка с видео не обрабатывается целиком."""
        self.out_dir = Path(out_dir)
        self.use_cache = use_cache
        self.lazy = lazy
        self._factory = pipeline_factory
        self.jobs: Dict[str, ClipJob] = {c.clip_id: ClipJob(c) for c in clips}
        self._wake = threading.Condition()
        if lazy:
            for job in self.jobs.values():

                job.state = "queued" if self._cached(job) else "idle"

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self.run_lazy if self.lazy else self.run, name="ui-clips", daemon=True)
        thread.start()
        return thread

    def request(self, clip_id: str) -> None:
        """Клип открыли в интерфейсе: поставить в очередь (в lazy-режиме), если ещё не обрабатывался."""
        job = self.jobs.get(clip_id)
        if job is not None and job.state == "idle":
            with self._wake:
                job.state = "queued"
                self._wake.notify()

    def run_lazy(self) -> None:
        pipeline = None
        while True:
            with self._wake:
                job = next((j for j in self.jobs.values() if j.state == "queued"), None)
                while job is None:
                    self._wake.wait()
                    job = next((j for j in self.jobs.values() if j.state == "queued"), None)
            try:
                if pipeline is None:
                    pipeline = self._factory()
                self.process(pipeline, job)
            except Exception as e:
                log.exception("[%s] не удалось подготовить клип", job.clip.clip_id)
                job.state, job.error = "error", f"{type(e).__name__}: {e}"

    def _cached(self, job: ClipJob) -> bool:
        return self.use_cache and (self.out_dir / "cache" / f"{job.clip.clip_id}_tracks.pkl").exists()

    def run(self) -> None:
        pipeline = None

        for job in sorted(self.jobs.values(), key=lambda j: not self._cached(j)):
            try:
                if pipeline is None:
                    pipeline = self._factory()
                self.process(pipeline, job)
            except Exception as e:
                log.exception("[%s] не удалось подготовить клип", job.clip.clip_id)
                job.state, job.error = "error", f"{type(e).__name__}: {e}"

    def process(self, pipeline, job: ClipJob) -> None:
        from ..tracking import stitch_tracks

        clip = job.clip
        if clip.video_path.startswith(("rtsp://", "http://", "https://")):
            raise ValueError("повтор работает только с видеофайлами")

        def on_progress(done: int, total: Optional[int]) -> None:
            job.progress = min(done / total, 1.0) if total else 0.0

        job.state = "tracking"
        ts = pipeline.track(clip, self.out_dir / "cache" if self.use_cache else None, on_progress=on_progress)
        job.state, job.progress = "analyzing", 1.0
        analysis = pipeline.analyze(clip, ts)
        job.video = browser_video(Path(clip.video_path), self.out_dir / "ui", clip.clip_id)
        tracks = stitch_tracks(ts.tracks, pipeline.cfg.tracking)
        payload = build_payload(clip, ts.meta, tracks, analysis, pipeline.cfg, f"/video/{quote(clip.clip_id)}")
        job.payload = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        job.events = payload["stats"]["events"]
        job.state = "ready"
        log.info("[%s] готов для интерфейса: людей %d, циклов %d, событий %d", clip.clip_id,
                 payload["stats"]["tracks"], payload["stats"]["cycles"], job.events)


def parse_range(header: Optional[str], size: int) -> Optional[Tuple[int, int]]:
    """Range "bytes=a-b" → (начало, конец включительно). None — заголовка нет или он не разобран (весь файл);
    ValueError — диапазон вне файла (ответ 416)."""
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


class UiServer(ThreadingHTTPServer):
    daemon_threads = True

    allow_reuse_address = sys.platform != "win32"

    def __init__(self, address: Tuple[str, int], store: ClipStore, static_dir: Path = STATIC):
        self.store = store
        self.static_dir = static_dir.resolve()
        super().__init__(address, Handler)


class Handler(BaseHTTPRequestHandler):
    server: UiServer
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt: str, *args) -> None:
        log.debug("%s %s", self.address_string(), fmt % args)

    def do_GET(self) -> None:
        self._route(head=False)

    def do_HEAD(self) -> None:
        self._route(head=True)

    def _route(self, head: bool) -> None:
        path = unquote(urlparse(self.path).path)
        jobs = self.server.store.jobs
        try:
            if path in ("/", "/index.html"):
                self._file(self.server.static_dir / "index.html", head)
            elif path.startswith("/static/"):
                target = (self.server.static_dir / path[len("/static/"):]).resolve()
                if self.server.static_dir in target.parents and target.is_file():
                    self._file(target, head)
                else:
                    self._error(HTTPStatus.NOT_FOUND, head)
            elif path == "/api/clips":
                self._json([j.summary() for j in jobs.values()], head)
            elif path.startswith("/api/clips/"):
                job = jobs.get(path[len("/api/clips/"):])
                if job is None:
                    self._error(HTTPStatus.NOT_FOUND, head)
                elif job.payload is None:
                    self.server.store.request(job.clip.clip_id)
                    self._json(job.summary(), head, HTTPStatus.CONFLICT)
                else:
                    self._send(job.payload, "application/json; charset=utf-8", head)
            elif path.startswith("/video/"):
                job = jobs.get(path[len("/video/"):])
                if job is None or job.video is None or not job.video.is_file():
                    self._error(HTTPStatus.NOT_FOUND, head)
                else:
                    self._file(job.video, head)
            else:
                self._error(HTTPStatus.NOT_FOUND, head)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _send(self, data: bytes, ctype: str, head: bool, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if not head:
            self.wfile.write(data)

    def _json(self, obj, head: bool, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", head, status)

    def _error(self, status: HTTPStatus, head: bool) -> None:
        self._json({"error": status.phrase}, head, status)

    def _file(self, path: Path, head: bool) -> None:
        size = path.stat().st_size
        try:
            rng = parse_range(self.headers.get("Range"), size)
        except ValueError:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{size}")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        start, end = rng or (0, size - 1)
        self.send_response(HTTPStatus.PARTIAL_CONTENT if rng else HTTPStatus.OK)
        self.send_header("Content-Type", CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream"))
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
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
