"""Проверка загруженных видео: очередь заданий «В очереди → Анализируется → Готово | Ошибка».

Анализатор выбирается при запуске:
- `pipeline` — настоящий пайплайн (`Pipeline.track` + `analyze`), эпизоды — события с `passed`;
- `realtime` — потоковый режим (`run_rt.py`, `smokedet/realtime.py`): кадр 1 Гц, поза на кропах, плотная проверка
  у людей с рукой у рта, правило `realtime.rule_overrides`; эпизоды — события с `passed`;
- `demo` — для показа интерфейса без моделей: эпизоды генерируются по имени и длительности файла и помечены
  в журнале как демо (`demo=1`). Сам по себе демо-режим не включается.

Фрагмент эпизода — запись с запасом `PRE_ROLL`/`POST_ROLL`. Если есть ffmpeg, он вырезается в отдельный mp4
(H.264, играет любой браузер); иначе фрагмент — окно в исходном файле (`clip_kind='range'`), который тогда
хранится столько же, сколько фрагменты.

Имеющиеся записи (`monitor.py --enqueue`) ставятся в очередь без загрузки через браузер: в папку загрузок кладётся
жёсткая ссылка на файл (или копия, если ссылку сделать нельзя), поэтому очистка по сроку не трогает исходник.
"""
from __future__ import annotations

import csv
import hashlib
import logging
import os
import queue
import random
import secrets
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

from .cameras import POST_ROLL, PRE_ROLL
from .journal import Journal

log = logging.getLogger(__name__)

VIDEO_EXT = {".mp4", ".m4v", ".mov", ".mkv", ".avi", ".webm", ".ts", ".mpg", ".mpeg", ".wmv"}
MAX_UPLOAD = 8 << 30  # 8 ГБ


class AnalyzerUnavailable(RuntimeError):
    pass


class Episode:
    def __init__(self, start: float, end: float, cycles: int, confidence: float, track: Optional[int] = None):
        self.start, self.end, self.cycles, self.confidence, self.track = start, end, cycles, confidence, track


class PipelineAnalyzer:
    """Настоящий анализ. Модели грузятся при первом задании; без torch/ultralytics/cv2 — понятная ошибка."""

    name = "pipeline"

    def __init__(self, config: str, overrides: List[str], out_dir: Path):
        self.config, self.overrides, self.out_dir = config, overrides, Path(out_dir)
        self._pipeline = None

    def available(self) -> str:
        """Пустая строка — можно анализировать, иначе причина."""
        missing = [m for m in ("torch", "ultralytics", "cv2") if not _importable(m)]
        return f"не установлены {', '.join(missing)}: анализ видео недоступен" if missing else ""

    def run(self, path: Path, job_id: str, on_progress: Callable[[float], None]) -> tuple:
        reason = self.available()
        if reason:
            raise AnalyzerUnavailable(reason)
        from ..config import load_config
        from ..pipeline import ClipInput, Pipeline

        if self._pipeline is None:
            self._pipeline = Pipeline(load_config(self.config, self.overrides), progress=False)
        clip = ClipInput(video_path=str(path), camera_id="upload", clip_id=job_id)

        def progress(done: int, total: Optional[int]) -> None:
            on_progress(0.95 * min(done / total, 1.0) if total else 0.0)

        ts = self._pipeline.track(clip, self.out_dir / "cache", on_progress=progress)
        analysis = self._pipeline.analyze(clip, ts)
        return _episodes(analysis.events), ts.meta.duration or None


class RealtimeAnalyzer(PipelineAnalyzer):
    """Потоковый режим, как `run_rt.py`, но без кэша отсчётов: у каждого задания свой файл. Модели грузятся
    при первом задании; конфиг — `load_rt_config` (YAML → realtime.rule_overrides → --set)."""

    name = "realtime"

    def __init__(self, config: str, overrides: List[str], out_dir: Path):
        super().__init__(config, overrides, out_dir)
        self._cfg = None
        self._detector = None

    def run(self, path: Path, job_id: str, on_progress: Callable[[float], None]) -> tuple:
        reason = self.available()
        if reason:
            raise AnalyzerUnavailable(reason)
        from ..io_video import probe_video
        from ..realtime import RealtimeDetector, analyze, load_rt_config
        from ..roi import Roi

        if self._detector is None:
            self._cfg = load_rt_config(self.config, self.overrides)
            self._detector = RealtimeDetector(self._cfg)
        cfg, det = self._cfg, self._detector
        total = probe_video(str(path), cfg.video.fallback_fps).duration
        shown = [0.0]

        def on_frame(f) -> None:  # кадры идут по ~5 в секунду видео: журнал обновляем не чаще 2 раз в секунду
            now = time.monotonic()
            if total and now - shown[0] >= 0.5:
                shown[0] = now
                on_progress(0.95 * min(f.t / total, 1.0))

        run, jobs = det.run(str(path), cfg, on_frame=on_frame)
        roi = Roi.load(None, run.meta.width, run.meta.height)
        _, _, events = analyze(run, cfg, roi, jobs, det.appearance, job_id, str(path))
        dur = run.meta.duration
        log.info("[%s] потоковый режим: %.1f с видео за %.1f с (загрузка %.2f), сообщений о курении %d", job_id, dur,
                 run.wall_time, run.wall_time / dur if dur else 0.0, len(run.alerts))  # fmt: skip
        return _episodes(events), dur or None


def _episodes(events) -> List[Episode]:
    return [
        Episode(e.start, e.end, e.to_dict()["n_movements"], float(e.confidence), e.track_id)
        for e in events if e.passed
    ]  # fmt: skip


class DemoAnalyzer:
    """Эпизоды без моделей — только для показа интерфейса. Воспроизводимы для одного и того же файла."""

    name = "demo"

    def __init__(self, min_work: float = 3.0, max_work: float = 20.0):
        self.min_work, self.max_work = min_work, max_work  # сколько секунд «идёт анализ»

    def available(self) -> str:
        return ""

    def run(self, path: Path, job_id: str, on_progress: Callable[[float], None], duration: Optional[float] = None):
        if not duration:
            raise AnalyzerUnavailable("длительность записи неизвестна: браузер не смог прочитать файл")
        seed = hashlib.sha1(f"{path.name}:{path.stat().st_size}".encode()).hexdigest()
        r = random.Random(int(seed[:8], 16))
        work = max(self.min_work, min(self.max_work, duration / 15))
        steps = max(1, int(work * 4))
        for i in range(steps):
            time.sleep(work / steps)
            on_progress((i + 1) / steps)
        episodes = []
        t = 5.0 + r.random() * 20
        while t + 12 < duration and len(episodes) < 6:
            if r.random() < 0.55:
                n = 2 + int(r.random() * 5)
                length = 6 + n * (6 + r.random() * 4)
                end = min(duration - 1, t + length)
                episodes.append(Episode(t, end, n, round(min(0.95, 0.55 + 0.06 * n + r.random() * 0.08), 3)))
                t = end + 30 + r.random() * 60
            else:
                t += 40 + r.random() * 80
        return episodes, duration


def new_job_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)


def video_sources(paths: Iterable[str]) -> List[Tuple[Path, str]]:
    """Имеющиеся записи для `--enqueue`: (файл, имя в журнале). Принимает видеофайлы, папки (видео в самой папке,
    имя — «папка/файл», чтобы 1.mp4 из true и false различались) и CSV со списком клипов, как у run.py
    (`video_path[,camera_id,clip_id,...]`, пути — от CSV, имя — clip_id). Нет файла — ValueError."""
    out: List[Tuple[Path, str]] = []
    for arg in paths:
        p = Path(arg)
        if p.is_dir():
            out += [(f, f"{p.name}/{f.name}") for f in sorted(p.iterdir()) if f.suffix.lower() in VIDEO_EXT]
        elif p.suffix.lower() == ".csv":
            with open(p, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    video = p.resolve().parent / row["video_path"].strip()
                    clip_id = (row.get("clip_id") or "").strip()
                    out.append((video, f"{clip_id}{video.suffix}" if clip_id else video.name))
        else:
            out.append((p, p.name))
    for f, _ in out:
        if not f.is_file():
            raise ValueError(f"нет файла {f}")
        if f.suffix.lower() not in VIDEO_EXT:
            raise ValueError(f"не видео: {f} (нужно {', '.join(sorted(VIDEO_EXT))})")
    return out


def _importable(module: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(module) is not None


class AnalysisQueue:
    """Один фоновый поток: задания обрабатываются по очереди (на GPU параллельно не выгоднее, см. CLAUDE.md)."""

    def __init__(self, journal: Journal, analyzer, uploads_dir: Path, clips_dir: Path):
        self.journal = journal
        self.analyzer = analyzer
        self.uploads_dir = Path(uploads_dir)
        self.clips_dir = Path(clips_dir)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self._queue: "queue.Queue[str]" = queue.Queue()
        self.ffmpeg = shutil.which("ffmpeg")

    def start(self) -> threading.Thread:
        thread = threading.Thread(target=self._loop, name="monitor-analysis", daemon=True)
        thread.start()
        return thread

    def submit(self, job_id: str) -> None:
        self._queue.put(job_id)

    def enqueue_file(self, src: Path, name: str) -> str:
        """Ставит имеющуюся запись в очередь. Файл задания — жёсткая ссылка на исходник (или копия): по истечении
        срока `Journal.purge` удаляет файл задания, а исходная запись должна остаться."""
        job_id = new_job_id()
        target = self.uploads_dir / f"{job_id}{src.suffix.lower()}"
        try:
            os.link(src, target)
        except OSError:
            shutil.copy2(src, target)
        self.journal.add_job(job_id, name, str(target), src.stat().st_size, None, self.analyzer.name)
        self.submit(job_id)
        return job_id

    def _loop(self) -> None:
        while True:
            job_id = self._queue.get()
            try:
                self.process(job_id)
            except Exception as e:  # ошибка одного файла не останавливает очередь
                if not isinstance(e, AnalyzerUnavailable):
                    log.exception("[%s] анализ не удался", job_id)
                self.journal.update_job(job_id, state="error", error=_short(e), finished_at=self.journal.clock(),
                                        expires_at=self.journal.clock() + self.journal.retention)  # fmt: skip

    def process(self, job_id: str) -> None:
        job = self.journal.job(job_id)
        if job is None:
            return
        path = Path(job["path"])
        self.journal.update_job(job_id, state="analyzing", progress=0.0, started_at=self.journal.clock())
        progress = lambda k: self.journal.set_progress(job_id, round(k, 3))  # noqa: E731
        if isinstance(self.analyzer, DemoAnalyzer):
            episodes, duration = self.analyzer.run(path, job_id, progress, job["duration"])
        else:
            episodes, duration = self.analyzer.run(path, job_id, progress)
        now = self.journal.clock()
        for ep in episodes:
            event_id = self.journal.add_event(
                source_type="file", source_id=job_id, source_name=job["filename"], track=ep.track,
                started_at=job["created_at"], offset_sec=round(ep.start, 2), duration=round(ep.end - ep.start, 2),
                cycles=ep.cycles, confidence=ep.confidence, demo=int(self.analyzer.name == "demo"),
            )  # fmt: skip
            a = max(0.0, ep.start - PRE_ROLL)
            b = ep.end + POST_ROLL if duration is None else min(duration, ep.end + POST_ROLL)
            cut = self._cut(path, event_id, a, b) if self.ffmpeg else None
            if cut:
                self.journal.attach_clip(event_id, "video", str(cut), a, b, now)
            else:
                self.journal.attach_clip(event_id, "range", None, a, b, now)
        self.journal.update_job(job_id, state="done", progress=1.0, duration=duration, finished_at=now,
                                expires_at=now + self.journal.retention)  # fmt: skip
        log.info("[%s] готово: эпизодов %d (%s)", job_id, len(episodes), self.analyzer.name)

    def _cut(self, src: Path, event_id: int, a: float, b: float) -> Optional[Path]:
        out = self.clips_dir / f"file_{event_id}.mp4"
        cmd = [self.ffmpeg, "-y", "-loglevel", "error", "-ss", f"{a:.2f}", "-i", str(src), "-t", f"{b - a:.2f}",
               "-map", "0:v:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(out)]  # fmt: skip
        if subprocess.run(cmd).returncode == 0 and out.exists():
            return out
        log.warning("ffmpeg не вырезал фрагмент эпизода %d, остаётся окно в исходном файле", event_id)
        return None


def _short(e: Exception) -> str:
    text = str(e) or type(e).__name__
    return text if isinstance(e, AnalyzerUnavailable) else f"{type(e).__name__}: {text}"[:300]
