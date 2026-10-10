"""Обработка клипов интерфейса потоковым режимом — тем же путём, что run_rt.py и serve.py (realtime.py, с VLM-арбитром
при vlm.enabled).

Данные интерфейса публикуются по ходу потока: пока клип обрабатывается, интерфейс показывает то, что поток уже знает
(люди, зафиксированные циклы, тревоги, запросы и ответы VLM), и видео можно смотреть сразу. Отсчёты кэшируются в
<out>/cache/<clip>_rt.pkl, как у run_rt.py: повтор из кэша даёт то же, что поток.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from urllib.parse import quote

from ..config import Config
from .payload import rt_payload

log = logging.getLogger(__name__)

PUBLISH_SEC = 1.0  # с: как часто пересобирать данные по ходу потока (и не чаще, чем 5× время сборки)


def encode(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


class StreamWorker:
    """Модели потока (RealtimeDetector) на весь процесс; process_job — один клип очереди ClipStore."""

    def __init__(self, cfg: Config):
        from ..realtime import RealtimeDetector

        self.cfg = cfg
        self.det = RealtimeDetector(cfg)

    def process_job(self, store, job) -> None:
        from ..realtime import analyze, load_run, person_cycles, person_signals, rt_cache_key, rt_events, save_run
        from ..roi import Roi
        from ..vlm import apply_veto
        from .server import browser_video

        clip, cfg, det = job.clip, self.cfg, self.det
        if clip.video_path.startswith(("rtsp://", "http://", "https://")):
            raise ValueError("интерфейс работает только с видеофайлами")
        url = f"/video/{quote(clip.clip_id)}"
        job.video = browser_video(Path(clip.video_path), store.out_dir / "ui", clip.clip_id)  # смотреть можно сразу
        key = rt_cache_key(clip.video_path, cfg)
        cache = store.out_dir / "cache" / f"{clip.clip_id}_rt.pkl" if store.use_cache else None
        run = load_run(cache, key) if cache else None
        jobs = None
        if run is None:
            live: dict = {"next": 0.0}

            def publish(t: float) -> None:
                t0 = time.perf_counter()
                ses = live["ses"]
                cycles = person_cycles(ses.jobs)
                signals = person_signals(ses.jobs, cfg, ses.roi)
                cycles = {tid: cs for tid, cs in cycles.items() if tid in signals}
                events = rt_events(cycles, signals, cfg)
                apply_veto(events, ses.alerts)
                data = rt_payload(clip, ses.meta, [j.record() for j in ses.jobs], ses.sparse, cycles, events, ses.alerts,
                                  cfg, url, processed=t)  # fmt: skip
                job.payload, job.events = encode(data), data["stats"]["events"]
                spent = time.perf_counter() - t0
                live["next"] = time.perf_counter() + max(PUBLISH_SEC, 5 * spent)

            def on_start(ses) -> None:
                live["ses"] = ses
                job.duration = ses.meta.duration
                publish(0.0)
                job.state = "streaming"

            def on_frame(f) -> None:
                job.processed = f.t
                job.progress = min(f.t / job.duration, 1.0) if job.duration else 0.0
                if time.perf_counter() >= live["next"]:
                    publish(f.t)

            run, jobs = det.run(clip.video_path, cfg, clip.roi_path, on_frame=on_frame, on_start=on_start)
            run.cache_key = key
            if cache:
                save_run(run, cache)
        else:
            log.info("[%s] отсчёты потока из кэша %s", clip.clip_id, cache)
        job.state = "analyzing"
        roi = Roi.load(clip.roi_path, run.meta.width, run.meta.height)
        cycles, _, events = analyze(run, cfg, roi, jobs, det.appearance, clip.clip_id, clip.video_path)
        data = rt_payload(clip, run.meta, run.jobs, run.sparse, cycles, events, run.alerts, cfg, url)
        job.payload, job.events = encode(data), data["stats"]["events"]
        job.duration = job.processed = run.meta.duration
        job.progress = 1.0
        job.state = "ready"
        dur = run.meta.duration
        log.info("[%s] поток: %.1f с видео за %.1f с (загрузка %.2f), тревог %d, ответов VLM %d, событий %d",
                 clip.clip_id, dur, run.wall_time, run.wall_time / dur if dur else 0.0, data["stats"]["alerts"],
                 sum(1 for a in run.alerts if "vlm" in a), job.events)  # fmt: skip
