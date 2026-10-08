"""Оркестрация обработки клипа: трекинг (с кэшем) → сигналы → циклы → верификатор → вспышки → пар → события → выход."""
from __future__ import annotations

import csv
import dataclasses
import hashlib
import logging
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from .config import Config
from .cycles import Cycle, detect_cycles
from .episodes import Event, build_events
from .flash import Flash, attach_flashes, find_flashes
from .geometry import TrackSignal, build_signal
from .io_video import frame_stride, probe_video
from .roi import Roi
from .tracking import PoseTracker, TrackSet, select_device, stitch_tracks
from .vapor import VaporMeter
from .verifier import ObjectVerifier

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


@dataclass
class ClipInput:
    video_path: str
    camera_id: str
    clip_id: str
    start_time: Optional[str] = None
    roi_path: Optional[str] = None


@dataclass
class Analysis:
    signals: Dict[int, TrackSignal]
    cycles: Dict[int, List[Cycle]]
    events: List[Event]
    flashes: Dict[int, List[Flash]] = field(default_factory=dict)


def project_path(path: str) -> str:
    """Относительные пути к весам и конфигам трекера считаются от корня проекта."""
    p = Path(path)
    return str(p if p.is_absolute() else ROOT / p)


def track_cache_key(video_path: str, cfg_hash: str, tracker_yaml: str) -> str:
    """Ключ кэша треков: видео (путь, размер, mtime), секции detector/video и содержимое YAML трекера.

    В конфиге трекер задан путём, поэтому правка самого bytetrack_cctv.yaml без хэша файла не сбросила бы кэш.
    """
    p = Path(video_path)
    stat = f"{p.stat().st_size}:{p.stat().st_mtime_ns}" if p.exists() else "stream"
    t = Path(tracker_yaml)
    tracker = hashlib.sha256(t.read_bytes()).hexdigest()[:16] if t.is_file() else tracker_yaml
    return hashlib.sha256(f"{p.resolve() if p.exists() else video_path}|{stat}|{cfg_hash}|{tracker}".encode()).hexdigest()[:16]


def read_inputs_csv(path: str) -> List[ClipInput]:
    """CSV: video_path,camera_id[,clip_id,start_time,roi]. Пути — относительно файла списка."""
    base = Path(path).resolve().parent
    clips = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            video = row["video_path"].strip()
            if not video.startswith(("rtsp://", "http://", "https://")) and not Path(video).is_absolute():
                video = str(base / video)
            roi = (row.get("roi") or "").strip() or None
            if roi and not Path(roi).is_absolute():
                roi = str(base / roi)
            clips.append(
                ClipInput(
                    video_path=video,
                    camera_id=row["camera_id"].strip(),
                    clip_id=(row.get("clip_id") or "").strip() or Path(video).stem,
                    start_time=(row.get("start_time") or "").strip() or None,
                    roi_path=roi,
                )
            )
    return clips


class Pipeline:
    def __init__(self, cfg: Config, progress: bool = True):
        self.cfg = cfg
        self.progress = progress  # полоса прогресса трекера (в параллельных процессах выключена)
        self.device = select_device(cfg.detector.device)
        self._tracker: Optional[PoseTracker] = None
        self.verifier: Optional[ObjectVerifier] = None
        if cfg.verifier.enabled:
            self._load_verifier(cfg)
        self.vapor = VaporMeter()

    def _load_verifier(self, cfg: Config) -> ObjectVerifier:
        """Модель верификатора грузится один раз; в tune.py сетка может включить его при выключенном в базе."""
        if self.verifier is None:
            vcfg = dataclasses.replace(
                cfg.verifier, model=project_path(cfg.verifier.model), fallback_model=project_path(cfg.verifier.fallback_model)
            )
            self.verifier = ObjectVerifier(vcfg, self.device)
        return self.verifier

    @property
    def tracker(self) -> PoseTracker:
        if self._tracker is None:
            dcfg = dataclasses.replace(
                self.cfg.detector, pose_model=project_path(self.cfg.detector.pose_model), tracker=project_path(self.cfg.detector.tracker)
            )
            self._tracker = PoseTracker(dcfg, self.device)
        return self._tracker

    def _cache_key(self, video_path: str) -> str:
        return track_cache_key(video_path, self.cfg.hash(["detector", "video"]), project_path(self.cfg.detector.tracker))

    def track(self, clip: ClipInput, cache_dir: Optional[Path] = None, on_progress=None) -> TrackSet:
        """Самая дорогая стадия; результат кэшируется (ключ — видео + секции detector/video конфига)."""
        key = self._cache_key(clip.video_path)
        cache_path = cache_dir / f"{clip.clip_id}_tracks.pkl" if cache_dir else None
        if cache_path is not None:
            cached = TrackSet.load(cache_path, key)
            if cached is not None:
                log.info("[%s] треки из кэша %s", clip.clip_id, cache_path)
                cached.extra["from_cache"] = "1"
                return cached
        meta = probe_video(clip.video_path, self.cfg.video.fallback_fps)
        stride = frame_stride(meta.fps, self.cfg.video.target_fps)
        log.info("[%s] %dx%d, %.2f fps, %.1f с, анализ каждого %d-го кадра, устройство %s",
                 clip.clip_id, meta.width, meta.height, meta.fps, meta.duration, stride, self.device)  # fmt: skip
        ts = self.tracker.run(meta, stride, progress=self.progress, on_progress=on_progress)
        ts.cache_key = key
        if cache_path is not None:
            ts.save(cache_path)
        return ts

    def analyze(self, clip: ClipInput, ts: TrackSet, cfg: Optional[Config] = None) -> Analysis:
        """Лёгкая стадия: всё, что зависит от порогов. Вызывается многократно в tune.py."""
        cfg = cfg or self.cfg
        roi = Roi.load(clip.roi_path, ts.meta.width, ts.meta.height)
        tracks = stitch_tracks(ts.tracks, cfg.tracking)
        signals = {
            tid: build_signal(tid, obs, roi, cfg.geometry, cfg.tracking, ts.sample_fps)
            for tid, obs in tracks.items()
            if len(obs) >= 3
        }
        cycles = {tid: detect_cycles(sig, cfg.cycle) for tid, sig in signals.items()}
        if cfg.verifier.enabled:
            self._load_verifier(cfg).verify(clip.clip_id, clip.video_path, cycles, signals, cfg.verifier,
                                            ts.meta.synthetic_from)  # fmt: skip
        flashes: Dict[int, List[Flash]] = {}
        if cfg.flash.enabled:  # после верификатора: вспышка отмечает только оставшиеся валидными циклы
            for tid, sig in signals.items():
                track_flashes = find_flashes(sig, cfg.flash)
                if track_flashes:
                    flashes[tid] = track_flashes
                    attach_flashes(cycles[tid], track_flashes, cfg.flash)
        if cfg.vapor.enabled:  # после верификатора: пар замеряется только у оставшихся валидными циклов
            self.vapor.measure(clip.clip_id, clip.video_path, cycles, signals, cfg.vapor, ts.meta.synthetic_from)
        events = build_events(cycles, signals, cfg.episode)
        return Analysis(signals=signals, cycles=cycles, events=events, flashes=flashes)

    def process(self, clip: ClipInput, out_dir: Path, render: bool = False, use_cache: bool = True):
        from .export import event_records, write_debug
        from .render import render_video

        t0 = time.time()
        ts = self.track(clip, out_dir / "cache" if use_cache else None)
        t1 = time.time()
        analysis = self.analyze(clip, ts)
        t2 = time.time()
        if render:
            roi = Roi.load(clip.roi_path, ts.meta.width, ts.meta.height)
            path = render_video(ts.meta, ts.stride, out_dir / "rendered" / f"{clip.clip_id}.mp4", analysis.signals,
                                analysis.cycles, analysis.events, roi, f"{clip.camera_id}/{clip.clip_id}", clip.start_time,
                                analysis.flashes)  # fmt: skip
            log.info("[%s] ролик с разметкой: %s", clip.clip_id, path)
        t3 = time.time()

        records = event_records(analysis.events, clip.camera_id, clip.clip_id)
        write_debug(out_dir, clip.clip_id, analysis.cycles, analysis.events, analysis.flashes)
        from_cache = ts.extra.get("from_cache") == "1"
        tracking_sec = ts.wall_time if from_cache else t1 - t0
        total = tracking_sec + (t3 - t1)
        n_cycles = sum(len(cs) for cs in analysis.cycles.values())
        n_valid = sum(c.valid for cs in analysis.cycles.values() for c in cs)
        n_flashes = sum(len(f) for f in analysis.flashes.values())
        log.info("[%s] треков %d, циклов %d (валидных %d), вспышек %d, событий %d", clip.clip_id, len(analysis.signals),
                 n_cycles, n_valid, n_flashes, len(records))  # fmt: skip
        log_row = {
            "camera_id": clip.camera_id,
            "clip_id": clip.clip_id,
            "video_path": clip.video_path,
            "start_time": clip.start_time or "",
            "duration_sec": round(ts.meta.duration, 3),
            "fps": round(ts.meta.fps, 3),
            "width": ts.meta.width,
            "height": ts.meta.height,
            "processed_frames": ts.processed_frames,
            "tracking_sec": round(tracking_sec, 2),
            "analysis_sec": round(t2 - t1, 2),
            "render_sec": round(t3 - t2, 2),
            "total_sec": round(total, 2),
            "realtime_factor": round(ts.meta.duration / total, 2) if total > 0 else "",
            "events": len(records),
            "device": self.device,
            "from_cache": int(from_cache),
        }
        return records, log_row, analysis

    def info(self) -> dict:
        """Устройство и фактические веса верификатора — для manifest.json."""
        return {"device": self.device, "verifier": self.verifier.describe() if self.verifier else None}


_worker: Optional[Pipeline] = None  # пайплайн процесса-обработчика (process_clips с workers > 1)


def _init_worker(cfg: Config, log_level: int) -> None:
    global _worker
    logging.basicConfig(level=log_level, format=LOG_FORMAT, datefmt="%H:%M:%S")
    _worker = Pipeline(cfg, progress=False)


def _process_in_worker(i: int, clip: ClipInput, out_dir: Path, render: bool, use_cache: bool):
    records, log_row, _ = _worker.process(clip, out_dir, render=render, use_cache=use_cache)
    return i, records, log_row, _worker.info()


def process_clips(
    cfg: Config, clips: List[ClipInput], out_dir: Path, render: bool = False, use_cache: bool = True, workers: int = 1
) -> Iterator[Tuple[int, list, dict, dict]]:
    """(номер клипа в clips, записи событий, строка журнала, info() пайплайна) по мере готовности клипов.

    workers > 1 — клипы обрабатываются параллельно в отдельных процессах, у каждого свои модели: пока один
    процесс декодирует кадры и считает на CPU, GPU занят другим. Процессы запускаются через spawn (CUDA
    не переносит fork); клипы раздаются по одному, поэтому длинный клип не задерживает остальные."""
    if workers <= 1 or len(clips) <= 1:
        pipeline = Pipeline(cfg)
        for i, clip in enumerate(clips):
            records, log_row, _ = pipeline.process(clip, out_dir, render=render, use_cache=use_cache)
            yield i, records, log_row, pipeline.info()
        return
    workers = min(workers, len(clips))
    log.info("Параллельная обработка: %d процесса(ов) на %d клипов", workers, len(clips))
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(workers, mp_context=ctx, initializer=_init_worker,
                             initargs=(cfg, logging.getLogger().getEffectiveLevel())) as pool:  # fmt: skip
        futures = [pool.submit(_process_in_worker, i, clip, out_dir, render, use_cache) for i, clip in enumerate(clips)]
        try:
            for future in as_completed(futures):
                yield future.result()
        finally:  # ошибка клипа или прерывание: ещё не начатые клипы не запускаются
            for future in futures:
                future.cancel()
