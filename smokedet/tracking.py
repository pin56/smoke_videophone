"""Детекция людей, ключевые точки (COCO-17) и трекинг: YOLO-pose + ByteTrack.

Результат — TrackSet: «сырые» наблюдения по каждому track_id. Он не зависит от ROI и порогов
анализа, поэтому кэшируется на диск и переиспользуется при подборе параметров (tune.py).
"""
from __future__ import annotations

import logging
import pickle
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np

from .config import DetectorConfig, TrackingConfig
from .io_video import FrameClock, VideoMeta, iter_sampled_frames

log = logging.getLogger(__name__)

CACHE_VERSION = 3  # 2: замер света для вспышек (flash.LightMeter); 3: время кадра по pts (io_video.FrameClock)
LIGHT_QUEUE = 2  # кадров ждут замера света в потоке (см. PoseTracker.run)


@dataclass
class PersonObs:
    frame_idx: int
    t: float
    box: np.ndarray  # (4,) xyxy в пикселях исходного кадра
    score: float
    kpts: np.ndarray  # (17, 3): x, y, conf
    light: Optional[np.ndarray] = None  # (3,): пересвет у рта/кистей и вокруг (px), scale; см. flash.py


@dataclass
class TrackSet:
    meta: VideoMeta
    stride: int
    tracks: Dict[int, List[PersonObs]]
    processed_frames: int = 0
    wall_time: float = 0.0
    cache_key: str = ""
    extra: Dict[str, str] = field(default_factory=dict)
    # с по частям трекинга: ожидание кадра (декодирование не успело), поза (пред-, инференс, постобработка),
    # ByteTrack, уточнение мелких, замер света; в кэшах до их появления поля нет — getattr(ts, "timings", {})
    timings: Dict[str, float] = field(default_factory=dict)

    @property
    def sample_fps(self) -> float:
        return self.meta.fps / self.stride

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"version": CACHE_VERSION, "trackset": self}, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path, cache_key: str) -> Optional["TrackSet"]:
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:  # повреждённый кэш — просто пересчитываем
            log.warning("Кэш треков %s не читается (%s), пересчёт", path, e)
            return None
        ts = data.get("trackset")
        if data.get("version") != CACHE_VERSION or ts is None or ts.cache_key != cache_key:
            return None
        return ts


def select_device(pref: str) -> str:
    if pref and pref != "auto":
        return pref
    import torch

    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def refine_model_path(pose_model: str) -> str:
    """Модель уточнения позы на кропах. Уточнение идёт батчем кропов, а TensorRT-engine позы собран на батч 1
    (tools/export_trt.py), поэтому для «<имя>-<суффикс>.engine» берётся «<имя>.pt» рядом с ним."""
    p = Path(pose_model)
    if p.suffix != ".engine":
        return pose_model
    base = p.stem.split("-fp16")[0].split("-int8")[0]
    return str(p.with_name(base + ".pt"))


def half_kwargs(enabled: bool, device: str) -> dict:
    """Аргументы FP16 для predict/track ultralytics: только на CUDA (на cpu и mps не используется).
    В ultralytics 8.4 точность задаёт quantize=16: half (и явный half=False) устарел и даёт предупреждение
    на каждом кадре. FP32 — аргумент не передаётся."""
    return {"quantize": 16} if enabled and str(device) not in ("cpu", "mps") else {}


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU набора рамок a (N,4) с одной рамкой b (4,)."""
    a = np.atleast_2d(a)
    x1 = np.maximum(a[:, 0], b[0])
    y1 = np.maximum(a[:, 1], b[1])
    x2 = np.minimum(a[:, 2], b[2])
    y2 = np.minimum(a[:, 3], b[3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / np.maximum(area_a + area_b - inter, 1e-9)


class PoseTracker:
    def __init__(self, cfg: DetectorConfig, device: str):
        from ultralytics import YOLO

        self.cfg = cfg
        self.device = device
        self.model = YOLO(cfg.pose_model)
        if self.model.task != "pose":
            raise ValueError(f"{cfg.pose_model}: нужна pose-модель, получена задача '{self.model.task}'")
        # отдельный экземпляр: predict() на основной модели запустил бы колбэки трекера на кропах
        self.refiner = YOLO(refine_model_path(cfg.pose_model)) if cfg.refine_small else None
        self._half = half_kwargs(cfg.half, device)

    def _track_frame(self, frame: np.ndarray, first: bool):
        cfg = self.cfg
        return self.model.track(
            frame,
            persist=not first,  # первый кадр клипа пересоздаёт трекер
            tracker=cfg.tracker,
            imgsz=cfg.imgsz,
            conf=cfg.conf,
            iou=cfg.iou,
            device=self.device,
            classes=[0],
            verbose=False,
            **self._half,
        )[0]

    def _refine(self, frame: np.ndarray, boxes: np.ndarray, kpts: np.ndarray) -> None:
        """Для мелких людей повторно оценивает позу на увеличенном кропе (top-down)."""
        cfg = self.cfg
        h_img, w_img = frame.shape[:2]
        heights = boxes[:, 3] - boxes[:, 1]
        small = np.where(heights < cfg.refine_max_height)[0]
        if len(small) == 0:
            return
        crops, offsets = [], []
        for i in small:
            x1, y1, x2, y2 = boxes[i]
            pw, ph = cfg.refine_pad * (x2 - x1), cfg.refine_pad * (y2 - y1)
            cx1, cy1 = int(max(0, x1 - pw)), int(max(0, y1 - ph))
            cx2, cy2 = int(min(w_img, x2 + pw)), int(min(h_img, y2 + ph))
            crops.append(frame[cy1:cy2, cx1:cx2])
            offsets.append((cx1, cy1))
        results = self.refiner.predict(
            crops, imgsz=cfg.refine_imgsz, conf=0.1, device=self.device, verbose=False, **self._half
        )
        for i, res, (ox, oy) in zip(small, results, offsets):
            if res.keypoints is None or len(res.boxes) == 0:
                continue
            rb = res.boxes.xyxy.cpu().numpy() + np.array([ox, oy, ox, oy])
            ious = box_iou(rb, boxes[i])
            j = int(np.argmax(ious))
            if ious[j] < 0.5:
                continue
            rk = res.keypoints.data[j].cpu().numpy().astype(np.float32).copy()
            rk[:, 0] += ox
            rk[:, 1] += oy
            # берём уточнённую позу целиком, если верх тела (лицо, плечи, руки) увереннее
            if rk[:11, 2].mean() > kpts[i, :11, 2].mean():
                kpts[i] = rk

    def run(
        self, meta: VideoMeta, stride: int, progress: bool = True, on_progress: Optional[Callable[[int, Optional[int]], None]] = None
    ) -> TrackSet:
        """on_progress(обработано кадров, всего или None) вызывается после каждого кадра (веб-интерфейс)."""
        from .flash import LightMeter  # flash → geometry → tracking: импорт здесь, иначе цикл

        meter = LightMeter(self.device)
        tracks: Dict[int, List[PersonObs]] = {}
        n_processed = 0
        start = time.time()
        total = meta.frame_count // stride if meta.frame_count else None
        clock = FrameClock(meta.fps)
        iterator = iter_sampled_frames(meta.path, stride, meta.fps, clock)
        if progress:
            try:
                from tqdm import tqdm

                iterator = tqdm(iterator, total=total, desc=Path(meta.path).name, unit="fr", leave=False)
            except ImportError:
                pass
        last_idx = -1
        tm =dict.fromkeys(("decode_wait", "pose", "tracker", "refine", "light"), 0.0)
        iterator = iter(iterator)
        # Замер света (CPU) идёт в своём потоке, пока GPU считает позу следующего кадра. Поток один:
        # кадры замеряются по порядку, как раньше (фон зависит от порядка); light дописывается в наблюдения,
        # когда замер готов. В tm["light"] — время самого замера, а не ожидания.
        light_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="light")
        pending: Deque[Tuple[Future, List[PersonObs]]] = deque()

        def measure(frame, t, boxes, kpts):
            a = time.perf_counter()
            return meter.measure(frame, t, boxes, kpts), time.perf_counter() - a

        def settle(limit: int) -> None:  # в очереди не больше limit кадров: кадр 2560x1440 — 11 МБ
            while len(pending) > limit:
                fut, obs_list = pending.popleft()
                lights, dt = fut.result()
                tm["light"] += dt
                for o, light in zip(obs_list, lights):
                    o.light = light

        try:
            while True:
                t0 = time.perf_counter()
                item = next(iterator, None)
                t1 = time.perf_counter()
                tm["decode_wait"] += t1 - t0
                if item is None:
                    break
                frame_idx, t, frame = item
                res = self._track_frame(frame, first=n_processed == 0)
                t2 = time.perf_counter()
                pose = sum(res.speed.get(k) or 0.0 for k in ("preprocess", "inference", "postprocess")) / 1000.0
                tm["pose"] += min(pose, t2 - t1)
                tm["tracker"] += max(0.0, t2 - t1 - pose)  # остаток вызова track(): обновление ByteTrack
                n_processed += 1
                last_idx = frame_idx
                if on_progress is not None:
                    on_progress(n_processed, total)
                if res.boxes is None or res.boxes.id is None or len(res.boxes) == 0:
                    pending.append((light_pool.submit(measure, frame, t, [], []), []))  # фон обновляется и без людей
                    settle(LIGHT_QUEUE)
                    continue
                boxes = res.boxes.xyxy.cpu().numpy().astype(np.float32)
                ids = res.boxes.id.int().cpu().numpy()
                scores = res.boxes.conf.cpu().numpy()
                if res.keypoints is not None:
                    kpts = res.keypoints.data.cpu().numpy().astype(np.float32).copy()
                else:
                    kpts = np.zeros((len(boxes), 17, 3), np.float32)
                t3 = time.perf_counter()
                if self.refiner is not None:
                    self._refine(frame, boxes, kpts)
                tm["refine"] += time.perf_counter() - t3
                obs_list = [PersonObs(frame_idx, t, box, float(score), kp) for box, score, kp in zip(boxes, scores, kpts)]
                for tid, o in zip(ids, obs_list):
                    tracks.setdefault(int(tid), []).append(o)
                pending.append((light_pool.submit(measure, frame, t, boxes, kpts), obs_list))
                settle(LIGHT_QUEUE)
            settle(0)
        finally:
            light_pool.shutdown(wait=True, cancel_futures=True)
        if last_idx >= 0 and last_idx + 1 > meta.frame_count:
            meta.frame_count = last_idx + 1  # поток или неверный FRAME_COUNT в контейнере
        meta.synthetic_from = clock.synthetic_from  # FrameReader не ищет такие кадры по pts
        return TrackSet(meta=meta, stride=stride, tracks=tracks, processed_frames=n_processed, wall_time=time.time() - start,
                        timings=tm)  # fmt: skip


def stitch_tracks(tracks: Dict[int, List[PersonObs]], cfg: TrackingConfig) -> Dict[int, List[PersonObs]]:
    """Склеивает треки, оборванные трекером (перекрытие, пропуск детекции).

    Трек B продолжает трек A, если B начинается не позже чем через stitch_max_gap после конца A
    и рядом с ним (расстояние центров <= stitch_max_dist высот bbox). Итоговый id — id первого трека.
    """
    if not tracks:
        return {}
    tracks = {tid: sorted(obs, key=lambda o: o.t) for tid, obs in tracks.items() if obs}
    info = [(obs[0].t, obs[-1].t, tid) for tid, obs in tracks.items()]
    info.sort()
    tail_of: Dict[int, int] = {}  # id цепочки -> id последнего трека в ней
    chain_of: Dict[int, int] = {}  # id трека -> id цепочки
    end_time = {tid: e for _, e, tid in info}
    for start, _, tid in info:
        first = tracks[tid][0]
        best, best_cost = None, None
        for chain, tail in tail_of.items():
            gap = start - end_time[tail]
            if gap <= 0 or gap > cfg.stitch_max_gap:
                continue
            last = tracks[tail][-1]
            h = max(last.box[3] - last.box[1], first.box[3] - first.box[1], 1.0)
            c_last = (last.box[:2] + last.box[2:]) / 2
            c_first = (first.box[:2] + first.box[2:]) / 2
            dist = float(np.linalg.norm(c_last - c_first)) / h
            if dist > cfg.stitch_max_dist:
                continue
            cost = dist + gap / cfg.stitch_max_gap
            if best_cost is None or cost < best_cost:
                best, best_cost = chain, cost
        if best is None:
            chain_of[tid] = tid
            tail_of[tid] = tid
        else:
            chain_of[tid] = best
            tail_of[best] = tid
    merged: Dict[int, List[PersonObs]] = {}
    for tid, obs in tracks.items():
        merged.setdefault(chain_of[tid], []).extend(obs)
    for obs in merged.values():
        obs.sort(key=lambda o: o.t)
    return merged
