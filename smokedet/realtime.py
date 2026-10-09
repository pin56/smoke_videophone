"""Потоковый (реалтайм) режим: кадры обрабатываются по мере поступления, без заглядывания в будущее.

Стадии (параметры — секция realtime конфига):
1. Раз в секунду (по pts кадра, io_video.FrameClock) лёгкий детектор людей (yolo11n, COCO класс 0) на всём
   кадре; простой трекер 1 Гц (жадно по IoU и расстоянию центров: курящие почти стоят) даёт id человека.
2. Сильная поза (yolo11m-pose) на кропах людей батчем за кадр; по 17 точкам — нормированное расстояние
   кисть→рот d (функции geometry.py, как у основного пайплайна).
3. Если d в отсчёте 1 Гц ниже trigger_dist (рука у рта или поднята), область человека проверяется плотно:
   кадры ~dense_fps назад из кольцевого буфера (dense_before с) и вперёд, пока срабатывает триггер, плюс
   dense_after с. На каждом шаге 1 Гц по последним отсчётам ищутся законченные циклы «подъём → пауза → отвод»
   (cycles.detect_cycles); цикл фиксируется, когда после его конца прошло settle с. Плотно проверяется не больше
   max_jobs людей сразу.
4. Опционально (realtime.objects): CLIP по кропам «рот + кисть» в паузе (appearance.py) — кропы берутся из
   памяти задания, видео не перечитывается. Сигарета (look) — прямой признак, питьё и касание лица — отказ.
Правило эпизода и объединение циклов — episodes.build_events, как у основного пайплайна.

Дорогие стадии 1–3 кэшируются (отсчёты поз заданий и моменты шагов 1 Гц), поэтому пороги циклов и правила
перебираются повтором из кэша без инференса: повтор проходит те же шаги фиксации циклов, что и поток.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import pickle
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Callable, Deque, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from .appearance import (AppearanceScorer, apply_score, crop_geometry, cycle_score, plan_frames, rescuable, square_crop,
                         view_plan)
from .config import Config, apply_overrides, config_from_dict, load_config
from .cycles import Cycle, detect_cycles
from .episodes import Event, build_events, has_evidence
from .geometry import TrackSignal, arm_plausible, build_signal, estimate_mouth, nearest_hand, raw_scale
from .io_video import FrameClock, VideoMeta, _open, prefetched, probe_video
from .ov_yolo import OvYolo, is_openvino_dir
from .roi import Roi, anchor_point
from .tracking import PersonObs, box_iou, half_kwargs, select_device
from .weights import ensure_for

log = logging.getLogger(__name__)

CACHE_VERSION = 1
# поля секции realtime, от которых не зависят отсчёты стадий 1–3 (не входят в ключ кэша, меняются повтором)
RULE_FIELDS = ("objects", "objects_per_event", "many_cycles", "many_min_score", "many_median", "rule_overrides",
               "finalize_window", "start_hold", "threads", "skip_confirmed")  # fmt: skip
# поля, добавленные после первых прогонов: в ключ кэша входят, только если отличаются от этих значений (старые кэши
# остаются действительны)
HASH_DEFAULTS = {"alert_hold": 0.0, "ov_precision": ""}
DUP_IOU = 0.8  # два задания плотной проверки с такими рамками — один человек (рамки позы сошлись на нём)
EPS = 1e-6
PREFETCH = 8  # кадров плотной сетки, декодированных заранее (поток чтения работает, пока занят GPU)
POSE_IOU = 0.7  # NMS позы: iou по умолчанию predict() ultralytics
POSE_BATCH = 24  # кропов в одном вызове позы (все размеры батча прогреваются при загрузке)
SETTLE_EXTRA = 1.0  # с сверх cycle.lookaround: после конца цикла столько ждать, прежде чем его зафиксировать
CROP_SIDE = 1.2  # стадия 4: в памяти хранится квадрат со стороной CROP_SIDE роста вокруг «рот + кисть»...
CROP_MAX = 576  # ...не больше CROP_MAX px (широкий кроп CLIP 0.7 роста — до 336 px, как вход модели)
CROP_KEEP_EXTRA = 3.0  # с сверх паузы и фаз подхода/отвода: столько хранятся кропы отсчётов
MOTION_SAMPLES = 12  # скорость рамки — по стольким последним отсчётам задания (в пределах realtime.motion_window)...
MOTION_MIN_SPAN = 0.3  # ...охватывающим не меньше стольких с
MOTION_MAX = 1.5  # с: дальше последнего найденного отсчёта рамка не сдвигается (человека потеряли надолго)


# ---------------------------------------------------------------- конфиг и кадры


def load_rt_config(path: Optional[str], overrides: Optional[List[str]] = None) -> Config:
    """Конфиг режима: YAML, затем realtime.rule_overrides (пороги общих секций для этого режима), затем --set."""
    overrides = list(overrides or [])
    base = load_config(path, overrides)
    data = {}
    if path:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    return config_from_dict(apply_overrides(data, list(base.realtime.rule_overrides) + overrides))


class Ticker:
    """Отбор кадров по времени: срабатывает на первом кадре каждого интервала сетки period (от 0 с). Время —
    pts, поэтому при переменном fps частота выдерживается по секундам, а не по номерам кадров; после пропуска
    кадров (поток) лишних срабатываний подряд нет."""

    def __init__(self, period: float):
        self.period = period
        self.next = 0.0

    def __call__(self, t: float) -> bool:
        if t + EPS < self.next:
            return False
        self.next = (math.floor((t + EPS) / self.period) + 1) * self.period
        return True


@dataclass
class StreamFrame:
    idx: int  # последовательный номер кадра (как у трекера основного пайплайна)
    t: float  # время кадра, с (pts)
    frame: np.ndarray
    sparse: bool  # кадр сетки 1 Гц (стадии 1–2); любой кадр — кадр плотной сетки


def iter_stream(path: str, fps: float, sparse_fps: float, dense_fps: float, stats: Optional[dict] = None) -> Iterator[StreamFrame]:
    """Все кадры декодируются (grab), в память попадают только кадры плотной сетки dense_fps; кадр сетки 1 Гц —
    всегда и кадр плотной сетки. Для RTSP — то же самое: кадры идут по мере поступления."""
    cap = _open(path)
    clock = FrameClock(fps)
    sparse, dense = Ticker(1.0 / sparse_fps), Ticker(1.0 / dense_fps)
    idx = 0
    try:
        while cap.grab():
            t = clock.time(idx, cap.get(cv2.CAP_PROP_POS_MSEC))
            is_dense = dense(t)
            is_sparse = sparse(t)
            if is_dense or is_sparse:
                ok, frame = cap.retrieve()
                if ok:
                    yield StreamFrame(idx, t, frame, is_sparse)
            idx += 1
    finally:
        cap.release()
        if stats is not None:
            stats["decoded_frames"] = idx
            stats["synthetic_from"] = clock.synthetic_from


class FrameRing:
    """Кольцевой буфер последних кадров плотной сетки за span с: плотная проверка начинается с прошлого —
    к моменту срабатывания триггера подход руки ко рту уже прошёл."""

    def __init__(self, span: float):
        self.span = span
        self.items: Deque[StreamFrame] = deque()

    def push(self, f: StreamFrame) -> None:
        self.items.append(f)
        while self.items and f.t - self.items[0].t > self.span + EPS:
            self.items.popleft()

    def between(self, t0: float, t1: float) -> List[StreamFrame]:
        """Кадры с t0 < t < t1 по возрастанию времени."""
        return [f for f in self.items if t0 + EPS < f.t < t1 - EPS]

    def __len__(self) -> int:
        return len(self.items)


# ---------------------------------------------------------------- трекер 1 Гц


@dataclass
class SparseTrack:
    tid: int
    box: np.ndarray
    last_t: float


class SparseTracker:
    """Трекер по кадрам 1 Гц: жадное сопоставление по IoU (затем по расстоянию центров в высотах рамки).
    Курящий почти стоит, поэтому рамки соседних секунд перекрываются; трек без рамки дольше max_missed с закрыт."""

    def __init__(self, iou_min: float, dist_max: float, max_missed: float):
        self.iou_min, self.dist_max, self.max_missed = iou_min, dist_max, max_missed
        self.tracks: List[SparseTrack] = []
        self.next_id = 1

    def update(self, boxes: np.ndarray, t: float) -> List[int]:
        """id трека для каждой рамки; новые рамки открывают треки."""
        self.tracks = [tr for tr in self.tracks if t - tr.last_t <= self.max_missed + EPS]
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        pairs = []
        for i, tr in enumerate(self.tracks):
            if not len(boxes):
                break
            ious = box_iou(boxes, tr.box)
            th = max(tr.box[3] - tr.box[1], 1.0)
            tc = np.array([(tr.box[0] + tr.box[2]) / 2, (tr.box[1] + tr.box[3]) / 2])
            for j, b in enumerate(boxes):
                bh = max(b[3] - b[1], 1.0)
                if not 0.5 <= bh / th <= 2.0:
                    continue
                dist = float(np.hypot((b[0] + b[2]) / 2 - tc[0], (b[1] + b[3]) / 2 - tc[1]) / max(bh, th))
                if ious[j] >= self.iou_min or dist <= self.dist_max:
                    pairs.append((-float(ious[j]), dist, i, j))
        ids = [-1] * len(boxes)
        used = set()
        for _, _, i, j in sorted(pairs):
            if i in used or ids[j] >= 0:
                continue
            used.add(i)
            ids[j] = self.tracks[i].tid
            self.tracks[i].box, self.tracks[i].last_t = boxes[j].copy(), t
        for j in range(len(boxes)):
            if ids[j] < 0:
                self.tracks.append(SparseTrack(self.next_id, boxes[j].copy(), t))
                ids[j] = self.next_id
                self.next_id += 1
        return ids

    def touch(self, tid: int, box: np.ndarray, t: float) -> None:
        """Человек найден позой в плотной проверке, хотя детектор его пропустил: трек жив, рамка обновлена."""
        for tr in self.tracks:
            if tr.tid == tid:
                tr.box, tr.last_t = np.asarray(box, dtype=np.float64).copy(), t
                return


# ---------------------------------------------------------------- геометрия отсчёта


def sample_features(k: np.ndarray, box: np.ndarray, geo) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """d (кисть→рот в единицах нормировки) одного отсчёта, рот и кисть — как geometry.build_signal, но без
    сглаживания по времени (для триггера 1 Гц). Неправдоподобная (чужая) рука не учитывается."""
    scale = raw_scale(k, box, geo)
    mouth, _ = estimate_mouth(k, scale, geo.min_kpt_conf)
    if mouth is None:
        return float("nan"), None, None
    skip = tuple(not ok for ok in arm_plausible(k, box, scale, geo))
    res = nearest_hand(k, mouth, scale, geo, skip)
    if res is None:
        return float("nan"), mouth, None
    return float(res[1]), mouth, res[0]


def interp_box(ts: List[float], boxes: List[np.ndarray], t: float) -> np.ndarray:
    """Рамка человека в момент t по отсчётам трека 1 Гц (линейно между соседними, за краями — крайняя):
    кадры кольцевого буфера идущего человека берутся по его прошлому положению, а не по рамке в момент триггера."""
    if t <= ts[0]:
        return np.asarray(boxes[0], dtype=np.float64)
    for i in range(1, len(ts)):
        if t <= ts[i]:
            a = (t - ts[i - 1]) / max(ts[i] - ts[i - 1], EPS)
            return (1 - a) * np.asarray(boxes[i - 1], dtype=np.float64) + a * np.asarray(boxes[i], dtype=np.float64)
    return np.asarray(boxes[-1], dtype=np.float64)


def crop_box(box: np.ndarray, pad: float, width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    pw, ph = pad * (x2 - x1), pad * (y2 - y1)
    return (int(max(0, x1 - pw)), int(max(0, y1 - ph)), int(min(width, x2 + pw)), int(min(height, y2 + ph)))


# ---------------------------------------------------------------- задание плотной проверки


@dataclass
class StoredCrop:
    x0: float  # левый верхний угол квадрата в координатах кадра
    y0: float
    k: float  # px хранимого кропа на px кадра
    img: np.ndarray
    t: float


class PersonJob:
    """Плотная проверка области одного человека: отсчёты позы ~dense_fps и фиксация законченных циклов на
    шагах 1 Гц. Шаги (момент, последний ли, сколько отсчётов было) записываются: повтор из кэша проходит их
    так же, как поток. Число отсчётов нужно, потому что после повторного открытия задания бэкфилл из кольцевого
    буфера дописывает отсчёты с временем раньше прошлого шага: по времени повтор отдал бы их не тому шагу."""

    def __init__(self, tid: int, box: np.ndarray, until: float, t0: float = 0.0, fps: float = 0.0,
                 stream_fps: float = 0.0):
        self.tid = tid
        # частота отсчётов плотной проверки (сервер снижает её под нагрузкой); 0 — каждый кадр плотной сетки потока
        self.fps = fps
        # частота плотной сетки потока при fps=0: ступень сервера меняет cfg.realtime.dense_fps на лету,
        # а сетка потока остаётся прежней; 0 — сетка равна cfg.realtime.dense_fps (повтор из кэша, run_rt)
        self.stream_fps = stream_fps
        self._tick = Ticker(1.0 / fps) if fps > 0 else None
        self.t0 = t0  # начало потока: пауза в первые cycle.lookaround с не отбрасывается за no_approach
        self.box = np.asarray(box, dtype=np.float64).copy()  # рамка человека для кропа следующего кадра
        self.until = until  # после этого момента без новых срабатываний триггера задание закрывается
        self.obs: List[PersonObs] = []
        self.ticks: List[Tuple[float, bool, int]] = []
        self.cycles: List[Cycle] = []  # зафиксированные циклы
        self.crops: Dict[int, StoredCrop] = {}  # стадия 4: кадр -> кроп «рот + кисть»
        self.closed = False
        self._cut = -math.inf  # новые циклы ищутся по отсчётам не раньше этого момента
        self._last_hold_end = -math.inf

    @property
    def last_t(self) -> float:
        return self.obs[-1].t if self.obs else -math.inf

    def rate(self, cfg: Config) -> float:
        if self.fps > 0:
            return self.fps
        return self.stream_fps if self.stream_fps > 0 else cfg.realtime.dense_fps

    def wants(self, t: float) -> bool:
        """Брать ли кадр плотной сетки t (вызывается по кадрам в порядке времени): при своей частоте задания —
        первый кадр каждого интервала её сетки."""
        return self._tick is None or self._tick(t)

    def add(self, o: PersonObs) -> None:
        self.obs.append(o)
        if self._tick is not None:
            self._tick(o.t)  # отсчёт кадра 1 Гц занимает свой интервал сетки задания
        self.box = o.box.astype(np.float64).copy()

    def predict_box(self, t: float, window: float) -> np.ndarray:
        """Рамка для кропа кадра t: последняя найденная, сдвинутая по скорости центра рамки за последние window с.
        Идущий человек за 0.2 с уходит на треть ширины рамки: без сдвига кроп отстаёт, поза его теряет и на
        следующих кадрах уже не находит (отсчёты обрываются как раз в паузе у рта, когда детектор тоже его не видит)."""
        if window <= 0 or len(self.obs) < 3:
            return self.box
        last = self.obs[-1].t
        pts = [o for o in self.obs[-MOTION_SAMPLES:] if o.t >= last - window - EPS]
        if len(pts) < 3 or pts[-1].t - pts[0].t < MOTION_MIN_SPAN:
            return self.box
        ts = np.array([o.t for o in pts])
        cx = np.array([(o.box[0] + o.box[2]) / 2 for o in pts])
        cy = np.array([(o.box[1] + o.box[3]) / 2 for o in pts])
        vx, vy = np.polyfit(ts, cx, 1)[0], np.polyfit(ts, cy, 1)[0]
        dt = min(max(t - last, 0.0), MOTION_MAX)
        return self.box + np.array([vx, vy, vx, vy]) * dt

    def finalize(self, now: float, final: bool, cfg: Config, roi: Roi) -> Tuple[List[Cycle], Optional[TrackSignal]]:
        """Новые законченные циклы по отсчётам последних finalize_window с (не раньше конца прошлого цикла).
        Цикл фиксируется, если после его конца прошло settle с (отвод руки и сглаживание уже не изменятся),
        а при закрытии задания (final) — сразу."""
        self.ticks.append((now, final, len(self.obs)))
        settle = cfg.cycle.lookaround + SETTLE_EXTRA
        start = max(self._cut, now - cfg.realtime.finalize_window)
        obs = [o for o in self.obs if o.t >= start - EPS]
        if len(obs) < 3:
            return [], None
        sig = build_signal(self.tid, obs, roi, cfg.geometry, cfg.tracking, self.rate(cfg))
        new = []
        for c in detect_cycles(sig, cfg.cycle):
            if c.hold_start <= self._last_hold_end + EPS:
                continue  # уже зафиксирован (или его хвост в начале окна)
            if not final and now - c.end < settle:
                break
            if cfg.realtime.start_hold and "no_approach" in c.reject_reason.split(",") and c.hold_start - self.t0 <= cfg.cycle.lookaround + EPS:  # fmt: skip
                # рука у рта уже к началу записи: подхода не видно, но цикл полный
                reasons = [r for r in c.reject_reason.split(",") if r != "no_approach"]
                c.reject_reason, c.valid = ",".join(reasons), not reasons
            new.append(c)
        if new:
            self._last_hold_end = new[-1].hold_end
            self._cut = new[-1].end - cfg.geometry.scale_window  # запас на скользящую медиану нормировки
            self.cycles.extend(new)
        if final:
            self.closed = True
        return new, sig

    def prune_crops(self, now: float, cfg: Config) -> None:
        keep = cfg.cycle.max_hold + 2 * cfg.cycle.lookaround + SETTLE_EXTRA + CROP_KEEP_EXTRA
        for f in [f for f, c in self.crops.items() if now - c.t > keep]:
            del self.crops[f]

    def record(self) -> dict:
        """Для кэша: только то, что нужно для повтора."""
        rec = {"tid": self.tid, "obs": self.obs, "ticks": self.ticks, "t0": self.t0}
        if self.fps > 0:
            rec["fps"] = self.fps
        return rec


def replay_job(rec: dict, cfg: Config, roi: Roi) -> PersonJob:
    """Повтор задания из кэша: те же отсчёты и шаги фиксации, что в потоке."""
    job = PersonJob(rec["tid"], rec["obs"][0].box if rec["obs"] else np.zeros(4), 0.0, rec.get("t0", 0.0),
                    rec.get("fps", 0.0))
    obs, i = rec["obs"], 0
    for tick in rec["ticks"]:
        now, final = tick[0], tick[1]
        if len(tick) > 2:  # отсчёты, которые были у задания на этом шаге
            job.obs.extend(obs[i : tick[2]])
            i = max(i, tick[2])
        else:  # кэш до записи числа отсчётов: по времени
            while i < len(obs) and obs[i].t <= now + EPS:
                job.obs.append(obs[i])
                i += 1
        job.finalize(now, final, cfg, roi)
    job.obs.extend(obs[i:])
    return job


def person_signals(jobs: List[PersonJob], cfg: Config, roi: Roi) -> Dict[int, TrackSignal]:
    """Сигнал человека по отсчётам всех его заданий (для рамки события и стадии 4 при повторе)."""
    by_tid: Dict[int, Dict[int, PersonObs]] = defaultdict(dict)
    rate: Dict[int, float] = {}
    for job in jobs:
        rate[job.tid] = min(rate.get(job.tid, math.inf), job.rate(cfg))
        for o in job.obs:
            by_tid[job.tid][o.frame_idx] = o
    out = {}
    for tid, d in by_tid.items():
        obs = [d[k] for k in sorted(d)]
        if len(obs) >= 3:
            out[tid] = build_signal(tid, obs, roi, cfg.geometry, cfg.tracking, rate[tid])
    return out


def person_cycles(jobs: List[PersonJob]) -> Dict[int, List[Cycle]]:
    out: Dict[int, List[Cycle]] = defaultdict(list)
    for job in jobs:
        out[job.tid].extend(job.cycles)
    for cs in out.values():
        cs.sort(key=lambda c: c.start)
    return dict(out)


# ---------------------------------------------------------------- модели


def project_path(path: str) -> str:
    from .pipeline import project_path as pp

    return pp(path)


class PersonDetector:
    """Стадия 1: лёгкий детектор людей на всём кадре."""

    def __init__(self, cfg: Config, device: str):
        rc = cfg.realtime
        self.rc, self.device = rc, device
        path = project_path(rc.det_model)
        self.ov = OvYolo(path, rc.threads) if device == "cpu" and is_openvino_dir(path) else None
        if self.ov is None:
            from ultralytics import YOLO

            self.model = YOLO(path)
        else:  # вход как у predict() для динамической модели: вписан в imgsz, поля до кратного 32
            from ultralytics.data.augment import LetterBox

            self.letterbox = LetterBox((rc.det_imgsz, rc.det_imgsz), auto=True, stride=32)
        self.half = half_kwargs(rc.half, device)

    def __call__(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.ov is not None:
            from ultralytics.utils.ops import scale_boxes

            img = self.letterbox(image=frame)
            boxes, conf, _ = self.ov.detect(img[None], self.rc.det_conf, self.rc.det_iou)[0]
            return scale_boxes(img.shape[:2], boxes, frame.shape[:2]).astype(np.float64), conf.astype(np.float64)
        r = self.model.predict(frame, imgsz=self.rc.det_imgsz, conf=self.rc.det_conf, iou=self.rc.det_iou, classes=[0],
                               device=self.device, verbose=False, **self.half)[0]  # fmt: skip
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 4)), np.zeros(0)
        return r.boxes.xyxy.cpu().numpy().astype(np.float64), r.boxes.conf.cpu().numpy().astype(np.float64)


def unpack_pose(res) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Results ultralytics -> (рамки xyxy, уверенность, точки (K, 17, 3)) в numpy."""
    if res.keypoints is None or res.boxes is None or len(res.boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros((0, 17, 3))
    return res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.keypoints.data.cpu().numpy()


def letterbox(img: np.ndarray, size: int) -> Tuple[np.ndarray, float, int, int]:
    """Квадрат size x size: изображение вписано с сохранением пропорций по центру, поля серые (как LetterBox
    ultralytics). Возвращает (квадрат, масштаб, сдвиг x, сдвиг y)."""
    h, w = img.shape[:2]
    r = size / max(h, w)
    nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    out[py : py + nh, px : px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, px, py


class CropPose:
    """Стадии 2–3: поза на кропах людей батчем. Из найденных в кропе поз берётся та, чья рамка лучше всех
    совпадает с рамкой человека (IoU >= match_iou): в кроп попадают и соседи.

    Кропы вписываются в квадрат pose_imgsz заранее: вход модели всегда одной формы. Первый вызов CUDA на новой
    форме входа стоит ~0.8 с (RTX 2060), а у одиночного кропа ultralytics выбрал бы прямоугольник по его
    пропорциям; поэтому все размеры батча 1..POSE_BATCH прогреваются при загрузке."""

    def __init__(self, cfg: Config, device: str):
        rc = cfg.realtime
        self.rc, self.device = rc, device
        path = project_path(rc.pose_model)
        self.ov = OvYolo(path, rc.threads) if device == "cpu" and is_openvino_dir(path) else None
        if self.ov is None:
            from ultralytics import YOLO

            self.model = YOLO(path)
        task = (self.ov or self.model).task
        if task != "pose":
            raise ValueError(f"{rc.pose_model}: нужна pose-модель, получена задача '{task}'")
        self.half = half_kwargs(rc.half, device)

    def warmup(self) -> None:
        blank = np.full((self.rc.pose_imgsz, self.rc.pose_imgsz, 3), 114, dtype=np.uint8)
        for bs in range(1, POSE_BATCH + 1):
            self._predict([blank] * bs)

    def _predict(self, imgs: List[np.ndarray]) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Квадраты pose_imgsz -> на каждый (рамки xyxy, уверенность, точки (K, 17, 3)) в координатах квадрата."""
        if self.ov is not None:
            return self.ov.detect(np.stack(imgs), self.rc.pose_conf, POSE_IOU)
        return [unpack_pose(r) for r in self._predict_ul(imgs)]

    def _predict_ul(self, imgs: List[np.ndarray]):
        return self.model.predict(imgs, imgsz=self.rc.pose_imgsz, conf=self.rc.pose_conf, device=self.device,
                                  verbose=False, **self.half)  # fmt: skip

    def __call__(self, items: List[Tuple[np.ndarray, np.ndarray]]) -> List[Optional[Tuple[np.ndarray, np.ndarray, float]]]:
        """items — (кадр, рамка человека); результат — (рамка, точки 17x3, уверенность) в координатах кадра или None."""
        out: List[Optional[Tuple[np.ndarray, np.ndarray, float]]] = [None] * len(items)
        imgs, geom, where = [], [], []
        for n, (frame, box) in enumerate(items):
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = crop_box(box, self.rc.crop_pad, w, h)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            img, r, px, py = letterbox(frame[y1:y2, x1:x2], self.rc.pose_imgsz)
            imgs.append(img)
            geom.append((r, px - x1 * r, py - y1 * r))  # кадр -> квадрат: x * r + сдвиг
            where.append(n)
        for s in range(0, len(imgs), POSE_BATCH):
            results = self._predict(imgs[s : s + POSE_BATCH])
            for (boxes, conf, kpts), (r, ox, oy), n in zip(results, geom[s : s + POSE_BATCH], where[s : s + POSE_BATCH]):
                if len(boxes) == 0:
                    continue
                rb = (boxes.astype(np.float64) - np.array([ox, oy, ox, oy])) / r
                ious = box_iou(rb, np.asarray(items[n][1], dtype=np.float64))
                j = int(np.argmax(ious))
                if ious[j] < self.rc.match_iou:
                    continue
                k = kpts[j].astype(np.float32).copy()
                k[:, 0] = (k[:, 0] - ox) / r
                k[:, 1] = (k[:, 1] - oy) / r
                out[n] = (rb[j], k, float(conf[j]))
        return out


# ---------------------------------------------------------------- стадия 4 в потоке


def keep_cycle(cfg: Config) -> Callable[[Cycle], bool]:
    """Какой цикл с оценкой CLIP ниже appearance.min_score остаётся валидным: с прямым признаком, а при оценке
    по событиям (realtime.objects_per_event) — любой: порог проверяет rt_events по лучшему циклу события."""
    if cfg.realtime.objects_per_event:
        return lambda c: True
    return lambda c: has_evidence(c, cfg.episode)


def rt_events(cycles: Dict[int, List[Cycle]], signals: Dict[int, TrackSignal], cfg: Config) -> List[Event]:
    """События режима: episodes.build_events, а при стадии 4 по событиям (realtime.objects_per_event) событие
    без прямого признака отклоняется, если лучшая оценка CLIP его циклов ниже appearance.min_score. Порог по
    каждому циклу (как в run.py) теряет true-клипы: у курящего вдалеке «похожа» на курение одна пауза из нескольких.
    Если CLIP оценил не меньше realtime.many_cycles циклов события, лучшая оценка должна быть не ниже
    many_min_score: человек много раз подносит руку ко рту, а сигареты не видно ни разу — питьё, касание лица.
    Или медиана оценок не ниже realtime.many_median (если > 0): у долгого курения сигарета видна в большинстве пауз,
    но ни в одной уверенно, а у питья и касания лица «похожих» пауз одна-две."""
    events = build_events(cycles, signals, cfg.episode)
    rc = cfg.realtime
    if not (rc.objects and rc.objects_per_event):
        return events
    for e in events:
        scores = [c.appearance for c in e.cycles if np.isfinite(c.appearance)]
        if not (e.passed and scores) or any(has_evidence(c, cfg.episode) for c in e.cycles):
            continue
        best = max(scores)
        if best < cfg.appearance.min_score:
            e.passed, e.reject_reason = False, f"appearance:{best:.2f}"
        elif (rc.many_cycles > 0 and len(scores) >= rc.many_cycles and best < rc.many_min_score
              and not (rc.many_median > 0 and np.median(scores) >= rc.many_median)):
            e.passed, e.reject_reason = False, f"many_cycles:{len(scores)}:{best:.2f}"
    return events


AlertState = Tuple[float, float, bool]  # начало и конец события человека, о котором сообщено; действует ли тревога


def alert_change(events: List[Event], st: Optional[AlertState]) -> Tuple[Optional[AlertState], Optional[str]]:
    """События человека (по времени) и прежнее состояние тревоги -> новое состояние и сообщение: "alert" —
    правило эпизода выполнилось (новое событие или снятое снова подтвердилось), "withdrawn" — событие, о котором
    сообщили, правило больше не выполняет (стадия 4: много циклов без сигареты), None — без изменений. Меняется
    только последнее событие: новые циклы дописываются в конец."""
    if st is not None and st[2] and events:
        last = events[-1]
        if not last.passed and last.start <= st[1] + EPS:
            return (st[0], last.end, False), "withdrawn"
    passed = [e for e in events if e.passed]
    if not passed:
        return st, None
    e = passed[-1]
    if st is not None and e.start <= st[1] + EPS and (st[2] or e.end <= st[1] + EPS):
        return ((e.start, e.end, True) if st[2] else st), None  # продолжение или прежнее событие до снятого
    return (e.start, e.end, True), "alert"


def dropped_alert(st: Optional[AlertState], left: List[Cycle]) -> bool:
    """Снять ли тревогу человека, чьё задание удалено как дубль: тревога действует, а среди оставшихся у него
    циклов (других заданий) нет валидных на отрезке события, о котором сообщено."""
    if st is None or not st[2]:
        return False
    return not any(c.valid and c.start <= st[1] + EPS and c.end >= st[0] - EPS for c in left)


def store_crop(frame: np.ndarray, o: PersonObs, mouth, hand) -> StoredCrop:
    """Квадрат вокруг «рот + кисть» отсчёта (рот не найден — верх рамки) — для CLIP после фиксации цикла."""
    h = float(o.box[3] - o.box[1])
    if mouth is not None:
        c = (np.asarray(mouth, float) + np.asarray(hand if hand is not None else mouth, float)) / 2
    else:
        c = np.array([(o.box[0] + o.box[2]) / 2, o.box[1] + 0.12 * h])
    side = max(8.0, CROP_SIDE * h)
    img = square_crop(frame, c[0], c[1], side)
    k = 1.0
    if img.shape[0] > CROP_MAX:
        k = CROP_MAX / img.shape[0]
        img = cv2.resize(img, (CROP_MAX, CROP_MAX), interpolation=cv2.INTER_AREA)
    s_int = max(8, int(round(side)))
    x0, y0 = int(round(c[0] - s_int / 2)), int(round(c[1] - s_int / 2))
    return StoredCrop(float(x0), float(y0), k, img, o.t)


def score_from_memory(job: PersonJob, cycles: List[Cycle], sig: TrackSignal, cfg: Config, encoder) -> List[float]:
    """Оценка CLIP циклов по кропам из памяти задания: кадры и центры — как в AppearanceScorer (plan_frames,
    crop_geometry), кроп вырезается из сохранённого квадрата вместо кадра."""
    ac = cfg.appearance
    plan = view_plan(ac)
    out = []
    for c in cycles:
        ks = plan_frames(sig, c, ac.max_frames)
        centers, height = crop_geometry(sig, c, ks)
        views: List[List[np.ndarray]] = [[] for _ in plan]
        for k, (cx, cy) in zip(ks, centers):
            sc = job.crops.get(int(sig.frame_idx[k]))
            if sc is None:
                continue
            for v, (s, _) in enumerate(plan):
                views[v].append(square_crop(sc.img, (cx - sc.x0) * sc.k, (cy - sc.y0) * sc.k, s * height * sc.k))
        out.append(cycle_score([encoder(views[v], tv) for v, (_, tv) in enumerate(plan)], ac.frame_agg)
                   if views[0] else float("nan"))  # fmt: skip
    return out


# ---------------------------------------------------------------- результат и кэш


@dataclass
class RtRun:
    """Результат стадий 1–3 по клипу (кэшируется): задания плотной проверки и замеры времени."""

    meta: VideoMeta
    jobs: List[dict]  # PersonJob.record()
    sparse: Dict[int, List[PersonObs]]  # отсчёты 1 Гц людей вне плотной проверки (разбор ошибок)
    stats: Dict[str, float] = field(default_factory=dict)
    timings: Dict[str, float] = field(default_factory=dict)
    wall_time: float = 0.0
    cache_key: str = ""
    alerts: List[dict] = field(default_factory=list)  # тревоги в потоке и их снятия (withdrawn), alert_change


def inference_hash(cfg: Config) -> str:
    """Параметры, от которых зависят отсчёты стадий 1–3: секция realtime (кроме правил и стадии 4), geometry и
    tracking (триггер считается по той же геометрии), видео."""
    rc = {f.name: getattr(cfg.realtime, f.name) for f in fields(cfg.realtime) if f.name not in RULE_FIELDS}
    rc = {k: v for k, v in rc.items() if not (k in HASH_DEFAULTS and v == HASH_DEFAULTS[k])}
    data = {"realtime": rc, "geometry": cfg.to_dict()["geometry"], "tracking": cfg.to_dict()["tracking"],
            "video": cfg.to_dict()["video"]}  # fmt: skip
    return hashlib.sha256(repr(sorted(data.items())).encode()).hexdigest()[:16]


def rt_cache_key(video_path: str, cfg: Config) -> str:
    p = Path(video_path)
    stat = f"{p.stat().st_size}:{p.stat().st_mtime_ns}" if p.exists() else "stream"
    return hashlib.sha256(f"{p.resolve() if p.exists() else video_path}|{stat}|{inference_hash(cfg)}".encode()).hexdigest()[:16]


def save_run(run: RtRun, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"version": CACHE_VERSION, "run": run}, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_run(path: Path, key: str) -> Optional[RtRun]:
    if not path.exists():
        return None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:  # noqa: BLE001 — повреждённый кэш пересчитывается
        log.warning("Кэш %s не читается (%s), пересчёт", path, e)
        return None
    run = data.get("run")
    if data.get("version") != CACHE_VERSION or run is None or run.cache_key != key:
        return None
    fix_frame_count(run)
    return run


def fix_frame_count(run: RtRun) -> None:
    """Число кадров — по декодированным: в webm CAP_PROP_FRAME_COUNT бывает в десятки раз больше настоящего,
    а от него считаются длительность клипа, load и часы набора."""
    n = run.stats.get("decoded_frames", 0)
    if n > 0:
        run.meta.frame_count = n


# ---------------------------------------------------------------- поток


STAGES = ("decode_wait", "detect", "pose_sparse", "pose_dense", "cycles", "objects")


def limit_threads(n: int, precision: str = "") -> None:
    """Ограничить инференс на CPU n потоками: torch, OpenCV и OpenVINO; precision — INFERENCE_PRECISION_HINT
    OpenVINO (f32 — как на процессорах без bf16), пусто — по умолчанию устройства.

    ultralytics компилирует модели OpenVINO на устройстве AUTO со своим конфигом без числа потоков (берёт все ядра),
    а настройка одного ov.Core на другие не действует, поэтому Core.compile_model подменяется: AUTO → CPU (AUTO
    не принимает INFERENCE_NUM_THREADS) и подставляется число потоков."""
    if n <= 0 and not precision:
        return
    if n > 0:
        # OpenMP torch по умолчанию крутит потоки в ожидании (~200 мс): на n ядрах это отнимает их у OpenVINO и декодера
        os.environ.setdefault("KMP_BLOCKTIME", "0")
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
        import torch

        torch.set_num_threads(n)
        cv2.setNumThreads(n)
    try:
        import openvino as ov
    except ImportError:
        return
    compile_model = ov.Core.compile_model
    extra = {**({"INFERENCE_NUM_THREADS": n} if n > 0 else {}), **({"INFERENCE_PRECISION_HINT": precision} if precision else {})}

    def compile_limited(self, model, device_name=None, config=None, *args, **kwargs):
        if device_name in (None, "CPU", "AUTO"):
            device_name, config = "CPU", {**extra, **(config or {})}
        return compile_model(self, model, device_name, config, *args, **kwargs)

    ov.Core.compile_model = compile_limited


class RealtimeSession:
    """Состояние обработки одного клипа (потока)."""

    def __init__(self, models: "RealtimeDetector", cfg: Config, meta: VideoMeta, roi: Roi, encoder=None):
        self.m, self.cfg, self.rc, self.meta, self.roi = models, cfg, cfg.realtime, meta, roi
        self.encoder = encoder  # стадия 4 (None — выключена)
        self.tracker = SparseTracker(self.rc.track_iou, self.rc.track_dist, self.rc.track_max_missed)
        self.ring = FrameRing(self.rc.dense_before)
        self.active: Dict[int, PersonJob] = {}
        self.jobs: List[PersonJob] = []
        self.last_job: Dict[int, PersonJob] = {}
        self.sparse: Dict[int, List[PersonObs]] = defaultdict(list)
        self.stats: Counter = Counter()
        self.tm = dict.fromkeys(STAGES, 0.0)
        self.alerts: List[dict] = []
        self.t0: Optional[float] = None  # время первого кадра потока
        self.stream_fps = self.rc.dense_fps  # частота плотной сетки потока (сервер меняет rc.dense_fps на лету)
        self._alerted: Dict[int, AlertState] = {}  # человек -> последнее событие, о котором сообщено (alert_change)

    # -- вспомогательное
    def _pose(self, items, key: str):
        t0 = time.perf_counter()
        res = self.m.pose(items) if items else []
        self.tm[key] += time.perf_counter() - t0
        self.stats[f"crops_{key.split('_')[1]}"] += len(items)
        return res

    def _add(self, job: PersonJob, f: StreamFrame, r) -> PersonObs:
        box, k, score = r
        o = PersonObs(f.idx, f.t, box, score, k)
        job.add(o)
        if self.encoder is not None:
            _, mouth, hand = sample_features(k, box, self.cfg.geometry)
            job.crops[f.idx] = store_crop(f.frame, o, mouth, hand)
        return o

    def _usable(self, box: np.ndarray) -> bool:
        if box[3] - box[1] < self.cfg.tracking.min_person_height:
            return False
        return self.roi.contains(*anchor_point(box, self.cfg.tracking.roi_anchor))

    # -- шаги
    def step(self, f: StreamFrame) -> None:
        if self.t0 is None:
            self.t0 = f.t
        if f.sparse:
            self._sparse_step(f)
        elif self.active:
            self._dense_step(f)
        self.ring.push(f)

    def _dense_step(self, f: StreamFrame) -> None:
        jobs = [j for j in self.active.values() if j.wants(f.t)]
        if not jobs:
            return
        res = self._pose([(f.frame, j.predict_box(f.t, self.rc.motion_window)) for j in jobs], "pose_dense")
        self.stats["dense_frames"] += 1
        for job, r in zip(jobs, res):
            if r is not None:
                self._add(job, f, r)

    def _sparse_step(self, f: StreamFrame) -> None:
        rc, geo = self.rc, self.cfg.geometry
        self.stats["sparse_frames"] += 1
        t0 = time.perf_counter()
        boxes, _ = self.m.detect(f.frame)
        self.tm["detect"] += time.perf_counter() - t0
        keep = [i for i in range(len(boxes)) if self._usable(boxes[i])]
        boxes = boxes[keep]
        ids = self.tracker.update(boxes, f.t)
        items, owners = [], []
        for b, tid in zip(boxes, ids):
            items.append((f.frame, b))
            owners.append(tid)
        for tid, job in self.active.items():  # детектор пропустил человека с плотной проверкой — ведём по позе
            if tid not in owners:
                items.append((f.frame, job.predict_box(f.t, rc.motion_window)))
                owners.append(tid)
        has_job = [tid in self.active for tid in owners]
        res = self._pose(items, "pose_sparse")
        self.stats["people"] += len(items)
        triggered, idle = [], []
        seen = set()
        for tid, r, dense in zip(owners, res, has_job):
            if r is None:
                continue
            seen.add(tid)
            d, _, _ = sample_features(r[1], r[0], geo)
            if dense:
                self._add(self.active[tid], f, r)
                self.tracker.touch(tid, r[0], f.t)
            else:
                self.sparse[tid].append(PersonObs(f.idx, f.t, r[0], r[2], r[1]))
            if np.isfinite(d) and d < rc.trigger_dist:
                self.stats["triggers"] += 1
                if dense:
                    self.active[tid].until = max(self.active[tid].until, f.t + rc.dense_after)
                else:
                    triggered.append((d, tid, r))
            elif rc.alert_hold > 0 and self._alerted.get(tid, (0, 0, False))[2]:
                # человек с действующей тревогой: плотная проверка, пока виден (после сработавших — по слотам)
                if dense:
                    self.active[tid].until = max(self.active[tid].until, f.t + rc.alert_hold)
                else:
                    triggered.append((rc.trigger_dist + (d if np.isfinite(d) else 0.0), tid, r))
                    self.stats["alert_holds"] += 1
            elif not dense:
                idle.append((d if np.isfinite(d) else math.inf, tid, r))
        self._drop_duplicates(f.t)
        dropped = self._start_jobs(f, sorted(triggered, key=lambda x: x[0]))
        if rc.fill_budget and not dropped:
            # свободный бюджет: плотно проверяются и люди без срабатывания (ближе рука ко рту — раньше);
            # задание продлевается, пока человек виден и слоты не нужны сработавшим
            for job in self.active.values():
                if job.tid in seen and job.until < f.t + 1.0 / rc.sparse_fps:
                    job.until = f.t + 1.0 / rc.sparse_fps + EPS
            self._start_jobs(f, sorted(idle, key=lambda x: x[0]), budget=True)
        self._finalize(f.t)

    def _drop_duplicates(self, now: float) -> None:
        """Два задания на одного человека (детектор дал две рамки, трекер — два id): остаётся старшее.
        События удалённого задания в итоги не попадут, поэтому его тревога снимается (dropped_alert)."""
        jobs = sorted(self.active.values(), key=lambda j: self.jobs.index(j))
        for i, a in enumerate(jobs):
            for b in jobs[i + 1 :]:
                if b.tid in self.active and a.tid in self.active and box_iou(a.box, b.box)[0] > DUP_IOU:
                    del self.active[b.tid]
                    self.jobs.remove(b)
                    if self.last_job.get(b.tid) is b:
                        del self.last_job[b.tid]
                    self.stats["jobs_duplicate"] += 1
                    left = [c for j in self.jobs if j.tid == b.tid for c in j.cycles]
                    if dropped_alert(self._alerted.get(b.tid), left):
                        self._withdraw(b.tid, now, self._alerted[b.tid], f"duplicate:{a.tid}")

    def _start_jobs(self, f: StreamFrame, candidates, budget: bool = False) -> int:
        """Задания для кандидатов (по порядку, пока есть слоты); возвращает число не получивших слот."""
        rc = self.rc
        started = []
        dropped = 0
        for d, tid, r in candidates:
            if len(self.active) >= rc.max_jobs:
                dropped += 1
                if not budget:
                    self.stats["dropped_triggers"] += 1
                continue
            if any(box_iou(j.box, r[0])[0] > DUP_IOU for j in self.active.values()):
                continue  # этого человека уже проверяет задание с другим id
            prev = self.last_job.get(tid)
            if prev is not None and f.t - prev.last_t <= rc.dense_before + EPS:
                job = prev  # человек снова поднял руку вскоре после закрытия — то же задание, без разрыва
                job.closed = False
                job.until = f.t + (1.0 / rc.sparse_fps + EPS if budget else rc.dense_after)
                self.stats["jobs_reopened"] += 1
            else:
                fps = rc.dense_fps if rc.dense_fps < self.stream_fps - EPS else 0.0
                job = PersonJob(tid, r[0], f.t + (1.0 / rc.sparse_fps + EPS if budget else rc.dense_after), self.t0, fps,
                                self.stream_fps)
                self.jobs.append(job)
                self.last_job[tid] = job
                self.stats["jobs_budget" if budget else "jobs"] += 1
            job.box = np.asarray(r[0], dtype=np.float64).copy()
            self.active[tid] = job
            started.append((job, r))
            self.stats["max_parallel"] = max(self.stats["max_parallel"], len(self.active))
        # кадры из кольцевого буфера (прошлое) всех новых заданий — одним батчем
        items, plan = [], []
        for job, r in started:
            t_from = max(job.last_t, f.t - rc.dense_before - EPS)
            hist = [(o.t, o.box) for o in self.sparse.get(job.tid, []) if t_from - 1.0 / rc.sparse_fps < o.t < f.t - EPS]
            ts, boxes = [h[0] for h in hist] + [f.t], [h[1] for h in hist] + [r[0]]
            for fr in self.ring.between(t_from, f.t):
                if not job.wants(fr.t):
                    continue
                items.append((fr.frame, interp_box(ts, boxes, fr.t)))
                plan.append((job, fr))
        res = self._pose(items, "pose_dense")
        for (job, fr), rr in zip(plan, res):
            if rr is not None:
                self._add(job, fr, rr)
        for job, r in started:
            self._add(job, f, r)
        return dropped

    def _finalize(self, now: float, final: bool = False) -> None:
        for tid, job in list(self.active.items()):
            last = final or now > job.until + EPS
            t0 = time.perf_counter()
            new, sig = job.finalize(now, last, self.cfg, self.roi)
            self.tm["cycles"] += time.perf_counter() - t0
            if new and self.encoder is not None:
                self._score(job, new, sig)
            if self.encoder is not None:
                job.prune_crops(now, self.cfg)
            if new:
                self._alert(tid, now, sig)
            if last:
                job.crops.clear()
                del self.active[tid]

    def _confirmed(self, tid: int, c: Cycle) -> bool:
        """Валидный цикл продолжает событие человека с действующей тревогой, а лучшая оценка CLIP этого события уже
        не ниже appearance.min_score и many_min_score: его оценка решения по событию не изменит (realtime.skip_confirmed)."""
        st = self._alerted.get(tid)
        if not (self.rc.skip_confirmed and c.valid and st is not None and st[2]):
            return False
        if c.start - st[1] > self.cfg.episode.merge_gap + EPS:
            return False  # новое событие
        scores = [x.appearance for j in self.jobs if j.tid == tid for x in j.cycles
                  if x.valid and np.isfinite(x.appearance) and st[0] - EPS <= x.start <= st[1] + EPS]  # fmt: skip
        need = max(self.cfg.appearance.min_score, self.rc.many_min_score if self.rc.many_cycles > 0 else -math.inf)
        return bool(scores) and max(scores) >= need

    def _score(self, job: PersonJob, new: List[Cycle], sig: TrackSignal) -> None:
        ac = self.cfg.appearance
        todo = [c for c in new if c.valid or rescuable(c, ac)]
        skip = [c for c in todo if self._confirmed(job.tid, c)]
        if skip:
            todo = [c for c in todo if c not in skip]
            self.stats["skipped_cycles"] += len(skip)
        if not todo:
            return
        t0 = time.perf_counter()
        for c, s in zip(todo, score_from_memory(job, todo, sig, self.cfg, self.encoder)):
            apply_score(c, s, ac, keep_cycle(self.cfg))
        self.tm["objects"] += time.perf_counter() - t0
        self.stats["scored_cycles"] += len(todo)

    def _alert(self, tid: int, now: float, sig: TrackSignal) -> None:
        """Сообщение о курении, как только у человека выполнилось правило эпизода (как в итоговом build_events);
        продолжение того же события повторно не сообщается, а если событие правило больше не выполняет — тревога
        снимается (alert_change). Задержка — от конца цикла, выполнившего правило."""
        cycles = sorted((c for j in self.jobs if j.tid == tid for c in j.cycles), key=lambda c: c.start)
        events = sorted(rt_events({tid: cycles}, {tid: sig}, self.cfg), key=lambda e: e.start)
        st, msg = alert_change(events, self._alerted.get(tid))
        if st is not None:
            self._alerted[tid] = st
        if msg is None:
            return
        if msg == "withdrawn":
            self._withdraw(tid, now, st, events[-1].reject_reason)
            return
        rec = {"t": round(now, 2), "track_id": tid, "start": round(st[0], 2), "end": round(st[1], 2)}
        self.alerts.append({**rec, "latency": round(now - st[1], 2)})
        log.info("t=%.1f с: похоже на курение, человек %d (%.1f–%.1f с)", now, tid, st[0], st[1])

    def _withdraw(self, tid: int, now: float, st: AlertState, reason: str) -> None:
        self._alerted[tid] = (st[0], st[1], False)
        rec = {"t": round(now, 2), "track_id": tid, "start": round(st[0], 2), "end": round(st[1], 2)}
        self.alerts.append({**rec, "withdrawn": True, "reason": reason})
        log.info("t=%.1f с: тревога снята, человек %d (%.1f–%.1f с): %s", now, tid, st[0], st[1], reason)

    def finish(self, t: float) -> None:
        self._finalize(t, final=True)


class RealtimeDetector:
    """Модели режима и обработка клипа. Детектор и поза грузятся при первом прогоне (повтору из кэша не нужны)
    и прогреваются: первый вызов CUDA-моделей долгий и не должен попадать во время первого клипа."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = select_device(cfg.detector.device)
        ensure_for(cfg, self.device)  # недостающие веса — из релиза GitHub (smokedet/weights.py)
        self._detect: Optional[PersonDetector] = None
        self._pose: Optional[CropPose] = None
        self.appearance = AppearanceScorer(self.device, project_path(cfg.appearance.weights_dir))

    def _load(self) -> None:
        if self._detect is not None:
            return
        t0 = time.perf_counter()
        self._detect = PersonDetector(self.cfg, self.device)
        self._pose = CropPose(self.cfg, self.device)
        self._detect(np.zeros((1080, 1920, 3), dtype=np.uint8))
        self._pose.warmup()
        log.info("модели загружены и прогреты за %.1f с", time.perf_counter() - t0)

    @property
    def detect(self) -> PersonDetector:
        self._load()
        return self._detect

    @property
    def pose(self) -> CropPose:
        self._load()
        return self._pose

    def encoder(self, cfg: Config):
        if not cfg.realtime.objects:
            return None
        return self.appearance._encoder(cfg.appearance)

    def run(self, video_path: str, cfg: Config, roi_path: Optional[str] = None,
            on_frame: Optional[Callable[[StreamFrame], None]] = None) -> Tuple[RtRun, List[PersonJob]]:  # fmt: skip
        """Стадии 1–3 (и 4 при realtime.objects) по клипу: отсчёты, задания с зафиксированными циклами."""
        rc = cfg.realtime
        meta = probe_video(video_path, cfg.video.fallback_fps)
        roi = Roi.load(roi_path, meta.width, meta.height)
        enc = self.encoder(cfg)
        ses = RealtimeSession(self, cfg, meta, roi, enc)
        stream_stats: dict = {}
        # форма входа детектора зависит от разрешения: первый вызов на новой форме долгий — до начала отсчёта
        self.detect(np.zeros((max(meta.height, 32), max(meta.width, 32), 3), dtype=np.uint8))
        start, cpu0 = time.perf_counter(), time.process_time()
        it = prefetched(iter_stream(video_path, meta.fps, rc.sparse_fps, rc.dense_fps, stream_stats), PREFETCH)
        last_t = 0.0
        while True:
            t0 = time.perf_counter()
            f = next(it, None)
            ses.tm["decode_wait"] += time.perf_counter() - t0
            if f is None:
                break
            ses.step(f)
            last_t = f.t
            if on_frame is not None:
                on_frame(f)
        ses.finish(last_t)
        wall, cpu = time.perf_counter() - start, time.process_time() - cpu0
        return session_result(ses, meta, stream_stats, wall, cpu), ses.jobs


def session_result(ses: RealtimeSession, meta: VideoMeta, stream_stats: dict, wall: float, cpu: float) -> RtRun:
    """Итог законченной сессии (после finish) для кэша и анализа."""
    meta.synthetic_from = stream_stats.get("synthetic_from")
    stats = dict(ses.stats)
    stats["cpu_sec"] = round(cpu, 2)  # процессорное время всех потоков: сколько ядер занято в среднем
    stats["decoded_frames"] = stream_stats.get("decoded_frames", 0)
    run = RtRun(meta=meta, jobs=[j.record() for j in ses.jobs], sparse=dict(ses.sparse), stats=stats,
                timings=dict(ses.tm), wall_time=wall, alerts=ses.alerts)  # fmt: skip
    fix_frame_count(run)
    return run


def analyze(run: RtRun, cfg: Config, roi: Roi, jobs: Optional[List[PersonJob]] = None,
            scorer: Optional[AppearanceScorer] = None, clip_id: str = "", video_path: str = ""):  # fmt: skip
    """Циклы, сигналы и события клипа. jobs — задания из потока (циклы уже зафиксированы и оценены CLIP);
    без них — повтор заданий из кэша, а стадия 4 (если realtime.objects) — по кадрам, перечитанным из видео."""
    if jobs is None:
        jobs = [replay_job(rec, cfg, roi) for rec in run.jobs]
        replayed = True
    else:
        replayed = False
    cycles = person_cycles(jobs)
    signals = person_signals(jobs, cfg, roi)
    cycles = {tid: cs for tid, cs in cycles.items() if tid in signals}
    if replayed and cfg.realtime.objects and scorer is not None:
        scorer.score(clip_id, video_path, cycles, signals, cfg.appearance, keep_cycle(cfg),
                     run.meta.synthetic_from)  # fmt: skip
    events = rt_events(cycles, signals, cfg)
    return cycles, signals, events
