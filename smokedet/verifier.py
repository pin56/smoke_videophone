"""Верификатор предметов: open-vocabulary детектор на кропах «рот + кисть» в момент паузы.

Позитивные классы (сигарета, вейп, дым) дают прямой признак; «чужие» предметы (бутылка, банка,
чашка, телефон, еда) отбрасывают цикл или снижают confidence. Каждый кадр смотрится в двух
масштабах: узкий кроп — для мелких предметов, широкий — для банок и бутылок, которые узкий
обрезает. Если open-vocab модель недоступна (нет CLIP / весов), используется COCO-детектор
только с негативными классами.

«Двойники» сигареты (чупа-чупс) ищутся по всему треку, а не только в паузах: в паузе детектор
принимает палочку за сигарету, а сама конфета хорошо видна, когда её держат в руке или у рта
между циклами.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import VerifierConfig
from .cycles import Cycle
from .geometry import TrackSignal
from .io_video import FrameReader

log = logging.getLogger(__name__)

COCO_NEGATIVE = {
    "bottle", "cup", "wine glass", "cell phone", "sandwich", "banana", "apple",
    "donut", "pizza", "hot dog", "fork", "spoon", "knife",
}  # fmt: skip


def crop_box(shape, mouth: np.ndarray, hand: np.ndarray, scale: float, margin: float):
    """Квадрат вокруг рта и кисти с полем margin * scale: (x1, y1, x2, y2) или None."""
    pts = [p for p in (mouth, hand) if p is not None and not np.isnan(p).any()]
    if not pts or np.isnan(scale):
        return None
    center = np.mean(pts, axis=0)
    span = np.linalg.norm(pts[0] - pts[-1]) / 2
    half = max(span + margin * scale, 12.0)
    h, w = shape[:2]
    x1, y1 = int(max(0, center[0] - half)), int(max(0, center[1] - half))
    x2, y2 = int(min(w, center[0] + half)), int(min(h, center[1] + half))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return x1, y1, x2, y2


def make_crop(frame: np.ndarray, mouth: np.ndarray, hand: np.ndarray, scale: float, margin: float):
    box = crop_box(frame.shape, mouth, hand, scale, margin)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return frame[y1:y2, x1:x2].copy()


def merge_detections(dets: List[Dict[str, float]]) -> Dict[str, float]:
    """Объединяет детекции одного кадра по нескольким кропам: максимум уверенности по классу."""
    out: Dict[str, float] = {}
    for det in dets:
        for name, conf in det.items():
            out[name] = max(out.get(name, 0.0), conf)
    return out


def pick_evenly(items: List[int], n: int) -> List[int]:
    if len(items) <= n:
        return list(items)
    pos = np.linspace(0, len(items) - 1, n)
    return [items[int(round(p))] for p in pos]


def scan_indices(sig: TrackSignal, step: float) -> List[int]:
    """Отсчёты трека не чаще чем раз в step секунд: человек в зоне, рот найден."""
    out: List[int] = []
    last = -np.inf
    for k in range(len(sig)):
        if sig.valid[k] and not np.isnan(sig.mouth[k]).any() and sig.t[k] - last >= step:
            out.append(k)
            last = sig.t[k]
    return out


def reject_by_track_objects(cycles: List[Cycle], hits: List[Tuple[float, str]], cfg: VerifierConfig) -> None:
    """Отбрасывает валидные циклы, рядом с которыми (± окно) у того же человека уверенно найден
    «двойник» сигареты. hits — (время, класс) кадров с находкой."""
    for c in cycles:
        if not c.valid:
            continue
        near = [name for t, name in hits if cfg.track_negative_window <= 0 or abs(t - c.peak) <= cfg.track_negative_window]
        if len(near) >= cfg.track_negative_frames:
            c.valid = False
            c.direct = False  # «сигарета» в паузе — это палочка конфеты
            c.reject_reason = "track_object:" + max(set(near), key=near.count)


class ObjectVerifier:
    def __init__(self, cfg: VerifierConfig, device: str):
        from ultralytics import YOLO

        self.device = device
        self.mode = "open_vocab"
        self.weights = cfg.model
        classes = list(dict.fromkeys([*cfg.positive_classes, *cfg.negative_classes, *cfg.track_negative_classes]))
        try:
            model = YOLO(cfg.model)
            model.set_classes(classes)
            self.positive = set(cfg.positive_classes)
            self.negative = set(cfg.negative_classes)
            self.track_negative = set(cfg.track_negative_classes)
        except Exception as e:  # нет CLIP/сети/весов — работаем без позитивных классов
            log.warning("Open-vocab верификатор недоступен (%s); переключаюсь на %s", e, cfg.fallback_model)
            model = YOLO(cfg.fallback_model)
            self.mode = "coco"
            self.weights = cfg.fallback_model
            self.positive = set()
            self.negative = COCO_NEGATIVE
            self.track_negative = set()
        self.model = model
        names = model.names
        self.names = names if isinstance(names, dict) else dict(enumerate(names))
        # кэш детекций по (clip_id, track_id, frame_idx, поле кропа): tune.py не перечитывает видео
        self.cache: Dict[Tuple[str, int, int, float], Dict[str, float]] = {}
        # то же для поиска по треку: (класс, уверенность, расстояние до кисти или рта в единицах нормировки)
        self.scan_cache: Dict[Tuple[str, int, int, float], List[Tuple[str, float, float]]] = {}

    def describe(self) -> dict:
        return {
            "weights": self.weights,
            "mode": self.mode,
            "positive_classes": sorted(self.positive),
            "negative_classes": sorted(self.negative),
            "track_negative_classes": sorted(self.track_negative),
        }

    def predict(self, crops: List[np.ndarray], cfg: VerifierConfig, batch: int = 16):
        """Для каждого кропа — список (класс, уверенность, xyxy в координатах кропа)."""
        out: List[List[Tuple[str, float, np.ndarray]]] = []
        for i in range(0, len(crops), batch):
            results = self.model.predict(
                crops[i : i + batch], imgsz=cfg.imgsz, conf=cfg.conf, device=self.device, verbose=False
            )
            for r in results:
                dets = []
                if r.boxes is not None:
                    for cls, conf, xyxy in zip(r.boxes.cls.int().tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.cpu().numpy()):
                        dets.append((self.names[cls], float(conf), xyxy))
                out.append(dets)
        return out

    def detect(self, crops: List[np.ndarray], cfg: VerifierConfig, batch: int = 16) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        for dets in self.predict(crops, cfg, batch):
            best: Dict[str, float] = {}
            for name, conf, _ in dets:
                best[name] = max(best.get(name, 0.0), conf)
            out.append(best)
        return out

    def verify(
        self,
        clip_id: str,
        video_path: str,
        cycles: Dict[int, List[Cycle]],
        signals: Dict[int, TrackSignal],
        cfg: VerifierConfig,
    ) -> None:
        """Заполняет у валидных циклов поля direct / pos_* / neg_fraction / strong_negative / objects,
        затем отбрасывает циклы людей, у которых по треку найден «двойник» сигареты."""
        margins = [cfg.crop_margin] + ([cfg.wide_crop_margin] if cfg.wide_crop_margin > 0 else [])
        requests = []  # (frame_idx, track_id, sample_idx, margin)
        for tid, track_cycles in cycles.items():
            sig = signals[tid]
            for c in track_cycles:
                if c.valid:
                    for k in pick_evenly(c.hold_idx, cfg.frames_per_cycle):
                        requests.extend((int(sig.frame_idx[k]), tid, k, m) for m in margins)
        todo = sorted({r for r in requests if (clip_id, r[1], r[0], r[3]) not in self.cache})
        if todo:
            crops, keys = [], []
            with FrameReader(video_path) as reader:
                frame_cache: Optional[Tuple[int, np.ndarray]] = None
                for frame_idx, tid, k, margin in todo:
                    key = (clip_id, tid, frame_idx, margin)
                    if frame_cache is None or frame_cache[0] != frame_idx:
                        frame_cache = (frame_idx, reader.get(frame_idx))
                    frame = frame_cache[1]
                    if frame is None:
                        self.cache[key] = {}
                        continue
                    sig = signals[tid]
                    crop = make_crop(frame, sig.mouth[k], sig.hand[k], sig.scale[k], margin)
                    if crop is None:
                        self.cache[key] = {}
                        continue
                    crops.append(crop)
                    keys.append(key)
            for key, det in zip(keys, self.detect(crops, cfg)):
                self.cache[key] = det

        for tid, track_cycles in cycles.items():
            sig = signals[tid]
            for c in track_cycles:
                if c.valid:
                    frames = [int(sig.frame_idx[k]) for k in pick_evenly(c.hold_idx, cfg.frames_per_cycle)]
                    dets = [merge_detections([self.cache.get((clip_id, tid, f, m), {}) for m in margins]) for f in frames]
                    self._aggregate(c, dets, cfg)
        if self.track_negative and cfg.track_scan_step > 0:
            self._scan_tracks(clip_id, video_path, cycles, signals, cfg, margins)

    def _scan_tracks(
        self,
        clip_id: str,
        video_path: str,
        cycles: Dict[int, List[Cycle]],
        signals: Dict[int, TrackSignal],
        cfg: VerifierConfig,
        margins: List[float],
    ) -> None:
        """Ищет «двойников» сигареты у кисти или рта по всему треку (только у людей с валидными циклами)."""
        plan = {tid: scan_indices(signals[tid], cfg.track_scan_step) for tid, cs in cycles.items() if any(c.valid for c in cs)}
        requests = [(int(signals[tid].frame_idx[k]), tid, k, m) for tid, ks in plan.items() for k in ks for m in margins]
        todo = sorted(r for r in requests if (clip_id, r[1], r[0], r[3]) not in self.scan_cache)
        if todo:
            crops, keys, metas = [], [], []
            with FrameReader(video_path) as reader:
                frame_cache: Optional[Tuple[int, np.ndarray]] = None
                for frame_idx, tid, k, margin in todo:
                    key = (clip_id, tid, frame_idx, margin)
                    if frame_cache is None or frame_cache[0] != frame_idx:
                        frame_cache = (frame_idx, reader.get(frame_idx))
                    frame = frame_cache[1]
                    sig = signals[tid]
                    box = None if frame is None else crop_box(frame.shape, sig.mouth[k], sig.hand[k], sig.scale[k], margin)
                    if box is None:
                        self.scan_cache[key] = []
                        continue
                    x1, y1, x2, y2 = box
                    crops.append(frame[y1:y2, x1:x2].copy())
                    keys.append(key)
                    metas.append((tid, k, np.array([x1, y1], dtype=float)))
            for key, (tid, k, offset), dets in zip(keys, metas, self.predict(crops, cfg)):
                sig = signals[tid]
                anchors = [p for p in (sig.mouth[k], sig.hand[k]) if not np.isnan(p).any()]
                found = []
                for name, conf, xyxy in dets:
                    if name in self.track_negative:
                        center = (xyxy[:2] + xyxy[2:]) / 2 + offset
                        dist = min(float(np.linalg.norm(center - p)) for p in anchors) / sig.scale[k]
                        found.append((name, conf, dist))
                self.scan_cache[key] = found

        for tid, ks in plan.items():
            sig = signals[tid]
            hits: List[Tuple[float, str]] = []
            for k in ks:
                f = int(sig.frame_idx[k])
                names = [
                    name
                    for m in margins
                    for name, conf, dist in self.scan_cache.get((clip_id, tid, f, m), [])
                    if conf >= cfg.track_negative_conf and dist <= cfg.track_negative_radius
                ]
                if names:
                    hits.append((float(sig.t[k]), max(set(names), key=names.count)))
            if len(hits) >= cfg.track_negative_frames:
                log.info("[%s] трек %d: %s у кисти или рта в %d кадрах (%.1f–%.1f с)", clip_id, tid,
                         hits[0][1], len(hits), hits[0][0], hits[-1][0])  # fmt: skip
            reject_by_track_objects(cycles[tid], hits, cfg)

    def _aggregate(self, c: Cycle, dets: List[Dict[str, float]], cfg: VerifierConfig) -> None:
        objects: Dict[str, float] = defaultdict(float)
        pos_frames = neg_frames = 0
        pos_conf = 0.0
        neg_names: Dict[str, int] = defaultdict(int)
        strong_frames: Dict[str, int] = defaultdict(int)
        strong = self.negative - set(cfg.weak_negative_classes)
        for det in dets:
            for name, conf in det.items():
                if name in strong and conf >= cfg.strong_negative_conf:
                    strong_frames[name] += 1
            for name, conf in det.items():
                objects[name] = max(objects[name], conf)
            p = max((v for k, v in det.items() if k in self.positive), default=0.0)
            n_name, n = max(((k, v) for k, v in det.items() if k in self.negative), key=lambda kv: kv[1], default=("", 0.0))
            if p > 0:
                pos_frames += 1
                pos_conf = max(pos_conf, p)
            if n > p:
                neg_frames += 1
                neg_names[n_name] += 1
        c.objects = dict(objects)
        c.pos_frames = pos_frames
        c.pos_conf = pos_conf
        c.neg_fraction = neg_frames / len(dets) if dets else 0.0
        c.direct = pos_frames >= cfg.direct_min_frames
        top = max(strong_frames, key=strong_frames.get, default="")
        c.strong_negative = top if top and strong_frames[top] >= cfg.strong_negative_frames else ""
        if pos_frames > 0:
            return  # при признаке сигареты «чужие» предметы цикл не отбрасывают
        if c.strong_negative:
            # один предмет (банка, бутылка...) уверенно виден в нескольких кадрах — это не курение
            c.valid = False
            c.reject_reason = "object:" + c.strong_negative
        elif dets and c.neg_fraction >= cfg.negative_min_fraction:
            c.valid = False
            c.reject_reason = "object:" + max(neg_names, key=neg_names.get)
