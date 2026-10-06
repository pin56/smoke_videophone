"""Верификатор предметов: open-vocabulary детектор на кропах «рот + кисть» в момент паузы.

Позитивные классы (сигарета, вейп, дым) дают прямой признак; «чужие» предметы (бутылка, банка,
чашка, телефон, еда) отбрасывают цикл или снижают confidence. Каждый кадр смотрится в двух
масштабах: узкий кроп — для мелких предметов, широкий — для банок и бутылок, которые узкий
обрезает. Если open-vocab модель недоступна (нет CLIP / весов), используется COCO-детектор
только с негативными классами.
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


def make_crop(frame: np.ndarray, mouth: np.ndarray, hand: np.ndarray, scale: float, margin: float):
    pts = [p for p in (mouth, hand) if p is not None and not np.isnan(p).any()]
    if not pts:
        return None
    center = np.mean(pts, axis=0)
    span = np.linalg.norm(pts[0] - pts[-1]) / 2
    half = max(span + margin * scale, 12.0)
    h, w = frame.shape[:2]
    x1, y1 = int(max(0, center[0] - half)), int(max(0, center[1] - half))
    x2, y2 = int(min(w, center[0] + half)), int(min(h, center[1] + half))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
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


class ObjectVerifier:
    def __init__(self, cfg: VerifierConfig, device: str):
        from ultralytics import YOLO

        self.device = device
        self.mode = "open_vocab"
        self.weights = cfg.model
        classes = list(cfg.positive_classes) + list(cfg.negative_classes)
        try:
            model = YOLO(cfg.model)
            model.set_classes(classes)
            self.positive = set(cfg.positive_classes)
            self.negative = set(cfg.negative_classes)
        except Exception as e:  # нет CLIP/сети/весов — работаем без позитивных классов
            log.warning("Open-vocab верификатор недоступен (%s); переключаюсь на %s", e, cfg.fallback_model)
            model = YOLO(cfg.fallback_model)
            self.mode = "coco"
            self.weights = cfg.fallback_model
            self.positive = set()
            self.negative = COCO_NEGATIVE
        self.model = model
        names = model.names
        self.names = names if isinstance(names, dict) else dict(enumerate(names))
        # кэш детекций по (clip_id, track_id, frame_idx, поле кропа): tune.py не перечитывает видео
        self.cache: Dict[Tuple[str, int, int, float], Dict[str, float]] = {}

    def describe(self) -> dict:
        return {
            "weights": self.weights,
            "mode": self.mode,
            "positive_classes": sorted(self.positive),
            "negative_classes": sorted(self.negative),
        }

    def detect(self, crops: List[np.ndarray], cfg: VerifierConfig, batch: int = 16) -> List[Dict[str, float]]:
        out: List[Dict[str, float]] = []
        for i in range(0, len(crops), batch):
            results = self.model.predict(
                crops[i : i + batch], imgsz=cfg.imgsz, conf=cfg.conf, device=self.device, verbose=False
            )
            for r in results:
                best: Dict[str, float] = {}
                if r.boxes is not None:
                    for cls, conf in zip(r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                        name = self.names[cls]
                        best[name] = max(best.get(name, 0.0), float(conf))
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
        """Заполняет у валидных циклов поля direct / pos_* / neg_fraction / strong_negative / objects."""
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
