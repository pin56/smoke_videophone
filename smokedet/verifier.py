"""Верификатор предметов: open-vocabulary детектор на кропах «рот + кисть» в момент паузы.

Позитивные классы (сигарета, вейп, дым) дают прямой признак; «чужие» предметы (бутылка, банка,
чашка, телефон, еда) отбрасывают цикл или снижают confidence. Каждый кадр смотрится в двух
масштабах: узкий кроп — для мелких предметов, широкий — для банок и бутылок, которые узкий
обрезает. Если open-vocab модель недоступна (нет CLIP / весов), используется COCO-детектор
только с негативными классами.

«Двойники» сигареты (чупа-чупс) и предметы для питья (бутылка, кружка, банка) ищутся по всему
треку, а не только в паузах: в паузе детектор принимает палочку за сигарету, а конфету или кружку
хорошо видно, когда их держат в руке или у рта между циклами.

Детекции кэшируются в памяти по рамке кропа, imgsz и словарю и хранятся с нижним порогом
уверенности: tune.py перебирает conf, геометрию и списки классов без устаревших результатов.
Словарь open-vocab модели «запекается» в weights/<модель>-vocab-<хэш>.pt: такой файл грузится
без текстового энкодера CLIP, поэтому прогон на площадке не требует интернета.
"""
from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

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
PREDICT_CONF = 0.05  # модель запускается с этим порогом, cfg.conf применяется при агрегации

# детекция в координатах кадра: (класс, уверенность, центр x, центр y)
Det = Tuple[str, float, float, float]


def raw_box(mouth: np.ndarray, hand: np.ndarray, scale: float, margin: float) -> Optional[Tuple[int, int, int, int]]:
    """Квадрат вокруг рта и кисти с полем margin * scale без обрезки по кадру — он же ключ кэша."""
    pts = [p for p in (mouth, hand) if p is not None and not np.isnan(p).any()]
    if not pts or np.isnan(scale):
        return None
    center = np.mean(pts, axis=0)
    span = np.linalg.norm(pts[0] - pts[-1]) / 2
    half = max(span + margin * scale, 12.0)
    return int(center[0] - half), int(center[1] - half), int(center[0] + half), int(center[1] + half)


def clip_box(shape, box: Optional[Tuple[int, int, int, int]]):
    if box is None:
        return None
    h, w = shape[:2]
    x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return x1, y1, x2, y2


def crop_box(shape, mouth: np.ndarray, hand: np.ndarray, scale: float, margin: float):
    """Квадрат вокруг рта и кисти с полем margin * scale в пределах кадра: (x1, y1, x2, y2) или None."""
    return clip_box(shape, raw_box(mouth, hand, scale, margin))


def make_crop(frame: np.ndarray, mouth: np.ndarray, hand: np.ndarray, scale: float, margin: float):
    box = crop_box(frame.shape, mouth, hand, scale, margin)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return frame[y1:y2, x1:x2].copy()


def best_by_class(dets: List[Det], min_conf: float) -> Dict[str, float]:
    best: Dict[str, float] = {}
    for name, conf, _, _ in dets:
        if conf >= min_conf:
            best[name] = max(best.get(name, 0.0), conf)
    return best


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


def vocabulary(cfg: VerifierConfig) -> List[str]:
    return list(dict.fromkeys([*cfg.positive_classes, *cfg.negative_classes, *cfg.track_negative_classes,
                               *cfg.track_drink_classes]))  # fmt: skip


def baked_path(model_path: str, classes: List[str]) -> Path:
    """Файл open-vocab модели с заданным словарём: грузится без CLIP."""
    digest = hashlib.sha256("\n".join(classes).encode("utf-8")).hexdigest()[:8]
    p = Path(model_path)
    return p.with_name(f"{p.stem}-vocab-{digest}{p.suffix}")


@dataclass
class TrackGroup:
    """Группа классов для поиска по треку и её пороги."""

    name: str
    classes: frozenset
    conf: float
    radius: float
    frames: int
    window: float


def track_groups(cfg: VerifierConfig, available: set) -> List[TrackGroup]:
    groups = [
        TrackGroup("lookalike", frozenset(cfg.track_negative_classes) & available, cfg.track_negative_conf,
                   cfg.track_negative_radius, cfg.track_negative_frames, cfg.track_negative_window),
        TrackGroup("drink", frozenset(cfg.track_drink_classes) & available, cfg.track_drink_conf,
                   cfg.track_drink_radius, cfg.track_drink_frames, cfg.track_drink_window),
    ]  # fmt: skip
    return [g for g in groups if g.classes and g.frames > 0]


def reject_by_track_objects(cycles: List[Cycle], hits: List[Tuple[float, str]], frames: int, window: float) -> None:
    """Отбрасывает валидные циклы, рядом с которыми (± окно) у того же человека уверенно найден
    «двойник» сигареты или предмет для питья. hits — (время, класс) кадров с находкой."""
    for c in cycles:
        if not c.valid:
            continue
        near = [name for t, name in hits if window <= 0 or abs(t - c.peak) <= window]
        if len(near) >= frames:
            c.valid = False
            c.direct = False  # «сигарета» в паузе — это палочка конфеты или край кружки
            c.reject_reason = "track_object:" + max(set(near), key=near.count)


class ObjectVerifier:
    def __init__(self, cfg: VerifierConfig, device: str, factory: Optional[Callable] = None):
        if factory is None:
            from ultralytics import YOLO

            factory = YOLO
        self.factory = factory
        self.device = device
        # пути здесь уже от корня проекта; в verify() приходит конфиг tune.py с относительными путями
        self.model_path = cfg.model
        self.fallback_path = cfg.fallback_model
        self._load(cfg)
        # детекции кропа по (clip_id, frame_idx, рамка кропа, imgsz, словарь, порог предсказания)
        self.cache: Dict[tuple, List[Det]] = {}

    def _load(self, cfg: VerifierConfig) -> None:
        classes = vocabulary(cfg)
        baked = baked_path(self.model_path, classes)
        self.base_weights = self.model_path
        try:
            if baked.is_file():
                model = self.factory(str(baked))
                if list(_names(model).values()) != classes:
                    raise RuntimeError(f"словарь в {baked} не совпадает с конфигом")
                self.weights, self.clip_used = str(baked), False
            else:
                model = self.factory(self.model_path)
                model.set_classes(classes)
                self.weights, self.clip_used = self.model_path, True
                _bake(model, baked)
            self.mode = "open_vocab"
            self.positive = set(cfg.positive_classes)
            self.negative = set(cfg.negative_classes)
        except Exception as e:  # нет CLIP/сети/весов — работаем без позитивных классов
            log.warning("Open-vocab верификатор недоступен (%s); переключаюсь на %s", e, self.fallback_path)
            model = self.factory(self.fallback_path)
            self.mode = "coco"
            self.weights = self.base_weights = self.fallback_path
            self.clip_used = False
            self.positive = set()
            self.negative = COCO_NEGATIVE
        self.model = model
        self.names = _names(model)
        self.vocab = tuple(classes) if self.mode == "open_vocab" else ("coco",)

    def _ensure_vocab(self, cfg: VerifierConfig) -> None:
        """Сетка tune.py может менять списки классов: тогда словарь модели задаётся заново."""
        if self.mode == "open_vocab" and tuple(vocabulary(cfg)) != self.vocab:
            log.info("Списки классов верификатора изменились — обновляю словарь модели")
            self._load(cfg)

    def describe(self) -> dict:
        track = [c for c in self.vocab if c not in self.positive | self.negative] if self.mode == "open_vocab" else []
        return {
            "weights": self.weights,
            "base_weights": self.base_weights,
            "mode": self.mode,
            "clip_used": self.clip_used,
            "positive_classes": sorted(self.positive),
            "negative_classes": sorted(self.negative),
            "track_negative_classes": sorted(track),
        }

    def predict(self, crops: List[np.ndarray], imgsz: int, conf: float, batch: int = 16):
        """Для каждого кропа — список (класс, уверенность, xyxy в координатах кропа)."""
        out: List[List[Tuple[str, float, np.ndarray]]] = []
        for i in range(0, len(crops), batch):
            results = self.model.predict(crops[i : i + batch], imgsz=imgsz, conf=conf, device=self.device, verbose=False)
            for r in results:
                dets = []
                if r.boxes is not None and len(r.boxes):
                    cls = r.boxes.cls.cpu().numpy().astype(int)
                    confs = r.boxes.conf.cpu().numpy()
                    for c, cf, xyxy in zip(cls, confs, r.boxes.xyxy.cpu().numpy()):
                        dets.append((self.names[int(c)], float(cf), xyxy))
                out.append(dets)
        return out

    def _detections(
        self, clip_id: str, video_path: str, signals: Dict[int, TrackSignal], requests, cfg: VerifierConfig
    ) -> Dict[tuple, List[Det]]:
        """requests — (track_id, sample_idx, margin). Детекции по каждому запросу; недостающие досчитываются."""
        pconf = min(PREDICT_CONF, cfg.conf)
        keyed = {}
        for tid, k, m in requests:
            sig = signals[tid]
            f = int(sig.frame_idx[k])
            box = raw_box(sig.mouth[k], sig.hand[k], sig.scale[k], m)
            keyed[(tid, k, m)] = (f, box, (clip_id, f, box, cfg.imgsz, self.vocab, pconf))
        todo = sorted({v for v in keyed.values() if v[2] not in self.cache}, key=lambda v: (v[0], str(v[1])))
        if todo:
            crops, keys, offsets = [], [], []
            with FrameReader(video_path) as reader:
                frame_cache: Optional[Tuple[int, np.ndarray]] = None
                for frame_idx, box, key in todo:
                    if frame_cache is None or frame_cache[0] != frame_idx:
                        frame_cache = (frame_idx, reader.get(frame_idx))
                    frame = frame_cache[1]
                    clipped = None if frame is None else clip_box(frame.shape, box)
                    if clipped is None:
                        self.cache[key] = []
                        continue
                    x1, y1, x2, y2 = clipped
                    crops.append(frame[y1:y2, x1:x2].copy())
                    keys.append(key)
                    offsets.append((x1, y1))
            for key, (ox, oy), dets in zip(keys, offsets, self.predict(crops, cfg.imgsz, pconf)):
                self.cache[key] = [
                    (name, conf, float((xyxy[0] + xyxy[2]) / 2 + ox), float((xyxy[1] + xyxy[3]) / 2 + oy))
                    for name, conf, xyxy in dets
                ]
        return {req: self.cache[v[2]] for req, v in keyed.items()}

    def verify(
        self,
        clip_id: str,
        video_path: str,
        cycles: Dict[int, List[Cycle]],
        signals: Dict[int, TrackSignal],
        cfg: VerifierConfig,
    ) -> None:
        """Заполняет у валидных циклов поля direct / pos_* / neg_fraction / strong_negative / objects,
        затем отбрасывает циклы людей, у которых по треку найден «двойник» сигареты или предмет для питья."""
        self._ensure_vocab(cfg)
        margins = [cfg.crop_margin] + ([cfg.wide_crop_margin] if cfg.wide_crop_margin > 0 else [])
        plan = [
            (tid, c, pick_evenly(c.hold_idx, cfg.frames_per_cycle))
            for tid, track_cycles in cycles.items() for c in track_cycles if c.valid
        ]  # fmt: skip
        requests = {(tid, k, m) for tid, _, ks in plan for k in ks for m in margins}
        dets = self._detections(clip_id, video_path, signals, requests, cfg)
        for tid, c, ks in plan:
            frames = [merge_detections([best_by_class(dets[(tid, k, m)], cfg.conf) for m in margins]) for k in ks]
            self._aggregate(c, frames, cfg)
        groups = track_groups(cfg, set(self.names.values()))
        if groups and cfg.track_scan_step > 0:
            self._scan_tracks(clip_id, video_path, cycles, signals, cfg, margins, groups)

    def _scan_tracks(
        self,
        clip_id: str,
        video_path: str,
        cycles: Dict[int, List[Cycle]],
        signals: Dict[int, TrackSignal],
        cfg: VerifierConfig,
        margins: List[float],
        groups: List[TrackGroup],
    ) -> None:
        """Ищет «двойников» сигареты и предметы для питья у кисти или рта по всему треку
        (только у людей с валидными циклами)."""
        plan = {tid: scan_indices(signals[tid], cfg.track_scan_step) for tid, cs in cycles.items() if any(c.valid for c in cs)}
        requests = {(tid, k, m) for tid, ks in plan.items() for k in ks for m in margins}
        dets = self._detections(clip_id, video_path, signals, requests, cfg)
        for tid, ks in plan.items():
            sig = signals[tid]
            for g in groups:
                hits: List[Tuple[float, str]] = []
                for k in ks:
                    anchors = [p for p in (sig.mouth[k], sig.hand[k]) if not np.isnan(p).any()]
                    names = [
                        name
                        for m in margins
                        for name, conf, cx, cy in dets[(tid, k, m)]
                        if name in g.classes and conf >= g.conf
                        and min(np.hypot(cx - p[0], cy - p[1]) for p in anchors) / sig.scale[k] <= g.radius
                    ]  # fmt: skip
                    if names:
                        hits.append((float(sig.t[k]), max(set(names), key=names.count)))
                if len(hits) >= g.frames:
                    log.info("[%s] трек %d: %s у кисти или рта в %d кадрах (%.1f–%.1f с)", clip_id, tid,
                             hits[0][1], len(hits), hits[0][0], hits[-1][0])  # fmt: skip
                reject_by_track_objects(cycles[tid], hits, g.frames, g.window)

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


def _bake(model, path: Path) -> None:
    """Сохраняет модель с заданным словарём. Кэшированный в модели CLIP (≈300 МБ) в файл не пишется."""
    inner = getattr(model, "model", None)
    clip = getattr(inner, "clip_model", None)
    try:
        if clip is not None:
            inner.clip_model = None
        model.save(str(path))
        log.info("Словарь верификатора сохранён в %s: дальше CLIP не нужен", path)
    except Exception as e:
        log.warning("Не удалось сохранить словарь в %s: %s", path, e)
    finally:
        if clip is not None:
            inner.clip_model = clip


def _names(model) -> Dict[int, str]:
    names = model.names
    return names if isinstance(names, dict) else dict(enumerate(names))
