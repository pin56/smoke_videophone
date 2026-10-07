"""Выдох пара или дыма после затяжки: светлая бледная дымка у рта сразу после отвода руки.

Сигарету на CCTV почти не видно, а выдох в холод виден: у лица на 1–2 с появляется облако,
светлее и бледнее того, что было на этом месте до жеста. По правилам кейса это прямой признак
(«жест и связанный с ним выдох»). Замер — в лёгкой стадии, только для валидных циклов: кадры
перечитываются из видео, как у верификатора, кэш треков не меняется.

Фон — медиана кадров перед подходом руки ко рту. В окне вокруг конца паузы считается доля «зоны
воздуха» у рта, ставшей светлее фона (с поправкой на общую яркость по кольцу вокруг человека)
и не насыщеннее. Зона воздуха — круг вокруг рта без головы, кисти, тела и рамок других людей.
Движение даёт и посветлевшие, и потемневшие пиксели, дымка — только посветлевшие, поэтому мера —
разность этих долей («чистая дымка»). Кадр не годится для замера, если человек сдвинулся
относительно фона (фон за идущим человеком другой) или от зоны мало что осталось (перекрытие
соседями). Пар — чистая дымка выше порога в двух годных кадрах подряд.

Пар не обязателен: на муляжах его нет, а люди, у которых он не виден (идут, стоят в группе),
просто не получают признака.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import VaporConfig
from .cycles import Cycle
from .geometry import TrackSignal
from .io_video import FrameReader

RING_PAD_X, RING_PAD_Y = 0.5, 0.2  # кольцо для поправки на общую яркость: рамка, расширенная на долю ширины/высоты
HEAD_UP = 0.1  # если центр лица неизвестен — он выше рта на столько (в единицах нормировки)

Rect = Tuple[int, int, int, int]


def _clip_rect(x1: float, y1: float, x2: float, y2: float, w: int, h: int) -> Rect:
    return int(max(0, np.floor(x1))), int(max(0, np.floor(y1))), int(min(w, np.ceil(x2))), int(min(h, np.ceil(y2)))


def _zone_rect(mouth: np.ndarray, scale: float, cfg: VaporConfig, w: int, h: int) -> Rect:
    r = cfg.zone_radius * scale
    return _clip_rect(mouth[0] - r, mouth[1] - r, mouth[0] + r + 1, mouth[1] + r + 1, w, h)


def _ring_rect(box: np.ndarray, w: int, h: int) -> Rect:
    bx1, by1, bx2, by2 = box
    px, py = RING_PAD_X * (bx2 - bx1), RING_PAD_Y * (by2 - by1)
    return _clip_rect(bx1 - px, by1 - py, bx2 + px, by2 + py, w, h)


def measure_frame(
    frame: np.ndarray,
    bg: np.ndarray,
    origin: Tuple[int, int],
    mouth: np.ndarray,
    face: Optional[np.ndarray],
    hand: Optional[np.ndarray],
    scale: float,
    box: np.ndarray,
    others: Sequence[np.ndarray],
    cfg: VaporConfig,
) -> Optional[Tuple[float, float]]:
    """(чистая дымка, доля оставшейся зоны) для одного кадра или None, если замерить нельзя.

    frame и bg — один и тот же прямоугольник кадра (BGR) с левым верхним углом origin
    в координатах кадра; остальные координаты — в координатах кадра."""
    h, w = frame.shape[:2]
    ox, oy = origin
    shift = np.array([ox, oy], np.float64)
    m = mouth - shift
    x1, y1, x2, y2 = _zone_rect(m, scale, cfg, w, h)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    yy, xx = np.mgrid[y1:y2, x1:x2]
    r = cfg.zone_radius * scale
    air = (xx - m[0]) ** 2 + (yy - m[1]) ** 2 <= r * r
    full = float(np.pi * r * r)
    head = (face - shift) if face is not None and not np.isnan(face).any() else m - np.array([0.0, HEAD_UP * scale])
    air &= (xx - head[0]) ** 2 + (yy - head[1]) ** 2 > (cfg.head_radius * scale) ** 2
    if hand is not None and not np.isnan(hand).any():
        p = hand - shift
        air &= (xx - p[0]) ** 2 + (yy - p[1]) ** 2 > (cfg.hand_radius * scale) ** 2
    bx1, by1, bx2, by2 = np.asarray(box, np.float64) - np.r_[shift, shift]
    air &= ~((yy > m[1] + cfg.body_offset * scale) & (xx >= bx1) & (xx <= bx2))
    for b in others:
        qx1, qy1, qx2, qy2 = np.asarray(b, np.float64) - np.r_[shift, shift]
        air &= ~((xx >= qx1) & (xx <= qx2) & (yy >= qy1) & (yy <= qy2))
    kept = float(air.sum()) / full
    if not air.any():
        return None

    # поправка на общую яркость (автоэкспозиция, облака): медиана изменения в кольце вокруг человека
    rx1, ry1, rx2, ry2 = _ring_rect(np.r_[bx1, by1, bx2, by2], w, h)
    ref = 0.0
    if rx2 > rx1 and ry2 > ry1:
        v_f = cv2.cvtColor(frame[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2HSV)[..., 2].astype(np.int16)
        v_b = cv2.cvtColor(bg[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2HSV)[..., 2].astype(np.int16)
        ryy, rxx = np.mgrid[ry1:ry2, rx1:rx2]
        ring = ~((rxx >= bx1) & (rxx <= bx2) & (ryy >= by1) & (ryy <= by2))
        if ring.any():
            ref = float(np.median((v_f - v_b)[ring]))

    hsv_f = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv_b = cv2.cvtColor(bg[y1:y2, x1:x2], cv2.COLOR_BGR2HSV).astype(np.int16)
    d_v = hsv_f[..., 2] - hsv_b[..., 2] - ref
    d_s = hsv_f[..., 1] - hsv_b[..., 1]
    haze = float(((d_v >= cfg.min_rise) & (d_s <= 0))[air].mean())
    dark = float((d_v <= -cfg.min_rise)[air].mean())
    return haze - dark, kept


def peak_pair(t: Sequence[float], net: Sequence[float], ok: Sequence[bool]) -> Tuple[float, float]:
    """Максимум по парам соседних годных отсчётов от меньшего из двух: одиночный кадр — помеха.
    Возвращает (значение, время первого отсчёта пары) или (nan, nan), если пар нет."""
    best, best_t = np.nan, np.nan
    for i in range(len(net) - 1):
        if ok[i] and ok[i + 1]:
            v = min(net[i], net[i + 1])
            if np.isnan(best) or v > best:
                best, best_t = v, t[i]
    return float(best), float(best_t)


def signals_digest(signals: Dict[int, TrackSignal]) -> str:
    """Отпечаток всего, что замер берёт из сигналов: рот, лицо, кисть, нормировка, рамки, кадры, годность."""
    h = hashlib.sha1()
    for tid in sorted(signals):
        sig = signals[tid]
        h.update(str(tid).encode())
        for arr in (sig.frame_idx, sig.t, sig.mouth, sig.face, sig.hand, sig.scale, sig.boxes, sig.valid):
            h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def plan_cycle(sig: TrackSignal, c: Cycle, cfg: VaporConfig) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Отсчёты фона (перед подходом руки) и окна замера (вокруг конца паузы) или None, если их мало."""
    known = ~np.isnan(sig.mouth).any(axis=1)
    bg_k = np.flatnonzero((sig.t >= c.start - cfg.bg_before) & (sig.t <= c.start - cfg.bg_gap))
    win_k = np.flatnonzero((sig.t >= c.hold_end - cfg.window_before) & (sig.t <= c.hold_end + cfg.window_after) & known)
    if len(bg_k) < cfg.bg_min_frames or len(win_k) < 2:
        return None
    return bg_k, win_k


class CycleFrames:
    """Кропы одного цикла, накопленные при общем проходе по кадрам клипа, и замер по ним.

    Кроп — общий для всех кадров цикла прямоугольник (зоны у рта и кольца вокруг человека в окне замера),
    он считается по размеру первого прочитанного кадра. Цикл не замеряется, если не прочитан первый кадр фона."""

    def __init__(self, sig: TrackSignal, c: Cycle, bg_k: np.ndarray, win_k: np.ndarray):
        self.sig, self.c, self.bg_k, self.win_k = sig, c, bg_k, win_k
        self.rect: Optional[Rect] = None
        self.failed = False
        self.crops: Dict[int, np.ndarray] = {}  # отсчёт сигнала -> кроп
        self.last = int(max(sig.frame_idx[bg_k[-1]], sig.frame_idx[win_k[-1]]))  # последний нужный кадр

    def add(self, k: int, frame: Optional[np.ndarray], cfg: VaporConfig) -> None:
        if self.failed:
            return
        if frame is None:
            self.failed = k == self.bg_k[0]
            return
        if self.rect is None:
            h, w = frame.shape[:2]
            rects = [_zone_rect(self.sig.mouth[j], self.sig.scale[j], cfg, w, h) for j in self.win_k]
            rects += [_ring_rect(self.sig.boxes[j], w, h) for j in self.win_k]
            self.rect = min(r[0] for r in rects), min(r[1] for r in rects), max(r[2] for r in rects), max(r[3] for r in rects)
            if self.rect[2] - self.rect[0] < 4 or self.rect[3] - self.rect[1] < 4:
                self.failed = True
                return
        x1, y1, x2, y2 = self.rect
        self.crops[k] = frame[y1:y2, x1:x2].copy()

    def finish(self, others, cfg: VaporConfig) -> Tuple[float, float]:
        """(чистая дымка, время) по накопленным кропам; кропы освобождаются."""
        crops, self.crops = self.crops, {}
        if self.failed or self.rect is None:
            return np.nan, np.nan
        sig = self.sig
        x1, y1 = self.rect[:2]
        bg_crops = [crops[k] for k in self.bg_k if k in crops]
        if len(bg_crops) < cfg.bg_min_frames:
            return np.nan, np.nan
        bg = np.median(np.stack(bg_crops), axis=0).astype(np.uint8)
        bg_box = np.median(sig.boxes[self.bg_k], axis=0)
        bg_center = (bg_box[:2] + bg_box[2:]) / 2
        bg_others = [b for k in self.bg_k for tid, b in others.get(int(sig.frame_idx[k]), []) if tid != sig.track_id]

        t, net, ok = [], [], []
        for k in self.win_k:
            crop = crops.get(k)
            if crop is None:
                continue
            box = sig.boxes[k]
            near = [b for tid, b in others.get(int(sig.frame_idx[k]), []) if tid != sig.track_id] + bg_others
            res = measure_frame(crop, bg, (x1, y1), sig.mouth[k], None if sig.face is None else sig.face[k],
                                sig.hand[k], sig.scale[k], box, near, cfg)  # fmt: skip
            if res is None:
                continue
            moved = float(np.linalg.norm((box[:2] + box[2:]) / 2 - bg_center)) / max(float(box[3] - box[1]), 1.0)
            t.append(float(sig.t[k]))
            net.append(res[0])
            ok.append(res[1] >= cfg.min_kept and moved <= cfg.max_shift)
        return peak_pair(t, net, ok)


class VaporMeter:
    """Замер выдоха для валидных циклов клипа. Результат кэшируется в памяти по циклу, параметрам
    замера и отпечатку сигналов клипа (рот, кисть, нормировка и рамки зависят от секций geometry и
    tracking): tune.py перебирает пороги без повторного чтения видео и без устаревших замеров."""

    def __init__(self) -> None:
        self.cache: Dict[tuple, Tuple[float, float]] = {}

    @staticmethod
    def _key(clip_id: str, c: Cycle, cfg: VaporConfig, digest: str = "") -> tuple:
        params = (cfg.zone_radius, cfg.head_radius, cfg.hand_radius, cfg.body_offset, cfg.min_rise, cfg.bg_before,
                  cfg.bg_gap, cfg.bg_min_frames, cfg.window_before, cfg.window_after, cfg.min_kept, cfg.max_shift)  # fmt: skip
        return clip_id, digest, c.track_id, round(c.start, 3), round(c.hold_end, 3), params

    def measure(
        self,
        clip_id: str,
        video_path: str,
        cycles: Dict[int, List[Cycle]],
        signals: Dict[int, TrackSignal],
        cfg: VaporConfig,
        synthetic_from: Optional[int] = None,
    ) -> None:
        """Заполняет у валидных циклов поля vapor / vapor_net / vapor_t. synthetic_from — см. VideoMeta."""
        digest = signals_digest(signals)
        todo = [c for cs in cycles.values() for c in cs if c.valid and self._key(clip_id, c, cfg, digest) not in self.cache]
        if todo:
            for c, res in zip(todo, self._measure_cycles(video_path, todo, signals, cfg, synthetic_from)):
                self.cache[self._key(clip_id, c, cfg, digest)] = res
        for cs in cycles.values():
            for c in cs:
                if c.valid:
                    c.vapor_net, c.vapor_t = self.cache[self._key(clip_id, c, cfg, digest)]
                    c.vapor = bool(c.vapor_net >= cfg.min_net)

    @staticmethod
    def _measure_cycles(
        video_path: str,
        todo: List[Cycle],
        signals: Dict[int, TrackSignal],
        cfg: VaporConfig,
        synthetic_from: Optional[int] = None,
    ) -> List[Tuple[float, float]]:
        """(чистая дымка, время) для каждого цикла todo. Кадры всех циклов читаются за один проход по
        возрастанию номера: переходы назад и seek дороже, чем прочитать файл подряд. Кропы цикла хранятся,
        пока не прочитан его последний кадр."""
        others = defaultdict(list)  # frame_idx -> [(track_id, рамка)]
        for tid, sig in signals.items():
            for f, b in zip(sig.frame_idx, sig.boxes):
                others[int(f)].append((tid, b))
        jobs: List[Optional[CycleFrames]] = []
        for c in todo:
            sig = signals[c.track_id]
            ks = plan_cycle(sig, c, cfg)
            jobs.append(None if ks is None else CycleFrames(sig, c, *ks))
        need: Dict[int, List[Tuple[int, int]]] = defaultdict(list)  # кадр -> [(номер цикла в todo, отсчёт)]
        times: Dict[int, float] = {}  # кадр -> его время у трекера (pts): по нему FrameReader ищет кадр
        for i, job in enumerate(jobs):
            if job is None:
                continue
            for k in np.union1d(job.bg_k, job.win_k):
                f = int(job.sig.frame_idx[k])
                need[f].append((i, int(k)))
                times[f] = float(job.sig.t[k])
        results: List[Tuple[float, float]] = [(np.nan, np.nan)] * len(jobs)
        if need:
            with FrameReader(video_path, synthetic_from=synthetic_from) as reader:
                for f in sorted(need):
                    frame = reader.get(f, times[f])
                    for i, k in need[f]:
                        jobs[i].add(k, frame, cfg)
                        if f == jobs[i].last:
                            results[i] = jobs[i].finish(others, cfg)
        return results
