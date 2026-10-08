"""Вспышка зажигалки у рта: локальный пересвет, которого нет в фоне кадра.

Пламя на CCTV — пересвеченное белое пятно без цвета: open-vocab детектор его не находит, зато
в темноте оно засвечивает лицо и кисти. Свет измеряется в проходе трекера, где кадр уже
декодирован. Фон — скользящая медиана кадров, «новый пересвет» — пиксели, яркие в кадре и
тёмные в фоне. Для каждого человека считаются две площади: у рта и кисти рядом с лицом и в
кольце вокруг рамки. Открытая дверь или фары освещают и окружение, пламя — только самого
человека. Маленький огонёк днём (несколько пикселей) таким способом не виден.

Анализ (лёгкая стадия): вспышка — несколько отсчётов подряд с большой площадью у рта, малой
в кольце и кистью у лица. Она отмечает валидные циклы того же человека рядом по времени:
прикуривание обычно идёт прямо перед первой затяжкой.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Sequence

import cv2
import numpy as np

from .config import FlashConfig, GeometryConfig
from .cycles import Cycle
from .geometry import TrackSignal, estimate_mouth, nearest_hand, raw_scale

# Параметры замера (стадия трекера, результат в кэше): менять только вместе с tracking.CACHE_VERSION
SAT = 240  # яркость пересвета
BG_MAX = 200  # в фоне пиксель не пересвечен...
BG_DIFF = 50  # ...и заметно темнее, чем в кадре
BG_PERIOD = 1.0  # с между кадрами буфера фона
BG_FRAMES = 31  # длина буфера: медиана примерно за 30 с
BG_MIN_FRAMES = 3  # с какого числа кадров фон считается готовым
DOWNSCALE = 2  # фон и маска пересвета — в половинном разрешении
ZONE_RADIUS = 0.7  # круги вокруг рта и кисти, в единицах нормировки
NEAR_HAND = 1.5  # кисть входит в зону, если не дальше от рта (в единицах нормировки)
RING_PAD_X, RING_PAD_Y = 0.5, 0.2  # кольцо: рамка, расширенная на долю ширины/высоты, минус рамка
_GEO = GeometryConfig()  # рот и кисть — как в geometry, с параметрами по умолчанию


@dataclass
class Flash:
    track_id: int
    start: float
    end: float
    area: float  # максимум пересвета у рта/кистей, scale^2
    ring: float  # максимум пересвета вокруг человека, scale^2

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "area": round(self.area, 3),
            "ring": round(self.ring, 3),
        }


def _rect_sum(integral: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> int:
    h, w = integral.shape[0] - 1, integral.shape[1] - 1
    x1, y1 = int(np.clip(np.floor(x1), 0, w)), int(np.clip(np.floor(y1), 0, h))
    x2, y2 = int(np.clip(np.ceil(x2), 0, w)), int(np.clip(np.ceil(y2), 0, h))
    if x2 <= x1 or y2 <= y1:
        return 0
    return int(integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1])


def _torch_device(device: str):
    """Устройство torch для медианы фона: CUDA трекера или CPU (MPS не умеет kthvalue/sort по uint8)."""
    import torch

    d = str(device).strip().lower()
    if d.isdigit():
        d = f"cuda:{d}"
    if d.startswith("cuda") and torch.cuda.is_available():
        return torch.device(d)
    return torch.device("cpu")


def background_median(frames: Sequence[np.ndarray], device: str = "cpu") -> np.ndarray:
    """Попиксельная медиана кадров в int16 — ровно np.median(stack, axis=0).astype(int16).

    np.median по оси кадров на 2560×1440/2 занимает ~0.2–0.3 с и съедала большую часть замера света;
    torch (сортировка на GPU, kthvalue на CPU) даёт тот же результат в 4–16 раз быстрее. При чётном числе
    кадров np.median берёт среднее двух средних, astype(int16) отбрасывает дробь — здесь это floor.
    """
    n = len(frames)
    try:
        import torch

        dev = _torch_device(device)
        x = torch.from_numpy(np.stack(frames)).to(dev).reshape(n, -1)
        if dev.type == "cuda":
            s = torch.sort(x, dim=0).values
            lo, hi = s[(n - 1) // 2], s[n // 2]
        else:
            lo = torch.kthvalue(x, (n + 1) // 2, dim=0).values
            hi = lo if n % 2 else torch.kthvalue(x, n // 2 + 1, dim=0).values
        med = torch.div(lo.to(torch.int16) + hi.to(torch.int16), 2, rounding_mode="floor")
        return med.reshape(frames[0].shape).cpu().numpy()
    except (ImportError, RuntimeError):
        return np.median(np.stack(frames), axis=0).astype(np.int16)


class LightMeter:
    """Пересвет у рта/кистей и вокруг человека относительно фона. Вызывается на каждом кадре трекера,
    в том числе без людей: так обновляется фон."""

    def __init__(self, device: str = "cpu") -> None:
        self.frames: Deque[np.ndarray] = deque(maxlen=BG_FRAMES)
        self.bg: Optional[np.ndarray] = None
        self.next_push = -np.inf
        self.device = device

    def measure(self, frame: np.ndarray, t: float, boxes: Sequence[np.ndarray], kpts: Sequence[np.ndarray]):
        """Возвращает для каждого человека (пиксели у рта, пиксели в кольце, scale) или None."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (gray.shape[1] // DOWNSCALE, gray.shape[0] // DOWNSCALE), interpolation=cv2.INTER_AREA)
        out: List[Optional[np.ndarray]] = [None] * len(boxes)
        if self.bg is not None and len(boxes):
            fg = (small >= SAT) & (self.bg <= BG_MAX) & (small.astype(np.int16) - self.bg >= BG_DIFF)
            integral = cv2.integral(fg.astype(np.uint8))
            out = [self._person(fg, integral, np.asarray(b, np.float64), k) for b, k in zip(boxes, kpts)]
        if t >= self.next_push:  # фон обновляется после замера: кадр со вспышкой не маскирует сам себя
            self.frames.append(small)
            self.next_push = t + BG_PERIOD
            if len(self.frames) >= BG_MIN_FRAMES:
                self.bg = background_median(self.frames, self.device)
        return out

    @staticmethod
    def _person(fg: np.ndarray, integral: np.ndarray, box: np.ndarray, k: np.ndarray) -> Optional[np.ndarray]:
        scale = raw_scale(k, box, _GEO)
        mouth, _ = estimate_mouth(k, scale, _GEO.min_kpt_conf)
        if mouth is None:
            return None
        centers = [mouth]
        res = nearest_hand(k, mouth, scale, _GEO)  # (кисть, расстояние до рта / scale, ...)
        if res is not None and res[1] <= NEAR_HAND:
            centers.append(res[0])
        pts = np.array(centers) / DOWNSCALE
        r = ZONE_RADIUS * scale / DOWNSCALE
        h, w = fg.shape
        x1, y1 = np.floor(pts.min(axis=0) - r).astype(int)
        x2, y2 = np.ceil(pts.max(axis=0) + r).astype(int) + 1
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        zone = 0
        if x2 > x1 and y2 > y1:
            yy, xx = np.mgrid[y1:y2, x1:x2]
            near = np.zeros(yy.shape, bool)
            for cx, cy in pts:
                near |= (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
            zone = int(np.count_nonzero(fg[y1:y2, x1:x2] & near))
        bx1, by1, bx2, by2 = box / DOWNSCALE
        px, py = RING_PAD_X * (bx2 - bx1), RING_PAD_Y * (by2 - by1)
        ring = _rect_sum(integral, bx1 - px, by1 - py, bx2 + px, by2 + py) - _rect_sum(integral, bx1, by1, bx2, by2)
        return np.array([zone * DOWNSCALE**2, ring * DOWNSCALE**2, scale], np.float32)


def find_flashes(sig: TrackSignal, cfg: FlashConfig) -> List[Flash]:
    """Вспышки трека: >= min_samples отсчётов подряд с пересветом у рта, без пересвета вокруг, кисть у лица."""
    if sig.light_area is None or sig.light_ring is None:
        return []
    hit = (
        sig.valid
        & (sig.light_area >= cfg.min_area)
        & (sig.light_ring <= cfg.max_ring)
        & (sig.d_raw <= cfg.max_hand_dist)
    )
    flashes: List[Flash] = []
    run: List[int] = []
    for i in list(np.flatnonzero(hit)) + [None]:
        if i is not None and (not run or sig.t[i] - sig.t[run[-1]] <= cfg.max_gap):
            run.append(i)
            continue
        if len(run) >= cfg.min_samples:
            flashes.append(
                Flash(
                    track_id=sig.track_id,
                    start=float(sig.t[run[0]]),
                    end=float(sig.t[run[-1]]),
                    area=float(np.max(sig.light_area[run])),
                    ring=float(np.max(sig.light_ring[run])),
                )
            )
        run = [i] if i is not None else []
    return flashes


def attach_flashes(cycles: List[Cycle], flashes: List[Flash], cfg: FlashConfig) -> None:
    """Отмечает валидные циклы, которые начались не позже attach_after после вспышки
    и закончились не раньше attach_before до неё."""
    for c in cycles:
        if not c.valid:
            continue
        near = [f for f in flashes if c.start <= f.end + cfg.attach_after and c.end >= f.start - cfg.attach_before]
        if near:
            c.flash = True
            c.flash_area = max(f.area for f in near)
