"""Чтение видео: метаданные, проход с шагом, чтение отдельных кадров по индексу."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np


@dataclass
class VideoMeta:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


def _open(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Не удалось открыть видео: {path}")
    return cap


def probe_video(path: str, fallback_fps: float = 25.0) -> VideoMeta:
    cap = _open(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1 or fps > 240:
        fps = fallback_fps
    meta = VideoMeta(
        path=str(path),
        fps=float(fps),
        frame_count=max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    cap.release()
    return meta


def frame_stride(fps: float, target_fps: float) -> int:
    return max(1, int(round(fps / max(target_fps, 1e-6))))


class FrameClock:
    """Время кадра: pts из контейнера (CAP_PROP_POS_MSEC после read/grab), иначе idx/fps.

    В CCTV-файлах с переменным fps idx/fps уходит от реального времени (до 0.7 с на клип). pts считается
    негодным, если он не растёт, после первого кадра равен 0 или дальше MAX_DRIFT от idx/fps (потоки,
    битые контейнеры). Тогда до конца клипа время — idx/fps со сдвигом последнего годного pts, без скачка.
    """

    MAX_DRIFT = 2.0  # с

    def __init__(self, fps: float):
        self.fps = fps
        self.offset: Optional[float] = None  # не None — pts отброшен, время = idx/fps + offset
        self.last = -1.0

    def time(self, idx: int, pos_msec: float) -> float:
        nominal = idx / self.fps
        if self.offset is None:
            t = pos_msec / 1000.0
            if t > self.last and (t > 0 or idx == t == 0) and abs(t - nominal) <= self.MAX_DRIFT:
                self.last = t
                return t
            self.offset = max(self.last, 0.0) - (idx - 1) / self.fps if idx > 0 else 0.0
        t = max(nominal + self.offset, self.last + 1e-3)
        self.last = t
        return t


def iter_sampled_frames(path: str, stride: int, fps: float) -> Iterator[Tuple[int, float, np.ndarray]]:
    """Отдаёт (индекс кадра, время, с; кадр) для каждого stride-го кадра; пропущенные кадры только grab()."""
    cap = _open(path)
    clock = FrameClock(fps)
    idx = 0
    try:
        while True:
            if idx % stride == 0:
                ok, frame = cap.read()
                if not ok:
                    break
                yield idx, clock.time(idx, cap.get(cv2.CAP_PROP_POS_MSEC)), frame
            elif not cap.grab():
                break
            else:
                clock.time(idx, cap.get(cv2.CAP_PROP_POS_MSEC))  # следим за pts и на пропущенных кадрах
            idx += 1
    finally:
        cap.release()


class FrameReader:
    """Случайный доступ к кадрам. Близкие кадры дочитываются вперёд, дальние — через seek."""

    def __init__(self, path: str, max_forward: int = 90):
        self.cap = _open(path)
        self.pos = 0  # индекс следующего кадра, который вернёт read()
        self.max_forward = max_forward

    def get(self, idx: int) -> Optional[np.ndarray]:
        if idx < self.pos or idx - self.pos > self.max_forward:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            self.pos = idx
        while self.pos < idx:
            if not self.cap.grab():
                return None
            self.pos += 1
        ok, frame = self.cap.read()
        self.pos += 1
        return frame if ok else None

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> "FrameReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
