"""Источник кадров камеры: поток чтения одного открытия потока или файла (сегмент сервера).

Все кадры декодируются (grab), в очередь попадают кадры плотной сетки (dense_fps потока) и сетки детектора —
её частота sparse_fps() читается на каждом кадре (ступень камеры меняет realtime.sparse_fps на лету). У кадра
есть arrival — время прихода по time.monotonic: у живого потока — момент чтения, у файла в темпе реального
времени (pace) — старт + (t − t0), то есть когда кадр пришёл бы с камеры, даже если декодер отстал. Без pace
файл читается так быстро, как его забирают (полная очередь ждёт): замер пропускной способности без потерь.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional

import cv2

from ..io_video import FrameClock, VideoMeta
from ..realtime import StreamFrame, Ticker

log = logging.getLogger(__name__)


@dataclass
class Arrived:
    f: StreamFrame
    arrival: float  # time.monotonic() прихода кадра


def open_capture(url: str, threads: int = 1) -> cv2.VideoCapture:
    params = [cv2.CAP_PROP_N_THREADS, threads] if threads > 0 else []
    cap = cv2.VideoCapture(str(url), cv2.CAP_FFMPEG, params)
    if not cap.isOpened():
        cap = cv2.VideoCapture(str(url))  # не FFmpeg (камера по индексу, другой бэкенд)
    if not cap.isOpened():
        raise IOError(f"Не удалось открыть поток: {url}")
    return cap


def capture_meta(cap: cv2.VideoCapture, url: str, fallback_fps: float) -> VideoMeta:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1 or fps > 240:
        fps = fallback_fps
    return VideoMeta(path=str(url), fps=float(fps), frame_count=max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))),
                     width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))  # fmt: skip


class CameraSource:
    """Открывает поток в конструкторе (IOError — не открылся), читает в своём потоке после start()."""

    def __init__(self, url: str, dense_fps: float, sparse_fps: Callable[[], float], queue_sec: float = 20.0,
                 pace: bool = False, fallback_fps: float = 25.0, threads: int = 1, name: str = "",
                 clock: Callable[[], float] = time.monotonic, cap=None):  # fmt: skip
        self.url, self.pace, self.clock = url, pace, clock
        self.cap = cap if cap is not None else open_capture(url, threads)
        self.meta = capture_meta(self.cap, url, fallback_fps)
        self.dense_fps, self.sparse_fps = dense_fps, sparse_fps
        self.max_frames = max(2, int(queue_sec * dense_fps) + 1)
        self.blocking = not pace and not _is_stream(url)  # файл без pace: полная очередь ждёт, кадры не теряются
        self.q: Deque[Arrived] = deque()
        self.cv = threading.Condition()
        self.done = False  # поток кончился (файл, обрыв) или остановлен
        self.error: Optional[str] = None
        self.stats = {"decoded_frames": 0, "synthetic_from": None, "dropped_queue": 0}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"src-{name or url}", daemon=True)

    def start(self) -> "CameraSource":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        with self.cv:
            self.cv.notify_all()
        self._thread.join(timeout=5)

    def get(self, timeout: Optional[float] = None) -> Optional[Arrived]:
        """Следующий кадр; None — поток кончился и очередь пуста или вышел timeout (смотреть self.done)."""
        with self.cv:
            if not self.q and not self.done:
                self.cv.wait(timeout)
            if not self.q:
                return None
            a = self.q.popleft()
            self.cv.notify_all()  # читатель без pace ждёт места
            return a

    def oldest_arrival(self) -> Optional[float]:
        with self.cv:
            return self.q[0].arrival if self.q else None

    def __len__(self) -> int:
        with self.cv:
            return len(self.q)

    def _put(self, a: Arrived) -> None:
        with self.cv:
            if self.blocking:
                while len(self.q) >= self.max_frames and not self._stop.is_set():
                    self.cv.wait(0.5)
            elif len(self.q) >= self.max_frames:  # отставание больше queue_sec: старый кадр плотной сетки — прочь
                i = next((k for k, x in enumerate(self.q) if not x.f.sparse), 0)
                del self.q[i]
                self.stats["dropped_queue"] += 1
            self.q.append(a)
            self.cv.notify_all()

    def _run(self) -> None:
        cap, clock = self.cap, FrameClock(self.meta.fps)
        dense, sparse = Ticker(1.0 / self.dense_fps), Ticker(1.0)
        idx, t0, wall0 = 0, None, None
        try:
            while not self._stop.is_set() and cap.grab():
                t = clock.time(idx, cap.get(cv2.CAP_PROP_POS_MSEC))
                idx += 1
                period = 1.0 / max(self.sparse_fps(), 1e-3)
                if abs(sparse.period - period) > 1e-9:
                    sparse.period = period
                is_dense, is_sparse = dense(t), sparse(t)
                if not (is_dense or is_sparse):
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    continue
                if t0 is None:
                    t0, wall0 = t, self.clock()
                if self.pace:
                    arrival = wall0 + (t - t0)
                    wait = arrival - self.clock()
                    if wait > 0 and self._stop.wait(wait):
                        break
                else:
                    arrival = self.clock()
                self._put(Arrived(StreamFrame(idx - 1, t, frame, is_sparse), arrival))
        except Exception as e:  # noqa: BLE001 — обрыв чтения закрывает сегмент, сервер переподключается
            self.error = f"{type(e).__name__}: {e}"
            log.warning("%s: чтение прервано: %s", self.url, self.error)
        finally:
            cap.release()
            self.stats["decoded_frames"] = idx
            self.stats["synthetic_from"] = clock.synthetic_from
            with self.cv:
                self.done = True
                self.cv.notify_all()


def _is_stream(url: str) -> bool:
    return "://" in str(url)
