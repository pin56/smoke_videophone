"""Общие модели сервера: запросы всех камер к детектору, позе и CLIP собираются в батчи.

Поток камеры вызывает модель синхронно, как RealtimeSession в run_rt.py, и ждёт свой результат; исполнитель
модели берёт первый запрос из очереди и добирает к нему запросы других камер с тем же ключом (вид CLIP), пока
батч не наберёт max_items или не пройдёт wait_ms. Под нагрузкой очередь растёт, пока модель занята, и батчи
укрупняются сами. На GPU у модели один исполнитель, на CPU — несколько (OpenVINO с NUM_STREAMS, запрос на поток).
"""
from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Deque, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)


class _Req:
    __slots__ = ("items", "key", "done", "result", "error")

    def __init__(self, items: list, key: Any):
        self.items, self.key = items, key
        self.done = threading.Event()
        self.result: Optional[list] = None
        self.error: Optional[BaseException] = None


@dataclass
class ExecStats:
    busy: float = 0.0  # с в fn (сумма по исполнителям)
    batches: int = 0
    requests: int = 0
    items: int = 0


class BatchExecutor:
    """fn(items, key) -> список результатов той же длины; вызов executor(items, key) блокирует до результата,
    исключение fn получают все запросы батча."""

    def __init__(self, fn: Callable[[list, Any], list], name: str, max_items: int = 24, wait_ms: float = 0.0,
                 workers: int = 1):  # fmt: skip
        self.fn, self.name, self.max_items, self.wait = fn, name, max(1, max_items), wait_ms / 1000.0
        self.n_workers = max(1, workers)
        self.cv = threading.Condition()
        self.pending: Deque[_Req] = deque()
        self.closed = False
        self.stats = ExecStats()
        self._threads = [threading.Thread(target=self._loop, name=f"{name}-{i}", daemon=True) for i in range(self.n_workers)]
        for t in self._threads:
            t.start()

    def __call__(self, items: list, key: Any = None) -> list:
        if not items:
            return []
        req = _Req(list(items), key)
        with self.cv:
            if self.closed:
                raise RuntimeError(f"{self.name}: исполнитель остановлен")
            self.pending.append(req)
            self.cv.notify()
        req.done.wait()
        if req.error is not None:
            raise req.error
        return req.result

    def close(self) -> None:
        with self.cv:
            self.closed = True
            self.cv.notify_all()
        for t in self._threads:
            t.join(timeout=5)

    def _take(self) -> Optional[List[_Req]]:
        with self.cv:
            while not self.pending and not self.closed:
                self.cv.wait()
            if not self.pending:
                return None
            first = self.pending.popleft()
            batch, n = [first], len(first.items)
            deadline = time.monotonic() + self.wait
            while n < self.max_items:
                for r in list(self.pending):
                    if r.key == first.key and n + len(r.items) <= self.max_items:
                        self.pending.remove(r)
                        batch.append(r)
                        n += len(r.items)
                left = deadline - time.monotonic()
                if n >= self.max_items or left <= 0 or self.closed:
                    break
                self.cv.wait(left)
            return batch

    def _loop(self) -> None:
        while True:
            batch = self._take()
            if batch is None:
                return
            items = [x for r in batch for x in r.items]
            t0 = time.perf_counter()
            try:
                out = list(self.fn(items, batch[0].key))
                if len(out) != len(items):
                    raise RuntimeError(f"{self.name}: {len(out)} результатов на {len(items)} входов")
            except BaseException as e:  # noqa: BLE001 — ошибку получает каждая камера батча
                out, err = None, e
            else:
                err = None
            dt = time.perf_counter() - t0
            with self.cv:
                st = self.stats
                st.busy += dt
                st.batches += 1
                st.requests += len(batch)
                st.items += len(items)
            i = 0
            for r in batch:
                if err is None:
                    r.result = out[i : i + len(r.items)]
                    i += len(r.items)
                else:
                    r.error = err
                r.done.set()

    def snapshot(self) -> ExecStats:
        with self.cv:
            s = self.stats
            return ExecStats(s.busy, s.batches, s.requests, s.items)


def auto_workers(gpu: bool, workers: int) -> int:
    if workers > 0:
        return workers
    return 1 if gpu else max(1, (os.cpu_count() or 4) // 4)


class Hub:
    """Детектор, поза и CLIP из RealtimeDetector за исполнителями. session_models() — объект с detect/pose для
    RealtimeSession, encoder() — кодировщик CLIP с интерфейсом encoder(crops, view)."""

    def __init__(self, models, cfg, gpu: bool):
        from .. import ov_yolo

        sc = cfg.serve
        self.gpu = gpu
        self.workers = auto_workers(gpu, sc.workers)
        if not gpu:
            ov_yolo.STREAMS = self.workers  # до загрузки моделей: компиляция на столько потоков исполнения
        self.models = models
        detect, pose = models.detect, models.pose  # загрузка и прогрев
        enc = models.encoder(cfg)
        self.execs: Dict[str, BatchExecutor] = {
            "detect": BatchExecutor(lambda fs, _: [detect(f) for f in fs], "detect", 1, 0.0, self.workers),
            "pose": BatchExecutor(lambda items, _: pose(items), "pose", sc.max_batch, sc.batch_wait_ms, self.workers),
        }
        self._enc = enc
        self._warm = set()
        self._lock = threading.Lock()
        if enc is not None:
            self.execs["clip"] = BatchExecutor(lambda crops, view: list(enc(crops, view)), "clip", sc.max_batch,
                                               sc.batch_wait_ms, self.workers)  # fmt: skip
        log.info("модели сервера: %s, исполнителей на модель %d", "GPU" if gpu else "CPU", self.workers)

    def warm_shape(self, height: int, width: int) -> None:
        """Первый вызов детектора на новой форме входа долгий: до первого кадра камеры (через исполнитель —
        модель не вызывается из двух потоков сразу)."""
        shape = (max(height, 32), max(width, 32))
        with self._lock:
            if shape in self._warm:
                return
            self._warm.add(shape)
        self.execs["detect"]([np.zeros((*shape, 3), dtype=np.uint8)])

    def session_models(self) -> SimpleNamespace:
        det, pose = self.execs["detect"], self.execs["pose"]
        return SimpleNamespace(detect=lambda frame: det([frame])[0], pose=lambda items: pose(items))

    def encoder(self) -> Optional[Callable[[List[np.ndarray], int], np.ndarray]]:
        if self._enc is None:
            return None
        clip = self.execs["clip"]
        return lambda crops, view: np.asarray(clip(crops, view), dtype=np.float32)

    def snapshot(self) -> Dict[str, ExecStats]:
        return {k: e.snapshot() for k, e in self.execs.items()}

    def utilization(self, prev: Dict[str, ExecStats], cur: Dict[str, ExecStats], dt: float) -> float:
        """Загрузка моделей за dt с: на GPU модели делят одно устройство (сумма), на CPU — наибольшая по исполнителям."""
        if dt <= 0:
            return 0.0
        busy = {k: cur[k].busy - prev[k].busy for k in cur}
        if self.gpu:
            return min(sum(busy.values()) / dt, 1.0)
        return min(max(b / (dt * self.execs[k].n_workers) for k, b in busy.items()), 1.0)

    def close(self) -> None:
        for e in self.execs.values():
            e.close()
