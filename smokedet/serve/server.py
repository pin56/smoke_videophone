"""Сервер на много камер: поток на камеру с RealtimeSession, общие модели (Hub), регулятор нагрузки (Controller).

Камера читает кадры из CameraSource и шагает свою сессию; модели вызываются через Hub, поэтому запросы камер
собираются в батчи. Раз в serve.tick с регулятор смотрит отставание камер и загрузку, ставит давление G, а каждая
камера между кадрами применяет свою ступень (Ladder) к своему конфигу. Отставание камеры больше serve.lag_max —
камера выбрасывает кадры плотной сетки (кадры детектора — нет), пока оно не станет меньше serve.lag_high.

Сегмент — одно открытие потока (живая камера до обрыва) или клип эмуляции; по его концу события считаются тем
же analyze, что у run_rt.py, с базовым конфигом профиля, и строки пишутся в predictions.csv и журналы run_rt.py.
"""
from __future__ import annotations

import copy
import csv
import json
import logging
import os
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..config import Config
from ..export import write_manifest, write_predictions, write_processing_log
from ..pipeline import ROOT, ClipInput
from ..realtime import EPS, RealtimeSession, session_result
from ..roi import Roi
from ..rt_export import clip_outputs, write_rt_log
from .control import PRIO_ALERT, PRIO_EMPTY, PRIO_JOBS, PRIO_PEOPLE, Controller, Ladder
from .hub import Hub
from .source import CameraSource

log = logging.getLogger(__name__)

SERVE_COLUMNS = [
    "time_sec", "G", "util", "util_models", "util_cpu", "cameras", "lag_mean", "lag_max", "levels", "prios",
    "dropped_dense", "dropped_queue", "alerts", "pose_batch", "clip_batch", "detect_per_sec",
]  # fmt: skip


@dataclass
class CameraSpec:
    camera_id: str
    url: str = ""  # живой поток или файл (без эмуляции)
    roi: Optional[str] = None
    playlist: List[ClipInput] = field(default_factory=list)  # эмуляция: клипы по очереди, с выходом
    filler: List[ClipInput] = field(default_factory=list)  # эмуляция: после своих клипов — повтор без выхода


class Camera:
    def __init__(self, srv: "Server", spec: CameraSpec):
        self.srv, self.spec, self.id = srv, spec, spec.camera_id
        self.cfg = copy.deepcopy(srv.base)  # конфиг камеры: ступень меняет его поля на лету
        self.level = 0
        self.target = 0
        self.prio = PRIO_EMPTY
        self.state = "starting"
        self.src: Optional[CameraSource] = None
        self.busy_arrival: Optional[float] = None  # приход кадра, который сейчас обрабатывается
        self.dropping = False
        self.counts: Counter = Counter()
        self.real_done = threading.Event()  # свои сегменты (не наполнитель) закончены
        self.thread = threading.Thread(target=self._run, name=f"cam-{self.id}", daemon=True)

    # -- для регулятора (другой поток: читаются только простые поля)
    def lag(self, now: float) -> float:
        src, busy = self.src, self.busy_arrival
        oldest = src.oldest_arrival() if src is not None else None
        cands = [x for x in (busy, oldest) if x is not None]
        return max(0.0, now - min(cands)) if cands else 0.0

    @property
    def active(self) -> bool:
        return self.thread.is_alive() and self.state in ("online", "filler")

    # -- поток камеры
    def _run(self) -> None:
        try:
            if self.spec.playlist or self.spec.filler:
                self._emulate()
            else:
                self._live()
        except Exception:  # noqa: BLE001 — камера не роняет сервер
            log.exception("[%s] камера остановлена ошибкой", self.id)
        finally:
            self.state = "done"
            self.real_done.set()

    def _emulate(self) -> None:
        for clip in self.spec.playlist:
            if self.srv.stopping.is_set():
                break
            self._segment(clip.video_path, clip, real=True)
        self.real_done.set()
        i = 0
        while self.spec.filler and not self.srv.stopping.is_set() and not self.srv.stop_filler.is_set():
            c = self.spec.filler[i % len(self.spec.filler)]
            self._segment(c.video_path, c, real=False)
            i += 1

    def _live(self) -> None:
        sc, url = self.srv.base.serve, self.spec.url
        attempt = 0
        while not self.srv.stopping.is_set():
            stamp = datetime.now()
            clip = ClipInput(url, self.id, f"{self.id}_{stamp:%Y%m%d_%H%M%S}", stamp.isoformat(timespec="seconds"),
                             self.spec.roi)  # fmt: skip
            frames = self._segment(url, clip, real=True)
            if "://" not in url:
                break  # файл — один проход
            attempt = 0 if frames > 0 else attempt + 1
            delay = sc.reconnect[min(attempt, len(sc.reconnect) - 1)] if sc.reconnect else 1.0
            self.state = "offline"
            log.info("[%s] поток недоступен, переподключение через %.0f с", self.id, delay)
            self.srv.stopping.wait(delay)

    def _apply_level(self) -> None:
        if self.target != self.level:
            self.srv.ladder.apply(self.cfg, self.target)
            self.level = self.target

    def _priority(self, ses: RealtimeSession, t: float, last_people: float) -> int:
        gap = self.cfg.episode.merge_gap
        if any(st[2] and t - st[1] <= gap + EPS for st in ses._alerted.values()):
            return PRIO_ALERT
        if ses.active:
            return PRIO_JOBS
        if t - last_people <= self.srv.base.serve.people_hold:
            return PRIO_PEOPLE
        return PRIO_EMPTY

    def _segment(self, url: str, clip: ClipInput, real: bool) -> int:
        """Один сегмент: открыть, читать и шагать сессию до конца потока или остановки; число обработанных кадров."""
        srv, sc = self.srv, self.srv.base.serve
        base_dense = srv.base.realtime.dense_fps
        try:
            src = srv.source(url, base_dense, lambda: self.cfg.realtime.sparse_fps, sc.queue_sec, srv.pace,
                               srv.base.video.fallback_fps, sc.decode_threads, self.id)  # fmt: skip
        except IOError as e:
            self.state = "offline"
            log.warning("[%s] %s", self.id, e)
            return 0
        meta = src.meta
        srv.hub.warm_shape(meta.height, meta.width)
        roi = Roi.load(clip.roi_path or self.spec.roi, meta.width, meta.height)
        self._apply_level()
        ses = RealtimeSession(srv.hub.session_models(), self.cfg, meta, roi, srv.hub.encoder())
        ses.stream_fps = base_dense  # сетка потока — базовая частота, ступень меняет частоту заданий
        self.src = src
        self.state = "online" if real else "filler"
        start, n, n_alerts, last_t, last_people = time.perf_counter(), 0, 0, 0.0, -1e9
        src.start()
        try:
            while not srv.stopping.is_set() and (real or not srv.stop_filler.is_set()):
                a = src.get(timeout=0.5)
                if a is None:
                    if src.done:
                        break
                    continue
                self._apply_level()
                lag = time.monotonic() - a.arrival
                if not src.blocking:  # у файла без темпа отставание — очередь чтения впрок, а не задержка
                    if lag > sc.lag_max:
                        self.dropping = True
                    elif lag < sc.lag_high:
                        self.dropping = False
                if self.dropping and not a.f.sparse:
                    self.counts["dropped_dense"] += 1
                    continue
                self.busy_arrival = a.arrival
                people0 = ses.stats["people"]
                try:
                    ses.step(a.f)
                except Exception:  # noqa: BLE001 — ошибка модели закрывает сегмент
                    log.exception("[%s] ошибка обработки кадра %.1f с, сегмент закрыт", self.id, a.f.t)
                    self.counts["errors"] += 1
                    break
                finally:
                    self.busy_arrival = None
                n += 1
                last_t = a.f.t
                if ses.stats["people"] > people0:
                    last_people = a.f.t
                if len(ses.alerts) > n_alerts:
                    done = time.monotonic()
                    for rec in ses.alerts[n_alerts:]:
                        srv.on_alert(self, clip, rec, done - a.arrival + (a.f.t - rec["end"]), real)
                    n_alerts = len(ses.alerts)
                self.prio = self._priority(ses, a.f.t, last_people)
        finally:
            src.stop()
            self.counts["dropped_queue"] += src.stats["dropped_queue"]
            self.counts["decoded_frames"] += src.stats["decoded_frames"]
            self.src = None
            self.prio = PRIO_EMPTY
        try:
            ses.finish(last_t)
        except Exception:  # noqa: BLE001
            log.exception("[%s] ошибка при закрытии сегмента", self.id)
        for rec in ses.alerts[n_alerts:]:
            srv.on_alert(self, clip, rec, rec.get("latency", 0.0), real)  # после конца потока: отставания нет
        if real and n > 0:
            srv.on_segment(self, clip, ses, meta, src.stats, time.perf_counter() - start)
        return n


def assign_clips(clips: List[ClipInput], durations: Dict[str, float], n: int) -> List[List[ClipInput]]:
    """Клипы по камерам: самые длинные — первыми, каждый — камере с наименьшей суммой (LPT)."""
    cams: List[List[ClipInput]] = [[] for _ in range(n)]
    load = [0.0] * n
    for c in sorted(clips, key=lambda c: -durations.get(c.clip_id, 0.0)):
        i = min(range(n), key=lambda k: (load[k], len(cams[k])))
        cams[i].append(c)
        load[i] += durations.get(c.clip_id, 0.0)
    return cams


def emulated_specs(clips: List[ClipInput], durations: Dict[str, float], n: int) -> List[CameraSpec]:
    groups = assign_clips(clips, durations, min(n, len(clips)))
    specs = [CameraSpec(f"cam{i + 1:02d}", playlist=g, filler=list(g)) for i, g in enumerate(groups)]
    for i in range(len(groups), n):  # камер больше, чем клипов: только наполнитель
        specs.append(CameraSpec(f"cam{i + 1:02d}", filler=[clips[i % len(clips)]]))
    return specs


def percentiles(xs: List[float]) -> dict:
    if not xs:
        return {"n": 0}
    a = np.asarray(xs, dtype=float)
    return {"n": len(xs), "mean": round(float(a.mean()), 2), "p50": round(float(np.percentile(a, 50)), 2),
            "p95": round(float(np.percentile(a, 95)), 2), "max": round(float(a.max()), 2)}  # fmt: skip


class Server:
    def __init__(self, base: Config, specs: List[CameraSpec], out_dir: Path, models, pace: bool = True,
                 fixed_level: Optional[int] = None, label: str = "", source=CameraSource, keep_raw: bool = False):  # fmt: skip
        self.base, self.out, self.pace, self.fixed, self.label = base, out_dir, pace, fixed_level, label
        self.keep_raw = keep_raw  # писать serve_raw.json (процесс из --procs)
        self.source = source  # фабрика источника с сигнатурой CameraSource (тесты подставляют свой capture)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.ladder = Ladder(base)
        self.gpu = models.device != "cpu"
        self.ctrl = Controller(base.serve, self.ladder.size, boost=bool(self.ladder.boost))
        self.hub = Hub(models, base, self.gpu)
        self.cams = [Camera(self, s) for s in specs]
        for c in self.cams:
            c.target = 0 if fixed_level is None else fixed_level
        self.stopping = threading.Event()
        self.stop_filler = threading.Event()
        self.lock = threading.Lock()
        self.rows: List[dict] = []
        self.log_rows: List[dict] = []
        self.rt_rows: List[dict] = []
        self.alerts: List[dict] = []
        self.tick_lags: List[float] = []  # отставания всех активных камер во всех тиках
        self.level_time: Dict[int, float] = defaultdict(float)
        self.g_time: Dict[int, float] = defaultdict(float)
        self._alerts_f = open(out_dir / "alerts.jsonl", "w", encoding="utf-8")
        self._log_f = open(out_dir / "serve_log.csv", "w", newline="", encoding="utf-8")
        self._log_w = csv.DictWriter(self._log_f, fieldnames=SERVE_COLUMNS)
        self._log_w.writeheader()

    # -- события камер
    def on_alert(self, cam: Camera, clip: ClipInput, rec: dict, latency_wall: float, real: bool) -> None:
        item = {"wall": datetime.now().isoformat(timespec="milliseconds"), "camera": cam.id, "clip_id": clip.clip_id,
                "filler": not real, **rec, "level": cam.level, "G": self.ctrl.G}  # fmt: skip
        if not rec.get("withdrawn"):
            item["latency_wall"] = round(latency_wall, 2)
        with self.lock:
            self._alerts_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            self._alerts_f.flush()
            if real:
                self.alerts.append(item)
        if not rec.get("withdrawn"):
            log.info("[%s] ТРЕВОГА %s: человек %d, %.1f–%.1f с, задержка %.1f с (ступень %d)", cam.id, clip.clip_id,
                     rec["track_id"], rec["start"], rec["end"], latency_wall, cam.level)  # fmt: skip

    def on_segment(self, cam: Camera, clip: ClipInput, ses: RealtimeSession, meta, stream_stats: dict, wall: float) -> None:
        run = session_result(ses, meta, stream_stats, wall, 0.0)
        records, log_row, rt_row = clip_outputs(run, ses.jobs, self.base, clip, self.out, self.hub.models.device)
        with self.lock:
            self.rows += records
            self.log_rows.append(log_row)
            self.rt_rows.append(rt_row)
            write_predictions(self.rows, self.out)
            write_processing_log(self.log_rows, self.out)
            write_rt_log(self.rt_rows, self.out)
        log.info("[%s] сегмент %s: %.1f с видео, событий %d", cam.id, clip.clip_id, meta.duration, len(records))

    # -- цикл
    def run(self) -> dict:
        t_start, cpu0 = time.monotonic(), time.process_time()
        for c in self.cams:
            c.thread.start()
        prev, prev_cpu, prev_t = self.hub.snapshot(), cpu0, t_start
        ncpu = os.cpu_count() or 1
        try:
            while any(c.thread.is_alive() for c in self.cams):
                if all(c.real_done.is_set() for c in self.cams):
                    self.stop_filler.set()
                time.sleep(max(0.0, prev_t + self.base.serve.tick - time.monotonic()))
                now = time.monotonic()
                dt = now - prev_t
                cur, cpu = self.hub.snapshot(), time.process_time()
                util_m = self.hub.utilization(prev, cur, dt)
                util_cpu = min((cpu - prev_cpu) / (dt * ncpu), 1.0) if dt > 0 else 0.0
                self._tick(now - t_start, now, dt, max(util_m, util_cpu), util_m, util_cpu, prev, cur)
                prev, prev_cpu, prev_t = cur, cpu, now
        except KeyboardInterrupt:
            log.info("остановка: сегменты закрываются")
            self.stopping.set()
            for c in self.cams:
                c.thread.join(timeout=30)
        wall, cpu = time.monotonic() - t_start, time.process_time() - cpu0
        return self._finish(wall, cpu)

    def _tick(self, t: float, now: float, dt: float, util: float, util_m: float, util_cpu: float, prev, cur) -> None:
        cams = [c for c in self.cams if c.active]
        lags = [c.lag(now) for c in cams]
        lag_max = max(lags, default=0.0)
        if self.fixed is None:
            self.ctrl.update(now, lag_max, util)
            for c in cams:
                c.target = self.ctrl.level(c.prio)
        else:
            for c in cams:
                c.target = self.fixed
        self.tick_lags += lags
        for c in cams:
            self.level_time[c.level] += dt
        self.g_time[self.ctrl.G if self.fixed is None else self.fixed] += dt

        def avg_batch(k: str) -> str:
            if k not in cur:
                return ""
            b = cur[k].batches - prev[k].batches
            return round((cur[k].items - prev[k].items) / b, 2) if b else ""

        row = {
            "time_sec": round(t, 1), "G": self.ctrl.G, "util": round(util, 3), "util_models": round(util_m, 3),
            "util_cpu": round(util_cpu, 3), "cameras": len(cams),
            "lag_mean": round(float(np.mean(lags)), 2) if lags else 0.0, "lag_max": round(lag_max, 2),
            "levels": " ".join(f"{k}:{v}" for k, v in sorted(Counter(c.level for c in cams).items())),
            "prios": " ".join(f"{k}:{v}" for k, v in sorted(Counter(c.prio for c in cams).items())),
            "dropped_dense": sum(c.counts["dropped_dense"] for c in self.cams),
            "dropped_queue": sum(c.counts["dropped_queue"] + (c.src.stats["dropped_queue"] if c.src else 0) for c in self.cams),
            "alerts": len(self.alerts), "pose_batch": avg_batch("pose"), "clip_batch": avg_batch("clip"),
            "detect_per_sec": round((cur["detect"].items - prev["detect"].items) / dt, 1) if dt > 0 else "",
        }  # fmt: skip
        self._log_w.writerow(row)
        self._log_f.flush()

    def _finish(self, wall: float, cpu: float) -> dict:
        self.hub.close()
        self._alerts_f.close()
        self._log_f.close()
        cfg = self.base
        appearance = self.hub.models.appearance.describe(cfg.appearance) if cfg.realtime.objects else None
        write_manifest(self.out, cfg, self.hub.models.device, None, ROOT, appearance)
        raw = self.raw(wall, cpu)
        if self.keep_raw:
            (self.out / "serve_raw.json").write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
        summary = summarize(raw)
        (self.out / "serve_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
        log_summary(summary, self.out)
        return summary

    def raw(self, wall: float, cpu: float) -> dict:
        """Сырые итоги прогона: из них summarize считает serve_summary.json, а --procs — сводку по процессам."""
        cfg, st = self.base, self.hub.snapshot()
        counts = Counter()
        for c in self.cams:
            counts.update({k: c.counts[k] for k in ("dropped_dense", "dropped_queue", "decoded_frames", "errors")})
        return {
            "label": self.label, "mode": "gpu" if self.gpu else "cpu", "device": self.hub.models.device,
            "cameras": len(self.cams), "procs": 1, "pace": self.pace, "fixed_level": self.fixed,
            "workers": self.hub.workers, "ladder": cfg.serve.ladder, "boost": cfg.serve.boost,
            "wall": wall, "cpu": cpu, "video_sec": sum(r["duration_sec"] for r in self.rt_rows),
            "segments": len(self.rt_rows), "events": len(self.rows), "tick_lags": self.tick_lags,
            "alerts": self.alerts, "level_time": {str(k): v for k, v in self.level_time.items()},
            "g_time": {str(k): v for k, v in self.g_time.items()}, "counts": dict(counts),
            "executors": {k: {"batches": s.batches, "items": s.items, "busy": s.busy,
                              "workers": self.hub.execs[k].n_workers} for k, s in st.items()},
        }  # fmt: skip


def summarize(raw: dict) -> dict:
    wall = raw["wall"]
    cam_time = sum(raw["level_time"].values()) or 1.0
    g_total = sum(raw["g_time"].values()) or 1.0
    live = [a for a in raw["alerts"] if not a.get("withdrawn")]
    keys = ("label", "mode", "device", "cameras", "procs", "pace", "fixed_level", "workers", "ladder", "boost")
    return {
        **{k: raw[k] for k in keys},
        "wall_sec": round(wall, 1), "video_sec": round(raw["video_sec"], 1),
        "segments": raw["segments"], "events": raw["events"],
        "cpu_cores": round(raw["cpu"] / wall, 2) if wall else 0.0,
        "lag": percentiles(raw["tick_lags"]),
        "alert_latency_wall": percentiles([a["latency_wall"] for a in live]),
        "alert_latency_stream": percentiles([a["latency"] for a in live]),
        "alerts_withdrawn": len(raw["alerts"]) - len(live),
        "level_share": {k: round(v / cam_time, 3) for k, v in sorted(raw["level_time"].items(), key=lambda x: int(x[0]))},
        "G_share": {k: round(v / g_total, 3) for k, v in sorted(raw["g_time"].items(), key=lambda x: int(x[0]))},
        **{k: raw["counts"].get(k, 0) for k in ("dropped_dense", "dropped_queue", "decoded_frames", "errors")},
        "executors": {k: {"batches": e["batches"], "avg_items": round(e["items"] / e["batches"], 2) if e["batches"] else 0.0,
                          "busy_share": round(e["busy"] / (wall * e["workers"]), 3) if wall else 0.0}
                      for k, e in raw["executors"].items()},
    }  # fmt: skip


def log_summary(summary: dict, out: Path) -> None:
    log.info("Готово: %d камер, %.0f с, видео %.0f с, задержка тревоги p95 %s с, отставание p95 %s с → %s",
             summary["cameras"], summary["wall_sec"], summary["video_sec"], summary["alert_latency_wall"].get("p95", "—"),
             summary["lag"].get("p95", "—"), out)  # fmt: skip
