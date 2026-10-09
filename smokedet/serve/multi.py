"""Сервер в нескольких процессах (serve.py --procs P): камеры делятся между процессами, у каждого свои модели,
Hub и регулятор. Один процесс упирается в GIL: потоки камер, трекер и обвязка ultralytics — Python, и на GPU
десятки камер в одном процессе ждут интерпретатор, а не модель. Регулятор каждого процесса видит своё отставание
— общий GPU или CPU перегружен у всех, и давление растёт везде.

Процесс k пишет свой выход в <out>/p<k> (serve_log.csv там же), родитель после завершения сливает predictions,
processing_log, realtime_log, debug/, alerts.jsonl и сводку (serve_summary.json по serve_raw.json процессов).
"""
from __future__ import annotations

import csv
import json
import logging
import multiprocessing as mp
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

from ..export import write_predictions, write_processing_log
from ..rt_export import write_rt_log
from .server import CameraSpec, log_summary, summarize

log = logging.getLogger(__name__)


def split_specs(specs: List[CameraSpec], procs: int) -> List[List[CameraSpec]]:
    """Камеры по процессам по очереди: у эмуляции камеры уже уравнены по длительности (LPT)."""
    procs = max(1, min(procs, len(specs)))
    return [specs[k::procs] for k in range(procs)]


def child_overrides(mode: str, procs: int, cpus: Optional[int] = None) -> List[str]:
    """На CPU OpenVINO каждого процесса считает своими все ядра: делим потоки и исполнителей между процессами."""
    if mode != "cpu" or procs <= 1:
        return []
    cpus = cpus or os.cpu_count() or 4
    return [f"realtime.threads={max(1, cpus // procs)}", f"serve.workers={max(1, cpus // 4 // procs)}"]


def _child(k: int, config: str, overrides: List[str], specs: List[CameraSpec], out: str, pace: bool,
           level: Optional[int], label: str) -> None:  # fmt: skip
    from ..pipeline import LOG_FORMAT
    from ..realtime import RealtimeDetector, limit_threads, load_rt_config
    from .server import Server

    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT.replace("%(message)s", f"[p{k}] %(message)s"),
                        datefmt="%H:%M:%S")  # fmt: skip
    cfg = load_rt_config(config, overrides)
    limit_threads(cfg.realtime.threads, cfg.realtime.ov_precision)
    srv = Server(cfg, specs, Path(out), RealtimeDetector(cfg), pace=pace, fixed_level=level, label=label, keep_raw=True)
    srv.run()


def run_multi(config: str, overrides: List[str], specs: List[CameraSpec], out: Path, pace: bool,
              level: Optional[int], label: str, procs: int, mode: str) -> dict:  # fmt: skip
    groups = split_specs(specs, procs)
    extra = child_overrides(mode, len(groups))
    out.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    parts = [out / f"p{k}" for k in range(len(groups))]
    ps = [ctx.Process(target=_child, name=f"serve-p{k}",
                      args=(k, config, list(overrides) + extra, g, str(parts[k]), pace, level, label))
          for k, g in enumerate(groups)]  # fmt: skip
    log.info("процессов %d: камер %s%s", len(ps), [len(g) for g in groups], f", {' '.join(extra)}" if extra else "")
    for p in ps:
        p.start()
    try:
        for p in ps:
            p.join()
    except KeyboardInterrupt:  # Ctrl+C получают и процессы: ждём, пока они закроют сегменты
        for p in ps:
            p.join()
    failed = [p.name for p in ps if p.exitcode != 0]
    if failed:
        log.error("процессы завершились с ошибкой: %s", failed)
    return merge_parts(out, [d for d in parts if (d / "serve_raw.json").exists()])


def merge_parts(out: Path, parts: List[Path]) -> dict:
    rows, logs, rts, alerts, raws = [], [], [], [], []
    for d in parts:
        rows += json.loads((d / "predictions.json").read_text(encoding="utf-8")) if (d / "predictions.json").exists() else []
        logs += _read_csv(d / "processing_log.csv")
        rts += _read_csv(d / "realtime_log.csv")
        with open(d / "alerts.jsonl", encoding="utf-8") as f:
            alerts += [json.loads(line) for line in f if line.strip()]
        raws.append(json.loads((d / "serve_raw.json").read_text(encoding="utf-8")))
        if (d / "debug").is_dir():
            (out / "debug").mkdir(exist_ok=True)
            for f in (d / "debug").iterdir():
                shutil.copy2(f, out / "debug" / f.name)
    if not raws:
        raise RuntimeError("ни один процесс не дописал итог")
    write_predictions(rows, out)
    write_processing_log(logs, out)
    write_rt_log(rts, out)
    alerts.sort(key=lambda a: a["wall"])
    with open(out / "alerts.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(a, ensure_ascii=False) + "\n" for a in alerts)
    shutil.copy2(parts[0] / "manifest.json", out / "manifest.json")
    summary = summarize(merge_raw(raws))
    (out / "serve_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    log_summary(summary, out)
    return summary


def merge_raw(raws: List[dict]) -> dict:
    """Сводка процессов как одного сервера: время — наибольшее, процессорное время и счётчики — сумма, занятость
    исполнителя — средняя по процессам (busy и число исполнителей складываются)."""
    m = {k: raws[0][k] for k in ("label", "mode", "device", "pace", "fixed_level", "workers", "ladder", "boost")}
    m.update(cameras=sum(r["cameras"] for r in raws), procs=len(raws), wall=max(r["wall"] for r in raws))
    for k in ("cpu", "video_sec", "segments", "events"):
        m[k] = sum(r[k] for r in raws)
    m["tick_lags"] = [x for r in raws for x in r["tick_lags"]]
    m["alerts"] = [a for r in raws for a in r["alerts"]]
    for k in ("level_time", "g_time", "counts"):
        acc = defaultdict(float)
        for r in raws:
            for key, v in r[k].items():
                acc[key] += v
        m[k] = {key: int(v) for key, v in acc.items()} if k == "counts" else dict(acc)
    ex = defaultdict(lambda: defaultdict(float))
    for r in raws:
        for name, e in r["executors"].items():
            for key, v in e.items():
                ex[name][key] += v
    m["executors"] = {k: {key: (v if key == "busy" else int(v)) for key, v in e.items()} for k, e in ex.items()}
    return m


def _read_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
