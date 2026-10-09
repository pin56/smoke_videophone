"""Строки выхода потокового режима по клипу (run_rt.py, serve.py): события → predictions, processing_log,
realtime_log и debug в формате run.py (export.PRED_COLUMNS/LOG_COLUMNS), поэтому работают evaluate.py и
tools/eval_markup.py."""
from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import List, Optional

from .config import Config
from .export import LOG_COLUMNS, event_records, write_debug
from .pipeline import ClipInput
from .realtime import STAGES, PersonJob, RtRun, analyze
from .roi import Roi

RT_COLUMNS = [
    "camera_id", "clip_id", "duration_sec", "wall_sec", "load", "cpu_sec", "cpu_cores", "from_cache",
    *[f"{s}_sec" for s in STAGES], *[f"{s}_load" for s in STAGES], "analysis_sec",
    "decoded_frames", "sparse_frames", "dense_frames", "people", "crops_sparse", "crops_dense", "triggers",
    "jobs", "jobs_reopened", "max_parallel", "dropped_triggers", "scored_cycles", "cycles", "valid_cycles", "events",
    "alerts", "alerts_withdrawn", "alert_latency_sec", "skipped_cycles", "alert_holds",
]  # fmt: skip


def clip_outputs(run: RtRun, jobs: Optional[List[PersonJob]], cfg: Config, clip: ClipInput, out_dir: Path, device: str,
                 scorer=None, from_cache: bool = False):  # fmt: skip
    """События клипа по итогу стадий 1–3 (jobs — задания из потока, None — повтор из кэша) → строки
    predictions/processing_log/realtime_log; debug пишется сразу."""
    t0 = time.perf_counter()
    roi = Roi.load(clip.roi_path, run.meta.width, run.meta.height)
    cycles, signals, events = analyze(run, cfg, roi, jobs, scorer, clip.clip_id, clip.video_path)
    analysis_sec = time.perf_counter() - t0
    records = event_records(events, clip.camera_id, clip.clip_id)
    write_debug(out_dir, clip.clip_id, cycles, events)
    dbg = out_dir / "debug" / f"{clip.clip_id}_rt.json"
    dbg.write_text(json.dumps({"alerts": run.alerts, "stats": run.stats, "timings": run.timings}, ensure_ascii=False, indent=1),
                   encoding="utf-8")  # fmt: skip

    meta, st, tm = run.meta, run.stats, run.timings
    dur = meta.duration
    wall = run.wall_time
    n_cycles = sum(len(cs) for cs in cycles.values())
    n_valid = sum(c.valid for cs in cycles.values() for c in cs)
    log_row = {
        "camera_id": clip.camera_id, "clip_id": clip.clip_id, "video_path": clip.video_path,
        "start_time": clip.start_time or "", "duration_sec": round(dur, 3), "fps": round(meta.fps, 3),
        "width": meta.width, "height": meta.height,
        "processed_frames": st.get("sparse_frames", 0) + st.get("dense_frames", 0),
        "tracking_sec": round(wall, 2), "analysis_sec": round(analysis_sec, 2), "render_sec": 0.0,
        "total_sec": round(wall + analysis_sec, 2),
        # как у run.py: во сколько раз быстрее реального времени (загрузка — в realtime_log.csv)
        "realtime_factor": round(dur / (wall + analysis_sec), 2) if wall + analysis_sec > 0 else "",
        "events": len(records), "device": device, "from_cache": int(from_cache),
        "decode_wait_sec": round(tm.get("decode_wait", 0.0), 2), "pose_sec": round(tm.get("pose_sparse", 0.0) + tm.get("pose_dense", 0.0), 2),
        "appearance_sec": round(tm.get("objects", 0.0), 2), "signals_sec": round(tm.get("cycles", 0.0), 2),
    }  # fmt: skip
    lat = [a["latency"] for a in run.alerts if not a.get("withdrawn")]
    rt_row = {
        "camera_id": clip.camera_id, "clip_id": clip.clip_id, "duration_sec": round(dur, 2), "wall_sec": round(wall, 2),
        "load": round(wall / dur, 3) if dur else "", "cpu_sec": st.get("cpu_sec", ""),
        "cpu_cores": round(st["cpu_sec"] / dur, 3) if dur and "cpu_sec" in st else "", "from_cache": int(from_cache), "analysis_sec": round(analysis_sec, 2),
        **{f"{s}_sec": round(tm.get(s, 0.0), 2) for s in STAGES},
        **{f"{s}_load": round(tm.get(s, 0.0) / dur, 4) if dur else "" for s in STAGES},
        **{k: st.get(k, 0) for k in ("decoded_frames", "sparse_frames", "dense_frames", "people", "crops_sparse",
                                      "crops_dense", "triggers", "jobs", "jobs_reopened", "max_parallel",
                                      "dropped_triggers", "scored_cycles")},
        "cycles": n_cycles, "valid_cycles": n_valid, "events": len(records), "alerts": len(lat),
        "alerts_withdrawn": len(run.alerts) - len(lat),
        "alert_latency_sec": round(sum(lat) / len(lat), 2) if lat else "",
        "skipped_cycles": st.get("skipped_cycles", 0), "alert_holds": st.get("alert_holds", 0),
    }  # fmt: skip
    logging.info("[%s] %.1f с видео за %.1f с (загрузка %.2f), заданий %d, циклов %d (валидных %d), событий %d",
                 clip.clip_id, dur, wall, wall / dur if dur else 0, st.get("jobs", 0), n_cycles, n_valid, len(records))  # fmt: skip
    return records, {k: log_row.get(k, "") for k in LOG_COLUMNS}, rt_row


def write_rt_log(rows, out_dir: Path) -> None:
    with open(out_dir / "realtime_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RT_COLUMNS)
        w.writeheader()
        w.writerows(rows)
