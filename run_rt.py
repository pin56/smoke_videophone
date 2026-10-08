#!/usr/bin/env python
"""Потоковый (реалтайм) режим: видео или RTSP → predictions.csv/json + processing_log.csv + realtime_log.csv.

Кадр раз в секунду: лёгкий детектор людей → поза на кропах людей → у людей с рукой у рта плотная проверка
области (~5 кадр/с, с кольцевым буфером назад) → циклы «рука ко рту → пауза → отвод» → правило эпизода.
Стадия 4 (realtime.objects, по умолчанию включена; --no-objects — без неё) — CLIP по кропам «рот + кисть» в паузе.
Подробности — smokedet/realtime.py.

Примеры:
    python run_rt.py --input videos/true/1.mp4 --camera-id cam1 --out runs/rt/one
    python run_rt.py --inputs runs/rt/all.csv --out runs/rt/base [--no-objects] [--set realtime.trigger_dist=1.0]

Выход в том же формате, что у run.py (export.PRED_COLUMNS/LOG_COLUMNS), поэтому работают evaluate.py и
tools/eval_markup.py. realtime_log.csv — время по стадиям и загрузка (время обработки / длительность видео:
меньше 1 — успевает за реальным временем). Отсчёты стадий 1–3 кэшируются в <out>/cache: пороги циклов и
правила (cycle.*, episode.*, appearance.*) перебираются повтором без инференса.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from smokedet.config import Config
from smokedet.export import LOG_COLUMNS, event_records, write_debug, write_manifest, write_predictions, write_processing_log
from smokedet.pipeline import LOG_FORMAT, ROOT, ClipInput, read_inputs_csv
from smokedet.realtime import STAGES, RealtimeDetector, analyze, load_rt_config, load_run, rt_cache_key, save_run
from smokedet.roi import Roi

RT_COLUMNS = [
    "camera_id", "clip_id", "duration_sec", "wall_sec", "load", "from_cache",
    *[f"{s}_sec" for s in STAGES], *[f"{s}_load" for s in STAGES], "analysis_sec",
    "decoded_frames", "sparse_frames", "dense_frames", "people", "crops_sparse", "crops_dense", "triggers",
    "jobs", "jobs_reopened", "max_parallel", "dropped_triggers", "scored_cycles", "cycles", "valid_cycles", "events",
    "alerts", "alerts_withdrawn", "alert_latency_sec",
]  # fmt: skip


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Детекция курения в потоковом режиме (1 Гц + плотная проверка областей)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="видеофайл или RTSP-поток")
    src.add_argument("--inputs", help="CSV со списком клипов: video_path,camera_id[,clip_id,start_time,roi]")
    p.add_argument("--camera-id", help="camera_id для --input")
    p.add_argument("--clip-id", help="clip_id для --input (по умолчанию имя файла)")
    p.add_argument("--start-time", help="время начала файла, ISO 8601 (для --input)")
    p.add_argument("--roi", help="маска зоны интереса: PNG или JSON с полигонами (для --input)")
    p.add_argument("--out", required=True, help="папка результатов")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="переопределить параметр")
    p.add_argument("--device", help="cpu | mps | 0 (по умолчанию auto)")
    p.add_argument("--objects", action=argparse.BooleanOptionalAction, default=None,
                   help="стадия 4: CLIP по кропам «рот + кисть» (realtime.objects, по умолчанию — из конфига)")
    p.add_argument("--no-cache", action="store_true", help="не использовать кэш отсчётов стадий 1–3")
    p.add_argument("--cache", help="папка кэша отсчётов (по умолчанию <out>/cache): перебор порогов повтором другого прогона")
    return p.parse_args(argv)


def process_clip(det: RealtimeDetector, cfg: Config, clip: ClipInput, out_dir: Path, cache_dir: Optional[Path]):
    """Клип: стадии 1–3 (или повтор из кэша) → события → строки predictions/processing_log/realtime_log."""
    key = rt_cache_key(clip.video_path, cfg)
    cache_path = cache_dir / f"{clip.clip_id}_rt.pkl" if cache_dir else None
    run = load_run(cache_path, key) if cache_path else None
    jobs = None
    from_cache = run is not None
    if from_cache:
        logging.info("[%s] отсчёты из кэша %s", clip.clip_id, cache_path)
    else:
        run, jobs = det.run(clip.video_path, cfg, clip.roi_path)
        run.cache_key = key
        if cache_path:
            save_run(run, cache_path)
    t0 = time.perf_counter()
    roi = Roi.load(clip.roi_path, run.meta.width, run.meta.height)
    cycles, signals, events = analyze(run, cfg, roi, jobs, det.appearance, clip.clip_id, clip.video_path)
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
        "events": len(records), "device": det.device, "from_cache": int(from_cache),
        "decode_wait_sec": round(tm.get("decode_wait", 0.0), 2), "pose_sec": round(tm.get("pose_sparse", 0.0) + tm.get("pose_dense", 0.0), 2),
        "appearance_sec": round(tm.get("objects", 0.0), 2), "signals_sec": round(tm.get("cycles", 0.0), 2),
    }  # fmt: skip
    lat = [a["latency"] for a in run.alerts if not a.get("withdrawn")]
    rt_row = {
        "camera_id": clip.camera_id, "clip_id": clip.clip_id, "duration_sec": round(dur, 2), "wall_sec": round(wall, 2),
        "load": round(wall / dur, 3) if dur else "", "from_cache": int(from_cache), "analysis_sec": round(analysis_sec, 2),
        **{f"{s}_sec": round(tm.get(s, 0.0), 2) for s in STAGES},
        **{f"{s}_load": round(tm.get(s, 0.0) / dur, 4) if dur else "" for s in STAGES},
        **{k: st.get(k, 0) for k in ("decoded_frames", "sparse_frames", "dense_frames", "people", "crops_sparse",
                                      "crops_dense", "triggers", "jobs", "jobs_reopened", "max_parallel",
                                      "dropped_triggers", "scored_cycles")},
        "cycles": n_cycles, "valid_cycles": n_valid, "events": len(records), "alerts": len(lat),
        "alerts_withdrawn": len(run.alerts) - len(lat),
        "alert_latency_sec": round(sum(lat) / len(lat), 2) if lat else "",
    }  # fmt: skip
    logging.info("[%s] %.1f с видео за %.1f с (загрузка %.2f), заданий %d, циклов %d (валидных %d), событий %d",
                 clip.clip_id, dur, wall, wall / dur if dur else 0, st.get("jobs", 0), n_cycles, n_valid, len(records))  # fmt: skip
    return records, {k: log_row.get(k, "") for k in LOG_COLUMNS}, rt_row


def write_rt_log(rows, out_dir: Path) -> None:
    with open(out_dir / "realtime_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RT_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")
    overrides = list(args.set) + ([f"detector.device={args.device}"] if args.device else [])
    overrides += [] if args.objects is None else [f"realtime.objects={str(args.objects).lower()}"]
    cfg = load_rt_config(args.config, overrides)

    if args.inputs:
        clips = read_inputs_csv(args.inputs)
    else:
        if not args.camera_id:
            sys.exit("--camera-id обязателен вместе с --input")
        clips = [ClipInput(args.input, args.camera_id, args.clip_id or Path(args.input).stem, args.start_time, args.roi)]
    ids = [c.clip_id for c in clips]
    if len(set(ids)) != len(ids):
        sys.exit("clip_id должны быть уникальны")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    det = RealtimeDetector(cfg)
    all_rows, log_rows, rt_rows = [], [], []
    for i, clip in enumerate(clips, start=1):
        cache_dir = None if args.no_cache else Path(args.cache) if args.cache else out_dir / "cache"
        records, log_row, rt_row = process_clip(det, cfg, clip, out_dir, cache_dir)
        all_rows += records
        log_rows.append(log_row)
        rt_rows.append(rt_row)
        # после каждого клипа: длинный прогон можно прервать, не потеряв готовое
        write_predictions(all_rows, out_dir)
        write_processing_log(log_rows, out_dir)
        write_rt_log(rt_rows, out_dir)
        logging.info("Клипов готово: %d из %d", i, len(clips))

    appearance = det.appearance.describe(cfg.appearance) if cfg.realtime.objects else None
    write_manifest(out_dir, cfg, det.device, None, ROOT, appearance)
    dur = sum(r["duration_sec"] for r in rt_rows)
    wall = sum(r["wall_sec"] for r in rt_rows)
    logging.info("Готово: %d клипов, %.1f с видео, обработка %.1f с (загрузка %.2f), событий %d → %s", len(clips), dur, wall,
                 wall / dur if dur else 0.0, len(all_rows), out_dir / "predictions.csv")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
