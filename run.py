#!/usr/bin/env python
"""Одна команда запуска: видео → predictions.csv/json + manifest.json + журнал времени [+ ролик].

Примеры:
    python run.py --input data/clip.mp4 --camera-id cam_fountain --roi data/roi_fountain.json --out runs/demo --render
    python run.py --inputs data/hidden.csv --out runs/final --workers 2
    python run.py --inputs data/ref.csv --out runs/ref --set episode.confidence_threshold=0.6
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from smokedet.config import load_config
from smokedet.export import write_manifest, write_predictions, write_processing_log
from smokedet.pipeline import LOG_FORMAT, ROOT, ClipInput, process_clips, read_inputs_csv
from smokedet.tracking import select_device


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Детекция эпизодов поведения, похожего на курение")
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
    p.add_argument("--render", action="store_true", help="сохранить ролик с отметками событий")
    p.add_argument("--no-cache", action="store_true", help="не использовать кэш треков")
    p.add_argument("--workers", type=int, default=1,
                   help="клипов параллельно, каждый в своём процессе со своими моделями (CUDA, 6 ГБ: 2)")  # fmt: skip
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")
    overrides = list(args.set) + ([f"detector.device={args.device}"] if args.device else [])
    cfg = load_config(args.config, overrides)

    if args.inputs:
        clips = read_inputs_csv(args.inputs)
    else:
        if not args.camera_id:
            sys.exit("--camera-id обязателен вместе с --input")
        clips = [
            ClipInput(
                video_path=args.input,
                camera_id=args.camera_id,
                clip_id=args.clip_id or Path(args.input).stem,
                start_time=args.start_time,
                roi_path=args.roi,
            )
        ]
    ids = [c.clip_id for c in clips]
    if len(set(ids)) != len(ids):
        sys.exit("clip_id должны быть уникальны")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    done, info, all_rows, log_rows = {}, {}, [], []
    for i, rows, log_row, info in process_clips(cfg, clips, out_dir, render=args.render, use_cache=not args.no_cache,
                                                workers=args.workers):  # fmt: skip
        done[i] = (rows, log_row)
        # пишем после каждого клипа, в порядке списка: длинный прогон можно прервать, не потеряв готовое
        all_rows = [r for k in sorted(done) for r in done[k][0]]
        log_rows = [done[k][1] for k in sorted(done)]
        write_predictions(all_rows, out_dir)
        write_processing_log(log_rows, out_dir)
        logging.info("Клипов готово: %d из %d", len(done), len(clips))

    write_manifest(out_dir, cfg, info.get("device") or select_device(cfg.detector.device), info.get("verifier"), ROOT,
                   info.get("appearance"))  # fmt: skip
    total = sum(r["duration_sec"] for r in log_rows)
    wall = time.time() - start
    logging.info("Готово: %d клипов, %.1f с видео за %.1f с (×%.2f реального времени), событий %d → %s", len(clips), total,
                 wall, total / wall if wall > 0 else 0.0, len(all_rows), out_dir / "predictions.csv")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
