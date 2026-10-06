#!/usr/bin/env python
"""Одна команда запуска: видео → predictions.csv/json + manifest.json + журнал времени [+ ролик].

Примеры:
    python run.py --input data/clip.mp4 --camera-id cam_fountain --roi data/roi_fountain.json --out runs/demo --render
    python run.py --inputs data/hidden.csv --out runs/final
    python run.py --inputs data/ref.csv --out runs/ref --set episode.confidence_threshold=0.6
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from smokedet.config import load_config
from smokedet.export import write_manifest, write_predictions, write_processing_log
from smokedet.pipeline import ROOT, ClipInput, Pipeline, read_inputs_csv


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
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
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
    pipeline = Pipeline(cfg)
    all_rows, log_rows = [], []
    for clip in clips:
        rows, log_row, _ = pipeline.process(clip, out_dir, render=args.render, use_cache=not args.no_cache)
        all_rows.extend(rows)
        log_rows.append(log_row)
        # пишем после каждого клипа: длинный прогон можно прервать, не потеряв готовое
        write_predictions(all_rows, out_dir)
        write_processing_log(log_rows, out_dir)

    verifier = pipeline.verifier.describe() if pipeline.verifier else None
    write_manifest(out_dir, cfg, pipeline.device, verifier, ROOT)
    total = sum(r["duration_sec"] for r in log_rows)
    wall = sum(r["total_sec"] for r in log_rows)
    logging.info("Готово: %d клипов, %.1f с видео за %.1f с, событий %d → %s", len(clips), total, wall,
                 len(all_rows), out_dir / "predictions.csv")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
