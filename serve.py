#!/usr/bin/env python
"""Сервер на много камер: потоковый режим run_rt.py для всех камер сразу, общие модели, регулятор нагрузки.

Режимы: --mode gpu (модели на CUDA, декодирование на CPU; профиль configs/serve_gpu.yaml) и --mode cpu (только
CPU через OpenVINO; configs/serve_cpu.yaml). Регулятор держит отставание обработки в serve.lag_high: при
перегрузке камеры без людей деградируют первыми (ступени serve.ladder), при запасе включается буст
(serve.boost). Подробности — smokedet/serve/server.py, docs/superpowers/specs/2026-10-09-multicam-serve-design.md.

Примеры:
    python serve.py --cameras cams.csv --mode gpu --out runs/serve            # cams.csv: camera_id,url[,roi]
    python serve.py --emulate 8 --inputs runs/rt/all.csv --mode cpu --out runs/serve/e8 --durations runs/cpu/l168/processing_log.csv
    python serve.py --emulate 4 --inputs runs/rt/all.csv --mode cpu --out runs/serve/x --no-pace   # пропускная способность
    python serve.py --cameras cams.csv --mode gpu --procs 4 --out runs/serve   # 30 камер: 4 процесса по 7–8 камер

Выход: predictions.csv/json, processing_log.csv, realtime_log.csv, debug/ (как у run_rt.py: tools/eval_markup.py),
alerts.jsonl (тревоги: камера, клип, задержка wall), serve_log.csv (тик регулятора), serve_summary.json (задержки,
отставание, доли ступеней, батчи).
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")  # декодер — 1 поток на камеру (и CAP_PROP_N_THREADS)

from smokedet.io_video import probe_video  # noqa: E402
from smokedet.pipeline import LOG_FORMAT, ROOT, read_inputs_csv  # noqa: E402
from smokedet.realtime import RealtimeDetector, limit_threads, load_rt_config  # noqa: E402
from smokedet.serve.multi import run_multi  # noqa: E402
from smokedet.serve.server import CameraSpec, Server, emulated_specs  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Детекция курения на многих камерах с регулятором нагрузки")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--cameras", help="CSV камер: camera_id,url[,roi] (url — RTSP/HTTP или файл)")
    src.add_argument("--emulate", type=int, metavar="N", help="эмуляция N камер клипами из --inputs")
    p.add_argument("--inputs", help="CSV клипов для --emulate: video_path,camera_id[,clip_id,start_time,roi]")
    p.add_argument("--durations", help="CSV с clip_id,duration_sec (processing_log.csv прогона): раздача клипов по длительности")
    p.add_argument("--mode", choices=("gpu", "cpu"), required=True)
    p.add_argument("--config", help="профиль (по умолчанию configs/serve_<mode>.yaml)")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="переопределить параметр")
    p.add_argument("--out", required=True, help="папка результатов")
    p.add_argument("--no-pace", action="store_true", help="файлы читать без темпа реального времени (замер пропускной способности)")
    p.add_argument("--level", type=int, help="постоянная ступень всех камер (регулятор выключен): -1 буст, 0 база, 1..K")
    p.add_argument("--procs", type=int, default=1, help="процессов с моделями: камеры делятся между ними (потолок одного "
                   "процесса — GIL; на CPU потоки OpenVINO делятся поровну)")  # fmt: skip
    return p.parse_args(argv)


def read_cameras(path: str):
    base = Path(path).resolve().parent
    specs = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            url = row["url"].strip()
            if "://" not in url and not Path(url).is_absolute():
                url = str(base / url)
            roi = (row.get("roi") or "").strip() or None
            if roi and not Path(roi).is_absolute():
                roi = str(base / roi)
            specs.append(CameraSpec(row["camera_id"].strip(), url=url, roi=roi))
    return specs


def read_durations(path: str):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["clip_id"]: float(r["duration_sec"]) for r in csv.DictReader(f) if r.get("duration_sec")}


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")
    config = args.config or str(ROOT / "configs" / f"serve_{args.mode}.yaml")
    overrides = list(args.set) + (["detector.device=cpu"] if args.mode == "cpu" else [])
    cfg = load_rt_config(config, overrides)
    limit_threads(cfg.realtime.threads, cfg.realtime.ov_precision)

    if args.emulate:
        if not args.inputs:
            sys.exit("--emulate требует --inputs")
        clips = read_inputs_csv(args.inputs)
        durations = read_durations(args.durations) if args.durations else {}
        for c in clips:
            if c.clip_id not in durations:
                durations[c.clip_id] = probe_video(c.video_path, cfg.video.fallback_fps).duration
        specs = emulated_specs(clips, durations, args.emulate)
        label = f"emulate {args.emulate} × {Path(args.inputs).name}"
    else:
        specs = read_cameras(args.cameras)
        label = Path(args.cameras).name
    if len({s.camera_id for s in specs}) != len(specs):
        sys.exit("camera_id должны быть уникальны")

    if args.no_pace and args.level is None:
        args.level = 0  # без темпа отставание не задержка: регулятор не работает
        logging.info("--no-pace: ступень 0 у всех камер (регулятор выключен)")
    if args.mode == "gpu":
        import torch

        if not torch.cuda.is_available():
            sys.exit("--mode gpu: CUDA недоступна")
    if args.procs > 1:
        run_multi(config, overrides, specs, Path(args.out), not args.no_pace, args.level, label, args.procs, args.mode)
        return 0
    models = RealtimeDetector(cfg)
    srv = Server(cfg, specs, Path(args.out), models, pace=not args.no_pace, fixed_level=args.level, label=label)
    srv.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
