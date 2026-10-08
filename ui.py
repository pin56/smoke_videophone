#!/usr/bin/env python
"""Веб-интерфейс повтора анализа для показа: видео с разметкой и ход анализа синхронно с ним.

Клипы обрабатываются в фоне по очереди (треки берутся из кэша <out>/cache, если он есть — тогда клип готов
сразу), страница открывается в браузере. Сервер слушает только 127.0.0.1: видео не покидает машину.

Примеры:
    python ui.py --videos videos
    python ui.py --input videos/true/1.mp4 --camera-id cam1 --out runs/demo
    python ui.py --inputs data/demo.csv --out runs/check_true --port 8765
"""
from __future__ import annotations

import argparse
import logging
import sys
import webbrowser
from pathlib import Path

from smokedet.config import load_config
from smokedet.pipeline import ROOT, ClipInput, Pipeline, read_inputs_csv
from smokedet.ui.server import ClipStore, UiServer


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Веб-интерфейс повтора анализа")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--videos", help="папка с видео (рекурсивно): клип анализируется, когда его открывают")
    src.add_argument("--input", help="видеофайл")
    src.add_argument("--inputs", help="CSV со списком клипов: video_path,camera_id[,clip_id,start_time,roi]")
    p.add_argument("--camera-id", default="cam", help="camera_id для --input")
    p.add_argument("--clip-id", help="clip_id для --input (по умолчанию имя файла)")
    p.add_argument("--start-time", help="время начала файла, ISO 8601 (для --input)")
    p.add_argument("--roi", help="маска зоны интереса: PNG или JSON с полигонами (для --input)")
    p.add_argument("--out", default="runs/ui", help="папка результатов: кэш треков и прокси-видео")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="переопределить параметр")
    p.add_argument("--device", help="cpu | mps | 0 (по умолчанию auto)")
    p.add_argument("--no-cache", action="store_true", help="не использовать кэш треков")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="не открывать браузер")
    return p.parse_args(argv)


VIDEO_EXT = {".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm"}


def scan_videos(root: Path, camera_id: str = "cam") -> list:
    """Все видео папки (с вложенными): clip_id — путь относительно папки без расширения, камера — по умолчанию."""
    clips = []
    for f in sorted(root.rglob("*")):
        if f.is_file() and f.suffix.lower() in VIDEO_EXT:
            cid = f.relative_to(root).with_suffix("").as_posix().replace("/", "__")
            clips.append(ClipInput(video_path=str(f), camera_id=camera_id, clip_id=cid))
    return clips


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    overrides = list(args.set) + ([f"detector.device={args.device}"] if args.device else [])
    cfg = load_config(args.config, overrides)
    if args.videos:
        clips = scan_videos(Path(args.videos), args.camera_id)
        if not clips:
            sys.exit(f"В {args.videos} нет видео ({', '.join(sorted(VIDEO_EXT))})")
    elif args.inputs:
        clips = read_inputs_csv(args.inputs)
    else:
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
    store = ClipStore(clips, out_dir, lambda: Pipeline(cfg, progress=False), use_cache=not args.no_cache,
                      lazy=bool(args.videos))
    try:
        server = UiServer(("127.0.0.1", args.port), store)
    except OSError as e:
        sys.exit(f"Порт {args.port} занят ({e}): укажите другой через --port")
    store.start()
    url = f"http://127.0.0.1:{args.port}/"
    logging.info("Интерфейс: %s (Ctrl+C — выход)", url)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
