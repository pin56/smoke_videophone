#!/usr/bin/env python
"""Скачать веса потокового режима из релиза GitHub в weights/ заранее (до площадки без интернета).

    python tools/fetch_weights.py                                   # все файлы релиза (GPU и CPU)
    python tools/fetch_weights.py --config configs/serve_cpu.yaml   # только то, что нужно профилю

run_rt.py и serve.py докачивают недостающее сами (smokedet/weights.py). Репозиторий закрытый: нужен
GH_TOKEN / GITHUB_TOKEN или `gh auth login`.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smokedet.pipeline import ROOT  # noqa: E402
from smokedet.weights import RELEASE, TAG, ensure, needed  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=f"Скачать веса из релиза {TAG}")
    p.add_argument("--config", help="профиль (configs/serve_gpu.yaml, configs/serve_cpu.yaml); без него — все файлы")
    p.add_argument("--device", help="cpu | 0 (по умолчанию — detector.device профиля)")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if args.config:
        from smokedet.realtime import load_rt_config
        from smokedet.tracking import select_device

        cfg = load_rt_config(args.config, args.set)
        paths = needed(cfg, args.device or select_device(cfg.detector.device))
    else:
        paths = [ROOT / "weights" / name for name in RELEASE]
    got = ensure(paths)
    for path in paths:
        print(("скачан " if path in got else "есть   ") + str(path) if path.exists() else f"нет    {path}")
    return 0 if all(path.exists() for path in paths) else 1


if __name__ == "__main__":
    sys.exit(main())
