#!/usr/bin/env python
"""Мониторинг камер: стена потоков с архивом, проверка загруженных видео, журнал эпизодов, хранение фрагментов.

Сервер слушает только 127.0.0.1: видео не покидает машину. Камеры — симуляция (видео объекта публиковать
нельзя), загруженные файлы анализирует пайплайн (`--analyzer pipeline`, нужны модели) или демо-анализ
(`--analyzer demo`: эпизоды генерируются и помечены в журнале как демо).
Примеры:
    python monitor.py
    python monitor.py --analyzer demo --retention-days 7 --port 8765
"""
from __future__ import annotations

import argparse
import logging
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from smokedet.monitor.analysis import AnalysisQueue, DemoAnalyzer, PipelineAnalyzer  # noqa: E402
from smokedet.monitor.cameras import CameraWall  # noqa: E402
from smokedet.monitor.journal import DAY, Journal, start_purger  # noqa: E402
from smokedet.monitor.server import MonitorServer  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Мониторинг камер и проверка видео")
    p.add_argument("--out", default="runs/monitor", help="журнал, загруженные файлы и фрагменты")
    p.add_argument("--analyzer", choices=["pipeline", "demo"], default="pipeline",
                   help="pipeline — настоящий анализ (нужны модели), demo — сгенерированные эпизоды для показа")
    p.add_argument("--retention-days", type=float, default=7.0, help="срок хранения фрагментов и загрузок")
    p.add_argument("--archive-min", type=float, default=30.0, help="сколько минут записи камер держать в архиве")
    p.add_argument("--prefill-min", type=float, default=10.0, help="история камер, подготовленная при запуске")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"), help="конфиг пайплайна")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE", help="параметр пайплайна")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true", help="не открывать браузер")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    out = Path(args.out)
    journal = Journal(out / "journal.sqlite", retention=args.retention_days * DAY)
    analyzer = (DemoAnalyzer() if args.analyzer == "demo"
                else PipelineAnalyzer(args.config, args.set, out))  # fmt: skip
    problem = analyzer.available()
    if problem:
        logging.warning("проверка видео: %s (задания завершатся ошибкой; для показа — --analyzer demo)", problem)
    analysis = AnalysisQueue(journal, analyzer, out / "uploads", out / "clips")
    wall = CameraWall(journal, out / "clips", archive_sec=args.archive_min * 60, prefill_sec=args.prefill_min * 60)
    try:
        server = MonitorServer(("127.0.0.1", args.port), journal, wall, analysis)
    except OSError as e:
        sys.exit(f"Порт {args.port} занят ({e}): укажите другой через --port")
    wall.start()
    analysis.start()
    start_purger(journal, interval=min(600.0, max(5.0, journal.retention / 20)))
    url = f"http://127.0.0.1:{args.port}/"
    logging.info("Мониторинг: %s (Ctrl+C — выход), анализ видео: %s", url, analyzer.name)
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
