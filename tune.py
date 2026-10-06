#!/usr/bin/env python
"""Подбор параметров на ЭТАЛОННОМ наборе (на скрытом наборе до фиксации результата — запрещено).

Трекинг считается один раз (кэш), затем для каждой комбинации сетки пересчитываются сигналы,
циклы, верификатор (детекции кэшируются) и события, и считаются метрики по протоколу.

    python tune.py --inputs data/ref.csv --gt data/ref_gt.csv --out runs/tune \
        --grid episode.confidence_threshold=0.3,0.4,0.5,0.6 --grid cycle.near_enter=0.5,0.6,0.7
"""
from __future__ import annotations

import argparse
import csv
import itertools
import logging
from pathlib import Path

from smokedet.config import load_config, with_overrides
from smokedet.evaluation import evaluate, load_rows, parse_items
from smokedet.export import event_records
from smokedet.pipeline import ROOT, Pipeline, read_inputs_csv


def parse_grid(items):
    grid = []
    for item in items:
        key, values = item.split("=", 1)
        grid.append([f"{key}={v}" for v in values.split(",")])
    return grid


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True, help="CSV клипов эталонного набора")
    p.add_argument("--gt", required=True, help="разметка эталонного набора")
    p.add_argument("--out", required=True)
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--set", action="append", default=[], help="фиксированные переопределения")
    p.add_argument("--grid", action="append", default=[], metavar="SECTION.KEY=V1,V2,...")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    base = load_config(args.config, args.set)
    clips = read_inputs_csv(args.inputs)
    gts = parse_items(load_rows(args.gt))
    out = Path(args.out)
    pipeline = Pipeline(base)
    tracksets = {c.clip_id: pipeline.track(c, out / "cache") for c in clips}
    durations = {(c.camera_id, c.clip_id): tracksets[c.clip_id].meta.duration for c in clips}

    results = []
    combos = list(itertools.product(*parse_grid(args.grid))) or [()]
    for n, combo in enumerate(combos, 1):
        cfg = with_overrides(base, list(combo))
        rows = []
        for c in clips:
            analysis = pipeline.analyze(c, tracksets[c.clip_id], cfg)
            rows.extend(event_records(analysis.events, c.camera_id, c.clip_id))
        rep = evaluate(parse_items(rows), gts, durations)
        results.append({"params": " ".join(combo), "f1": rep["f1"], "precision": rep["precision"],
                        "recall": rep["recall"], "tp": rep["tp"], "fp": rep["fp"], "fn": rep["fn"],
                        "fp_per_hour": rep["fp_per_hour"]})  # fmt: skip
        logging.info("[%d/%d] %s → F1=%.3f P=%.3f R=%.3f", n, len(combos), " ".join(combo) or "(база)",
                     rep["f1"], rep["precision"], rep["recall"])  # fmt: skip

    results.sort(key=lambda r: (-r["f1"], r["fp_per_hour"] if r["fp_per_hour"] is not None else 0))
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "tune_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    print("\nЛучшие комбинации:")
    for r in results[:5]:
        print(f"  F1={r['f1']:.3f} P={r['precision']:.3f} R={r['recall']:.3f} FP={r['fp']}  {r['params']}")
    best = results[0]["params"].split()
    if best:
        print("\nЗапуск с лучшими параметрами:\n  python run.py ... " + " ".join(f"--set {b}" for b in best))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
