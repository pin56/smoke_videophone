#!/usr/bin/env python
"""Метрики по протоколу кейса: TP/FP/FN, precision, recall, F1, FP/час, задержка + разбор ошибок.

    python evaluate.py --pred runs/ref/predictions.csv --gt data/ref_gt.csv \
        --durations runs/ref/processing_log.csv --out runs/ref/report
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from smokedet.evaluation import EvalParams, evaluate, load_durations, load_rows, parse_items, summary_table


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred", required=True, help="predictions.csv / .json")
    p.add_argument("--gt", required=True, help="эталон: CSV/JSON с label POSITIVE|NEGATIVE|IGNORE")
    p.add_argument("--durations", help="processing_log.csv — длительности клипов для FP/час")
    p.add_argument("--out", help="папка для report.json и matches.csv")
    p.add_argument("--min-iou", type=float, default=0.30)
    p.add_argument("--expand", type=float, default=3.0, help="расширение эталона, с")
    p.add_argument("--min-overlap", type=float, default=1.0, help="минимальное пересечение, с")
    args = p.parse_args(argv)

    preds = parse_items(load_rows(args.pred))
    gts = parse_items(load_rows(args.gt))
    params = EvalParams(gt_expand=args.expand, min_overlap=args.min_overlap, min_iou=args.min_iou)
    report = evaluate(preds, gts, load_durations(args.durations), params)
    print(summary_table(report))

    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        matches = report.pop("matches")
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if matches:
            with open(out / "matches.csv", "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(matches[0]))
                w.writeheader()
                w.writerows(matches)
            print(f"Отчёт: {out / 'report.json'}, сопоставления: {out / 'matches.csv'}")
        else:
            (out / "matches.csv").unlink(missing_ok=True)  # не оставлять таблицу прошлого запуска
            print(f"Отчёт: {out / 'report.json'} (сопоставлений нет)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
