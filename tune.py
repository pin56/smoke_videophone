#!/usr/bin/env python
"""Подбор параметров на ЭТАЛОННОМ наборе (на скрытом наборе до фиксации результата — запрещено).

Трекинг считается один раз (кэш), затем для каждой комбинации сетки пересчитываются сигналы,
циклы, верификатор (детекции кэшируются) и события, и считаются метрики по протоколу — по каждому клипу.

Выбирается не лучшая точка сетки, а лучшая по сглаженному F1 (среднее точки и её соседей по осям;
значения в --grid задавайте по возрастанию). Перекрёстная проверка (--cv) повторяет выбор без одного клипа
или без одной камеры и оценивает его на отложенных: разница с F1 на всех клипах — оценка переобучения.
Рекомендуемые параметры всё равно выбираются по всем клипам.

    python tune.py --inputs data/ref.csv --gt data/ref_gt.csv --out runs/tune \
        --grid episode.confidence_threshold=0.3,0.4,0.5,0.6 --grid cycle.near_enter=0.5,0.6,0.7

Результаты: tune_results.csv (все точки по порядку выбора), tune_cv.csv (фолды), tune_summary.json.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, List

from smokedet.config import load_config, with_overrides
from smokedet.evaluation import load_rows, parse_items
from smokedet.export import event_records
from smokedet.pipeline import ROOT, Pipeline, read_inputs_csv
from smokedet.tuning import (
    CV_MODES, ClipKey, Counts, base_point, clip_counts, cross_validate, cv_modes, cv_rows, explain_choice,
    format_cv, grid_points, make_folds, parse_grid, point_label, point_overrides, pool, rank_points, result_rows,
    summarize_cv,
)  # fmt: skip

log = logging.getLogger("tune")
DEFAULT_CONFIG = str(ROOT / "configs" / "default.yaml")


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:  # не оставлять таблицу прошлого запуска
        path.unlink(missing_ok=True)
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True, help="CSV клипов эталонного набора")
    p.add_argument("--gt", required=True, help="разметка эталонного набора")
    p.add_argument("--out", required=True)
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--set", action="append", default=[], help="фиксированные переопределения")
    p.add_argument("--grid", action="append", default=[], metavar="SECTION.KEY=V1,V2,...")
    p.add_argument("--cv", choices=CV_MODES, default="auto",
                   help="перекрёстная проверка: clip — без одного клипа, camera — без одной камеры; "
                        "auto — clip при >= 2 клипах и camera при >= 2 камерах (если у камеры больше одного клипа)")  # fmt: skip
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    t_start = time.time()

    base = load_config(args.config, args.set)
    axes = parse_grid(args.grid)
    for a in axes:  # опечатки в ключах — до трекинга
        for v in a.values:
            with_overrides(base, [f"{a.key}={v}"])

    clips = read_inputs_csv(args.inputs)
    keys: List[ClipKey] = [(c.camera_id, c.clip_id) for c in clips]
    if len({c.clip_id for c in clips}) != len(clips):
        raise SystemExit("clip_id в списке клипов должны быть уникальны: по ним кэшируются треки и детекции")
    gts = parse_items(load_rows(args.gt))
    known = set(keys)
    unknown = sorted({g.key for g in gts} - known)
    if unknown:
        log.warning("В эталоне есть клипы, которых нет в списке, они не учитываются: %s",
                    ", ".join(f"{cam}/{clip}" for cam, clip in unknown))  # fmt: skip
    gts = [g for g in gts if g.key in known]
    n_pos = sum(g.label == "POSITIVE" for g in gts)
    log.info("Клипов %d (камер %d), эталонных эпизодов %d", len(clips), len({k[0] for k in keys}), n_pos)

    out = Path(args.out)
    pipeline = Pipeline(base)
    tracksets = [pipeline.track(c, out / "cache") for c in clips]
    durations = {k: ts.meta.duration for k, ts in zip(keys, tracksets)}
    t_tracked = time.time()

    def run_config(cfg) -> Dict[ClipKey, Counts]:
        res = {}
        for c, k, ts in zip(clips, keys, tracksets):
            analysis = pipeline.analyze(c, ts, cfg)
            preds = parse_items(event_records(analysis.events, c.camera_id, c.clip_id))
            res[k] = clip_counts(preds, gts, k, durations[k])
        return res

    points = grid_points(axes)
    per_clip = {}
    for n, point in enumerate(points, 1):
        per_clip[point] = run_config(with_overrides(base, point_overrides(axes, point)))
        c = pool(per_clip[point].values())
        log.info("[%d/%d] %s — F1=%.3f P=%.3f R=%.3f FP=%d", n, len(points), point_label(axes, point), c.f1,
                 c.precision, c.recall, c.fp)  # fmt: skip

    bp = base_point(axes, base)
    if all(i is not None for i in bp):
        base_counts = pool(per_clip[bp].values())
    else:
        log.info("Базовый конфиг не входит в сетку — считаю его отдельно")
        base_counts = pool(run_config(base).values())

    ranked = rank_points(axes, {pt: pool(by_clip.values()) for pt, by_clip in per_clip.items()}, bp)
    chosen = ranked[0]
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "tune_results.csv", result_rows(axes, ranked))

    print("\nЛучшие точки (по сглаженному F1):")
    for r in ranked[:5]:
        c = r.counts
        print(f"  F1~={r.f1_smooth:.3f} F1={c.f1:.3f} P={c.precision:.3f} R={c.recall:.3f} FP={c.fp}  "
              f"{point_label(axes, r.point)}")  # fmt: skip
    reason = explain_choice(axes, ranked)
    print("\n" + reason)
    print(f"  базовый конфиг: F1={base_counts.f1:.3f} P={base_counts.precision:.3f} R={base_counts.recall:.3f} "
          f"FP={base_counts.fp}")  # fmt: skip

    summary = {
        "grid": {a.key: list(a.values) for a in axes},
        "clips": [f"{cam}/{clip}" for cam, clip in keys],
        "chosen": {"params": point_overrides(axes, chosen.point), "f1_smooth": round(chosen.f1_smooth, 4),
                   "n_smooth": chosen.n_smooth, **chosen.counts.to_dict()},  # fmt: skip
        "reason": reason,
        "base": base_counts.to_dict(),
        "cv": {},
    }
    cv_table = []
    for mode in cv_modes(keys, args.cv):
        folds = make_folds(keys, mode)
        if not folds:
            log.warning("Перекрёстная проверка «%s» невозможна: нужно минимум две группы клипов", mode)
            continue
        s = summarize_cv(axes, mode, cross_validate(axes, per_clip, folds, bp), chosen.point)
        print("\n" + format_cv(axes, s, chosen.counts, base_counts))
        summary["cv"][mode] = {"folds": len(s.folds), **s.pooled.to_dict(), "same_as_full": s.same_as_full,
                               "value_counts": {k: dict(v) for k, v in s.value_counts.items()}}  # fmt: skip
        cv_table.extend(cv_rows(axes, s))
    write_csv(out / "tune_cv.csv", cv_table)
    (out / "tune_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    fixed = ([f"--config {args.config}"] if args.config != DEFAULT_CONFIG else []) + [f"--set {o}" for o in args.set]
    best = fixed + [f"--set {o}" for o in point_overrides(axes, chosen.point)]
    if best:
        print("\nЗапуск с выбранными параметрами:\n  python run.py ... " + " ".join(best))
    t_end = time.time()
    print(f"\nВремя: модели и треки {t_tracked - t_start:.1f} с, анализ {len(points)} точек {t_end - t_tracked:.1f} с. "
          f"Результаты: {out / 'tune_results.csv'}, {out / 'tune_summary.json'}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
