#!/usr/bin/env python
"""Перебор параметров лёгкой стадии с оценкой по разметке пользователя (rasmetka_gt.csv).

    python tools/tune_markup.py --inputs runs/goal/all.csv --cache runs/goal/cache \
        [--gt rasmetka_gt.csv] [--set section.key=value ...] [--grid section.key=v1,v2 ...] [--diag] [--out <папка>]

Треки берутся из кэша (или считаются и кладутся в него), затем для каждой точки сетки вызывается
Pipeline.analyze и предсказания оцениваются как в tools/eval_markup.py (только по времени, без рамок).
Кэши верификатора, пара и appearance живут в памяти процесса, поэтому вторая и следующие точки сетки
считаются быстро. --diag печатает FN/FP лучшей точки с причинами отказа циклов; --out пишет debug-JSON
лучшей точки (для eval_markup.py) и таблицу сетки.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from eval_markup import RunClip, evaluate_run, load_gt  # noqa: E402
from match_markup import canonical_stem  # noqa: E402
from smokedet.config import load_config, with_overrides  # noqa: E402
from smokedet.export import event_records, write_debug  # noqa: E402
from smokedet.pipeline import Pipeline, read_inputs_csv  # noqa: E402

DEFAULT_CONFIG = str(ROOT / "configs" / "default.yaml")


def parse_grid(items: List[str]) -> List[Tuple[str, List[str]]]:
    axes = []
    for it in items:
        key, _, vals = it.partition("=")
        if not vals:
            raise SystemExit(f"--grid {it}: нужен вид section.key=v1,v2")
        axes.append((key.strip(), [v.strip() for v in vals.split(";" if ";" in vals else ",")]))
    return axes


def _f1(tp: int, fp: int, fn: int) -> float:
    return 2 * tp / (2 * tp + fp + fn) if tp else 0.0


def leave_camera_out(per_clip: List[Dict[str, Tuple[int, int, int]]], camera: Dict[str, str], results: List[dict]) -> str:
    """Перекрёстная проверка «без камеры»: точка выбирается по остальным камерам и считается на отложенной.
    Итог — F1 по сумме отложенных счётов; разница с F1 лучшей точки — оптимизм подбора на этих данных."""
    cams = sorted({camera.get(stem, stem) for pc in per_clip for stem in pc})
    tp = fp = fn = 0
    picks = []
    for cam in cams:
        def score(i, inside):
            s = [v for stem, v in per_clip[i].items() if (camera.get(stem, stem) == cam) == inside]
            return tuple(sum(x[k] for x in s) for k in range(3))

        best = max(range(len(per_clip)), key=lambda i: (_f1(*score(i, False)), -i))
        t, p, n = score(best, True)
        tp, fp, fn = tp + t, fp + p, fn + n
        picks.append(best)
    common = max(set(picks), key=picks.count)
    return (f"без камеры ({len(cams)} камер): F1 {_f1(tp, fp, fn):.3f} TP {tp} FP {fp} FN {fn}; "
            f"чаще всего выбрана точка {common + 1} ({picks.count(common)}/{len(cams)}): {results[common]['params']}")


def debug_dict(analysis) -> dict:
    return {
        "events": [e.to_dict() for e in analysis.events],
        "cycles": [c.to_dict() for cs in analysis.cycles.values() for c in cs],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True, help="CSV клипов (video_path,camera_id,clip_id)")
    p.add_argument("--gt", default=str(ROOT / "rasmetka_gt.csv"))
    p.add_argument("--cache", required=True, help="папка кэша треков")
    p.add_argument("--config", default=DEFAULT_CONFIG)
    p.add_argument("--set", action="append", default=[])
    p.add_argument("--grid", action="append", default=[], metavar="SECTION.KEY=V1,V2",
                   help="значения через запятую; для списков — через ';' (элементы списка — через '|')")
    p.add_argument("--diag", action="store_true", help="печатать FN/FP лучшей точки")
    p.add_argument("--out", help="папка: grid.csv и debug/ лучшей точки")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    base = load_config(args.config, args.set)
    axes = parse_grid(args.grid)
    points = list(itertools.product(*[vals for _, vals in axes])) or [()]

    def overrides(point) -> List[str]:
        return [f"{k}={v.replace('|', ',')}" for (k, _), v in zip(axes, point)]

    for pt in points:  # опечатки в ключах — до трекинга
        with_overrides(base, overrides(pt))

    gt = load_gt(args.gt)
    clips = read_inputs_csv(args.inputs)
    pipeline = Pipeline(base, progress=False)
    t0 = time.time()
    tracksets = [pipeline.track(c, Path(args.cache)) for c in clips]
    print(f"треки: {len(clips)} клипов за {time.time() - t0:.0f} с", file=sys.stderr)

    camera = {}  # stem видео -> camera_id
    for c in clips:
        camera[canonical_stem(Path(c.video_path.replace("\\", "/")).stem)] = c.camera_id
    results = []
    per_clip: List[Dict[str, Tuple[int, int, int]]] = []  # точка -> stem -> (TP, FP, FN)
    best = None
    for n, pt in enumerate(points, 1):
        cfg = with_overrides(base, overrides(pt))
        t1 = time.time()
        run: Dict[str, RunClip] = {}
        debug = {}
        for c, ts in zip(clips, tracksets):
            an = pipeline.analyze(c, ts, cfg)
            vp = Path(c.video_path.replace("\\", "/"))
            rc = RunClip(canonical_stem(vp.stem), c.clip_id, vp.parent.name, ts.meta.duration)
            rc.preds = [{k: str(v) for k, v in r.items()} for r in event_records(an.events, c.camera_id, c.clip_id)]
            run[rc.stem] = rc
            debug[c.clip_id] = (an, debug_dict(an))
        res = evaluate_run(gt, run)
        tot = res["total"]
        label = " ".join(overrides(pt)) or "база"
        print(f"[{n}/{len(points)}] F1 {tot['f1']:.3f} P {tot['precision']:.3f} R {tot['recall']:.3f} "
              f"TP {tot['tp']} FP {tot['fp']} FN {tot['fn']} dup {tot['dup']}  ({time.time() - t1:.0f} с)  {label}", flush=True)  # fmt: skip
        results.append({"params": label, **{k: tot[k] for k in ("f1", "precision", "recall", "tp", "fp", "fn", "dup")}})
        per_clip.append({r["video"]: (r["TP"], r["FP"], r["FN"]) for r in res["per_clip"]})
        if best is None or tot["f1"] > best[0]["total"]["f1"]:
            best = (res, run, debug, label)

    res, run, debug, label = best
    print(f"\nлучшая точка: {label} — F1 {res['total']['f1']:.3f}")
    if len(points) > 1:
        print(leave_camera_out(per_clip, camera, results))
    if args.diag:
        for m in res["overall"]["matches"]:
            if m["result"] not in ("FN", "FP"):
                continue
            from eval_markup import diagnose_fn

            clip_id = run[m["clip_id"]].clip_id
            if m["result"] == "FN":
                note = diagnose_fn(debug[clip_id][1], m["gt_start"], m["gt_end"], 3.0)
                print(f"FN {m['folder']}/{m['clip_id']} {m['gt_start']:g}–{m['gt_end']:g}: {note}")
            else:
                print(f"FP {m['folder']}/{m['clip_id']} {m['pred_start']:.1f}–{m['pred_end']:.1f} "
                      f"conf {m.get('pred_conf', '')}: {m.get('note', '')}")  # fmt: skip
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "grid.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0]))
            w.writeheader()
            w.writerows(results)
        for clip_id, (an, _) in debug.items():
            write_debug(out, clip_id, an.cycles, an.events, an.flashes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
