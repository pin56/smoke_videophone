#!/usr/bin/env python
"""Перебор порогов потокового режима (run_rt.py) повтором из кэша отсчётов, с оценкой по разметке пользователя.

    python tools/tune_rt.py --inputs runs/rt/all.csv --cache runs/rt/base/cache \\
        --grid cycle.near_enter=0.6,0.8,1.0 --grid episode.min_cycles=1,2 [--set ...] [--no-objects]

Повтор проходит те же шаги фиксации циклов, что и поток (realtime.replay_job), поэтому меняться могут только
пороги сигнала, циклов, правила и стадии 4 (cycle.*, episode.*, appearance.*, realtime.finalize_window); пороги
стадий 1–3 (триггер, плотная проверка) входят в ключ кэша — для них нужен новый прогон run_rt.py. Стадия 4
(realtime.objects, --no-objects — без неё) при повторе перечитывает кадры пауз из видео, оценки CLIP кэшируются в памяти между точками сетки.

Оценка — как tools/eval_markup.py (rasmetka_gt.csv, только время). На каждую точку: TP/FP/FN, F1, сколько
true-видео с разметкой имеют хотя бы одно совпавшее событие, покрытие (затяжки, пересечённые событиями, и доля
времени эпизодов курения внутри событий), FP на false-видео, пропущенные true-видео.
"""
from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from eval_markup import MERGE_GAP, RunClip, evaluate_run, load_gt  # noqa: E402
from match_markup import canonical_stem  # noqa: E402
from smokedet.export import event_records  # noqa: E402
from smokedet.pipeline import read_inputs_csv  # noqa: E402
from smokedet.realtime import RealtimeDetector, analyze, load_rt_config, load_run, rt_cache_key  # noqa: E402
from smokedet.roi import Roi  # noqa: E402


def parse_grid(items):
    keys, values = [], []
    for item in items:
        key, raw = item.split("=", 1)
        keys.append(key)
        values.append(raw.split(","))
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)] if keys else [{}]


def score_point(gt, clips, runs, cfg, det, gt_stems):
    rc = {}
    for clip in clips:
        run = runs[clip.clip_id]
        roi = Roi.load(clip.roi_path, run.meta.width, run.meta.height)
        _, _, events = analyze(run, cfg, roi, None, det.appearance if det else None, clip.clip_id, clip.video_path)
        vp = Path(clip.video_path)
        c = RunClip(canonical_stem(vp.stem), clip.clip_id, vp.parent.name, run.meta.duration)
        c.preds = event_records(events, clip.camera_id, clip.clip_id)
        rc[c.stem] = c
    res = evaluate_run(gt, rc)
    hit = {m["clip_id"] for m in res["overall"]["matches"] if m["result"] == "TP"}
    true_stems = sorted(s for s in gt_stems if s in rc and rc[s].folder == "true")
    neg = res["per_folder"].get("false", {})
    return res, [s for s in true_stems if s not in hit], len(true_stems), neg.get("fp", 0)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--inputs", required=True)
    p.add_argument("--cache", required=True, help="папка кэша прогона run_rt.py (<out>/cache)")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    p.add_argument("--grid", action="append", default=[], metavar="SECTION.KEY=V1,V2")
    p.add_argument("--objects", action=argparse.BooleanOptionalAction, default=None,
                   help="стадия 4 (CLIP) при повторе (по умолчанию — realtime.objects из конфига)")
    p.add_argument("--gt", default=str(ROOT / "rasmetka_gt.csv"))
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    base = list(a.set) + ([] if a.objects is None else [f"realtime.objects={str(a.objects).lower()}"])
    cfg0 = load_rt_config(a.config, base)
    clips = read_inputs_csv(a.inputs)
    runs = {}
    for clip in clips:
        run = load_run(Path(a.cache) / f"{clip.clip_id}_rt.pkl", rt_cache_key(clip.video_path, cfg0))
        if run is None:
            sys.exit(f"{clip.clip_id}: нет кэша с этими параметрами стадий 1–3 в {a.cache}")
        runs[clip.clip_id] = run
    det = RealtimeDetector(cfg0) if cfg0.realtime.objects else None
    gt = load_gt(a.gt, MERGE_GAP)
    gt_stems = {s for s, g in gt.items() if g.episodes}
    print("| точка | TP | FP | FN | F1 | true с TP | затяжек | время | FP в false | дублей | пропущены |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for point in parse_grid(a.grid):
        cfg = load_rt_config(a.config, base + [f"{k}={v}" for k, v in point.items()])
        res, missed, n_true, fp_false = score_point(gt, clips, runs, cfg, det, gt_stems)
        t = res["total"]
        name = " ".join(f"{k.split('.')[-1]}={v}" for k, v in point.items()) or "база"
        cover = "—" if t["time_cover"] is None else f"{t['time_cover']:.0%}"
        print(f"| {name} | {t['tp']} | {t['fp']} | {t['fn']} | {t['f1']:.3f} | {n_true - len(missed)}/{n_true} "
              f"| {t['marks_hit']}/{t['marks']} | {cover} | {fp_false} "
              f"| {t['dup']} | {', '.join(missed)} |", flush=True)  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
