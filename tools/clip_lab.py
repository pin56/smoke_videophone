#!/usr/bin/env python
"""Лаборатория стадии 4 потокового режима (CLIP по кропам паузы): векторы CLIP всех кадров паузы каждого цикла
считаются один раз, дальше число и выбор кадров, виды кропа, агрегация по кадрам и пороги перебираются без CLIP.

    python tools/clip_lab.py extract --inputs runs/rt_noobj/all.csv --cache runs/cpu/full/cache \\
        --set realtime.det_model=... --model ViT-L/14@336px --views 0.35,0.7,0.2 --out runs/clip_lab/L336.pkl
    python tools/clip_lab.py score --inputs runs/rt_noobj/all.csv --cache runs/cpu/full/cache --set ... \\
        --emb runs/clip_lab/L336.pkl --grid lab.frames=u8,u4,p2 --grid lab.views=0+1,0 --grid appearance.min_score=0,0.5

Циклы — повтор заданий из кэша run_rt.py (как tools/tune_rt.py); кропы вырезаются из кадров видео (в потоке —
из сохранённых квадратов задания, поэтому оценки чуть отличаются). Параметры лаборатории (ключи lab.*):
frames — u<k> k кадров равномерно по паузе (как plan_frames), p<k> — k кадров с наименьшим d (пик), all — все;
views — номера видов из --views через «+» (0 — узкий 0.35, 1 — широкий 0.7, 2 — тесный 0.2 со словарём узкого);
agg — max (как в appearance.cycle_score), top2 (среднее двух лучших), mean; hand — вес лучшей оценки кисти в покое
(кадры до подхода и после отвода, кроп по кисти, HAND_PROMPTS), diff — вес вычитаемой средней оценки рта в покое;
prompts — словарь из PROMPT_SETS (base — как в appearance.py); bridge — валидный цикл с оценкой ниже не входит в события (none — входит).
Остальные ключи — как --set.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import logging
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from eval_markup import MERGE_GAP, RunClip, evaluate_run, load_gt  # noqa: E402
from match_markup import canonical_stem  # noqa: E402
from smokedet import appearance as ap  # noqa: E402
from smokedet.export import event_records  # noqa: E402
from smokedet.io_video import FrameReader  # noqa: E402
from smokedet.pipeline import read_inputs_csv  # noqa: E402
from smokedet.realtime import analyze, keep_cycle, load_rt_config, load_run, project_path, rt_cache_key, rt_events  # noqa: E402
from smokedet.roi import Roi  # noqa: E402

log = logging.getLogger("clip_lab")
REST_SPAN = 3.0  # с до подхода и после отвода: кадры «рука в покое»
REST_VIEWS = (0.25, 0.35)  # кроп по кисти (сигарета в руке между затяжками) и по рту (базовая линия человека)
HAND_PROMPTS = (
    ["a hand holding a cigarette", "fingers holding a lit cigarette", "a person holding a cigarette in their hand",
     "a hand holding a vape"],
    ["an empty hand", "a hand holding a bottle", "a hand holding a cup", "a hand holding a can",
     "a hand holding a mobile phone", "a hand in a pocket", "a hand holding a bag", "a hand holding food"],
)  # fmt: skip


def cycle_key(clip_id: str, c) -> tuple:
    return clip_id, c.track_id, round(c.start, 3), round(c.hold_start, 3), round(c.hold_end, 3)


def hold_samples(sig, c) -> np.ndarray:
    idx = np.nonzero((sig.t >= c.hold_start) & (sig.t <= c.hold_end))[0]
    return idx if len(idx) else np.array([sig.index_at((c.hold_start + c.hold_end) / 2)])


def rest_samples(sig, c, cfg, span: float = REST_SPAN, per_side: int = 2) -> np.ndarray:
    """Отсчёты «рука в покое» рядом с циклом: до per_side последних перед подходом и первых после отвода (в пределах
    span с), где кисть далеко ото рта (d > cycle.near_exit)."""
    far = np.isfinite(sig.d) & (sig.d > cfg.cycle.near_exit)
    before = np.nonzero(far & (sig.t >= c.start - span) & (sig.t < c.start))[0][-per_side:]
    after = np.nonzero(far & (sig.t > c.end) & (sig.t <= c.end + span))[0][:per_side]
    return np.concatenate([before, after]).astype(int)


def replay_clips(a):
    cfg = load_rt_config(a.config, a.set)
    out = []
    for clip in read_inputs_csv(a.inputs):
        run = load_run(Path(a.cache) / f"{clip.clip_id}_rt.pkl", rt_cache_key(clip.video_path, cfg))
        if run is None:
            sys.exit(f"{clip.clip_id}: нет кэша с этими параметрами стадий 1–3 в {a.cache}")
        roi = Roi.load(clip.roi_path, run.meta.width, run.meta.height)
        cycles, signals, _ = analyze(run, cfg, roi)
        cycles = {tid: cs for tid, cs in cycles.items() if tid in signals}
        out.append((clip, run, cycles, signals))
    return cfg, out


def extract(a) -> int:
    cfg, clips = replay_clips(a)
    acfg = copy.deepcopy(cfg.appearance)
    acfg.model = a.model
    enc = ap.ClipEncoder(acfg, a.device, project_path(acfg.weights_dir))
    if a.device == "cpu" and acfg.openvino and enc.visual is None:
        ap.export_openvino(enc.model, enc.weights)
        enc = ap.ClipEncoder(acfg, a.device, project_path(acfg.weights_dir))
    scales = [float(s) for s in a.views.split(",")]
    recs, t0, n_crops = {}, time.perf_counter(), 0
    for clip, run, cycles, signals in clips:
        todo = [c for cs in cycles.values() for c in cs if c.valid or ap.rescuable(c, cfg.appearance)]
        need = defaultdict(list)
        plans, times, rests = [], {}, []
        for i, c in enumerate(todo):
            sig = signals[c.track_id]
            ks = hold_samples(sig, c)
            centers, height = ap.crop_geometry(sig, c, ks)
            frames = [int(sig.frame_idx[k]) for k in ks]
            rk = rest_samples(sig, c, cfg)
            rframes = [int(sig.frame_idx[k]) for k in rk]
            plans.append((ks, centers, height, max(frames + rframes)))
            for j, (k, f) in enumerate(zip(ks, frames)):
                need[f].append((i, j))
                times[f] = float(sig.t[k])
            for j, (k, f) in enumerate(zip(rk, rframes)):
                need[f].append((i, -1 - j))
                times[f] = float(sig.t[k])
            rests.append(rk)
        crops = [[None] * len(p[0]) for p in plans]
        rcrops = [[None] * len(r) for r in rests]
        with FrameReader(clip.video_path, synthetic_from=run.meta.synthetic_from) as reader:
            for f in sorted(need):
                frame = reader.get(f, times[f])
                for i, j in need[f]:
                    ks, centers, height, last = plans[i]
                    if frame is not None and j >= 0:
                        cx, cy = centers[j]
                        crops[i][j] = [ap.square_crop(frame, cx, cy, s * height) for s in scales]
                    elif frame is not None:
                        sig, k = signals[todo[i].track_id], rests[i][-1 - j]
                        m = sig.mouth[k] if np.isfinite(sig.mouth[k]).all() else None
                        hd = sig.hand[k] if np.isfinite(sig.hand[k]).all() else None
                        if m is not None and hd is not None:
                            rcrops[i][-1 - j] = [ap.square_crop(frame, hd[0], hd[1], REST_VIEWS[0] * height),
                                                 ap.square_crop(frame, m[0], m[1], REST_VIEWS[1] * height)]  # fmt: skip
                    if f != last:
                        continue
                    c, sig = todo[i], signals[todo[i].track_id]
                    ok = [j2 for j2 in range(len(ks)) if crops[i][j2] is not None]
                    if ok:
                        flat = [crops[i][j2][v] for j2 in ok for v in range(len(scales))]
                        e = enc.embed(flat).numpy().reshape(len(ok), len(scales), -1).astype(np.float16)
                        n_crops += len(flat)
                        ks_ok = ks[ok]
                        rec = {"t": sig.t[ks_ok].astype(np.float32), "d": sig.d[ks_ok].astype(np.float32), "emb": e}
                        rok = [j2 for j2 in range(len(rests[i])) if rcrops[i][j2] is not None]
                        if rok:
                            flat = [rcrops[i][j2][v] for j2 in rok for v in range(len(REST_VIEWS))]
                            rec["rest"] = enc.embed(flat).numpy().reshape(len(rok), len(REST_VIEWS), -1).astype(np.float16)
                            rec["rest_t"] = sig.t[rests[i][rok]].astype(np.float32)
                            n_crops += len(flat)
                        recs[cycle_key(clip.clip_id, c)] = rec
                    crops[i] = rcrops[i] = None
        log.info("%s: циклов %d, всего кропов %d, %.0f с", clip.clip_id, len(todo), n_crops, time.perf_counter() - t0)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as fh:
        pickle.dump({"model": a.model, "views": scales, "recs": recs}, fh)
    print(f"{a.out}: циклов {len(recs)}, кропов {n_crops}, {time.perf_counter() - t0:.0f} с")
    return 0


def _prefixed(prompts, prefix):
    return tuple([prefix + q[0].lower() + q[1:] if q.startswith("A") else prefix + q for q in side] for side in prompts)


def _extended(prompts, pos, neg):
    return list(prompts[0]) + pos, list(prompts[1]) + neg


# словари для lab.prompts: имя -> {вид: (курит, не курит)}
PROMPT_SETS = {
    "base": {"mouth": ap.MOUTH_PROMPTS, "wide": ap.WIDE_PROMPTS, "hand": HAND_PROMPTS},
    "ext": {
        "mouth": _extended(ap.MOUTH_PROMPTS,
                           ["a cigarette between the lips", "a person puffing on a cigarette",
                            "smoke coming out of a person's mouth", "a person holding a cigarette near their face"],
                           ["a person talking", "a person coughing into their hand", "a person holding a phone to their ear",
                            "a person biting their finger", "a person licking their fingers", "a person whistling"]),
        "wide": _extended(ap.WIDE_PROMPTS,
                          ["a person standing and smoking a cigarette", "a person with a cigarette in their hand",
                           "a smoker taking a puff"],
                          ["a person holding a phone to their ear", "a person looking at a mobile phone",
                           "a person coughing", "a person drinking water", "a person standing with hands near their face"]),
        "hand": HAND_PROMPTS,
    },
    "cctv": {k: _prefixed(v, "a blurry low-resolution CCTV image of ") for k, v in
             (("mouth", ap.MOUTH_PROMPTS), ("wide", ap.WIDE_PROMPTS), ("hand", HAND_PROMPTS))},
    "core": {"mouth": (["a person smoking a cigarette"], ["a person drinking", "a person touching their face",
                                                          "a person talking on the phone", "a person eating"]),
             "wide": (["a person smoking a cigarette"], ["a person drinking", "a person touching their face",
                                                         "a person talking on the phone", "a person eating"]),
             "hand": HAND_PROMPTS},
}  # fmt: skip


class TextBank:
    """Векторы описаний всех словарей PROMPT_SETS: «mouth» для узких видов, «wide» для широкого, «hand» для кисти."""

    def __init__(self, model: str, weights_dir: str):
        import clip
        import torch

        m, _ = clip.load(model, device="cpu", download_root=weights_dir)
        self.scale = float(m.logit_scale.detach().exp())
        self.sets = {}
        with torch.no_grad():
            for pset, views in PROMPT_SETS.items():
                for name, (pos, neg) in views.items():
                    t = m.encode_text(clip.tokenize(list(pos) + list(neg))).float()
                    self.sets[pset, name] = ((t / t.norm(dim=-1, keepdim=True)).numpy(), len(pos))

    def frame_scores(self, emb: np.ndarray, names, pset: str = "base") -> np.ndarray:
        """(кадры, виды, D) и словарь каждого вида -> (кадры, виды) оценок «курит»."""
        out = np.empty(emb.shape[:2], dtype=np.float64)
        for v, name in enumerate(names):
            txt, npos = self.sets[pset, name]
            lg = self.scale * emb[:, v].astype(np.float32) @ txt.T
            out[:, v] = _lse(lg[:, :npos]) - _lse(lg[:, npos:])
        return out


def _lse(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=1, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis=1, keepdims=True)))[:, 0]


def pick_frames(rec: dict, spec: str) -> np.ndarray:
    n = len(rec["t"])
    if spec == "all":
        return np.arange(n)
    k = int(spec[1:])
    if spec[0] == "u":
        return np.unique(np.linspace(0, n - 1, min(k, n)).round().astype(int))
    if spec[0] == "p":
        d = np.where(np.isfinite(rec["d"]), rec["d"], np.inf)
        return np.sort(np.argsort(d, kind="stable")[:k])
    raise ValueError(f"lab.frames: {spec}")


def aggregate(s: np.ndarray, agg: str) -> float:
    if agg == "max":
        return float(s.max())
    if agg == "top2":
        return float(np.sort(s)[-2:].mean())
    if agg == "mean":
        return float(s.mean())
    raise ValueError(f"lab.agg: {agg}")


def cycle_appearance(rec: dict, fs, lab: dict) -> float:
    """Оценка цикла: сумма по видам агрегата по выбранным кадрам паузы; плюс hand × лучшая оценка кисти в покое,
    минус diff × средняя оценка рта в покое (базовая линия человека и камеры)."""
    hold, rest = fs
    ks = pick_frames(rec, lab["frames"])
    s = sum(aggregate(hold[ks, v], lab["agg"]) for v in lab["views"])
    if rest is not None and len(rest):
        s += lab["hand"] * float(rest[:, 0].max()) - lab["diff"] * float(rest[:, 1].mean())
    return s


def parse_grid(items):
    keys, values = [], []
    for item in items:
        key, raw = item.split("=", 1)
        keys.append(key)
        values.append(raw.split(","))
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)] if keys else [{}]


def score(a) -> int:
    cfg0, clips = replay_clips(a)
    with open(a.emb, "rb") as fh:
        bank = pickle.load(fh)
    text = TextBank(bank["model"], project_path(cfg0.appearance.weights_dir))
    names = ["wide" if v >= 0.5 else "mouth" for v in bank["views"]]
    by_set = {}

    def fscores(pset):
        if pset not in by_set:
            by_set[pset] = {k: (text.frame_scores(r["emb"], names, pset),
                                text.frame_scores(r["rest"], ["hand", "mouth"], pset) if "rest" in r else None)
                            for k, r in bank["recs"].items()}  # fmt: skip
        return by_set[pset]

    gt = load_gt(a.gt, MERGE_GAP)
    gt_stems = {s for s, g in gt.items() if g.episodes}
    print(f"эмбеддинги {a.emb}: {bank['model']}, виды {bank['views']}, циклов {len(bank['recs'])}")
    print("| точка | TP | FP | FN | F1 | true с TP | затяжек | время | FP в false | дублей | кропов | пропущены |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|---|")  # fmt: skip
    for point in parse_grid(a.grid):
        lab = {"frames": "u8", "views": "0+1", "agg": "max", "hand": 0.0, "diff": 0.0, "bridge": "none", "prompts": "base"}
        sets = []
        for k, v in point.items():
            if k.startswith("lab."):
                lab[k[4:]] = v
            else:
                sets.append(f"{k}={v}")
        lab["views"] = [int(v) for v in str(lab["views"]).split("+")]
        lab["hand"], lab["diff"] = float(lab["hand"]), float(lab["diff"])
        lab["bridge"] = None if lab["bridge"] == "none" else float(lab["bridge"])
        cfg = load_rt_config(a.config, a.set + sets)
        res, missed, n_true, fp_false, n_crops = evaluate_point(gt, gt_stems, clips, cfg, bank, fscores(lab["prompts"]), lab)
        t = res["total"]
        name = " ".join(f"{k.split('.')[-1]}={v}" for k, v in point.items()) or "база"
        cover = "—" if t["time_cover"] is None else f"{t['time_cover']:.0%}"
        print(f"| {name} | {t['tp']} | {t['fp']} | {t['fn']} | {t['f1']:.3f} | {n_true - len(missed)}/{n_true} "
              f"| {t['marks_hit']}/{t['marks']} | {cover} | {fp_false} | {t['dup']} | {n_crops} "
              f"| {', '.join(missed)} |", flush=True)  # fmt: skip
    return 0


def evaluate_point(gt, gt_stems, clips, cfg, bank, fscores, lab):
    rc, n_crops = {}, 0
    keep = keep_cycle(cfg)
    for clip, run, cycles0, signals in clips:
        cycles = copy.deepcopy(cycles0)
        for cs in cycles.values():
            for c in cs:
                if not (c.valid or ap.rescuable(c, cfg.appearance)):
                    continue
                key = cycle_key(clip.clip_id, c)
                rec = bank["recs"].get(key)
                if rec is None:
                    continue
                n_crops += len(pick_frames(rec, lab["frames"])) * len(lab["views"])
                ap.apply_score(c, cycle_appearance(rec, fscores[key], lab), cfg.appearance, keep)
                if lab["bridge"] is not None and c.valid and c.appearance < lab["bridge"]:
                    c.valid, c.reject_reason = False, "bridge"  # не курение на вид: не продлевает и не склеивает события
        events = rt_events(cycles, signals, cfg)
        vp = Path(clip.video_path)
        c = RunClip(canonical_stem(vp.stem), clip.clip_id, vp.parent.name, run.meta.duration)
        c.preds = event_records(events, clip.camera_id, clip.clip_id)
        rc[c.stem] = c
    res = evaluate_run(gt, rc)
    hit = {m["clip_id"] for m in res["overall"]["matches"] if m["result"] == "TP"}
    true_stems = sorted(s for s in gt_stems if s in rc and rc[s].folder == "true")
    neg = res["per_folder"].get("false", {})
    return res, [s for s in true_stems if s not in hit], len(true_stems), neg.get("fp", 0), n_crops


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("extract", "score"))
    p.add_argument("--inputs", required=True)
    p.add_argument("--cache", required=True, help="папка кэша прогона run_rt.py (<out>/cache)")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    p.add_argument("--model", default="ViT-L/14@336px", help="extract: CLIP")
    p.add_argument("--views", default="0.35,0.7,0.2", help="extract: стороны кропов, доли роста")
    p.add_argument("--device", default="cpu", help="extract: cpu (OpenVINO, как в потоке) или 0")
    p.add_argument("--out", help="extract: файл эмбеддингов (.pkl)")
    p.add_argument("--emb", help="score: файл эмбеддингов")
    p.add_argument("--grid", action="append", default=[], metavar="KEY=V1,V2")
    p.add_argument("--gt", default=str(ROOT / "rasmetka_gt.csv"))
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("smokedet").setLevel(logging.WARNING)
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    return extract(a) if a.mode == "extract" else score(a)


if __name__ == "__main__":
    sys.exit(main())
