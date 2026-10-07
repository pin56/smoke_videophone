#!/usr/bin/env python
"""Обучение классификатора циклов «курит / не курит» по меткам веб-разметчика.

    python tools/train_cycle_clf.py --inputs data/labeling/all31.csv --cache data/labeling/ref/cache \
        --labels data/labeling/ref/labels.jsonl --ref data/labeling/ref/cycles.jsonl \
        --soft data/labeling/soft/cycles.jsonl --gt rasmetka_gt.csv --out data/training

Образцы трёх источников (колонка `source`):
  web      — циклы из выгрузки ref с меткой разметчика (smoking/not_smoking; unsure пропускаются), главные;
  rasmetka — циклы мягкого экспорта (soft), пересекающие эпизод ручной разметки rasmetka_gt.csv (SMOKE/NOSMOKE);
  window   — эпизоды ручной разметки, для которых детектор не нашёл цикл: окно берётся прямо из разметки
             (подъём руки → затяжка → опускание), чтобы обрезанные циклы тоже попали в обучение.
Признаки считаются только по сигналу кисть→рот (без моделей верификатора), поэтому одинаковы для всех источников.
Качество — перекрёстная проверка «оставить клип» (LeaveOneGroupOut); итоговая модель — на всех образцах.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smokedet.config import load_config  # noqa: E402
from smokedet.pipeline import Pipeline, read_inputs_csv  # noqa: E402

log = logging.getLogger("train_cycle_clf")

FEATURES = [
    "hold", "total", "min_d", "med_d_hold", "std_d_hold", "d_before", "d_after", "d_drop_in", "d_drop_out",
    "ear_ratio", "wrist_above", "wrist_dy", "elbow_raise", "face_visible", "kconf", "head_tilt",
    "hand_move", "light_max", "light_ring_max", "valid_frac",
]


def _med(x) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float("nan")


def _slice(sig, t0: float, t1: float):
    m = (sig.t >= t0) & (sig.t <= t1)
    return np.nonzero(m)[0]


def cycle_features(sig, start: float, hold_start: float, hold_end: float, end: float) -> dict:
    ih = _slice(sig, hold_start, hold_end)
    if len(ih) == 0:
        ih = np.array([sig.index_at((hold_start + hold_end) / 2)])
    ia = _slice(sig, start, end)
    ib = _slice(sig, start - 1.5, start)
    ie = _slice(sig, end, end + 1.5)
    d = sig.d
    hold = hold_end - hold_start
    hand = sig.hand[ih] / np.maximum(sig.scale[ih, None], 1e-6)
    hm = float(np.nanstd(hand)) if np.isfinite(hand).any() else float("nan")
    if sig.head_pitch is not None:
        tilt = _med(sig.head_pitch[ih]) - _med(sig.head_pitch[ib])
    else:
        tilt = float("nan")
    d_before, d_after = _med(d[ib]), _med(d[ie])
    med_hold = _med(d[ih])
    return {
        "hold": hold, "total": end - start,
        "min_d": float(np.nanmin(d[ih])) if np.isfinite(d[ih]).any() else float("nan"),
        "med_d_hold": med_hold, "std_d_hold": float(np.nanstd(d[ih])) if np.isfinite(d[ih]).any() else float("nan"),
        "d_before": d_before, "d_after": d_after,
        "d_drop_in": d_before - med_hold, "d_drop_out": d_after - med_hold,
        "ear_ratio": _med(sig.ear_ratio[ih]), "wrist_above": _med(sig.wrist_above[ih]),
        "wrist_dy": _med(sig.wrist_dy[ih]), "elbow_raise": _med(sig.elbow_raise[ih]),
        "face_visible": float(np.nanmean(sig.face_visible[ih])), "kconf": _med(sig.kconf[ih]),
        "head_tilt": tilt, "hand_move": hm,
        "light_max": float(np.nanmax(sig.light_area[ia])) if sig.light_area is not None and len(ia) and np.isfinite(sig.light_area[ia]).any() else float("nan"),
        "light_ring_max": float(np.nanmax(sig.light_ring[ia])) if sig.light_ring is not None and len(ia) and np.isfinite(sig.light_ring[ia]).any() else float("nan"),
        "valid_frac": float(np.mean(sig.valid[ia])) if len(ia) else float("nan"),
    }  # fmt: skip


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def best_signal(signals: dict, track_id: int, t0: float, t1: float):
    """Сигнал трека; для эпизода разметки (без трека) — тот, где у рта руки больше всего достоверных отсчётов."""
    if track_id in signals:
        return track_id, signals[track_id]
    best, best_n = None, -1
    for tid, s in signals.items():
        i = _slice(s, t0, t1)
        n = int(np.isfinite(s.d[i]).sum()) if len(i) else 0
        if n > best_n:
            best, best_n = (tid, s), n
    return best if best else (None, None)


def overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def build_dataset(args):
    cfg = load_config(str(ROOT / "configs" / "default.yaml"),
                      ["verifier.enabled=false", "flash.enabled=false", "vapor.enabled=false"])  # fmt: skip
    pipe = Pipeline(cfg)
    clips = {c.clip_id: c for c in read_inputs_csv(args.inputs)}
    labels = {}
    for r in read_jsonl(args.labels):
        labels[r["id"]] = r  # последняя метка побеждает
    ref = {r["id"]: r for r in read_jsonl(args.ref)}
    soft = read_jsonl(args.soft)
    gt = {}
    with open(args.gt, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["label"] in ("SMOKE", "NOSMOKE") and r["up_sec"] and r["down_sec"]:
                up, down = float(r["up_sec"]), float(r["down_sec"])
                puff = float(r["puff_sec"]) if r["puff_sec"] else (up + down) / 2
                gt.setdefault(f'{r["folder"]}_{r["video"]}', []).append((r["label"], up, puff, down))

    rows, covered = [], {}
    for cid, clip in clips.items():
        ts = pipe.track(clip, Path(args.cache))
        an = pipe.analyze(clip, ts)
        sigs = an.signals
        used_gt = set()
        gt_rows = gt.get(cid, [])

        def match_gt(t0, t1):
            best, bo = None, 0.0
            for k, (lab, up, puff, down) in enumerate(gt_rows):
                o = overlap(t0, t1, up - 1.0, down + 1.0)
                if o > bo:
                    best, bo = k, o
            return best

        def add(src, sid, label, tid, start, h0, h1, end):
            tid, sig = best_signal(sigs, tid, start, end)
            if sig is None:
                return
            f = cycle_features(sig, start, h0, h1, end)
            rows.append({"id": sid, "clip_id": cid, "source": src, "label": label, "track_id": tid,
                         "start": start, "end": end, **f})  # fmt: skip

        seen = set()
        for r in ref.values():  # 1) разметчик
            if r["clip_id"] != cid:
                continue
            lab = labels.get(r["id"])
            if not lab or lab["label"] not in ("smoking", "not_smoking"):
                continue
            seen.add(r["id"])
            add("web", r["id"], int(lab["label"] == "smoking"), r["track_id"], r["start"], r["hold_start"], r["hold_end"], r["end"])
            k = match_gt(r["start"], r["end"])
            if k is not None:
                used_gt.add(k)
        for r in soft:  # 2) мягкий экспорт + ручная разметка
            if r["clip_id"] != cid or r["id"] in seen:
                continue
            if any(overlap(r["start"], r["end"], x["start"], x["end"]) > 0.3 for x in ref.values()
                   if x["clip_id"] == cid and x["id"] in labels):  # fmt: skip
                continue  # этот момент уже размечен человеком
            k = match_gt(r["start"], r["end"])
            if k is None:
                continue
            used_gt.add(k)
            add("rasmetka", r["id"], int(gt_rows[k][0] == "SMOKE"), r["track_id"], r["start"], r["hold_start"], r["hold_end"], r["end"])
        for k, (lab, up, puff, down) in enumerate(gt_rows):  # 3) окна по разметке там, где цикла нет
            if k in used_gt:
                continue
            h0, h1 = max(up, puff - 0.5), min(down, puff + 0.5)
            if h1 <= h0:
                h0, h1 = puff - 0.25, puff + 0.25
            add("window", f"{cid}__gt{k}", int(lab == "SMOKE"), None, up, h0, h1, max(down, h1 + 0.1))
    return rows


def train(rows, out: Path):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.array([[r[f] for f in FEATURES] for r in rows], float)
    y = np.array([r["label"] for r in rows], int)
    g = np.array([r["clip_id"] for r in rows])
    src = np.array([r["source"] for r in rows])
    models = {
        "logreg": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                        LogisticRegression(C=0.5, class_weight="balanced", max_iter=2000)),  # fmt: skip
        "forest": lambda: make_pipeline(SimpleImputer(strategy="median"),
                                        RandomForestClassifier(300, min_samples_leaf=3, class_weight="balanced", random_state=0)),  # fmt: skip
    }
    report = {"n": int(len(y)), "pos": int(y.sum()), "by_source": {s: int((src == s).sum()) for s in np.unique(src)}, "models": {}}
    for name, mk in models.items():
        proba = np.zeros(len(y))
        for tr, te in LeaveOneGroupOut().split(X, y, g):
            if len(np.unique(y[tr])) < 2:
                proba[te] = float(y[tr].mean())
                continue
            m = mk().fit(X[tr], y[tr])
            proba[te] = m.predict_proba(X[te])[:, 1]
        pred = proba >= 0.5
        res = {}
        for label, mask in (("all", np.ones(len(y), bool)), ("web", src == "web")):
            if mask.sum() < 5 or len(np.unique(y[mask])) < 2:
                continue
            res[label] = {
                "auc": round(roc_auc_score(y[mask], proba[mask]), 3), "ap": round(average_precision_score(y[mask], proba[mask]), 3),
                "precision": round(precision_score(y[mask], pred[mask]), 3), "recall": round(recall_score(y[mask], pred[mask]), 3),
                "f1": round(f1_score(y[mask], pred[mask]), 3), "n": int(mask.sum()),
            }  # fmt: skip
        report["models"][name] = res
        for r, p in zip(rows, proba):
            r[f"cv_{name}"] = round(float(p), 3)
    best = max(report["models"], key=lambda k: report["models"][k].get("all", {}).get("auc", 0))
    final = models[best]().fit(X, y)
    import joblib

    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "features": FEATURES, "name": best}, out / "cycle_clf.joblib")
    report["best"] = best
    if best == "forest":
        report["importance"] = dict(sorted(zip(FEATURES, map(lambda v: round(float(v), 3), final[-1].feature_importances_)), key=lambda kv: -kv[1]))
    else:
        report["coef"] = dict(sorted(zip(FEATURES, map(lambda v: round(float(v), 3), final[-1].coef_[0])), key=lambda kv: -abs(kv[1])))
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Обучение классификатора циклов")
    p.add_argument("--inputs", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--soft", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    rows = build_dataset(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = train(rows, out)
    keys = ["id", "clip_id", "source", "label", "track_id", "start", "end"] + FEATURES + ["cv_logreg", "cv_forest"]
    with open(out / "dataset.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
