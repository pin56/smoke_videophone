#!/usr/bin/env python
"""Оценка прогонов по разметке пользователя (rasmetka_gt.csv из tools/match_markup.py).

    python tools/eval_markup.py --run runs/research/base_fp16 [--run runs/research/base_fp32 ...]
        [--gt rasmetka_gt.csv] [--out <папка отчёта>]

Протокол — как в smokedet/evaluation.py (кейс), но только по времени: в разметке нет рамок и личности
человека, поэтому IoU рамки не проверяется, а отметки SMOKE клипа объединяются в эпизод независимо от того,
чьи они. Эталонный эпизод — отметки SMOKE клипа (интервал [подъём руки, опускание]), объединённые при
зазоре <= 15 с; эпизод расширяется на ±3 с, пара засчитывается при пересечении >= 1 с, сопоставление один
к одному жадно по наибольшему пересечению. Предсказание без пары не FP, если >= 50 % его длительности
лежит на IGNORE и оно не пересекает положительный эпизод. IGNORE — строки «игнорировать» и промежутки
«продолжают курить» (строка SMOKE без времени: от конца предыдущей отметки до начала следующей, человек
курит, но циклы не размечены).

Клипы связываются по имени видеофайла (stem), а не по clip_id: clip_id зависит от прогона. clip_id
предсказания переводится в видео по processing_log.csv прогона (video_path). Папка видео (true/false) —
по video_path: в false курения нет нигде, любое предсказание без IGNORE — FP, даже без строк NOSMOKE.
Клипы из true без строк разметки исключаются из метрик (список выводится).

Выход в --out (по умолчанию <run>/markup_eval): summary.md, matches.csv, per_clip.csv; сводка — в stdout.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from match_markup import canonical_stem  # noqa: E402
from smokedet.evaluation import EvalParams, Item, evaluate, overlap  # noqa: E402

MERGE_GAP = 15.0  # с: отметки одного клипа с меньшим зазором — один эпизод (правило кейса)
NEGATIVE_FOLDERS = ("false",)
# что происходит в NOSMOKE-циклах клипа (в разметке тип не указан): по имени файла и словам пользователя
SCENARIO = {
    "1_drink": "питьё",
    "drinking_Jar": "питьё",
    "распитие": "питьё",
    "распитие (2)": "питьё",
    "2025_11_06_08_15_57": "касание лица",
    "2025_11_06_08_35_37": "касание лица",
}


def _num(v) -> Optional[float]:
    v = (v or "").strip()
    return None if v in ("", "None") else float(v)


@dataclass
class ClipGT:
    stem: str
    folder: str
    episodes: List[Tuple[float, float, int]] = field(default_factory=list)  # начало, конец, число отметок
    ignores: List[Tuple[float, float, str]] = field(default_factory=list)  # начало, конец, вид
    nosmoke: List[Tuple[float, float]] = field(default_factory=list)


def mark_interval(up, puff, down) -> Optional[Tuple[float, float]]:
    """[подъём, опускание] отметки; недостающее время — по соседним полям; None — времени нет."""
    vals = [v for v in (up, puff, down) if v is not None]
    return (min(vals), max(vals)) if vals else None


def merge_marks(intervals: List[Tuple[float, float]], gap: float = MERGE_GAP) -> List[Tuple[float, float, int]]:
    """Отметки -> эпизоды (начало, конец, число отметок): объединение при зазоре <= gap."""
    out: List[List[float]] = []
    for s, e in sorted(intervals):
        if out and s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
            out[-1][2] += 1
        else:
            out.append([s, e, 1])
    return [(s, e, int(n)) for s, e, n in out]


def load_gt(path: str, gap: float = MERGE_GAP) -> Dict[str, ClipGT]:
    """rasmetka_gt.csv -> разметка по stem видео. Порядок строк важен: «продолжают курить» (SMOKE без
    времени) — промежуток между соседними по файлу отметками SMOKE того же клипа."""
    clips: Dict[str, ClipGT] = {}
    smoke_seq: Dict[str, List[Optional[Tuple[float, float]]]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            stem = canonical_stem(r["video"])
            clip = clips.setdefault(stem, ClipGT(stem, r.get("folder", "")))
            iv = mark_interval(_num(r["up_sec"]), _num(r["puff_sec"]), _num(r["down_sec"]))
            label = r["label"].strip().upper()
            if label == "SMOKE":
                smoke_seq[stem].append(iv)
            elif iv is None:
                continue
            elif label == "IGNORE":
                clip.ignores.append((iv[0], iv[1], "IGNORE"))
            else:
                clip.nosmoke.append(iv)
    for stem, seq in smoke_seq.items():
        clip = clips[stem]
        for i, iv in enumerate(seq):
            if iv is not None:
                continue
            prev = next((x for x in reversed(seq[:i]) if x is not None), None)
            nxt = next((x for x in seq[i + 1 :] if x is not None), None)
            start = prev[1] if prev else 0.0
            end = nxt[0] if nxt else float("inf")  # до конца клипа
            if end > start:
                clip.ignores.append((start, end, "continue"))
        clip.episodes = merge_marks([iv for iv in seq if iv is not None], gap)
    return clips


@dataclass
class RunClip:
    stem: str
    clip_id: str
    folder: str
    duration: float
    preds: List[dict] = field(default_factory=list)


def load_run(run_dir: str) -> Dict[str, RunClip]:
    """processing_log.csv + predictions.csv прогона -> клипы по stem видео."""
    run = Path(run_dir)
    by_id: Dict[str, RunClip] = {}
    with open(run / "processing_log.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            vp = Path(r["video_path"].replace("\\", "/"))
            by_id[r["clip_id"]] = RunClip(canonical_stem(vp.stem), r["clip_id"], vp.parent.name, float(r["duration_sec"]))
    pred_path = run / "predictions.csv"
    if pred_path.exists():
        with open(pred_path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["clip_id"] not in by_id:
                    print(f"предупреждение: {r['clip_id']} нет в processing_log.csv, предсказание пропущено", file=sys.stderr)
                    continue
                by_id[r["clip_id"]].preds.append(r)
    out = {}
    for c in by_id.values():
        if c.stem in out:
            raise SystemExit(f"видео {c.stem} в прогоне дважды")
        out[c.stem] = c
    return out


def _debug(run_dir: str, clip_id: str) -> Optional[dict]:
    p = Path(run_dir) / "debug" / f"{clip_id}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def diagnose_fn(dbg: Optional[dict], start: float, end: float, expand: float) -> str:
    """Что пайплайн нашёл около пропущенного эпизода: циклы (валидные / причины отказа), непрошедшие события."""
    if dbg is None:
        return "нет debug"
    s, e = start - expand, end + expand
    cyc = [c for c in dbg["cycles"] if overlap(s, e, c["start"], c["end"]) > 0]
    if not cyc:
        return "циклов нет"
    valid = [c for c in cyc if c["valid"]]
    reasons = Counter((c["reject_reason"] or "?").split(":")[0] for c in cyc if not c["valid"])
    app = [c["appearance"] for c in cyc if c.get("appearance") is not None]
    parts = [f"циклов {len(cyc)}, валидных {len(valid)}"]
    if reasons:
        parts.append("отказ: " + ", ".join(f"{k}×{n}" for k, n in reasons.most_common()))
    if app:
        parts.append(f"CLIP max {max(app):.1f}")
    evs = [ev for ev in dbg["events"] if not ev["passed"] and overlap(s, e, ev["start"], ev["end"]) > 0]
    for ev in evs:
        why = ev["reject_reason"] or "conf"
        parts.append(f"событие не прошло: conf {ev['confidence']:.2f}, циклов {ev['n_movements']}, {why}")
    return "; ".join(parts)


def evaluate_run(gt: Dict[str, ClipGT], run: Dict[str, RunClip], run_dir: str = "",
                 params: EvalParams = EvalParams()) -> dict:  # fmt: skip
    included, excluded = {}, []
    for stem, rc in run.items():
        if stem in gt or rc.folder in NEGATIVE_FOLDERS:
            included[stem] = rc
        else:
            excluded.append(stem)
    missing = sorted(s for s in gt if s not in run)
    preds, gts, durations = [], [], {}
    for stem, rc in included.items():
        durations[("", stem)] = rc.duration
        for r in rc.preds:
            preds.append(Item("", stem, float(r["start_sec"]), float(r["end_sec"]), None, "POSITIVE",
                              float(r.get("confidence") or 1.0), r))  # fmt: skip
        g = gt.get(stem)
        if g is None:
            continue
        for s, e, n in g.episodes:
            gts.append(Item("", stem, s, e, None, "POSITIVE", row={"n_marks": n}))
        for s, e, kind in g.ignores:
            gts.append(Item("", stem, s, min(e, rc.duration), None, "IGNORE", row={"scenario": kind}))
        for s, e in g.nosmoke:
            gts.append(Item("", stem, s, e, None, "NEGATIVE"))
    rep = evaluate(preds, gts, durations, params)
    folder = {stem: rc.folder for stem, rc in included.items()}

    matched = [(m["clip_id"], m["gt_start"], m["gt_end"]) for m in rep["matches"] if m["result"] == "TP"]
    for m in rep["matches"]:
        stem = m["clip_id"]
        m["folder"] = folder[stem]
        m["clip"] = included[stem].clip_id
        if m["result"] == "FP":
            # второе предсказание на уже сопоставленный эпизод: по протоколу FP, но в разметке нет личности,
            # и отметки разных людей слиты в один эпизод — считается отдельно (dup)
            dup = next(((a, b) for st, a, b in matched if st == stem and overlap(a - params.gt_expand, b + params.gt_expand,
                        m["pred_start"], m["pred_end"]) >= params.min_overlap), None)  # fmt: skip
            m["dup"] = dup is not None
            s, e = m["pred_start"] - params.gt_expand, m["pred_end"] + params.gt_expand
            hits = [(a, b) for a, b in (gt[stem].nosmoke if stem in gt else []) if overlap(s, e, a, b) > 0]
            if dup:
                m["note"] = f"дубль: второе предсказание на эпизод {dup[0]:g}–{dup[1]:g}"
            elif hits:
                m["note"] = SCENARIO.get(stem, "не курит") + ": " + ", ".join(f"{a:g}–{b:g}" for a, b in hits)
            else:
                m["note"] = "без NOSMOKE-отметки рядом" if folder[stem] in NEGATIVE_FOLDERS else "клип с курением, вне эпизодов"
        elif m["result"] == "FN":
            n = next(it.row["n_marks"] for it in gts if it.label == "POSITIVE" and it.clip_id == stem and it.start == m["gt_start"])
            m["gt_marks"] = n
            m["note"] = diagnose_fn(_debug(run_dir, included[stem].clip_id) if run_dir else None, m["gt_start"],
                                    m["gt_end"], params.gt_expand)  # fmt: skip

    def subset(cond) -> dict:
        tp = sum(1 for m in rep["matches"] if m["result"] == "TP" and cond(m["clip_id"]))
        fp = sum(1 for m in rep["matches"] if m["result"] == "FP" and cond(m["clip_id"]))
        fn = sum(1 for m in rep["matches"] if m["result"] == "FN" and cond(m["clip_id"]))
        dup = sum(1 for m in rep["matches"] if m["result"] == "FP" and m["dup"] and cond(m["clip_id"]))
        hours = sum(rc.duration for st, rc in included.items() if cond(st)) / 3600.0
        p, r, f1 = prf(tp, fp, fn)
        return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1, "dup": dup,
                "f1_nodup": prf(tp, fp - dup, fn)[2], "hours": hours, "fp_per_hour": fp / hours if hours else None}  # fmt: skip

    total = subset(lambda st: True)
    per_folder = {fo: subset(lambda st, fo=fo: folder[st] == fo) for fo in sorted(set(folder.values()))}
    per_clip = []
    for stem in sorted(included, key=lambda st: (folder[st], st)):
        ms = [m for m in rep["matches"] if m["clip_id"] == stem]
        cnt = Counter(m["result"] for m in ms)
        per_clip.append({"folder": folder[stem], "video": stem, "clip_id": included[stem].clip_id,
                         "duration_sec": included[stem].duration, "gt_episodes": len(gt[stem].episodes) if stem in gt else 0,
                         "preds": len(included[stem].preds), "TP": cnt["TP"], "FN": cnt["FN"], "FP": cnt["FP"],
                         "ignored": cnt["IGNORED"]})  # fmt: skip
    single = [(st, s, e) for st, g in gt.items() if st in included for s, e, n in g.episodes if n == 1]
    return {"overall": rep, "total": total, "per_folder": per_folder, "per_clip": per_clip, "excluded": sorted(excluded),
            "missing": missing, "single_mark_episodes": single}  # fmt: skip


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def _f(v, spec=".3f") -> str:
    return "—" if v is None else format(v, spec)


def summary_md(name: str, res: dict) -> str:
    o = res["overall"]
    lines = [f"### {name}", "",
             f"P {_f(o['precision'])}  R {_f(o['recall'])}  F1 {_f(o['f1'])}  —  TP {o['tp']}, FP {o['fp']}, FN {o['fn']}, "
             f"в IGNORE {o['ignored_preds']}; FP/ч вне эпизодов и IGNORE {_f(o['fp_per_hour'], '.2f')}; "
             f"из FP дублей эпизода {res['total']['dup']} (F1 без них {_f(res['total']['f1_nodup'])})", "",
             "| папка | TP | FP | FN | P | R | F1 | часов | FP/ч |", "|---|---|---|---|---|---|---|---|---|"]  # fmt: skip
    for fo, s in res["per_folder"].items():
        lines.append(f"| {fo} | {s['tp']} | {s['fp']} | {s['fn']} | {_f(s['precision'])} | {_f(s['recall'])} | "
                     f"{_f(s['f1'])} | {s['hours']:.2f} | {_f(s['fp_per_hour'], '.1f')} |")  # fmt: skip
    if res["excluded"]:
        lines += ["", "Исключены (true без разметки): " + ", ".join(res["excluded"])]
    if res["missing"]:
        lines += ["", "Есть в разметке, нет в прогоне: " + ", ".join(res["missing"])]
    lines += ["", "| результат | видео | эталон, с | предсказание, с | conf | примечание |", "|---|---|---|---|---|---|"]
    for m in sorted(res["overall"]["matches"], key=lambda m: (m["result"], m["folder"], m["clip_id"])):
        if m["result"] not in ("FN", "FP"):
            continue
        g = f"{m['gt_start']:g}–{m['gt_end']:g} ({m['gt_marks']} отм.)" if m["result"] == "FN" else ""
        p = f"{m['pred_start']:.1f}–{m['pred_end']:.1f}" if m["pred_start"] != "" else ""
        c = f"{m['pred_conf']:.2f}" if m["pred_conf"] != "" else ""
        lines.append(f"| {m['result']} | {m['folder']}/{m['clip_id']} | {g} | {p} | {c} | {m.get('note', '')} |")
    return "\n".join(lines)


def write_outputs(out: Path, name: str, res: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(summary_md(name, res) + "\n", encoding="utf-8")
    cols = ["result", "dup", "folder", "clip_id", "clip", "gt_start", "gt_end", "gt_marks", "gt_label", "gt_scenario",
            "pred_event_id", "pred_track", "pred_start", "pred_end", "pred_conf", "note"]  # fmt: skip
    with open(out / "matches.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(res["overall"]["matches"])
    with open(out / "per_clip.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(res["per_clip"][0]) if res["per_clip"] else ["video"])
        w.writeheader()
        w.writerows(res["per_clip"])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", required=True, help="папка прогона run.py (можно несколько)")
    p.add_argument("--gt", default=str(ROOT / "rasmetka_gt.csv"))
    p.add_argument("--out", help="папка отчёта (для одного --run; по умолчанию <run>/markup_eval)")
    p.add_argument("--gap", type=float, default=MERGE_GAP, help="объединение отметок в эпизод, с")
    a = p.parse_args(argv)
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    gt = load_gt(a.gt, a.gap)
    eps = [(st, s, e, n) for st, g in gt.items() for s, e, n in g.episodes]
    single = [x for x in eps if x[3] == 1]
    print(f"Эталон: {len(eps)} эпизодов SMOKE в {len({x[0] for x in eps})} клипах, из них с одной отметкой {len(single)} "
          f"(по правилу им нужен прямой признак): " + ", ".join(f"{st} {s:g}–{e:g}" for st, s, e, _ in single) + "\n")
    rows = []
    for run_dir in a.run:
        res = evaluate_run(gt, load_run(run_dir), run_dir)
        name = Path(run_dir).name
        out = Path(a.out) if a.out and len(a.run) == 1 else Path(run_dir) / "markup_eval"
        write_outputs(out, name, res)
        o = res["overall"]
        neg = res["per_folder"].get("false", {})
        rows.append(f"| {name} | {_f(o['precision'])} | {_f(o['recall'])} | {_f(o['f1'])} | {o['tp']} | {o['fp']} | {o['fn']} "
                    f"| {res['total']['dup']} | {_f(res['total']['f1_nodup'])} | {neg.get('fp', '—')} "
                    f"| {_f(neg.get('fp_per_hour'), '.1f')} |")  # fmt: skip
        print(summary_md(name, res) + "\n")
    print("| прогон | P | R | F1 | TP | FP | FN | дублей | F1 без дублей | FP в false | FP/ч в false |\n"
          "|---|---|---|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
