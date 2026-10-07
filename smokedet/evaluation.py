"""Оценка по протоколу кейса.

Пара (эталон, предсказание): совпадают camera_id и clip_id; интервалы пересекаются не менее
чем на 1 с после расширения эталона на 3 с в обе стороны; IoU рамок человека >= 0.30.
Сопоставление один к одному, жадно по наибольшему временному пересечению.
Предсказание без пары не считается FP, если >= 50% его длительности попадает на IGNORE
и оно не пересекает положительный эталон.

Формат эталона (CSV/JSON): camera_id, clip_id, label (POSITIVE|NEGATIVE|IGNORE), start_sec, end_sec,
[person_gt_id, peak_sec, x1, y1, x2, y2, scenario]. Рамка эталона сравнивается с рамкой предсказания
в его peak_sec. Если у эталона нет рамки, проверка IoU для него пропускается.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class EvalParams:
    gt_expand: float = 3.0
    min_overlap: float = 1.0
    min_iou: float = 0.30
    ignore_fraction: float = 0.5


@dataclass
class Item:
    camera_id: str
    clip_id: str
    start: float
    end: float
    box: Optional[Tuple[float, float, float, float]]
    label: str = "POSITIVE"
    confidence: float = 1.0
    row: dict = field(default_factory=dict)

    @property
    def key(self) -> Tuple[str, str]:
        return self.camera_id, self.clip_id


def load_rows(path: str) -> List[dict]:
    p = Path(path)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("events") or data.get("annotations") or data.get("items") or []
        return list(data)
    with open(p, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _box(row: dict):
    try:
        vals = [float(row[k]) for k in ("x1", "y1", "x2", "y2")]
    except (KeyError, TypeError, ValueError):
        return None
    return tuple(vals)


def parse_items(rows: List[dict], default_label: str = "POSITIVE") -> List[Item]:
    items = []
    for r in rows:
        conf = r.get("confidence")
        items.append(
            Item(
                camera_id=str(r["camera_id"]),
                clip_id=str(r["clip_id"]),
                start=float(r["start_sec"]),
                end=float(r["end_sec"]),
                box=_box(r),
                label=str(r.get("label") or default_label).upper(),
                confidence=1.0 if conf is None or conf == "" else float(conf),  # число 0 в JSON — не «нет»
                row=dict(r),
            )
        )
    return items


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def union_length(intervals: List[Tuple[float, float]]) -> float:
    total, cur = 0.0, None
    for s, e in sorted(intervals):
        if cur is None or s > cur[1]:
            if cur is not None:
                total += cur[1] - cur[0]
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    if cur is not None:
        total += cur[1] - cur[0]
    return total


def evaluate(
    preds: List[Item],
    gts: List[Item],
    durations: Optional[Dict[Tuple[str, str], float]] = None,
    params: EvalParams = EvalParams(),
    min_confidence: float = 0.0,
) -> dict:
    preds = [p for p in preds if p.confidence >= min_confidence]
    positives = [g for g in gts if g.label == "POSITIVE"]
    ignores = [g for g in gts if g.label == "IGNORE"]
    negatives = [g for g in gts if g.label == "NEGATIVE"]

    candidates = []
    for gi, g in enumerate(positives):
        g0, g1 = g.start - params.gt_expand, g.end + params.gt_expand
        for pi, p in enumerate(preds):
            if p.key != g.key:
                continue
            ov = overlap(g0, g1, p.start, p.end)
            if ov < params.min_overlap:
                continue
            if g.box is not None and p.box is not None and iou(g.box, p.box) < params.min_iou:
                continue
            candidates.append((ov, gi, pi))
    candidates.sort(key=lambda x: -x[0])
    gt_match: Dict[int, int] = {}
    pred_match: Dict[int, int] = {}
    for ov, gi, pi in candidates:
        if gi not in gt_match and pi not in pred_match:
            gt_match[gi] = pi
            pred_match[pi] = gi

    rows = []
    latencies = []
    for gi, g in enumerate(positives):
        if gi in gt_match:
            p = preds[gt_match[gi]]
            latencies.append(max(0.0, p.start - g.start))
            rows.append(_row("TP", g, p))
        else:
            rows.append(_row("FN", g, None))

    fp = ignored = 0
    for pi, p in enumerate(preds):
        if pi in pred_match:
            continue
        dur = max(p.end - p.start, 1e-6)
        ign = [(max(i.start, p.start), min(i.end, p.end)) for i in ignores if i.key == p.key and overlap(i.start, i.end, p.start, p.end) > 0]
        hits_pos = any(g.key == p.key and overlap(g.start, g.end, p.start, p.end) > 0 for g in positives)
        if union_length(ign) / dur >= params.ignore_fraction and not hits_pos:
            ignored += 1
            rows.append(_row("IGNORED", None, p))
            continue
        fp += 1
        neg = max((n for n in negatives if n.key == p.key), key=lambda n: overlap(n.start, n.end, p.start, p.end), default=None)
        if neg is not None and overlap(neg.start, neg.end, p.start, p.end) <= 0:
            neg = None
        rows.append(_row("FP", neg, p))

    tp = len(gt_match)
    fn = len(positives) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    neg_hours = None
    if durations:
        neg_sec = 0.0
        for key, dur in durations.items():
            busy = [(g.start, g.end) for g in positives + ignores if g.key == key]
            neg_sec += max(0.0, dur - union_length([(max(0.0, s), min(dur, e)) for s, e in busy]))
        neg_hours = neg_sec / 3600.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ignored_preds": ignored,
        "n_gt_positive": len(positives),
        "n_pred": len(preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "negative_hours": neg_hours,
        "fp_per_hour": (fp / neg_hours) if neg_hours else None,
        "latency_median_sec": float(np.median(latencies)) if latencies else None,
        "matches": rows,
    }


def _row(kind: str, g: Optional[Item], p: Optional[Item]) -> dict:
    ref = g or p
    return {
        "result": kind,
        "camera_id": ref.camera_id,
        "clip_id": ref.clip_id,
        "gt_label": g.label if g else "",
        "gt_scenario": (g.row.get("scenario") or "") if g else "",
        "gt_person": (g.row.get("person_gt_id") or "") if g else "",
        "gt_start": g.start if g else "",
        "gt_end": g.end if g else "",
        "pred_event_id": p.row.get("event_id", "") if p else "",
        "pred_track": p.row.get("person_track_id", "") if p else "",
        "pred_start": p.start if p else "",
        "pred_end": p.end if p else "",
        "pred_conf": p.confidence if p else "",
        "iou": round(iou(g.box, p.box), 3) if g and p and g.box and p.box else "",
    }


def load_durations(path: Optional[str]) -> Optional[Dict[Tuple[str, str], float]]:
    """Длительности клипов из processing_log.csv (или любого CSV с camera_id, clip_id, duration_sec)."""
    if not path:
        return None
    out = {}
    for r in load_rows(path):
        out[(str(r["camera_id"]), str(r["clip_id"]))] = float(r["duration_sec"])
    return out


def summary_table(report: dict) -> str:
    def fmt(v, spec=".3f"):
        return "n/a" if v is None else format(v, spec)

    lines = [
        f"TP={report['tp']}  FP={report['fp']}  FN={report['fn']}  (в IGNORE: {report['ignored_preds']})",
        f"precision={fmt(report['precision'])}  recall={fmt(report['recall'])}  F1={fmt(report['f1'])}",
        f"FP/час={fmt(report['fp_per_hour'], '.2f')}  (отрицательного видео {fmt(report['negative_hours'], '.3f')} ч)  "
        f"задержка (медиана)={fmt(report['latency_median_sec'], '.2f')} с",
    ]
    by_kind: Dict[str, Dict[str, int]] = {}
    for r in report["matches"]:
        if r["result"] in ("FP", "FN") and r["gt_scenario"]:
            by_kind.setdefault(r["result"], {}).setdefault(r["gt_scenario"], 0)
            by_kind[r["result"]][r["gt_scenario"]] += 1
    for kind, counts in by_kind.items():
        lines.append(f"{kind} по сценариям: " + ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())))
    return "\n".join(lines)
