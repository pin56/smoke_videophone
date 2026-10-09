#!/usr/bin/env python
"""Выгрузка циклов потокового прогона (run_rt.py) в кадры для ручной разметки (tools/label_server.py).

То же, что tools/export_cycles.py, но без повторного трекинга: циклы берутся из <run>/debug/<clip>.json,
рамки человека — из отсчётов плотной проверки в кэше <run>/cache/<clip>_rt.pkl. Кадры между отсчётами
не рисуются (частота кадров набора — realtime.dense_fps).

    python tools/export_rt_cycles.py --run runs/students/l168 --out data/labeling/students [--rejected all]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from export_cycles import render_cycle  # noqa: E402
from smokedet.io_video import FrameReader  # noqa: E402
from smokedet.pipeline import LOG_FORMAT  # noqa: E402

log = logging.getLogger("export_rt_cycles")


class ObsSignal:
    """Минимум TrackSignal для render_cycle: время, рамки и индексы кадров отсчётов одного человека."""

    def __init__(self, obs):
        self.t = np.array([o.t for o in obs])
        self.boxes = np.array([o.box for o in obs], dtype=np.float64)
        self.frame_idx = np.array([o.frame_idx for o in obs])

    def index_at(self, t: float) -> int:
        return int(np.argmin(np.abs(self.t - t)))


def person_obs(run) -> dict:
    """Отсчёты плотной проверки по людям, без повторов кадров (задания одного человека перекрываются)."""
    by_tid: dict = {}
    for rec in run.jobs:
        d = by_tid.setdefault(rec["tid"], {})
        for o in rec["obs"]:
            d[o.frame_idx] = o
    return {tid: [d[k] for k in sorted(d)] for tid, d in by_tid.items()}


def want(c: dict, mode: str) -> bool:
    if c["valid"] or mode == "all":
        return True
    return mode == "object" and "object" in c["reject_reason"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Выгрузка циклов потокового прогона для разметки")
    p.add_argument("--run", required=True, help="папка прогона run_rt.py (debug/, cache/, processing_log.csv)")
    p.add_argument("--out", required=True, help="папка набора для разметки")
    p.add_argument("--rejected", choices=["none", "object", "all"], default="all",
                   help="какие отброшенные циклы тоже выгружать (по умолчанию все)")  # fmt: skip
    p.add_argument("--pad", type=float, default=1.0, help="секунд контекста до и после цикла")
    p.add_argument("--clips", nargs="*", help="только эти clip_id")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")

    run_dir, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    index_path = out / "cycles.jsonl"
    records = {}
    if index_path.exists():
        records = {r["id"]: r for r in map(json.loads, index_path.read_text(encoding="utf-8").splitlines()) if r}

    with open(run_dir / "processing_log.csv", encoding="utf-8") as f:
        clips = [(r["clip_id"], r["camera_id"], r["video_path"]) for r in csv.DictReader(f)]
    for clip_id, camera_id, video_path in clips:
        if args.clips and clip_id not in args.clips:
            continue
        with open(run_dir / "cache" / f"{clip_id}_rt.pkl", "rb") as f:
            run = pickle.load(f)["run"]
        debug = json.loads((run_dir / "debug" / f"{clip_id}.json").read_text(encoding="utf-8"))
        obs = person_obs(run)
        w, h = run.meta.width, run.meta.height
        k_by_tid: dict = {}
        todo = []
        for c in sorted(debug["cycles"], key=lambda c: (c["track_id"], c["start"])):
            k = k_by_tid[c["track_id"]] = k_by_tid.get(c["track_id"], -1) + 1
            if want(c, args.rejected) and len(obs.get(c["track_id"], [])) >= 2:
                todo.append((c["start"], k, c))
        todo.sort(key=lambda x: x[0])
        log.info("[%s] циклов к выгрузке: %d", clip_id, len(todo))
        with FrameReader(video_path, synthetic_from=run.meta.synthetic_from) as reader:
            for _, k, c in todo:
                tid = c["track_id"]
                cid = f"{clip_id}__t{tid}_{k}"
                frames_dir = out / "frames" / cid
                if cid in records and frames_dir.is_dir():
                    continue
                times = render_cycle(reader, ObsSignal(obs[tid]), SimpleNamespace(**c), args.pad, frames_dir, w, h)
                if times is None:
                    log.warning("[%s] цикл %s: кадры не прочитаны", clip_id, cid)
                    continue
                in_event = any(e["passed"] and e["track_id"] == tid and e["start"] - 0.5 <= c["start"] <= e["end"] + 0.5
                               for e in debug["events"])  # fmt: skip
                records[cid] = {
                    "id": cid, "clip_id": clip_id, "camera_id": camera_id, "track_id": tid,
                    "start": round(c["start"], 2), "end": round(c["end"], 2),
                    "hold_start": round(c["hold_start"], 2), "hold_end": round(c["hold_end"], 2),
                    "times": times, "valid": c["valid"], "reject_reason": c["reject_reason"], "in_event": in_event,
                    "auto": {
                        "direct": c["direct"], "flash": c["flash"], "vapor": c["vapor"], "vapor_net": c["vapor_net"],
                        "neg_fraction": c["neg_fraction"], "strong_negative": c["strong_negative"],
                        "min_dist": c["min_dist"], "quality": c["quality"], "appearance": c.get("appearance"),
                        "look": c.get("look"), "rescued_from": c.get("rescued_from", ""),
                    },
                }  # fmt: skip
        index_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records.values()), encoding="utf-8")
    log.info("готово: %d циклов в %s", len(records), index_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
