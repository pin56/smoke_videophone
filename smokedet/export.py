"""Запись результатов: предсказания (CSV/JSON), manifest.json, журнал времени, отладочные данные."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from . import __version__
from .config import Config
from .cycles import Cycle
from .episodes import Event
from .flash import Flash

PRED_COLUMNS = [
    "camera_id", "clip_id", "event_id", "start_sec", "end_sec", "confidence", "label",
    "person_track_id", "peak_sec", "x1", "y1", "x2", "y2",
]  # fmt: skip
LOG_COLUMNS = [
    "camera_id", "clip_id", "video_path", "start_time", "duration_sec", "fps", "width", "height",
    "processed_frames", "tracking_sec", "analysis_sec", "render_sec", "total_sec", "realtime_factor",
    "events", "device", "from_cache",
]  # fmt: skip


def event_records(events: List[Event], camera_id: str, clip_id: str) -> List[dict]:
    rows = []
    for n, e in enumerate((e for e in events if e.passed), start=1):
        x1, y1, x2, y2 = (round(float(v), 1) for v in e.box)
        rows.append(
            {
                "camera_id": camera_id,
                "clip_id": clip_id,
                "event_id": f"{clip_id}_{n:03d}",
                "start_sec": round(e.start, 3),
                "end_sec": round(e.end, 3),
                "confidence": round(e.confidence, 4),
                "label": "smoking_like",
                "person_track_id": int(e.track_id),
                "peak_sec": round(e.peak, 3),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            }  # fmt: skip
        )
    return rows


def write_predictions(rows: List[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PRED_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    (out_dir / "predictions.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def write_processing_log(rows: List[dict], out_dir: Path) -> None:
    with open(out_dir / "processing_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        w.writeheader()
        w.writerows(rows)


def write_debug(
    out_dir: Path,
    clip_id: str,
    cycles: Dict[int, List[Cycle]],
    events: List[Event],
    flashes: Optional[Dict[int, List[Flash]]] = None,
) -> None:
    path = out_dir / "debug" / f"{clip_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "events": [e.to_dict() for e in events],
        "cycles": [c.to_dict() for cs in cycles.values() for c in cs],
        "flashes": [f.to_dict() for fs in (flashes or {}).values() for f in fs],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def file_sha256(path: str) -> Optional[str]:
    p = Path(path)
    if not p.is_absolute():
        p = Path(__file__).resolve().parent.parent / p
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(repo: Path) -> Optional[str]:
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True)
        return sha.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return None


def hardware_info(device: str) -> dict:
    info = {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
            "device": device}  # fmt: skip
    try:
        import torch

        info["torch"] = torch.__version__
        if device not in ("cpu", "mps"):
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import ultralytics

        info["ultralytics"] = ultralytics.__version__
    except Exception:
        pass
    return info


def write_manifest(out_dir: Path, cfg: Config, device: str, verifier: Optional[dict], repo: Path) -> dict:
    ep = cfg.episode
    manifest = {
        "team": cfg.report.team,
        "model_version": cfg.report.model_version,
        "code_version": __version__,
        "git_commit": git_commit(repo),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "label": "smoking_like",
        "confidence_threshold": ep.confidence_threshold,
        "merge_rules": {
            "positive_rule": f">= {ep.min_cycles} полных цикла в окне {ep.window} с ИЛИ 1 цикл + прямой признак",
            "window_sec": ep.window,
            "min_cycles": ep.min_cycles,
            "merge_gap_sec": ep.merge_gap,
            "merge_across_people": False,
            "hold_sec": [cfg.cycle.min_hold, cfg.cycle.max_hold],
            "min_person_height_px": cfg.tracking.min_person_height,
        },
        "models": {
            "pose": {"weights": cfg.detector.pose_model, "sha256": file_sha256(cfg.detector.pose_model),
                     "license": "AGPL-3.0 (Ultralytics)", "imgsz": cfg.detector.imgsz},
            "tracker": {"config": cfg.detector.tracker, "type": "ByteTrack (Ultralytics)"},
            "verifier": None if verifier is None else {**verifier, "sha256": file_sha256(verifier["weights"]),
                                                       "license": "AGPL-3.0 (Ultralytics)"},
        },  # fmt: skip
        "prompts": None if verifier is None else (verifier.get("positive_classes", []) + verifier.get("negative_classes", [])
                                                  + verifier.get("track_negative_classes", [])),
        "data_sources": cfg.report.data_sources,
        "fine_tuning": cfg.report.fine_tuning,
        "hardware": hardware_info(device),
        "config_sha256": cfg.hash(),
        "config": cfg.to_dict(),
        "notes": cfg.report.notes,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
