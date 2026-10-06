"""Обработанный ролик: зона интереса, рамки людей, подсветка паузы у рта, вспышки и отметка события."""
from __future__ import annotations

import logging
import shutil
import subprocess
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .cycles import Cycle
from .episodes import Event
from .flash import Flash
from .geometry import TrackSignal
from .io_video import VideoMeta
from .roi import Roi

log = logging.getLogger(__name__)

RED, YELLOW, GREEN, GRAY, WHITE = (0, 0, 255), (0, 220, 255), (80, 200, 80), (150, 150, 150), (255, 255, 255)
ORANGE = (0, 140, 255)


def _label(frame, text, org, color, scale, thick):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = int(org[0]), int(max(org[1], h + 4))
    cv2.rectangle(frame, (x, y - h - base - 2), (x + w + 4, y + 2), color, -1)
    cv2.putText(frame, text, (x + 2, y - base), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)


def _start_datetime(start_time: Optional[str]) -> Optional[datetime]:
    if not start_time:
        return None
    try:
        return datetime.fromisoformat(start_time)
    except ValueError:
        return None


def render_video(
    meta: VideoMeta,
    stride: int,
    out_path: Path,
    signals: Dict[int, TrackSignal],
    cycles: Dict[int, List[Cycle]],
    events: List[Event],
    roi: Roi,
    title: str,
    start_time: Optional[str] = None,
    flashes: Optional[Dict[int, List[Flash]]] = None,
) -> Path:
    by_frame: Dict[int, list] = defaultdict(list)
    for tid, sig in signals.items():
        for i, f in enumerate(sig.frame_idx):
            by_frame[int(f)].append((tid, i))
    holds = {tid: [(c.hold_start, c.hold_end) for c in cs if c.valid] for tid, cs in cycles.items()}
    lit = {tid: [(f.start, f.end) for f in fs] for tid, fs in (flashes or {}).items()}
    shown = [e for e in events if e.passed]
    t0 = _start_datetime(start_time)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.mp4")
    cap = cv2.VideoCapture(meta.path)
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), meta.fps, (meta.width, meta.height))
    thick = max(1, round(meta.height / 540))
    fscale = 0.5 * thick
    current: list = []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            current = by_frame.get(idx, [])
        t = idx / meta.fps
        roi.draw(frame, thickness=thick)
        alarms = 0
        for tid, i in current:
            sig = signals[tid]
            x1, y1, x2, y2 = sig.boxes[i].astype(int)
            event = next((e for e in shown if e.track_id == tid and e.start <= t <= e.end), None)
            in_hold = any(a <= t <= b for a, b in holds.get(tid, []))
            if event is not None:
                alarms += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 2 * thick)
                _label(frame, f"SMOKING-LIKE {event.confidence:.2f} id{tid}", (x1, y1 - 2), RED, fscale, thick)
            else:
                color = YELLOW if in_hold else (GREEN if sig.valid[i] else GRAY)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
                _label(frame, f"id{tid}", (x1, y1 - 2), color, fscale * 0.8, thick)
            if any(a <= t <= b for a, b in lit.get(tid, [])):
                _label(frame, "FLASH", (x1, y2 + 4), ORANGE, fscale * 0.8, thick)
            if not np.isnan(sig.mouth[i]).any():
                cv2.circle(frame, tuple(sig.mouth[i].astype(int)), 2 * thick, WHITE, -1)
            if not np.isnan(sig.hand[i]).any():
                cv2.circle(frame, tuple(sig.hand[i].astype(int)), 2 * thick, YELLOW if in_hold else WHITE, -1)
        stamp = (t0 + timedelta(seconds=t)).strftime("%Y-%m-%d %H:%M:%S") if t0 else f"t={t:8.2f}s"
        banner = f"{title}  {stamp}" + (f"  ALARM x{alarms}" if alarms else "")
        _label(frame, banner, (8, 8), RED if alarms else WHITE, fscale, thick)
        writer.write(frame)
        idx += 1
    cap.release()
    writer.release()

    # mp4v не воспроизводится в браузерах — перекодируем в H.264, если есть ffmpeg
    if shutil.which("ffmpeg"):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264",
               "-pix_fmt", "yuv420p", "-crf", "23", str(out_path)]  # fmt: skip
        if subprocess.run(cmd).returncode == 0:
            tmp.unlink()
            return out_path
        log.warning("ffmpeg не смог перекодировать ролик, оставляю mp4v")
    tmp.replace(out_path)
    return out_path
