"""Обработанный ролик: зона интереса, рамки людей, подсветка паузы у рта, вспышки и отметка события."""
from __future__ import annotations

import logging
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .cycles import Cycle
from .episodes import Event
from .flash import Flash
from .geometry import TrackSignal
from .io_video import FrameClock, VideoMeta
from .roi import Roi
from .tracking import PersonObs

log = logging.getLogger(__name__)

RED, YELLOW, GREEN, GRAY, WHITE = (0, 0, 255), (0, 220, 255), (80, 200, 80), (150, 150, 150), (255, 255, 255)
ORANGE = (0, 140, 255)


@dataclass
class Mark:
    """Состояние человека, известное потоку с t0 по t1: рамка цвета color и подпись text ({dt} — секунд с t0).
    Заменяет отметку события из events: показ «как в потоке», без знания будущего."""

    t0: float
    t1: float
    text: str
    color: tuple
    alarm: bool = False  # тревога: толстая рамка и ALARM в шапке


def _label(frame, text, org, color, scale, thick):
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = int(org[0]), int(max(org[1], h + 4))
    cv2.rectangle(frame, (x, y - h - base - 2), (x + w + 4, y + 2), color, -1)
    cv2.putText(frame, text, (x + 2, y - base), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)


def _phase(cs: List[Cycle], t: float):
    """Фаза цикла человека в момент t: (текст, цвет, цикл) или None."""
    for c in cs:
        if c.start <= t <= c.end + (2.0 if not c.valid else 0.0):
            if not c.valid:
                return f"REJECT {c.reject_reason}", GRAY, c
            if t < c.hold_start:
                return "HAND TO MOUTH", YELLOW, c
            if t <= c.hold_end:
                return f"HOLD {t - c.hold_start:.1f}s", ORANGE, c
            return "HAND DOWN", YELLOW, c
    return None


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
    hold: int = 0,
    others: Optional[Dict[int, List[PersonObs]]] = None,
    marks: Optional[Dict[int, List[Mark]]] = None,
    panel: Optional[Callable[[float], List[Tuple[str, tuple]]]] = None,
    tail: float = 0.0,
) -> Path:
    """hold > 0 (потоковый режим: отсчёты не на каждом stride-м кадре) — последняя рамка человека держится hold кадров;
    others — отсчёты людей вне плотной проверки (только рамка). marks — состояние людей по времени вместо отметки
    событий из events, panel(t) — строки панели внизу слева. tail — секунд стоп-кадра после конца видео (ответы,
    пришедшие после конца клипа: поток ждёт их, как живая камера)."""
    by_frame: Dict[int, list] = defaultdict(list)
    for tid, sig in signals.items():
        for i, f in enumerate(sig.frame_idx):
            by_frame[int(f)].append((tid, i))
    other_by_frame: Dict[int, list] = defaultdict(list)
    for tid, obs in (others or {}).items():
        for o in obs:
            other_by_frame[o.frame_idx].append((tid, o.box))
    last: Dict[int, tuple] = {}  # tid -> (индекс отсчёта, кадр), при hold
    last_other: Dict[int, tuple] = {}
    holds = {tid: [(c.hold_start, c.hold_end) for c in cs if c.valid] for tid, cs in cycles.items()}
    lit = {tid: [(f.start, f.end) for f in fs] for tid, fs in (flashes or {}).items()}
    puffs = {tid: [(c.vapor_t, c.vapor_t + 1.0) for c in cs if c.vapor] for tid, cs in cycles.items()}
    shown = [e for e in events if e.passed]
    all_cycles = {tid: sorted(cs, key=lambda c: c.start) for tid, cs in cycles.items()}
    t0 = _start_datetime(start_time)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.mp4")
    cap = cv2.VideoCapture(meta.path)
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), meta.fps, (meta.width, meta.height))
    thick = max(1, round(meta.height / 540))
    fscale = 0.5 * thick
    current: list = []
    clock = FrameClock(meta.fps)
    idx = 0
    raw, t, t_end = None, 0.0, None
    while True:
        ok, frame = cap.read()
        if ok:
            t = clock.time(idx, cap.get(cv2.CAP_PROP_POS_MSEC))  # то же время, что у наблюдений трекера
            raw = frame.copy() if tail > 0 else None
        else:
            if raw is None:
                break
            t_end = t if t_end is None else t_end
            t += 1.0 / meta.fps
            if t > t_end + tail:
                break
            frame = raw.copy()  # стоп-кадр: рамки людей застывают на последнем отсчёте
        roi.draw(frame, thickness=thick)
        if hold and t_end is None:
            for tid, i in by_frame.get(idx, []):
                last[tid] = (i, idx)
            for tid, box in other_by_frame.get(idx, []):
                last_other[tid] = (box, idx)
            current = [(tid, i) for tid, (i, f) in last.items() if idx - f <= hold]
            seen = {tid for tid, _ in current}
            for tid, (box, f) in last_other.items():
                if idx - f <= hold and tid not in seen:
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, thick)
                    _label(frame, f"id{tid}", (x1, y1 - 2), GREEN, fscale * 0.8, thick)
        elif t_end is None and idx % stride == 0:
            current = by_frame.get(idx, [])
        alarms = 0
        for tid, i in current:
            sig = signals[tid]
            x1, y1, x2, y2 = sig.boxes[i].astype(int)
            event = next((e for e in shown if e.track_id == tid and e.start <= t <= e.end), None)
            in_hold = any(a <= t <= b for a, b in holds.get(tid, []))
            phase = _phase(all_cycles.get(tid, []), t)
            active = [m for m in (marks or {}).get(tid, []) if m.t0 <= t <= m.t1]
            mark = max(reversed(active), key=lambda m: m.t0, default=None)  # при равном t0 — более поздняя запись
            if marks is not None:
                event = None
            if mark is not None:
                alarms += mark.alarm
                cv2.rectangle(frame, (x1, y1), (x2, y2), mark.color, (2 if mark.alarm else 1) * thick)
                _label(frame, f"{mark.text.format(dt=t - mark.t0)} id{tid}", (x1, y1 - 2), mark.color, fscale, thick)
            elif event is not None:
                alarms += 1
                cv2.rectangle(frame, (x1, y1), (x2, y2), RED, 2 * thick)
                _label(frame, f"SMOKING-LIKE {event.confidence:.2f} id{tid}", (x1, y1 - 2), RED, fscale, thick)
            else:
                color = YELLOW if in_hold else (GREEN if sig.valid[i] else GRAY)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
                _label(frame, f"id{tid}", (x1, y1 - 2), color, fscale * 0.8, thick)
            small = fscale * 0.7
            row = y2 + 4
            step = int(18 * small * 1.6) + 2
            if any(a <= t <= b for a, b in lit.get(tid, [])):
                _label(frame, "FLASH", (x1, row), ORANGE, small, thick)
                row += step
            elif any(a <= t <= b for a, b in puffs.get(tid, [])):
                _label(frame, "VAPOR", (x1, row), WHITE, small, thick)
                row += step
            if phase is not None:
                text, pcolor, c = phase
                _label(frame, text, (x1, row), pcolor, small, thick)
                row += step
                info = []
                if not np.isnan(c.appearance):
                    info.append(f"clip {c.appearance:.1f}")
                if c.objects:
                    name, conf = max(c.objects.items(), key=lambda kv: kv[1])
                    info.append(f"{name} {conf:.2f}")
                if info:
                    _label(frame, " ".join(info), (x1, row), pcolor, small, thick)
                    row += step
            if not np.isnan(sig.mouth[i]).any():
                cv2.circle(frame, tuple(sig.mouth[i].astype(int)), 2 * thick, WHITE, -1)
            if not np.isnan(sig.hand[i]).any():
                hand = tuple(sig.hand[i].astype(int))
                cv2.circle(frame, hand, 2 * thick, YELLOW if in_hold else WHITE, -1)
                if not np.isnan(sig.mouth[i]).any():
                    cv2.line(frame, hand, tuple(sig.mouth[i].astype(int)), YELLOW if in_hold else GRAY, thick, cv2.LINE_AA)
                    if not np.isnan(sig.d[i]):
                        _label(frame, f"d={sig.d[i]:.2f}", (hand[0] + 6, hand[1]), WHITE, small, thick)
        n_cyc = sum(1 for cs in cycles.values() for c in cs if c.valid and c.end <= t)
        n_rej = sum(1 for cs in cycles.values() for c in cs if not c.valid and c.end <= t)
        n_ev = sum(1 for e in shown if e.end <= t or e.start <= t)
        if marks is not None:
            n_ev = sum(1 for ms in marks.values() for m in ms if m.alarm and m.t0 <= t)
        if panel is not None:
            lines = panel(t)
            y = meta.height - 8 - int(36 * fscale * 0.8) * (len(lines) - 1)
            for text, color in lines:
                _label(frame, text, (8, y), color, fscale * 0.8, thick)
                y += int(36 * fscale * 0.8)
        _label(frame, f"people {len(current)}  cycles {n_cyc}  rejected {n_rej}  events {n_ev}", (8, 8 + int(40 * fscale)),
               GREEN, fscale * 0.8, thick)
        stamp = (t0 + timedelta(seconds=t)).strftime("%Y-%m-%d %H:%M:%S") if t0 else f"t={t:8.2f}s"
        banner = f"{title}  {stamp}" + ("  END OF CLIP" if t_end is not None else "") + (f"  ALARM x{alarms}" if alarms else "")
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
