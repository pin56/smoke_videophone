"""Поиск полных циклов жеста: кисть подходит ко рту → пауза ~0.5–5 с → кисть отводится.

Конечный автомат с гистерезисом по d(t): вход «у рта» при d < near_enter, выход при d > near_exit.
Начало цикла — начало подъёма руки (последний момент перед паузой, когда кисть была далеко),
конец — завершение отвода. Отброшенные циклы сохраняются с причиной для разбора ошибок.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .config import CycleConfig
from .geometry import TrackSignal

HOLD_LEVEL = 0.5  # пауза: d <= d_min + HOLD_LEVEL * (near_enter - d_min)


@dataclass
class Cycle:
    track_id: int
    start: float  # начало движения руки ко рту
    hold_start: float
    hold_end: float
    end: float  # завершение отвода руки
    peak: float  # момент наибольшего сближения кисти и рта
    hold_idx: List[int]  # индексы отсчётов сигнала внутри паузы
    min_dist: float
    kconf: float
    ear_ratio: float
    wrist_above: float
    quality: float = 0.0
    valid: bool = True
    reject_reason: str = ""
    # заполняет верификатор предметов
    direct: bool = False
    pos_frames: int = 0
    pos_conf: float = 0.0
    neg_fraction: float = 0.0
    objects: Dict[str, float] = field(default_factory=dict)

    @property
    def hold(self) -> float:
        return self.hold_end - self.hold_start

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "start": round(self.start, 3),
            "hold_start": round(self.hold_start, 3),
            "hold_end": round(self.hold_end, 3),
            "end": round(self.end, 3),
            "peak": round(self.peak, 3),
            "hold": round(self.hold, 3),
            "min_dist": round(self.min_dist, 3),
            "kconf": round(self.kconf, 3),
            "ear_ratio": None if np.isnan(self.ear_ratio) else round(self.ear_ratio, 3),
            "wrist_above": None if np.isnan(self.wrist_above) else round(self.wrist_above, 3),
            "quality": round(self.quality, 3),
            "valid": self.valid,
            "reject_reason": self.reject_reason,
            "direct": self.direct,
            "pos_frames": self.pos_frames,
            "pos_conf": round(self.pos_conf, 3),
            "neg_fraction": round(self.neg_fraction, 3),
            "objects": {k: round(v, 3) for k, v in self.objects.items()},
        }


def _nanmedian(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    return float(np.median(x)) if x.size else float("nan")


def cycle_quality(hold: float, min_dist: float, kconf: float, cfg: CycleConfig) -> float:
    """0..1: правдоподобие длительности паузы × глубина сближения × уверенность точек."""
    if hold < 0.8:
        f_hold = 0.6 + 0.4 * np.clip((hold - cfg.min_hold) / max(0.8 - cfg.min_hold, 1e-6), 0, 1)
    elif hold > 3.5:
        f_hold = 0.6 + 0.4 * np.clip((cfg.max_hold - hold) / max(cfg.max_hold - 3.5, 1e-6), 0, 1)
    else:
        f_hold = 1.0
    f_depth = 0.5 + 0.5 * float(np.clip(1 - min_dist / cfg.near_enter, 0, 1))
    f_conf = float(np.clip(kconf / 0.6, 0.3, 1.0))
    return float(f_hold * f_depth * f_conf)


def _make_cycle(sig: TrackSignal, cfg: CycleConfig, hs: int, he: int, exit_i: int, prev_end: int):
    t, d = sig.t, sig.d
    n = len(t)
    seg = d[hs : he + 1]
    d_min = float(np.nanmin(seg))
    peak_i = hs + int(np.nanargmin(seg))
    # пауза — где кисть почти неподвижна у рта; полоса гистерезиса захватывает и части подъёма/отвода
    level = d_min + HOLD_LEVEL * max(cfg.near_enter - d_min, 0.0)
    hold_idx = [k for k in range(hs, he + 1) if not np.isnan(d[k]) and d[k] <= level]
    hold_first, hold_last = hold_idx[0], hold_idx[-1]
    reasons = []

    # фаза подхода: кисть до паузы должна быть далеко от рта
    pre = [j for j in range(prev_end, hs) if t[hs] - t[j] <= cfg.lookaround and not np.isnan(d[j])]
    if pre and max(d[j] for j in pre) >= cfg.near_exit:
        d_max = max(d[j] for j in pre)
        thr = d_min + 0.9 * (d_max - d_min)
        start_i = max(j for j in pre if d[j] >= thr)
    else:
        start_i = pre[0] if pre else hs
        reasons.append("no_approach")

    # фаза отвода: до следующего входа к рту или конца окна
    post = []
    for k in range(exit_i, n):
        if t[k] - t[exit_i] > cfg.lookaround:
            break
        if np.isnan(d[k]):
            continue
        if k > exit_i and d[k] < cfg.near_enter:
            break
        post.append(k)
    d_max = max(d[k] for k in post)
    thr = d_min + 0.9 * (d_max - d_min)
    end_i = next(k for k in post if d[k] >= thr)

    hold = float(t[hold_last] - t[hold_first]) + sig.dt
    if hold < cfg.min_hold:
        reasons.append("short_hold")
    elif hold > cfg.max_hold:
        reasons.append("long_hold")

    idx = np.array(hold_idx)
    ear = _nanmedian(sig.ear_ratio[idx])
    above = _nanmedian(sig.wrist_above[idx])
    kconf = float(np.median(sig.kconf[idx]))
    if not np.isnan(ear) and ear < cfg.phone_ear_ratio:
        reasons.append("phone_like")
    if not np.isnan(above) and above < cfg.min_wrist_above_elbow:
        reasons.append("hand_low")

    cycle = Cycle(
        track_id=sig.track_id,
        start=float(t[start_i]),
        hold_start=float(t[hold_first]),
        hold_end=float(t[hold_last]) + sig.dt,
        end=float(t[end_i]),
        peak=float(t[peak_i]),
        hold_idx=hold_idx,
        min_dist=d_min,
        kconf=kconf,
        ear_ratio=ear,
        wrist_above=above,
        quality=cycle_quality(hold, d_min, kconf, cfg),
        valid=not reasons,
        reject_reason=",".join(reasons),
    )
    return cycle, end_i


def detect_cycles(sig: TrackSignal, cfg: CycleConfig) -> List[Cycle]:
    t, d = sig.t, sig.d
    cycles: List[Cycle] = []
    near = False
    hs = last_near = 0
    prev_end = 0
    for i in range(len(t)):
        di = d[i]
        if np.isnan(di):
            if near and t[i] - t[last_near] > cfg.max_missing:
                near = False  # кисть или лицо пропали посреди паузы — цикл неполный
            continue
        if not near:
            if di < cfg.near_enter:
                near, hs, last_near = True, i, i
        elif di > cfg.near_exit:
            cycle, end_i = _make_cycle(sig, cfg, hs, last_near, i, prev_end)
            cycles.append(cycle)
            prev_end = end_i
            near = False
        else:
            last_near = i
    return cycles
