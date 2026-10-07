"""Правило положительного эпизода и объединение циклов в события.

Цикл «квалифицирован», если в пределах окна 20 с у того же человека есть ещё полный цикл
(всего >= min_cycles в окне) или у самого цикла есть прямой признак (предмет/дым от верификатора,
если flash_as_direct — вспышка зажигалки рядом по времени, если vapor_as_direct — выдох пара,
если appearance_as_direct — сигарета у рта по CLIP, look).
Цикл, начавшийся сразу после конца предыдущего (зазор <= min_cycle_gap: конец отвода и начало
подхода — один отсчёт), считается продолжением того же движения: кисть лишь качнулась от рта
без паузы вдали. В правиле min_cycles такая пара идёт за один цикл (в debug — joined_prev).
Квалифицированные циклы одного человека объединяются в событие при паузе <= 15 с.
События разных людей не объединяются. Событие без прямого признака отклоняется, если «чужие»
предметы (питьё, еда) видны в большинстве его циклов: по отдельности слабые детекции
не отбрасывают циклы, но их повторяемость по событию — надёжный признак.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .config import EpisodeConfig
from .cycles import Cycle
from .geometry import TrackSignal


@dataclass
class Event:
    track_id: int
    start: float
    end: float
    peak: float
    box: np.ndarray
    confidence: float
    cycles: List[Cycle]
    passed: bool  # confidence >= порога и событие не отклонено
    reject_reason: str = ""

    @property
    def direct(self) -> bool:
        return any(c.direct for c in self.cycles)

    @property
    def flash(self) -> bool:
        return any(c.flash for c in self.cycles)

    @property
    def vapor(self) -> bool:
        return any(c.vapor for c in self.cycles)

    @property
    def look(self) -> bool:
        return any(c.look for c in self.cycles)

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "peak": round(self.peak, 3),
            "box": [round(float(v), 1) for v in self.box],
            "confidence": round(self.confidence, 4),
            "passed": self.passed,
            "n_cycles": len(self.cycles),
            "n_movements": len(self.cycles) - sum(c.joined_prev for c in self.cycles[1:]),
            "direct": self.direct,
            "flash": self.flash,
            "vapor": self.vapor,
            "look": self.look,
            "reject_reason": self.reject_reason,
        }


def has_evidence(c: Cycle, cfg: EpisodeConfig) -> bool:
    """Прямой признак у цикла: предмет от верификатора, вспышка зажигалки, выдох пара или сигарета по CLIP."""
    return (c.direct or (cfg.flash_as_direct and c.flash) or (cfg.vapor_as_direct and c.vapor)
            or (cfg.appearance_as_direct and c.look))  # fmt: skip


def mark_joined(good: List[Cycle], cfg: EpisodeConfig) -> List[int]:
    """Отмечает joined_prev у валидных циклов (по времени) и возвращает номер движения каждого цикла."""
    movement, ids = -1, []
    for j, c in enumerate(good):
        c.joined_prev = j > 0 and c.start - good[j - 1].end <= cfg.min_cycle_gap
        movement += 0 if c.joined_prev else 1
        ids.append(movement)
    return ids


def qualifying_cycles(cycles: List[Cycle], cfg: EpisodeConfig) -> List[Cycle]:
    for c in cycles:
        c.joined_prev = False
    good = sorted((c for c in cycles if c.valid), key=lambda c: c.start)
    movement = mark_joined(good, cfg)
    qual = [has_evidence(c, cfg) for c in good]
    for a in good:
        members = [j for j, b in enumerate(good) if b.start >= a.start and b.end <= a.start + cfg.window]
        if len({movement[j] for j in members}) >= cfg.min_cycles:
            for j in members:
                qual[j] = True
    return [c for c, q in zip(good, qual) if q]


def negative_cycle_fraction(cycles: List[Cycle]) -> float:
    """Доля циклов, в которых верификатор видел «чужой» предмет."""
    return float(np.mean([c.neg_fraction > 0 for c in cycles])) if cycles else 0.0


def event_confidence(cycles: List[Cycle], cfg: EpisodeConfig) -> float:
    miss = 1.0
    for c in cycles:
        miss *= 1.0 - cfg.cycle_weight * c.quality
    conf = 1.0 - miss
    evidence = max((c.pos_conf for c in cycles if c.direct), default=0.0)
    ev = min(evidence / cfg.evidence_conf_ref, 1.0)
    if any(c.flash for c in cycles):  # вспышка повышает confidence и при flash_as_direct=false
        ev = max(ev, cfg.flash_evidence)
    if any(c.vapor for c in cycles):  # пар — тоже, независимо от vapor_as_direct
        ev = max(ev, cfg.vapor_evidence)
    if any(c.look for c in cycles):  # сигарета по CLIP — тоже, независимо от appearance_as_direct
        ev = max(ev, cfg.appearance_evidence)
    if ev > 0:
        conf = 1.0 - (1.0 - conf) * (1.0 - cfg.evidence_weight * ev)
    neg = float(np.mean([c.neg_fraction for c in cycles])) if cycles else 0.0
    conf *= 1.0 - cfg.negative_penalty * neg
    return float(np.clip(conf, 0.0, 1.0))


def event_box(sig: TrackSignal, peak: float, cfg: EpisodeConfig) -> np.ndarray:
    """Рамка человека в момент пика: покоординатная медиана рамок трека в ±box_window с
    (одиночная рамка дрожит и может быть искажена перекрытием), иначе — рамка ближайшего отсчёта."""
    if cfg.box_window > 0:
        near = sig.valid & (np.abs(sig.t - peak) <= cfg.box_window)
        if near.any():
            return np.median(sig.boxes[near], axis=0)
    return sig.boxes[sig.index_at(peak)].copy()


def build_events(
    cycles: Dict[int, List[Cycle]], signals: Dict[int, TrackSignal], cfg: EpisodeConfig
) -> List[Event]:
    events: List[Event] = []
    for tid, track_cycles in cycles.items():
        groups: List[List[Cycle]] = []
        for c in qualifying_cycles(track_cycles, cfg):
            if groups and c.start - max(x.end for x in groups[-1]) <= cfg.merge_gap:
                groups[-1].append(c)
            else:
                groups.append([c])
        sig = signals[tid]
        for group in groups:
            best = max(group, key=lambda c: c.quality)
            conf = event_confidence(group, cfg)
            reason = ""
            neg = negative_cycle_fraction(group)
            if not any(has_evidence(c, cfg) for c in group) and neg >= cfg.event_negative_fraction:
                reason = f"objects_in_{neg:.0%}_cycles"
            events.append(
                Event(
                    track_id=tid,
                    start=group[0].start,
                    end=max(c.end for c in group),
                    peak=best.peak,
                    box=event_box(sig, best.peak, cfg),
                    confidence=conf,
                    cycles=group,
                    passed=conf >= cfg.confidence_threshold and not reason,
                    reject_reason=reason,
                )
            )
    events.sort(key=lambda e: (e.start, e.track_id))
    return events
