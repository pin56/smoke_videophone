"""Правило положительного эпизода и объединение циклов в события.

Цикл «квалифицирован», если в пределах окна 20 с у того же человека есть ещё полный цикл
(всего >= min_cycles в окне) или у самого цикла есть прямой признак (предмет/дым).
Квалифицированные циклы одного человека объединяются в событие при паузе <= 15 с.
События разных людей не объединяются.
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
    passed: bool  # confidence >= порога

    @property
    def direct(self) -> bool:
        return any(c.direct for c in self.cycles)

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
            "direct": self.direct,
        }


def qualifying_cycles(cycles: List[Cycle], cfg: EpisodeConfig) -> List[Cycle]:
    good = sorted((c for c in cycles if c.valid), key=lambda c: c.start)
    qual = [c.direct for c in good]
    for a in good:
        members = [j for j, b in enumerate(good) if b.start >= a.start and b.end <= a.start + cfg.window]
        if len(members) >= cfg.min_cycles:
            for j in members:
                qual[j] = True
    return [c for c, q in zip(good, qual) if q]


def event_confidence(cycles: List[Cycle], cfg: EpisodeConfig) -> float:
    miss = 1.0
    for c in cycles:
        miss *= 1.0 - cfg.cycle_weight * c.quality
    conf = 1.0 - miss
    evidence = max((c.pos_conf for c in cycles if c.direct), default=0.0)
    if evidence > 0:
        ev = min(evidence / cfg.evidence_conf_ref, 1.0)
        conf = 1.0 - (1.0 - conf) * (1.0 - cfg.evidence_weight * ev)
    neg = float(np.mean([c.neg_fraction for c in cycles])) if cycles else 0.0
    conf *= 1.0 - cfg.negative_penalty * neg
    return float(np.clip(conf, 0.0, 1.0))


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
            events.append(
                Event(
                    track_id=tid,
                    start=group[0].start,
                    end=max(c.end for c in group),
                    peak=best.peak,
                    box=sig.boxes[sig.index_at(best.peak)].copy(),
                    confidence=conf,
                    cycles=group,
                    passed=conf >= cfg.confidence_threshold,
                )
            )
    events.sort(key=lambda e: (e.start, e.track_id))
    return events
