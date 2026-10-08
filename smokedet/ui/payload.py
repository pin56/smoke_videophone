"""Данные для веб-интерфейса: треки + Analysis → JSON для повтора анализа синхронно с видео.

У циклов, событий и строк ленты есть reveal_t — момент видео, к которому анализ о них «знает»: цикл — после
отвода руки, событие — когда по закончившимся циклам выполнено правило эпизода. В режиме повтора интерфейс
показывает только открытое к текущему времени, поэтому повтор выглядит как живой анализ.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from ..config import Config, EpisodeConfig
from ..cycles import Cycle
from ..episodes import Event, has_evidence
from ..geometry import TrackSignal
from ..io_video import VideoMeta
from ..roi import Roi
from ..tracking import PersonObs

# причины отказа циклов (cycles.py) — понятным языком
CYCLE_REASONS = {
    "no_approach": "рука не подходила ко рту, а уже была у лица",
    "short_hold": "слишком короткая пауза у рта",
    "long_hold": "слишком долгая пауза у рта",
    "phone_like": "кисть у уха — похоже на телефон",
    "hand_low": "запястье ниже локтя",
    "hand_above_mouth": "кисть выше рта — поправляет волосы или очки",
    "face_hidden": "лицо не видно",
    "head_tilt": "голова запрокинута — похоже на питьё",
}

# классы верификатора (config.VerifierConfig)
OBJECT_NAMES = {
    "cigarette": "сигарета",
    "vape": "вейп",
    "electronic cigarette": "электронная сигарета",
    "smoke": "дым",
    "vapor cloud": "облако пара",
    "bottle": "бутылка",
    "water bottle": "бутылка воды",
    "cup": "стакан",
    "mug": "кружка",
    "drinking glass": "стакан",
    "beverage can": "банка напитка",
    "can": "банка",
    "jar": "банка",
    "glass jar": "стеклянная банка",
    "thermos": "термос",
    "cell phone": "телефон",
    "sandwich": "бутерброд",
    "food": "еда",
    "spoon": "ложка",
    "fork": "вилка",
    "lollipop": "чупа-чупс",
}

CYCLE_FORMS = ("цикл", "цикла", "циклов")


def plural(n: int, forms: Sequence[str]) -> str:
    n = abs(n) % 100
    if 11 <= n <= 14:
        return forms[2]
    n %= 10
    return forms[0] if n == 1 else forms[1] if 2 <= n <= 4 else forms[2]


def object_name(name: str) -> str:
    return OBJECT_NAMES.get(name, name)


def reason_text(reason: str) -> str:
    """reject_reason цикла или события (через запятую) → русский текст."""
    parts = []
    for r in filter(None, reason.split(",")):
        if r.startswith("track_object:"):
            parts.append(f"в руке {object_name(r.split(':', 1)[1])}")
        elif r.startswith("object:"):
            parts.append(f"у рта {object_name(r.split(':', 1)[1])}")
        elif r.startswith("objects_in_") and r.endswith("_cycles"):
            parts.append(f"«чужие» предметы в {r[len('objects_in_'):-len('_cycles')]} циклов")
        else:
            parts.append(CYCLE_REASONS.get(r, r))
    return "; ".join(parts)


def evidence(c: Cycle, positive: Sequence[str]) -> List[str]:
    """Прямые признаки цикла: предмет курения от верификатора, вспышка, выдох."""
    out = []
    if c.direct:
        found = {k: v for k, v in c.objects.items() if k in positive}
        name = max(found, key=found.get) if found else ""
        out.append(f"{object_name(name)} {found[name]:.2f}" if name else "предмет курения")
    if c.flash:
        out.append("вспышка зажигалки")
    if c.vapor:
        out.append("выдох дыма или пара")
    return out


def movements(cycles: Sequence[Cycle]) -> int:
    """Число отдельных движений руки: цикл с joined_prev — продолжение предыдущего (episodes.mark_joined)."""
    return sum(1 for i, c in enumerate(cycles) if i == 0 or not c.joined_prev)


def event_reveal_t(event: Event, ep: EpisodeConfig) -> float:
    """Когда анализ «узнаёт» о событии. Принятое — конец цикла, после которого выполнено правило эпизода
    (прямой признак или min_cycles движений); отклонённое — конец события: уверенность и доля «чужих»
    предметов считаются по всем его циклам."""
    if not event.passed:
        return event.end
    done: List[Cycle] = []
    for c in sorted(event.cycles, key=lambda c: c.end):
        done.append(c)
        if has_evidence(c, ep) or movements(done) >= ep.min_cycles:
            return c.end
    return event.end


def basis_text(cycles: Sequence[Cycle], positive: Sequence[str]) -> str:
    """«2 цикла + сигарета 0.71»: на чём основано событие."""
    ev: List[str] = []
    for c in cycles:
        ev.extend(x for x in evidence(c, positive) if x not in ev)
    n = movements(cycles)
    return f"{n} {plural(n, CYCLE_FORMS)}" + (" + " + ", ".join(ev) if ev else "")


def roi_contours(roi: Roi) -> Optional[List[List[List[int]]]]:
    """Контуры маски ROI (внешние и дыры исключённых зон) в пикселях; None — анализ по всему кадру."""
    if roi.mask is None:
        return None
    contours, _ = cv2.findContours(roi.mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return [c.reshape(-1, 2).tolist() for c in contours if len(c) >= 3]


def _num(x, nd: int = 3):
    """Число для JSON: округлённое; nan и inf → None (JSON их не допускает)."""
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def _points(a: np.ndarray) -> list:
    return [None if np.isnan(p).any() else [round(float(p[0]), 1), round(float(p[1]), 1)] for p in a]


def track_payload(sig: TrackSignal, obs: Sequence[PersonObs]) -> dict:
    """Отсчёты трека: время, рамка, рот, кисть, d, valid и 17 точек позы (kp — x,y по точкам, kc — conf·100)."""
    data = {
        "id": int(sig.track_id),
        "t": [round(float(v), 3) for v in sig.t],
        "box": np.round(sig.boxes).astype(int).tolist(),
        "valid": sig.valid.astype(int).tolist(),
        "mouth": _points(sig.mouth),
        "hand": _points(sig.hand),
        "d": [_num(v) for v in sig.d],
        "kp": None,
        "kc": None,
    }
    if len(obs) == len(sig):  # build_signal даёт по отсчёту на наблюдение
        k = np.stack([o.kpts for o in obs])
        data["kp"] = np.round(k[:, :, :2]).astype(int).reshape(len(obs), -1).tolist()
        data["kc"] = np.round(k[:, :, 2] * 100).astype(int).tolist()
    return data


def build_payload(
    clip,
    meta: VideoMeta,
    tracks: Dict[int, List[PersonObs]],
    analysis,
    cfg: Config,
    video_url: str,
) -> dict:
    """clip — pipeline.ClipInput, tracks — треки после stitch_tracks, analysis — pipeline.Analysis."""
    ep = cfg.episode
    positive = cfg.verifier.positive_classes
    roi = Roi.load(clip.roi_path, meta.width, meta.height)

    track_list = [track_payload(sig, tracks.get(tid, [])) for tid, sig in analysis.signals.items()]
    track_list.sort(key=lambda tr: tr["t"][0] if tr["t"] else 0.0)

    cycles, cycle_ids = [], {}
    for tid in analysis.cycles:
        for c in sorted(analysis.cycles[tid], key=lambda c: c.start):
            cycle_ids[id(c)] = len(cycles)
            cycles.append(
                {
                    "id": len(cycles),
                    "track_id": int(c.track_id),
                    "start": _num(c.start),
                    "hold_start": _num(c.hold_start),
                    "hold_end": _num(c.hold_end),
                    "end": _num(c.end),
                    "peak": _num(c.peak),
                    "valid": bool(c.valid),
                    "reason": c.reject_reason,
                    "reason_text": reason_text(c.reject_reason),
                    "evidence": evidence(c, positive) if c.valid else [],
                    "direct": bool(c.valid and has_evidence(c, ep)),
                    "joined_prev": bool(c.valid and c.joined_prev),
                    "flash": bool(c.flash),
                    "vapor": bool(c.vapor),
                    "vapor_t": _num(c.vapor_t),
                    "objects": {k: _num(v, 2) for k, v in c.objects.items()},
                    "quality": _num(c.quality, 2),
                    "min_dist": _num(c.min_dist),
                    "kconf": _num(c.kconf, 2),
                    "face_visible": _num(c.face_visible, 2),
                    "head_tilt": _num(c.head_tilt),
                    "wrist_dy": _num(c.wrist_dy),
                    "neg_fraction": _num(c.neg_fraction, 2),
                    "appearance": _num(c.appearance, 2),
                    "look": bool(c.look),
                    "vapor_net": _num(c.vapor_net),
                    "rescued_from": c.rescued_from,
                    "reveal_t": _num(c.end),
                }
            )

    events = []
    for e in analysis.events:
        ev = []
        for c in e.cycles:
            ev.extend(x for x in evidence(c, positive) if x not in ev)
        reveal = event_reveal_t(e, ep)
        events.append(
            {
                "id": len(events),
                "track_id": int(e.track_id),
                "start": _num(e.start),
                "end": _num(e.end),
                "peak": _num(e.peak),
                "box": np.round(e.box).astype(int).tolist(),
                "confidence": _num(e.confidence, 2),
                "passed": bool(e.passed),
                "reason": e.reject_reason,
                "reason_text": reason_text(e.reject_reason),
                "cycle_ids": [cycle_ids[id(c)] for c in e.cycles if id(c) in cycle_ids],
                "n_movements": movements(e.cycles),
                "evidence": ev,
                "basis": basis_text(e.cycles, positive),
                # основание на момент появления: циклы, закончившиеся к reveal_t
                "reveal_basis": basis_text([c for c in e.cycles if c.end <= reveal], positive),
                "reveal_t": _num(reveal),
            }
        )

    flashes = [
        {"track_id": int(f.track_id), "start": _num(f.start), "end": _num(f.end)}
        for fs in analysis.flashes.values()
        for f in fs
    ]
    return {
        "clip": {
            "id": clip.clip_id,
            "camera_id": clip.camera_id,
            "start_time": clip.start_time,
            "video_name": clip.video_path.replace("\\", "/").rsplit("/", 1)[-1],
            "video_url": video_url,
            "duration": _num(meta.duration),
            "fps": _num(meta.fps, 3),
            "width": meta.width,
            "height": meta.height,
        },
        "thresholds": {
            "near_enter": cfg.cycle.near_enter,
            "near_exit": cfg.cycle.near_exit,
            "min_hold": cfg.cycle.min_hold,
            "max_hold": cfg.cycle.max_hold,
            "window": ep.window,
            "min_cycles": ep.min_cycles,
            "merge_gap": ep.merge_gap,
            "confidence_threshold": ep.confidence_threshold,
            "min_appearance": cfg.appearance.min_score,
            "direct_appearance": cfg.appearance.direct_score,
            "flash_as_direct": ep.flash_as_direct,
            "vapor_as_direct": ep.vapor_as_direct,
        },
        "roi": roi_contours(roi),
        "tracks": track_list,
        "cycles": cycles,
        "events": events,
        "flashes": flashes,
        "feed": build_feed(cycles, events, flashes, ep.confidence_threshold),
        "stats": {
            "tracks": len(track_list),
            "cycles": sum(c["valid"] for c in cycles),
            "events": sum(e["passed"] for e in events),
        },
    }


def build_feed(cycles: List[dict], events: List[dict], flashes: List[dict], threshold: float) -> List[dict]:
    """Лента анализа: строки с моментом появления t, человеком и текстом."""
    feed = []
    for c in cycles:
        hold = (c["hold_end"] or 0.0) - (c["hold_start"] or 0.0)
        if c["valid"]:
            what = "продолжение того же движения" if c["joined_prev"] else "цикл засчитан"
            text = f"рука у рта {hold:.1f} с — {what}"
            if c["evidence"]:
                text += ": " + ", ".join(c["evidence"])
            feed.append({"t": c["reveal_t"], "track_id": c["track_id"], "kind": "cycle", "text": text, "ref": c["id"]})
        else:
            if "short_hold" in c["reason"].split(","):
                text = f"касание {hold:.1f} с — слишком короткое для цикла"
            else:
                text = f"рука у рта {hold:.1f} с — не цикл: {c['reason_text']}"
            feed.append({"t": c["reveal_t"], "track_id": c["track_id"], "kind": "reject", "text": text, "ref": c["id"]})
    for f in flashes:
        feed.append({"t": f["end"], "track_id": f["track_id"], "kind": "flash", "text": "вспышка у лица — зажигалка?",
                     "ref": None})  # fmt: skip
    for e in events:
        row = {"track_id": e["track_id"], "ref": e["id"]}
        if e["passed"]:
            feed.append({**row, "t": e["reveal_t"], "kind": "event", "text": f"ЭПИЗОД: {e['reveal_basis']}"})
            length = (e["end"] or 0.0) - (e["start"] or 0.0)
            text = f"эпизод завершён: {length:.0f} с, {e['basis']}; уверенность {e['confidence']:.2f}"
            feed.append({**row, "t": e["end"], "kind": "event_end", "text": text})
        else:
            why = e["reason_text"] or f"уверенность {e['confidence']:.2f} < {threshold:.2f}"
            feed.append({**row, "t": e["reveal_t"], "kind": "event_reject", "text": f"кандидат ({e['basis']}) отклонён: {why}"})
    order = {"reject": 0, "flash": 1, "cycle": 2, "event_reject": 3, "event": 4, "event_end": 5}
    feed.sort(key=lambda r: (r["t"] if r["t"] is not None else 0.0, order[r["kind"]]))
    return feed
