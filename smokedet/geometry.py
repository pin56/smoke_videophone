"""Сигнал по треку: нормированное расстояние «кисть → рот» и вспомогательные признаки.

В COCO-17 нет точки рта, поэтому рот оценивается от носа (или глаз, или ушей) со смещением
в сторону середины плеч. Кисть — продолжение предплечья за запястье. Расстояние нормируется
на размер человека (ширина плеч или доля высоты bbox), чтобы пороги не зависели от дистанции.
В толпе поза может «пришить» человеку чужую руку: анатомически неправдоподобная рука
(слишком длинная или с запястьем далеко за рамкой) в отсчёте не учитывается.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .config import GeometryConfig, TrackingConfig
from .roi import Roi, anchor_point
from .tracking import PersonObs

NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHO, R_SHO, L_ELB, R_ELB, L_WRI, R_WRI = 5, 6, 7, 8, 9, 10

# смещение рта от опорной точки лица вдоль направления «вниз» (к плечам), в единицах нормировки
MOUTH_OFFSET = {"nose": 0.08, "eyes": 0.18, "ears": 0.15}


@dataclass
class TrackSignal:
    track_id: int
    t: np.ndarray  # (N,) сек от начала клипа
    frame_idx: np.ndarray  # (N,)
    boxes: np.ndarray  # (N, 4)
    valid: np.ndarray  # (N,) человек в зоне и достаточно крупный
    mouth: np.ndarray  # (N, 2), nan если неизвестно
    hand: np.ndarray  # (N, 2) ближайшая ко рту кисть
    scale: np.ndarray  # (N,) нормировка, px
    d_raw: np.ndarray  # (N,) расстояние кисть→рот / scale
    d: np.ndarray  # (N,) сглаженное d_raw
    ear_ratio: np.ndarray  # (N,) d(кисть, ухо) / d(кисть, рот)
    wrist_above: np.ndarray  # (N,) 1 — запястье выше локтя, 0 — ниже, nan — нет точек
    kconf: np.ndarray  # (N,) min(уверенность лица, уверенность запястья)
    wrist_dy: np.ndarray  # (N,) (y запястья - y рта) / scale: > 0 — запястье ниже рта
    elbow_raise: np.ndarray  # (N,) (y плеча - y локтя) / scale: > 0 — локоть выше плеча
    face_visible: np.ndarray  # (N,) 1 — нос найден уверенно, 0 — рот оценён по глазам/ушам
    light_area: Optional[np.ndarray] = None  # (N,) новый пересвет у рта/кистей / scale^2 (flash.py)
    light_ring: Optional[np.ndarray] = None  # (N,) новый пересвет вокруг человека / scale^2
    face: Optional[np.ndarray] = None  # (N, 2) центр уверенных точек лица (нос, глаза, уши), nan если нет
    head_pitch: Optional[np.ndarray] = None  # (N,) (y ушей - y носа) / scale: растёт, когда голова запрокинута
    arm_dropped: Optional[np.ndarray] = None  # (N,) сколько рук отброшено как неправдоподобные (arm_plausible)

    def __len__(self) -> int:
        return len(self.t)

    def index_at(self, t: float) -> int:
        return int(np.argmin(np.abs(self.t - t)))

    @property
    def dt(self) -> float:
        return float(np.median(np.diff(self.t))) if len(self.t) > 1 else 0.1


def rolling_nanmedian(x: np.ndarray, win: int) -> np.ndarray:
    if win <= 1 or len(x) == 0:
        return x.copy()
    half = win // 2
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        seg = x[max(0, i - half) : i + half + 1]
        seg = seg[~np.isnan(seg)]
        if seg.size:
            out[i] = np.median(seg)
    return out


def _pt(k: np.ndarray, i: int, thr: float) -> Optional[np.ndarray]:
    return k[i, :2].astype(np.float64) if k[i, 2] >= thr else None


def _mean(points) -> Optional[np.ndarray]:
    pts = [p for p in points if p is not None]
    return np.mean(pts, axis=0) if pts else None


def raw_scale(k: np.ndarray, box: np.ndarray, cfg: GeometryConfig) -> float:
    h = float(box[3] - box[1])
    ls, rs = _pt(k, L_SHO, cfg.min_kpt_conf), _pt(k, R_SHO, cfg.min_kpt_conf)
    shoulders = float(np.linalg.norm(ls - rs)) if ls is not None and rs is not None else 0.0
    return max(shoulders, cfg.scale_bbox_ratio * h, 1.0)


def estimate_mouth(k: np.ndarray, scale: float, thr: float) -> Tuple[Optional[np.ndarray], float]:
    sho = _mean([_pt(k, L_SHO, thr), _pt(k, R_SHO, thr)])
    nose = _pt(k, NOSE, thr)
    if nose is not None:
        ref, kind, conf = nose, "nose", float(k[NOSE, 2])
    else:
        eyes = _mean([_pt(k, L_EYE, thr), _pt(k, R_EYE, thr)])
        ears = _mean([_pt(k, L_EAR, thr), _pt(k, R_EAR, thr)])
        if eyes is not None:
            ref, kind, conf = eyes, "eyes", float(max(k[L_EYE, 2], k[R_EYE, 2]))
        elif ears is not None:
            ref, kind, conf = ears, "ears", float(max(k[L_EAR, 2], k[R_EAR, 2]))
        else:
            return None, 0.0
    down = np.array([0.0, 1.0])
    if sho is not None:
        v = sho - ref
        n = np.linalg.norm(v)
        if n > 1e-3:
            down = v / n
    return ref + down * MOUTH_OFFSET[kind] * scale, conf


ARMS = ((L_WRI, L_ELB, L_SHO), (R_WRI, R_ELB, R_SHO))


def arm_plausible(k: np.ndarray, box: np.ndarray, scale: float, cfg: GeometryConfig) -> Tuple[bool, bool]:
    """Правдоподобна ли левая и правая рука. Неправдоподобна, если плечо (плечо→локоть) или предплечье
    (локоть→запястье) длиннее k*scale или запястье далеко за рамкой человека: так отсекается рука
    соседа, которую поза «пришила» в толпе, и рамки на двоих. Отношение предплечья к плечу
    не проверяется: в проекции оно меняется от 0.1 до 30 у своей руки (локоть к камере у рта).
    Проверяются только найденные точки; порог 0 отключает проверку."""
    thr = cfg.min_kpt_conf
    out = []
    for wri, elb, sho in ARMS:
        w, e, s = _pt(k, wri, thr), _pt(k, elb, thr), _pt(k, sho, thr)
        ok = True
        if cfg.max_upper_arm > 0 and e is not None and s is not None:
            ok = ok and float(np.linalg.norm(e - s)) <= cfg.max_upper_arm * scale
        if cfg.max_forearm > 0 and w is not None and e is not None:
            ok = ok and float(np.linalg.norm(w - e)) <= cfg.max_forearm * scale
        if cfg.max_wrist_outside > 0 and w is not None:
            dx = max(box[0] - w[0], 0.0, w[0] - box[2])
            dy = max(box[1] - w[1], 0.0, w[1] - box[3])
            ok = ok and float(np.hypot(dx, dy)) <= cfg.max_wrist_outside * scale
        out.append(ok)
    return out[0], out[1]


def head_pitch(k: np.ndarray, scale: float, thr: float) -> float:
    """Высота носа над линией ушей, в единицах scale; nan без носа или обоих ушей. При запрокидывании
    головы (питьё из бутылки, кружки) нос поднимается относительно ушей, и значение растёт."""
    nose = _pt(k, NOSE, thr)
    ears = [p for p in (_pt(k, L_EAR, thr), _pt(k, R_EAR, thr)) if p is not None]
    if nose is None or not ears:
        return float("nan")
    return float((np.mean([p[1] for p in ears]) - nose[1]) / scale)


def nearest_hand(k: np.ndarray, mouth: np.ndarray, scale: float, cfg: GeometryConfig, skip: Tuple[bool, bool] = (False, False)):
    """Возвращает (кисть, расстояние до рта в единицах scale, ear_ratio, wrist_above, conf запястья,
    wrist_dy, elbow_raise) для руки, ближайшей ко рту. skip — не рассматривать левую/правую руку."""
    thr = cfg.min_kpt_conf
    best = None
    for (wri, elb, sho), skipped in zip(ARMS, skip):
        if skipped:
            continue
        w = _pt(k, wri, thr)
        if w is None:
            continue
        e = _pt(k, elb, thr)
        hand = w + cfg.hand_extrapolation * (w - e) if e is not None else w
        dist = min(np.linalg.norm(hand - mouth), np.linalg.norm(w - mouth)) / scale
        if best is None or dist < best[1]:
            above = float(w[1] < e[1]) if e is not None else np.nan
            s = _pt(k, sho, thr)
            raise_ = float((s[1] - e[1]) / scale) if e is not None and s is not None else np.nan
            best = (hand, dist, w, above, float(k[wri, 2]), raise_)
    if best is None:
        return None
    hand, dist, w, above, conf, elbow_raise = best
    wrist_dy = float((w[1] - mouth[1]) / scale)
    ears = [p for p in (_pt(k, L_EAR, thr), _pt(k, R_EAR, thr)) if p is not None]
    if ears:
        d_ear = min(min(np.linalg.norm(hand - p), np.linalg.norm(w - p)) for p in ears) / scale
        ear_ratio = d_ear / max(dist, 1e-3)
    else:
        ear_ratio = np.nan
    return hand, float(dist), float(ear_ratio), above, conf, wrist_dy, elbow_raise


def build_signal(
    track_id: int,
    obs: List[PersonObs],
    roi: Roi,
    geo: GeometryConfig,
    trk: TrackingConfig,
    sample_fps: float,
) -> TrackSignal:
    n = len(obs)
    t = np.array([o.t for o in obs], dtype=np.float64)
    frame_idx = np.array([o.frame_idx for o in obs], dtype=np.int64)
    boxes = np.stack([o.box for o in obs]).astype(np.float64)
    heights = boxes[:, 3] - boxes[:, 1]
    in_roi = np.array([roi.contains(*anchor_point(b, trk.roi_anchor)) for b in boxes])
    valid = (heights >= trk.min_person_height) & in_roi

    s_raw = np.array([raw_scale(o.kpts, o.box, geo) for o in obs])
    scale = rolling_nanmedian(s_raw, max(1, int(round(geo.scale_window * sample_fps))) | 1)

    mouth = np.full((n, 2), np.nan)
    hand = np.full((n, 2), np.nan)
    d_raw = np.full(n, np.nan)
    ear_ratio = np.full(n, np.nan)
    wrist_above = np.full(n, np.nan)
    kconf = np.zeros(n)
    wrist_dy = np.full(n, np.nan)
    elbow_raise = np.full(n, np.nan)
    face_visible = np.array([float(o.kpts[NOSE, 2] >= geo.min_kpt_conf) for o in obs])
    face = np.full((n, 2), np.nan)
    pitch = np.full(n, np.nan)
    arm_dropped = np.zeros(n)
    for i, o in enumerate(obs):
        c = _mean([_pt(o.kpts, j, geo.min_kpt_conf) for j in (NOSE, L_EYE, R_EYE, L_EAR, R_EAR)])
        if c is not None:
            face[i] = c
        pitch[i] = head_pitch(o.kpts, scale[i], geo.min_kpt_conf)
        m, face_conf = estimate_mouth(o.kpts, scale[i], geo.min_kpt_conf)
        if m is None:
            continue
        mouth[i] = m
        # только здесь, а не в flash.LightMeter: его замеры уже лежат в кэше треков
        skip = tuple(not ok for ok in arm_plausible(o.kpts, o.box, scale[i], geo))
        arm_dropped[i] = sum(skip)
        res = nearest_hand(o.kpts, m, scale[i], geo, skip)
        if res is None:
            continue
        hand[i], d_raw[i], ear_ratio[i], wrist_above[i], wconf, wrist_dy[i], elbow_raise[i] = res
        kconf[i] = min(face_conf, wconf)

    light_area = np.full(n, np.nan)
    light_ring = np.full(n, np.nan)
    for i, o in enumerate(obs):
        if o.light is not None:
            zone, ring, s = (float(v) for v in o.light)
            light_area[i], light_ring[i] = zone / s**2, ring / s**2

    d_raw[~valid] = np.nan
    d = rolling_nanmedian(d_raw, geo.smooth_window)
    d[~valid] = np.nan
    return TrackSignal(
        track_id=track_id,
        t=t,
        frame_idx=frame_idx,
        boxes=boxes,
        valid=valid,
        mouth=mouth,
        hand=hand,
        scale=scale,
        d_raw=d_raw,
        d=d,
        ear_ratio=ear_ratio,
        wrist_above=wrist_above,
        kconf=kconf,
        wrist_dy=wrist_dy,
        elbow_raise=elbow_raise,
        face_visible=face_visible,
        light_area=light_area,
        light_ring=light_ring,
        face=face,
        head_pitch=pitch,
        arm_dropped=arm_dropped,
    )
