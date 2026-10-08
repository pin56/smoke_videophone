#!/usr/bin/env python
"""Моделирование схемы «1 кадр/с + плотная проверка по подозрению» на готовом кэше треков (только CPU).

    python tools/sim_1hz.py --inputs runs/rt/all.csv --cache <папка с <clip_id>_tracks.pkl>
        [--gt rasmetka_gt.csv] [--q2-trigger 1.5] [--mark-pad 0.5] [--trigger-min-height 72]

Кэш — полнокадровая поза ~10 Гц (yolo11m-pose + ByteTrack, треки склеиваются stitch_tracks). Новый пайплайн
делает позу на кропах (точнее для мелких людей), поэтому результат — нижняя оценка.

1. Триггер. Берётся один кадр в секунду (кадр кэша, ближайший к сетке фаза + k с; фазы 0.0–0.9 с), сигнал
   build_signal строится по этим кадрам (sample_fps = 1), используется несглаженное d_raw. Триггер у человека —
   d_raw <= порога и/или запястье выше плеча (y запястья < y плеча, точки с уверенностью >= min_kpt_conf).
   Видео поймано, если хотя бы один триггер попал в отметку SMOKE [up, down] ± --mark-pad (0.5 с) (кто именно — не
   проверяется: в разметке нет личности). Нагрузка — триггеры (человеко-секунды) на false и на true вне отметок.
2. Плотная проверка. Вокруг каждого триггера человека — окно [t - before, t + after], окна одного трека
   объединяются; в окне берутся кадры кэша 5 Гц, по ним build_signal + detect_cycles, затем правило эпизода
   (qualifying_cycles: min_cycles движений за 20 с, без прямых признаков) и объединение циклов при паузе
   <= 15 с. События оцениваются по протоколу tools/eval_markup.py (эпизод ±3 с, пересечение >= 1 с).
   «Мягк.» — причины отказа цикла из SOFT_REASONS не учитываются, «без фильтров» — ещё и long_hold.
   Вариант «без гейта» — то же правило на всём треке 5 Гц (верхняя граница при идеальном триггере).
3. Диагностика непойманных true-видео: есть ли люди в отметках, их рост, минимум d_raw на 10 Гц.
4. Нагрузка: людей на кадр 1 Гц, одновременных плотных окон (сред/макс), плотных человеко-секунд на минуту видео.
"""
from __future__ import annotations

import argparse
import copy
import csv
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from eval_markup import RunClip, _num, evaluate_run, load_gt, mark_interval  # noqa: E402
from match_markup import canonical_stem  # noqa: E402
from smokedet.config import Config  # noqa: E402
from smokedet.cycles import detect_cycles  # noqa: E402
from smokedet.episodes import event_confidence, qualifying_cycles  # noqa: E402
from smokedet.geometry import build_signal  # noqa: E402
from smokedet.roi import Roi  # noqa: E402
from smokedet.tracking import stitch_tracks  # noqa: E402

PHASES = [round(0.1 * i, 1) for i in range(10)]
MARK_PAD = 0.5  # с: допуск вокруг отметки SMOKE для попадания триггера
DENSE_HZ = 5.0
# «мягкие» варианты: эти причины отказа цикла не учитываются (остаётся только long_hold)
SOFT_REASONS = frozenset({"no_approach", "short_hold", "phone_like", "hand_low", "hand_above_mouth", "face_hidden"})
L_SH, R_SH, L_WR, R_WR = 5, 6, 9, 10


class Clip:
    def __init__(self, clip_id: str, stem: str, folder: str, ts, cfg: Config):
        self.clip_id, self.stem, self.folder = clip_id, stem, folder  # stem — имя видео, ключ разметки
        self.duration = float(ts.meta.duration)
        self.sample_fps = float(ts.sample_fps)
        self.roi = Roi.full()
        self.tracks = {tid: obs for tid, obs in stitch_tracks(ts.tracks, cfg.tracking).items() if obs}
        frames: Dict[int, float] = {}
        for obs in self.tracks.values():
            for o in obs:
                frames[o.frame_idx] = o.t
        order = sorted(frames, key=frames.get)
        self.frame_idx = np.array(order, dtype=np.int64)
        self.frame_t = np.array([frames[i] for i in order])
        self.people = defaultdict(int)  # кадр -> людей в кэше
        for obs in self.tracks.values():
            for o in obs:
                self.people[o.frame_idx] += 1

    def grid_frames(self, phase: float) -> Tuple[np.ndarray, np.ndarray]:
        """Сетка 1 Гц: время узла и ближайший кадр кэша (-1, если рядом нет кадра с людьми)."""
        grid = np.arange(phase, self.duration, 1.0)
        out = np.full(len(grid), -1, dtype=np.int64)
        if len(self.frame_t):
            tol = 0.6 / self.sample_fps
            j = np.clip(np.searchsorted(self.frame_t, grid), 1, len(self.frame_t) - 1) if len(self.frame_t) > 1 else np.zeros(len(grid), int)
            for k, g in enumerate(grid):
                cand = [c for c in (j[k] - 1, j[k]) if 0 <= c < len(self.frame_t)]
                best = min(cand, key=lambda c: abs(self.frame_t[c] - g))
                if abs(self.frame_t[best] - g) <= tol:
                    out[k] = self.frame_idx[best]
        return grid, out


def wrist_above_shoulder(k: np.ndarray, thr: float) -> bool:
    for sh, wr in ((L_SH, L_WR), (R_SH, R_WR)):
        if k[sh, 2] >= thr and k[wr, 2] >= thr and k[wr, 1] < k[sh, 1]:
            return True
    return False


def samples_1hz(clip: Clip, phase: float, cfg: Config):
    """Отсчёты 1 Гц по людям: [(tid, t, valid, d_raw, выше плеча, рост)]."""
    _, frames = clip.grid_frames(phase)
    chosen = set(int(f) for f in frames if f >= 0)
    out = []
    for tid, obs in clip.tracks.items():
        sub = [o for o in obs if o.frame_idx in chosen]
        if not sub:
            continue
        sig = build_signal(tid, sub, clip.roi, cfg.geometry, cfg.tracking, 1.0)
        for i, o in enumerate(sub):
            h = float(o.box[3] - o.box[1])
            out.append((tid, o.t, h, sig.d_raw[i], wrist_above_shoulder(o.kpts, cfg.geometry.min_kpt_conf)))
    return out


def fires(s, thr: float, shoulder: bool, min_h: float) -> bool:
    _, _, h, d, above = s
    if h < min_h:
        return False
    hit = thr > 0 and np.isfinite(d) and d <= thr
    return hit or (shoulder and above)


def in_marks(t: float, marks: List[Tuple[float, float]], pad: Optional[float] = None) -> bool:
    pad = MARK_PAD if pad is None else pad
    return any(a - pad <= t <= b + pad for a, b in marks)


def merge_windows(ws: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out: List[List[float]] = []
    for a, b in sorted(ws):
        if out and a <= out[-1][1] + 1e-6:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def dense_obs(obs, a: float, b: float, hz: float = DENSE_HZ):
    """Кадры трека в [a, b], прореженные до hz (ближайший к узлу сетки от a)."""
    ts = np.array([o.t for o in obs])
    lo, hi = np.searchsorted(ts, a - 1e-6), np.searchsorted(ts, b + 1e-6)
    if hi - lo < 1:
        return []
    seg_t = ts[lo:hi]
    picked = []
    for g in np.arange(a, b + 1e-6, 1.0 / hz):
        j = int(np.argmin(np.abs(seg_t - g)))
        if abs(seg_t[j] - g) <= 0.6 / hz and (not picked or picked[-1] != lo + j):
            picked.append(lo + j)
    return [obs[i] for i in picked]


def events_for(clip: Clip, windows: Dict[int, List[Tuple[float, float]]], cfg: Config, relax=frozenset()):
    """Плотные окна -> события [(start, end, conf)] и плотные человеко-секунды. relax — причины отказа
    цикла, которые не учитываются (цикл с причинами только из relax считается валидным)."""
    evs, dense_sec = [], 0.0
    for tid, ws in windows.items():
        cycles = []
        for a, b in ws:
            dense_sec += b - a
            sub = dense_obs(clip.tracks[tid], a, b)
            if len(sub) < 3:
                continue
            sig = build_signal(tid, sub, clip.roi, cfg.geometry, cfg.tracking, DENSE_HZ)
            for c in detect_cycles(sig, cfg.cycle):
                if not c.valid and set(c.reject_reason.split(",")) <= relax:
                    c.valid = True
                cycles.append(c)
        groups: List[list] = []
        for c in qualifying_cycles(cycles, cfg.episode):
            if groups and c.start - max(x.end for x in groups[-1]) <= cfg.episode.merge_gap:
                groups[-1].append(c)
            else:
                groups.append([c])
        for g in groups:
            evs.append((g[0].start, max(c.end for c in g), event_confidence(g, cfg.episode)))
    return evs, dense_sec


def load_marks(path: str) -> Dict[str, List[Tuple[float, float]]]:
    marks: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["label"].strip().upper() != "SMOKE":
                continue
            iv = mark_interval(_num(r["up_sec"]), _num(r["puff_sec"]), _num(r["down_sec"]))
            if iv is not None:
                marks[canonical_stem(r["video"])].append(iv)
    return marks


def load_clips(inputs: str, cache: Path, cfg: Config) -> List[Clip]:
    clips = []
    with open(inputs, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            vp = Path(r["video_path"])
            cid = r.get("clip_id") or vp.stem
            p = cache / f"{cid}_tracks.pkl"
            if not p.exists():
                print(f"нет кэша: {p}")
                continue
            with open(p, "rb") as fh:
                ts = pickle.load(fh)["trackset"]
            clips.append(Clip(cid, canonical_stem(vp.stem), vp.parent.name, ts, cfg))
    return clips


def q1(clips, marks, cfg, triggers, min_h):
    true_clips = [c for c in clips if c.folder == "true" and marks.get(c.stem)]
    false_min = sum(c.duration for c in clips if c.folder == "false") / 60
    true_min = sum(c.duration for c in true_clips) / 60
    samples = {(c.clip_id, ph): samples_1hz(c, ph, cfg) for c in clips for ph in PHASES}
    print(f"\n## 1. Триггер 1 Гц (true с отметками: {len(true_clips)}, false: {false_min:.1f} мин, true: {true_min:.1f} мин)")
    print("триггер | поймано true (худш/сред по фазам) | триггеров/мин false | true вне отметок/мин | не пойманы (худшая фаза)")
    for name, thr, sh in triggers:
        caught, fl, tr = [], [], []
        worst_missed = None
        for ph in PHASES:
            n, missed, f_cnt, t_cnt = 0, [], 0, 0
            for c in clips:
                ss = [s for s in samples[(c.clip_id, ph)] if fires(s, thr, sh, min_h)]
                if c.folder == "false":
                    f_cnt += len(ss)
                elif c in true_clips:
                    mk = marks[c.stem]
                    if any(in_marks(s[1], mk) for s in ss):
                        n += 1
                    else:
                        missed.append(c.stem)
                    t_cnt += sum(1 for s in ss if not in_marks(s[1], mk))
            caught.append(n)
            fl.append(f_cnt / false_min)
            tr.append(t_cnt / true_min)
            if worst_missed is None or len(missed) > len(worst_missed):
                worst_missed = missed
        print(f"{name} | {min(caught)}/{np.mean(caught):.1f} | {np.mean(fl):.1f} | {np.mean(tr):.1f} | {','.join(worst_missed) or '-'}")
    return samples, true_clips


def trigger_windows(clip, samples, thr, sh, min_h, before, after):
    ws = defaultdict(list)
    for s in samples:
        if fires(s, thr, sh, min_h):
            ws[s[0]].append((max(0.0, s[1] - before), min(clip.duration, s[1] + after)))
    return {tid: merge_windows(w) for tid, w in ws.items()}


def q2(clips, gt, samples, cfg, thr, sh, min_h, variants, conf_thr):
    print(f"\n## 2. Плотная проверка 5 Гц после триггера d<={thr}{' или плечо' if sh else ''}"
          f"{f', conf>={conf_thr}' if conf_thr > 0 else ''}")
    print("вариант | true с TP (худш/сред) | FP false (сред) | FP true не-дубль (сред) | плотн. чел-с/мин | окон одновр. сред/макс")
    total_min = sum(c.duration for c in clips) / 60
    for name, ne, nx, mc, before, after, relax, hold in variants:
        vc = copy.deepcopy(cfg)
        vc.cycle.near_enter, vc.cycle.near_exit, vc.episode.min_cycles, vc.cycle.min_hold = ne, nx, mc, hold
        phases = PHASES if before is not None else [0.0]
        tp_n, fp_f, fp_t, load, conc_mean, conc_max = [], [], [], [], [], 0
        missed_any = defaultdict(int)
        for ph in phases:
            run, dense_total, conc = {}, 0.0, []
            for c in clips:
                if before is None:  # без гейта: весь трек
                    ws = {tid: [(obs[0].t, obs[-1].t)] for tid, obs in c.tracks.items()}
                else:
                    ws = trigger_windows(c, samples[(c.clip_id, ph)], thr, sh, min_h, before, after)
                    for g in np.arange(ph, c.duration, 1.0):
                        conc.append(sum(1 for w in ws.values() for a, b in w if a <= g <= b))
                evs, ds = events_for(c, ws, vc, relax)
                dense_total += ds
                preds = [{"start_sec": a, "end_sec": b, "confidence": cf} for a, b, cf in evs if cf >= conf_thr]
                run[c.stem] = RunClip(c.stem, c.clip_id, c.folder, c.duration, preds)
            res = evaluate_run(gt, run)
            per = {p["video"]: p for p in res["per_clip"]}
            hit = [v for v, p in per.items() if p["folder"] == "true" and p["TP"] > 0]
            for v, p in per.items():
                if p["folder"] == "true" and p["TP"] == 0:
                    missed_any[v] += 1
            tp_n.append(len(hit))
            fp_f.append(res["per_folder"].get("false", {}).get("fp", 0))
            pt = res["per_folder"].get("true", {})
            fp_t.append(pt.get("fp", 0) - pt.get("dup", 0))
            load.append(dense_total / total_min)
            if conc:
                conc_mean.append(float(np.mean(conc)))
                conc_max = max(conc_max, max(conc))
        cm = f"{np.mean(conc_mean):.2f}/{conc_max}" if conc_mean else "-"
        miss = ",".join(f"{v}({n})" for v, n in sorted(missed_any.items()))
        print(f"{name} | {min(tp_n)}/{np.mean(tp_n):.1f} | {np.mean(fp_f):.1f} | {np.mean(fp_t):.1f} | {np.mean(load):.1f} | {cm}")
        print(f"    без TP (число фаз): {miss or '-'}")


def q3(clips, true_clips, marks, cfg, samples, thr, sh, min_h):
    print("\n## 3. Диагностика true-видео (в отметках ±0.5 с)")
    print("видео | людей в отм. | макс. рост | min d_raw 10Гц (h>=72) | min d_raw 1Гц по фазам | фаз с триггером | 10Гц циклы")
    for c in true_clips:
        mk = marks[c.stem]
        ids, hmax, dmin, ncyc = set(), 0.0, np.inf, 0
        for tid, obs in c.tracks.items():
            inside = [o for o in obs if in_marks(o.t, mk)]
            if not inside:
                continue
            ids.add(tid)
            hmax = max(hmax, max(float(o.box[3] - o.box[1]) for o in inside))
            if len(obs) >= 3:
                sig = build_signal(tid, obs, c.roi, cfg.geometry, cfg.tracking, c.sample_fps)
                m = np.array([in_marks(t, mk) for t in sig.t])
                if np.isfinite(sig.d_raw[m]).any():
                    dmin = min(dmin, float(np.nanmin(sig.d_raw[m])))
                ncyc += sum(1 for cy in detect_cycles(sig, cfg.cycle) if cy.valid and in_marks(cy.peak, mk))
        d1 = []
        fired = 0
        for ph in PHASES:
            ss = [s for s in samples[(c.clip_id, ph)] if in_marks(s[1], mk) and s[2] >= min_h and np.isfinite(s[3])]
            d1.append(min((s[3] for s in ss), default=np.inf))
            fired += any(fires(s, thr, sh, min_h) for s in samples[(c.clip_id, ph)] if in_marks(s[1], mk))
        print(f"{c.stem} | {len(ids)} | {hmax:.0f} | {dmin:.2f} | {min(d1):.2f}..{max(d1):.2f} | {fired}/10 | {ncyc}")


def q4(clips, samples, min_h):
    print("\n## 4. Людей на кадр 1 Гц (фаза 0.0)")
    allc, allv = [], []
    for c in clips:
        ss = samples[(c.clip_id, 0.0)]
        cnt = defaultdict(int)
        cntv = defaultdict(int)
        for s in ss:
            cnt[round(s[1])] += 1
            cntv[round(s[1])] += s[2] >= min_h
        n = int(np.ceil(c.duration))
        allc.extend([cnt.get(k, 0) for k in range(n)])
        allv.extend([cntv.get(k, 0) for k in range(n)])
    print(f"все люди: сред {np.mean(allc):.2f}, макс {max(allc)}; рост >= {min_h:.0f}: сред {np.mean(allv):.2f}, макс {max(allv)}")


def main(argv=None) -> int:
    global MARK_PAD
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inputs", default=str(ROOT / "runs/rt/all.csv"))
    ap.add_argument("--cache", default=str(ROOT.parents[0] / "goal-max/runs/goal/cache"))
    ap.add_argument("--gt", default=str(ROOT / "rasmetka_gt.csv"))
    ap.add_argument("--trigger-min-height", type=float, default=None, help="мин. рост для триггера, по умолч. min_person_height")
    ap.add_argument("--q2-trigger", type=float, default=1.5, help="порог d_raw триггера для п. 2")
    ap.add_argument("--q2-shoulder", action="store_true", help="в п. 2 триггер ещё и по «запястье выше плеча»")
    ap.add_argument("--q2-conf", type=float, default=0.0, help="порог confidence события в п. 2 (0 — все)")
    ap.add_argument("--mark-pad", type=float, default=MARK_PAD, help="с: допуск вокруг отметки SMOKE в п. 1 и 3")
    ap.add_argument("--skip-q2", action="store_true")
    args = ap.parse_args(argv)

    cfg = Config()
    min_h = args.trigger_min_height if args.trigger_min_height is not None else cfg.tracking.min_person_height
    cfg.tracking.min_person_height = int(min_h)
    clips = load_clips(args.inputs, Path(args.cache), cfg)
    marks = load_marks(args.gt)
    gt = load_gt(args.gt)
    MARK_PAD = args.mark_pad
    triggers = [(f"d<={t}", t, False) for t in (0.6, 0.8, 1.0, 1.2, 1.3, 1.5, 2.0)]
    triggers += [("плечо", 0.0, True), ("d<=1.0|плечо", 1.0, True), ("d<=1.5|плечо", 1.5, True)]
    samples, true_clips = q1(clips, marks, cfg, triggers, min_h)
    if not args.skip_q2:
        no, soft = frozenset(), SOFT_REASONS
        variants = [  # имя, near_enter, near_exit, min_cycles, окно до, окно после (None — без гейта), relax, min_hold
            ("±3с, near0.6, ≥2ц", 0.6, 0.9, 2, 3.0, 3.0, no, 0.4),
            ("±3с, near0.6, ≥1ц", 0.6, 0.9, 1, 3.0, 3.0, no, 0.4),
            ("±3с, near0.8, ≥1ц", 0.8, 1.1, 1, 3.0, 3.0, no, 0.4),
            ("±3с, near0.8, ≥1ц, мягк.", 0.8, 1.1, 1, 3.0, 3.0, soft, 0.2),
            ("±3с, near1.0, ≥1ц, мягк.", 1.0, 1.3, 1, 3.0, 3.0, soft, 0.2),
            ("±2с, near1.0, ≥1ц, мягк.", 1.0, 1.3, 1, 2.0, 2.0, soft, 0.2),
            ("±3с, near1.0, ≥1ц, без фильтров", 1.0, 1.3, 1, 3.0, 3.0, soft | {"long_hold"}, 0.2),
            ("±3с, near1.0, ≥2ц, мягк.", 1.0, 1.3, 2, 3.0, 3.0, soft, 0.2),
            ("-3/+20с, near0.6, ≥2ц", 0.6, 0.9, 2, 3.0, 20.0, no, 0.4),
            ("без гейта, near0.6, ≥2ц", 0.6, 0.9, 2, None, None, no, 0.4),
            ("без гейта, near1.0, ≥1ц, мягк.", 1.0, 1.3, 1, None, None, soft, 0.2),
        ]
        q2(clips, gt, samples, cfg, args.q2_trigger, args.q2_shoulder, min_h, variants, args.q2_conf)
    q3(clips, true_clips, marks, cfg, samples, args.q2_trigger, args.q2_shoulder, min_h)
    q4(clips, samples, min_h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
