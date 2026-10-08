"""Обмен разметкой с CVAT (формат «CVAT for video 1.1»).

Экспорт (tools/cvat_export.py): на каждое видео — XML с треками людей из кэша трекера (метка person) и
эпизодами из разметки rasmetka_gt.csv (метки smoking / not_smoking / ignore). Эпизод — трек CVAT от подъёма
руки первой отметки до опускания последней; рамка идёт за человеком, у которого кисть ближе всего ко рту
в окне отметок. Отметки одного человека и одной метки объединяются при паузе <= episode.merge_gap.

Импорт (tools/cvat_import.py): XML/ZIP из CVAT -> эталон для evaluate.py: camera_id, clip_id, label,
start_sec, end_sec, person_gt_id, peak_sec, x1..y2. Каждый видимый отрезок трека эпизода — одно событие,
рамка — рамка трека в peak_sec (атрибут peak_sec, иначе середина отрезка).

Кадры CVAT нумеруются подряд, как у трекера (io_video). Секунды переводятся в кадры и обратно по pts
наблюдений трекера (FrameTime), а не idx/fps: в CCTV-файлах с переменным fps они расходятся до 0.7 с.
"""
from __future__ import annotations

import csv
import dataclasses
import io
import pickle
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import Config
from .geometry import TrackSignal, build_signal
from .roi import Roi
from .tracking import PersonObs, TrackSet, stitch_tracks

Box = Tuple[float, float, float, float]

PERSON = "person"
# метка CVAT -> метка эталона evaluate.py
EVENT_LABELS = {"smoking": "POSITIVE", "not_smoking": "NEGATIVE", "ignore": "IGNORE"}
# метка rasmetka_gt.csv -> метка CVAT
MARK_LABELS = {"SMOKE": "smoking", "NOSMOKE": "not_smoking", "IGNORE": "ignore"}
COLORS = {"smoking": "#e53935", "not_smoking": "#43a047", "ignore": "#9e9e9e", PERSON: "#1e88e5"}
# атрибуты эпизода (все неизменяемые): имя -> (input_type, значение по умолчанию)
EVENT_ATTRS = {"peak_sec": ("text", ""), "person": ("text", ""), "note": ("text", ""), "check": ("checkbox", "false")}
PERSON_ATTRS = {"person": ("text", "")}

MARK_WINDOW = 1.0  # с: окно вокруг отметки при поиске человека
TRACK_GAP = 1.0  # с: пропуск наблюдений длиннее — человек скрыт (outside) до следующего наблюдения
KEY_IOU = 0.85  # ключевой кадр, если линейная интерполяция рамки даёт IoU ниже
KEY_SPAN = 5.0  # с: ключевые кадры не реже


# --- время кадров -------------------------------------------------------------------------------------


class FrameTime:
    """Номер кадра <-> секунды по времени наблюдений трекера (pts); за краями — с шагом 1/fps."""

    def __init__(self, frames: Sequence[int], times: Sequence[float], fps: float):
        self.fps = fps if fps > 0 else 25.0
        f = np.asarray(frames, dtype=np.float64)
        t = np.asarray(times, dtype=np.float64)
        f, idx = np.unique(f, return_index=True)
        self.f, self.t = f, t[idx]

    @classmethod
    def from_trackset(cls, ts: TrackSet) -> "FrameTime":
        pairs = {o.frame_idx: o.t for obs in ts.tracks.values() for o in obs}
        return cls(list(pairs), list(pairs.values()), ts.meta.fps)

    def sec(self, frame: float) -> float:
        if len(self.f) == 0:
            return frame / self.fps
        if frame <= self.f[0]:
            return float(self.t[0] - (self.f[0] - frame) / self.fps)
        if frame >= self.f[-1]:
            return float(self.t[-1] + (frame - self.f[-1]) / self.fps)
        return float(np.interp(frame, self.f, self.t))

    def frame(self, sec: float) -> int:
        if len(self.f) == 0:
            return int(round(sec * self.fps))
        if sec <= self.t[0]:
            return int(round(self.f[0] - (self.t[0] - sec) * self.fps))
        if sec >= self.t[-1]:
            return int(round(self.f[-1] + (sec - self.t[-1]) * self.fps))
        return int(round(np.interp(sec, self.t, self.f)))


# --- рамки и ключевые кадры ---------------------------------------------------------------------------


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU построчно для (N, 4) xyxy."""
    x1, y1 = np.maximum(a[:, 0], b[:, 0]), np.maximum(a[:, 1], b[:, 1])
    x2, y2 = np.minimum(a[:, 2], b[:, 2]), np.minimum(a[:, 3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]) - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def lerp_boxes(frames: np.ndarray, boxes: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Рамки в кадрах at: линейно между известными, за краями — крайняя рамка."""
    return np.stack([np.interp(at, frames, boxes[:, k]) for k in range(4)], axis=1)


def thin_keyframes(frames: np.ndarray, boxes: np.ndarray, min_iou: float = KEY_IOU, max_span: int = 0) -> List[int]:
    """Индексы ключевых кадров: CVAT интерполирует рамку линейно, поэтому лишние точки отбрасываются,
    пока интерполированная рамка совпадает с настоящей с IoU >= min_iou. max_span > 0 — предел в кадрах
    между ключевыми (рамку проще поправить)."""
    n = len(frames)
    if n <= 2:
        return list(range(n))
    keys, i = [0], 0
    for j in range(2, n):  # отрезок i..j: точки между ними восстанавливаются интерполяцией?
        mid = np.arange(i + 1, j)
        interp = lerp_boxes(frames[[i, j]], boxes[[i, j]], frames[mid])
        if (max_span > 0 and frames[j] - frames[i] > max_span) or np.min(_iou(interp, boxes[mid])) < min_iou:
            keys.append(j - 1)
            i = j - 1
    keys.append(n - 1)
    return keys


# --- модель трека CVAT --------------------------------------------------------------------------------


@dataclass
class Shape:
    frame: int
    box: Box
    outside: bool = False
    keyframe: bool = True


@dataclass
class CvatTrack:
    label: str
    shapes: List[Shape]
    attributes: Dict[str, str] = field(default_factory=dict)
    track_id: int = -1

    def sorted_shapes(self) -> List[Shape]:
        return sorted(self.shapes, key=lambda s: s.frame)

    def segments(self, stop_frame: int) -> List[Tuple[int, int]]:
        """Видимые отрезки [первый, последний кадр]. Как в CVAT: трек без завершающей рамки outside
        виден до конца видео."""
        out, start, last = [], None, None
        for s in self.sorted_shapes():
            if s.outside:
                if start is not None:
                    out.append((start, s.frame - 1))
                    start = None
            elif start is None:
                start = s.frame
            last = s.frame
        if start is not None:
            out.append((start, max(stop_frame, last)))
        return [(a, b) for a, b in out if b >= a]

    def box_at(self, frame: int) -> Optional[Box]:
        shapes = self.sorted_shapes()
        before = [s for s in shapes if s.frame <= frame]
        if not before or before[-1].outside:
            return None
        a = before[-1]
        after = next((s for s in shapes if s.frame > frame), None)
        if after is None or after.outside or a.frame == frame:
            return a.box
        w = (frame - a.frame) / (after.frame - a.frame)
        return tuple(float(p + (q - p) * w) for p, q in zip(a.box, after.box))


def obs_shapes(frames: np.ndarray, boxes: np.ndarray, n_frames: int, gap: int, max_span: int) -> List[Shape]:
    """Рамки трека по наблюдениям: ключевые кадры по отрезкам без пропусков > gap, после отрезка — outside."""
    shapes: List[Shape] = []
    cuts = np.flatnonzero(np.diff(frames) > gap) + 1
    for seg_f, seg_b in zip(np.split(frames, cuts), np.split(boxes, cuts)):
        for k in thin_keyframes(seg_f, seg_b, max_span=max_span):
            shapes.append(Shape(int(seg_f[k]), tuple(float(v) for v in seg_b[k])))
        end = int(seg_f[-1]) + 1
        if end < n_frames:
            shapes.append(Shape(end, shapes[-1].box, outside=True))
    return shapes


# --- XML ----------------------------------------------------------------------------------------------


def _label_attrs(label: str) -> Dict[str, Tuple[str, str]]:
    return PERSON_ATTRS if label == PERSON else EVENT_ATTRS


def labels_spec() -> List[dict]:
    """Метки для конструктора CVAT (Project → Raw): все рамки, атрибуты неизменяемые."""
    return [
        {
            "name": name,
            "color": COLORS[name],
            "type": "rectangle",
            "attributes": [
                {"name": a, "mutable": False, "input_type": kind, "default_value": dv, "values": [dv]}
                for a, (kind, dv) in _label_attrs(name).items()
            ],
        }
        for name in [*EVENT_LABELS, PERSON]
    ]


def _sub(parent: ET.Element, tag: str, text: str = "", **attrib) -> ET.Element:
    el = ET.SubElement(parent, tag, {k: str(v) for k, v in attrib.items()})
    if text:
        el.text = text
    return el


def write_xml(path: Path, name: str, source: str, n_frames: int, width: int, height: int, tracks: List[CvatTrack]) -> None:
    root = ET.Element("annotations")
    _sub(root, "version", "1.1")
    task = _sub(_sub(root, "meta"), "task")
    for tag, val in (("name", name), ("size", n_frames), ("mode", "interpolation"), ("start_frame", 0),
                     ("stop_frame", max(n_frames - 1, 0))):  # fmt: skip
        _sub(task, tag, str(val))
    labels = _sub(task, "labels")
    for spec in labels_spec():
        lab = _sub(labels, "label")
        _sub(lab, "name", spec["name"])
        _sub(lab, "color", spec["color"])
        _sub(lab, "type", spec["type"])
        attrs = _sub(lab, "attributes")
        for a in spec["attributes"]:
            el = _sub(attrs, "attribute")
            for k in ("name", "mutable", "input_type", "default_value"):
                _sub(el, k, str(a[k]))
            _sub(el, "values", "\n".join(a["values"]))
    size = _sub(task, "original_size")
    _sub(size, "width", str(width))
    _sub(size, "height", str(height))
    _sub(task, "source", source)
    for i, tr in enumerate(tracks):
        el = _sub(root, "track", id=i, label=tr.label, source="auto")
        for s in tr.sorted_shapes():
            x1, y1, x2, y2 = s.box
            box = _sub(el, "box", frame=s.frame, keyframe=int(s.keyframe), outside=int(s.outside), occluded=0,
                       xtl=f"{x1:.2f}", ytl=f"{y1:.2f}", xbr=f"{x2:.2f}", ybr=f"{y2:.2f}", z_order=0)  # fmt: skip
            # CVAT берёт неизменяемые атрибуты трека из его рамок, поэтому они пишутся в каждую
            for k, v in tr.attributes.items():
                _sub(box, "attribute", v, name=k)
    ET.indent(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


@dataclass
class CvatFile:
    path: str
    origin: str = ""  # имя файла выгрузки без расширения
    name: str = ""  # имя задачи
    source: str = ""  # имя видеофайла задачи
    stop_frame: Optional[int] = None
    tracks: List[CvatTrack] = field(default_factory=list)


def read_xml_bytes(data: bytes, path: str = "", origin: str = "") -> CvatFile:
    root = ET.fromstring(data)
    out = CvatFile(path=path, origin=origin)
    meta = root.find("meta")
    if meta is not None:
        out.name = (meta.findtext(".//task/name") or meta.findtext(".//job/name") or "").strip()
        out.source = next((el.text.strip() for el in meta.iter("source") if el.text and el.text.strip()), "")
        stop = meta.findtext(".//stop_frame") or ""
        size = meta.findtext(".//size") or ""
        if stop.strip().isdigit():
            out.stop_frame = int(stop)
        elif size.strip().isdigit():
            out.stop_frame = int(size) - 1
    for tr in root.iter("track"):
        shapes, attrs = [], {}
        for b in tr.iter("box"):
            box = tuple(float(b.get(k)) for k in ("xtl", "ytl", "xbr", "ybr"))
            shapes.append(Shape(int(b.get("frame")), box, b.get("outside") == "1", b.get("keyframe", "1") == "1"))
            if not attrs:  # неизменяемые атрибуты одинаковы во всех рамках трека
                attrs = {a.get("name"): (a.text or "").strip() for a in b.iter("attribute")}
        out.tracks.append(CvatTrack(tr.get("label", ""), shapes, attrs, int(tr.get("id", -1))))
    return out


def read_cvat(path: Path) -> List[CvatFile]:
    """XML или ZIP-выгрузка CVAT (annotations.xml внутри; в архиве проекта — по файлу на задачу)."""
    if path.suffix.lower() != ".zip":
        return [read_xml_bytes(path.read_bytes(), str(path), path.stem)]
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".xml")]
        return [read_xml_bytes(z.read(n), f"{path}:{n}", path.stem) for n in names]


# --- разметка rasmetka_gt.csv -------------------------------------------------------------------------


@dataclass
class Mark:
    label: str  # метка CVAT
    start: float
    end: float  # inf — до конца клипа
    puff: Optional[float] = None
    text: str = ""  # «подъём/затяжка/опускание» для заметки
    kind: str = ""  # continue — «продолжают курить»: SMOKE без времени, циклы не размечены
    person: Optional[int] = None  # id склеенного трека
    check: bool = False
    note: str = ""


def _num(v) -> Optional[float]:
    v = (v or "").strip()
    return None if v in ("", "None") else float(v)


def load_marks(path: str) -> Dict[str, List[Mark]]:
    """rasmetka_gt.csv -> отметки по имени видео (stem). Порядок строк важен: SMOKE без времени —
    «продолжают курить» между соседними по файлу отметками SMOKE (как в eval_markup.py), это IGNORE."""
    rows: Dict[str, List[dict]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.setdefault(r["video"].strip(), []).append(r)
    out: Dict[str, List[Mark]] = {}
    for stem, rs in rows.items():
        marks, smoke = [], []  # smoke: отметки SMOKE по порядку, None — без времени
        for r in rs:
            label = r["label"].strip().upper()
            up, puff, down = _num(r["up_sec"]), _num(r["puff_sec"]), _num(r["down_sec"])
            vals = [v for v in (up, puff, down) if v is not None]
            text = "/".join("-" if v is None else f"{v:g}" for v in (up, puff, down))
            m = Mark(MARK_LABELS[label], min(vals), max(vals), puff, text) if vals else None
            if label == "SMOKE":
                smoke.append(m)
            if m is not None:
                marks.append(m)
        for i, m in enumerate(smoke):
            if m is not None:
                continue
            prev = next((x for x in reversed(smoke[:i]) if x is not None), None)
            nxt = next((x for x in smoke[i + 1 :] if x is not None), None)
            start, end = (prev.end if prev else 0.0), (nxt.start if nxt else float("inf"))
            if end > start:
                marks.append(Mark("ignore", start, end, kind="continue", note="продолжают курить, циклы не размечены"))
                marks[-1].text = "после " + (prev.text if prev else "начала")
        out[stem] = sorted(marks, key=lambda m: (m.start, m.end))
    return out


# --- привязка отметок к людям и эпизоды ---------------------------------------------------------------


def clip_signals(ts: TrackSet, cfg: Config) -> Tuple[Dict[int, List[PersonObs]], Dict[int, TrackSignal]]:
    """Склеенные треки и сигнал d(t) по всему кадру: эталон не ограничивается ROI и ростом человека."""
    tracks = stitch_tracks(ts.tracks, cfg.tracking)
    trk = dataclasses.replace(cfg.tracking, min_person_height=0)
    signals = {
        tid: build_signal(tid, obs, Roi.full(), cfg.geometry, trk, ts.sample_fps)
        for tid, obs in tracks.items()
        if len(obs) >= 3
    }
    return tracks, signals


def assign_person(mark: Mark, signals: Dict[int, TrackSignal], cfg: Config) -> None:
    """Человек отметки — с наименьшим d (кисть ближе всего ко рту) в окне отметки ±MARK_WINDOW.
    Сомнительные привязки помечаются check: рука далеко ото рта, рядом второй кандидат, поза не найдена."""
    t0, t1 = mark.start - MARK_WINDOW, mark.end + MARK_WINDOW
    best: List[Tuple[float, int]] = []
    present: List[Tuple[int, float, int]] = []  # наблюдений в окне, рост, id
    for tid, s in signals.items():
        m = (s.t >= t0) & (s.t <= t1)
        if not m.any():
            continue
        present.append((int(m.sum()), float(np.median(s.boxes[m, 3] - s.boxes[m, 1])), tid))
        d = s.d[m]
        if np.isfinite(d).any():
            best.append((float(np.nanmin(d)), tid))
    best.sort()
    notes = []
    if best:
        dmin, mark.person = best[0]
        if dmin > cfg.cycle.near_exit:
            notes.append(f"рука далеко ото рта (d={dmin:.2f})")
        if len(best) > 1 and best[1][0] < cfg.cycle.near_enter:
            notes.append(f"второй кандидат: человек {best[1][1]} (d={best[1][0]:.2f})")
    elif present:
        mark.person = max(present)[2]
        notes.append("поза не найдена, взят человек, дольше всех видимый в окне")
    else:
        notes.append("человек не найден — нарисуйте рамку")
    if notes:
        mark.check = True
        mark.note = "; ".join([mark.note, *notes]) if mark.note else f"{mark.text}: " + ", ".join(notes)


@dataclass
class Episode:
    label: str
    person: Optional[int]
    marks: List[Mark]

    @property
    def start(self) -> float:
        return min(m.start for m in self.marks)

    @property
    def end(self) -> float:
        return max(m.end for m in self.marks)

    @property
    def peak(self) -> float:
        """Затяжка средней по времени отметки (если её нет — середина отметки)."""
        m = self.marks[len(self.marks) // 2]
        return m.puff if m.puff is not None else (m.start + min(m.end, m.start + 60.0)) / 2


def group_episodes(marks: List[Mark], merge_gap: float) -> List[Episode]:
    """Отметки одной метки и одного человека подряд с паузой <= merge_gap — один эпизод."""
    groups: Dict[Tuple[str, Optional[int]], List[Mark]] = {}
    for m in sorted(marks, key=lambda m: m.start):
        groups.setdefault((m.label, m.person), []).append(m)
    out = []
    for (label, person), ms in groups.items():
        cur: List[Mark] = []
        for m in ms:
            if cur and m.start - max(x.end for x in cur) > merge_gap:
                out.append(Episode(label, person, cur))
                cur = []
            cur.append(m)
        out.append(Episode(label, person, cur))
    return sorted(out, key=lambda e: (e.start, e.label))


def episode_track(ep: Episode, tracks: Dict[int, List[PersonObs]], ft: FrameTime, n_frames: int,
                  width: int, height: int, max_span: int) -> CvatTrack:  # fmt: skip
    f0 = min(max(ft.frame(ep.start), 0), n_frames - 1)
    f1 = min(max(ft.frame(ep.end) if np.isfinite(ep.end) else n_frames - 1, f0), n_frames - 1)
    obs = sorted(tracks.get(ep.person, []), key=lambda o: o.frame_idx) if ep.person is not None else []
    if obs:
        of = np.array([o.frame_idx for o in obs], dtype=np.float64)
        ob = np.stack([o.box for o in obs]).astype(np.float64)
        inner = of[(of > f0) & (of < f1)]
        frames = np.concatenate([[f0], inner, [f1]]) if f1 > f0 else np.array([f0], dtype=np.float64)
        boxes = lerp_boxes(of, ob, frames)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, width)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, height)
    else:  # заглушка в левом верхнем углу: человека нужно отметить в CVAT
        frames = np.array([f0], dtype=np.float64)
        boxes = np.array([[0.0, 0.0, width * 0.1, height * 0.2]])
    shapes = [Shape(int(frames[k]), tuple(float(v) for v in boxes[k])) for k in thin_keyframes(frames, boxes, max_span=max_span)]
    if shapes[-1].frame != f1:  # последняя рамка без интерполяции держится до f1
        shapes.append(Shape(f1, shapes[-1].box))
    if f1 + 1 < n_frames:
        shapes.append(Shape(f1 + 1, shapes[-1].box, outside=True))
    notes = [m.note for m in ep.marks if m.note]
    attrs = {
        "peak_sec": f"{min(max(ep.peak, ep.start), ep.end):.2f}",
        "person": "" if ep.person is None else str(ep.person),
        "note": "; ".join(["отметки " + ", ".join(m.text for m in ep.marks), *dict.fromkeys(notes)]),
        "check": "true" if any(m.check for m in ep.marks) else "false",
    }
    return CvatTrack(ep.label, shapes, attrs)


def load_trackset(path: Path) -> TrackSet:
    """Кэш треков без проверки ключа: для разметки годится кэш любого прогона того же видео."""
    with open(path, "rb") as f:
        return pickle.load(f)["trackset"]


def n_frames_of(ts: TrackSet) -> int:
    last = max((o.frame_idx for obs in ts.tracks.values() for o in obs), default=-1)
    return max(ts.meta.frame_count, last + 1, 1)


def build_clip(ts: TrackSet, marks: List[Mark], cfg: Config, persons: bool = True) -> Tuple[List[CvatTrack], List[Episode]]:
    """Треки CVAT клипа: эпизоды разметки, затем люди."""
    ft = FrameTime.from_trackset(ts)
    n = n_frames_of(ts)
    fps = ts.meta.fps if ts.meta.fps > 0 else 25.0
    span = int(round(KEY_SPAN * fps))
    tracks, signals = clip_signals(ts, cfg)
    marks = [dataclasses.replace(m) for m in marks]
    prev_smoke: Optional[Mark] = None
    for m in marks:
        if m.kind == "continue" and prev_smoke is not None and prev_smoke.person is not None:
            m.person, m.check = prev_smoke.person, True
        else:
            assign_person(m, signals, cfg)
        if m.label == "smoking":
            prev_smoke = m
    episodes = group_episodes(marks, cfg.episode.merge_gap)
    out = [episode_track(e, tracks, ft, n, ts.meta.width, ts.meta.height, span) for e in episodes]
    if persons:
        for tid, obs in sorted(tracks.items()):
            obs = sorted(obs, key=lambda o: o.frame_idx)
            if len(obs) < 3:
                continue
            frames = np.array([o.frame_idx for o in obs], dtype=np.float64)
            boxes = np.stack([o.box for o in obs]).astype(np.float64)
            shapes = obs_shapes(frames, boxes, n, int(round(TRACK_GAP * fps)), span)
            out.append(CvatTrack(PERSON, shapes, {"person": str(tid)}))
    return out, episodes


# --- импорт -------------------------------------------------------------------------------------------


GT_FIELDS = ["camera_id", "clip_id", "label", "start_sec", "end_sec", "person_gt_id", "peak_sec",
             "x1", "y1", "x2", "y2", "scenario", "video", "note", "check"]  # fmt: skip


def gt_rows(cf: CvatFile, camera_id: str, clip_id: str, video: str, ft: FrameTime, stop_frame: int) -> List[dict]:
    """Эталонные события из треков эпизодов: по событию на видимый отрезок трека."""
    rows = []
    for tr in cf.tracks:
        label = EVENT_LABELS.get(tr.label)
        if label is None:
            continue
        for a, b in tr.segments(stop_frame):
            start, end = ft.sec(a), ft.sec(b)
            try:
                peak = float(tr.attributes.get("peak_sec", ""))
            except ValueError:
                peak = float("nan")
            if not (start <= peak <= end):
                peak = (start + end) / 2
            box = tr.box_at(min(max(ft.frame(peak), a), b))
            person = tr.attributes.get("person", "") or f"cvat{tr.track_id}"
            rows.append({
                "camera_id": camera_id, "clip_id": clip_id, "label": label,
                "start_sec": f"{start:.2f}", "end_sec": f"{end:.2f}", "person_gt_id": person,
                "peak_sec": f"{peak:.2f}",
                **{k: f"{v:.1f}" for k, v in zip(("x1", "y1", "x2", "y2"), box or ())},
                "scenario": tr.attributes.get("scenario", ""), "video": video,
                "note": tr.attributes.get("note", ""), "check": tr.attributes.get("check", ""),
            })  # fmt: skip
    return sorted(rows, key=lambda r: float(r["start_sec"]))


def write_gt(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=GT_FIELDS, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    path.write_text(buf.getvalue(), encoding="utf-8", newline="")
