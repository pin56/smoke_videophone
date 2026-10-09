"""Обмен разметкой с CVAT: время кадров, ключевые кадры, разбор XML, экспорт -> импорт без моделей."""
import numpy as np
import pytest

from smokedet.config import Config
from smokedet.cvat import (
    CvatTrack, FrameTime, Mark, Shape, assign_person, build_clip, gt_rows, lerp_boxes, load_marks,
    read_cvat, thin_keyframes, write_xml, _iou,
)  # fmt: skip
from smokedet.geometry import TrackSignal
from smokedet.io_video import VideoMeta
from smokedet.tracking import PersonObs, TrackSet

FPS, STRIDE, N = 30.0, 3, 1800  # 60 с


def vfr_time(idx):
    return idx / FPS * 1.01 + 0.2 * np.sin(idx / 300)  # переменный fps: idx/fps уходит от pts


def make_trackset(boxes_by_track):
    """boxes_by_track: tid -> функция кадр -> рамка или None (человека нет)."""
    tracks = {}
    for tid, fn in boxes_by_track.items():
        obs = []
        for i in range(0, N, STRIDE):
            b = fn(i)
            if b is not None:
                obs.append(PersonObs(i, float(vfr_time(i)), np.array(b, float), 0.9, np.zeros((17, 3))))
        tracks[tid] = obs
    return TrackSet(VideoMeta("x.mp4", FPS, N, 1920, 1080), STRIDE, tracks)


def test_frame_time_roundtrip():
    frames = np.arange(0, N, STRIDE)
    ft = FrameTime(frames, vfr_time(frames), FPS)
    for f in (0, 1, 100, 1001, N - 1):
        assert ft.sec(f) == pytest.approx(vfr_time(f), abs=0.01)
        assert ft.frame(ft.sec(f)) == f
    assert ft.frame(-1.0) < 0 and ft.sec(N + 30) == pytest.approx(ft.sec(N - 1) + 31 / FPS, abs=0.05)


def test_thin_keyframes():
    frames = np.arange(0, 300, 3, dtype=float)
    static = np.tile([100, 100, 200, 400], (len(frames), 1)).astype(float)
    assert thin_keyframes(frames, static) == [0, len(frames) - 1]
    keys = thin_keyframes(frames, static, max_span=60)
    assert all(frames[b] - frames[a] <= 60 for a, b in zip(keys, keys[1:]))
    # шаг посередине: интерполяция по ключевым кадрам восстанавливает все рамки
    moving = static.copy()
    moving[50:, [0, 2]] += 300
    keys = thin_keyframes(frames, moving)
    rebuilt = lerp_boxes(frames[keys], moving[keys], frames)
    assert len(keys) < 6 and _iou(rebuilt, moving).min() >= 0.85


def test_segments_and_box_at():
    tr = CvatTrack("smoking", [Shape(10, (0, 0, 10, 10)), Shape(20, (10, 0, 20, 10)), Shape(31, (10, 0, 20, 10), outside=True),
                               Shape(40, (0, 0, 5, 5))])  # fmt: skip
    assert tr.segments(99) == [(10, 30), (40, 99)]
    assert tr.box_at(15) == pytest.approx((5, 0, 15, 10))
    assert tr.box_at(25) == (10, 0, 20, 10) and tr.box_at(35) is None and tr.box_at(5) is None


def test_load_marks_continue(tmp_path):
    p = tmp_path / "m.csv"
    p.write_text(
        "video,folder,clip_id,label,up_sec,puff_sec,down_sec\n"
        "v,true,true_0,SMOKE,10,12,14\nv,true,true_0,SMOKE,,,\nv,true,true_0,SMOKE,40,41,42\n"
        "v,true,true_0,NOSMOKE,50,,52\nw,false,false_1,IGNORE,1,2,3\n",
        encoding="utf-8",
    )
    m = load_marks(str(p))
    got = [(x.label, x.start, x.end, x.kind) for x in m["v"]]
    assert got == [("smoking", 10, 14, ""), ("ignore", 14, 40, "continue"), ("smoking", 40, 42, ""),
                   ("not_smoking", 50, 52, "")]  # fmt: skip
    assert m["v"][0].puff == 12 and m["v"][3].text == "50/-/52" and m["w"][0].label == "ignore"


def _signal(tid, t, d):
    n = len(t)
    nan = np.full(n, np.nan)
    return TrackSignal(tid, np.asarray(t, float), np.arange(n), np.zeros((n, 4)) + [0, 0, 10, 100], np.ones(n, bool),
                       np.zeros((n, 2)), np.zeros((n, 2)), np.ones(n), np.asarray(d, float), np.asarray(d, float),
                       nan, nan, np.ones(n), nan, nan, np.ones(n))  # fmt: skip


def test_assign_person_picks_closest_hand():
    t = np.arange(0, 30, 0.1)
    near = np.where((t > 10) & (t < 13), 0.3, 1.5)
    signals = {1: _signal(1, t, np.full(len(t), 1.4)), 2: _signal(2, t, near)}
    m = Mark("smoking", 10, 13, 11.5, "10/11.5/13")
    assign_person(m, signals, Config())
    assert m.person == 2 and not m.check
    far = Mark("smoking", 20, 22, 21, "20/21/22")
    assign_person(far, signals, Config())
    assert far.check and "далеко" in far.note
    lost = Mark("smoking", 100, 102, 101, "100/101/102")
    assign_person(lost, signals, Config())
    assert lost.person is None and lost.check


def test_export_import_roundtrip(tmp_path):
    def walker(i):  # идёт вправо весь клип
        return (100 + i * 0.5, 300, 250 + i * 0.5, 700)

    def visitor(i):  # виден 5 с
        return (1500, 200, 1600, 500) if 900 <= i < 1050 else None

    ts = make_trackset({1: walker, 2: visitor})
    marks = [Mark("smoking", 10.0, 13.0, 11.5, "10/11.5/13"), Mark("smoking", 20.0, 24.0, 22.0, "20/22/24"),
             Mark("not_smoking", 50.0, 52.0, 51.0, "50/51/52")]  # fmt: skip
    tracks, episodes = build_clip(ts, marks, Config())
    # поз нет (точки с нулевой уверенностью) — берётся человек, дольше всех видимый в окне, с флажком check
    assert [(e.label, e.person, len(e.marks)) for e in episodes] == [("smoking", 1, 2), ("not_smoking", 1, 1)]
    assert sum(t.label == "person" for t in tracks) == 2

    path = tmp_path / "clip.xml"
    write_xml(path, "clip", "x.mp4", N, 1920, 1080, tracks)
    (cf,) = read_cvat(path)
    assert cf.source == "x.mp4" and cf.stop_frame == N - 1 and len(cf.tracks) == 4
    ft = FrameTime.from_trackset(ts)
    rows = gt_rows(cf, "cam", "clip", "x.mp4", ft, cf.stop_frame)
    assert [r["label"] for r in rows] == ["POSITIVE", "NEGATIVE"]
    pos = rows[0]
    assert float(pos["start_sec"]) == pytest.approx(10.0, abs=0.05) and float(pos["end_sec"]) == pytest.approx(24.0, abs=0.05)
    peak = float(pos["peak_sec"])
    assert peak == pytest.approx(22.0, abs=0.05)  # затяжка средней отметки
    want = np.array(walker(ft.frame(peak)))
    got = np.array([float(pos[k]) for k in ("x1", "y1", "x2", "y2")])
    assert _iou(got[None], want[None])[0] > 0.9
    assert pos["person_gt_id"] == "1" and pos["check"] == "true"


def test_import_cvat_style_every_frame(tmp_path):
    """Выгрузка CVAT: рамка на каждом кадре (keyframe=0 у интерполированных), outside в конце."""
    xml = ['<?xml version="1.0" encoding="utf-8"?><annotations><version>1.1</version><meta><task><name>t</name>'
           '<size>100</size><stop_frame>99</stop_frame></task></meta><track id="7" label="ignore" source="manual">']  # fmt: skip
    for f in range(20, 31):
        xml.append(f'<box frame="{f}" keyframe="{int(f == 20)}" outside="0" occluded="0" xtl="{f}" ytl="0" xbr="{f + 10}" '
                   f'ybr="10" z_order="0"><attribute name="note">a</attribute></box>')  # fmt: skip
    xml.append('<box frame="31" keyframe="1" outside="1" occluded="0" xtl="30" ytl="0" xbr="40" ybr="10" z_order="0"/>')
    xml.append("</track></annotations>")
    p = tmp_path / "t.xml"
    p.write_text("".join(xml), encoding="utf-8")
    (cf,) = read_cvat(p)
    ft = FrameTime([], [], 10.0)
    (row,) = gt_rows(cf, "cam", "t", "t.mp4", ft, cf.stop_frame)
    assert row["label"] == "IGNORE" and row["start_sec"] == "2.00" and row["end_sec"] == "3.00"
    assert row["peak_sec"] == "2.50" and row["x1"] == "25.0" and row["person_gt_id"] == "cvat7"
