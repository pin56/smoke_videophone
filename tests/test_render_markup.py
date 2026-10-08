"""tools/render_markup.py: рамки треков CVAT по кадрам, фазы отметок, ролик на синтетическом видео."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import render_markup as rm  # noqa: E402
from smokedet.cvat import CvatFile, CvatTrack, Mark, Shape  # noqa: E402


def make_track():
    shapes = [Shape(10, (0, 0, 10, 10)), Shape(20, (10, 0, 20, 10)), Shape(30, (10, 0, 20, 10), outside=True),
              Shape(40, (0, 0, 5, 5))]  # fmt: skip
    return CvatTrack("smoking", shapes, {"person": "3", "check": "true"})


def test_track_boxes_match_box_at():
    tr = make_track()
    tb = rm.TrackBoxes(tr)
    for f in range(0, 50):
        a, b = tb.at(f), tr.box_at(f)
        assert (a is None) == (b is None), f
        if a is not None:
            np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(tb.at(15), (5, 0, 15, 10))
    assert tb.at(25) is not None and tb.at(30) is None and tb.at(45) is not None


def test_phase_and_lines():
    m = Mark("smoking", 10.0, 14.0, 12.0, "10/12/14")
    assert rm.phase(m, 9.0) is None
    assert rm.phase(m, 10.0) == "рука ко рту"
    assert rm.phase(m, 12.4) == "затяжка"
    assert rm.phase(m, 13.5) == "отвод руки"
    point = Mark("smoking", 17.0, 17.0, 17.0, "17/17/17")  # нулевая длина видна ±MARK_PAD
    assert rm.phase(point, 17.3) == "затяжка"
    cont = Mark("ignore", 14.0, float("inf"), kind="continue", text="после 10/12/14")
    lines = rm.mark_lines([m, point, cont], 17.0)
    assert lines == [("КУРИТ — цикл 2/2: затяжка (17/17/17)", "smoking"),
                     ("ИГНОР: продолжают курить (после 10/12/14)", "ignore")]  # fmt: skip
    assert rm.track_text(make_track()) == "КУРИТ · чел. 3 · проверить"


def test_label_clipped_at_frame_edge():
    p = rm.Painter(120)
    frame = np.zeros((120, 160, 3), np.uint8)
    h = p.label(frame, "КУРИТ · чел. 12 · проверить", 150, 5, rm.BGR["smoking"])  # шире кадра, у края
    assert h > 0 and frame.any()


def test_render_clip(tmp_path):
    video = tmp_path / "v.avi"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    for _ in range(20):
        w.write(np.full((48, 64, 3), 90, np.uint8))
    w.release()
    cf = CvatFile(path="v.xml", stop_frame=19, tracks=[make_track()])
    out = tmp_path / "out.mp4"
    n = rm.render_clip(video, cf, [Mark("smoking", 0.5, 1.5, 1.0, "0.5/1/1.5")], out, "клип", max_width=0)
    assert n == 20
    cap = cv2.VideoCapture(str(out))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape[:2] == (48, 64)
