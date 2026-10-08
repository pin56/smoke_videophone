"""Рендер ролика: на синтетическом видео рисуются фазы цикла, причины отказа и счётчики."""
import cv2
import numpy as np

from smokedet.cycles import detect_cycles
from smokedet.config import Config
from smokedet.episodes import build_events
from smokedet.io_video import VideoMeta
from smokedet.render import _phase, render_video
from smokedet.roi import Roi
from tests.test_logic import FPS, make_signal, puffs


def test_phase_and_render(tmp_path):
    sig = make_signal(puffs([2.0, 8.0]))
    cfg = Config()
    cycles = {1: detect_cycles(sig, cfg.cycle)}
    c = cycles[1][0]
    assert _phase(cycles[1], c.start - 1.0) is None
    assert _phase(cycles[1], (c.hold_start + c.hold_end) / 2)[0].startswith("HOLD")
    c.valid, c.reject_reason = False, "appearance:low"
    assert _phase(cycles[1], c.end + 1.0)[0] == "REJECT appearance:low"

    video = tmp_path / "in.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (320, 400))
    for _ in range(120):
        w.write(np.zeros((400, 320, 3), np.uint8))
    w.release()
    meta = VideoMeta(str(video), FPS, 120, 320, 400)
    events = build_events(cycles, {1: sig}, cfg.episode)
    out = render_video(meta, 3, tmp_path / "out.mp4", {1: sig}, cycles, events, Roi.full(), "t")
    cap = cv2.VideoCapture(str(out))
    ok, frame = cap.read()
    assert ok and frame.shape == (400, 320, 3) and frame.any()
