"""Оценка внешнего вида цикла (appearance.py) без CLIP: кодировщик подменён оценкой по яркости кропа."""
import numpy as np
import pytest
from test_logic import make_signal, puffs

from smokedet import appearance
from smokedet.appearance import PAD_VALUE, AppearanceScorer, cycle_score, plan_frames, square_crop, view_plan
from smokedet.config import AppearanceConfig, CycleConfig, EpisodeConfig
from smokedet.cycles import detect_cycles
from smokedet.episodes import build_events


def test_square_crop_pads_outside_frame():
    frame = np.arange(100, dtype=np.uint8).reshape(10, 10, 1).repeat(3, axis=2)
    crop = square_crop(frame, 1, 1, 8)  # квадрат (-3..5, -3..5): левый верхний угол за кадром
    assert crop.shape == (8, 8, 3)
    assert (crop[:3, :, 0] == PAD_VALUE).all() and (crop[:, :3, 0] == PAD_VALUE).all()
    assert crop[3, 3, 0] == frame[0, 0, 0] and crop[7, 7, 0] == frame[4, 4, 0]
    assert (square_crop(frame, 50, 50, 8) == PAD_VALUE).all()  # целиком за кадром


def test_plan_frames_inside_hold():
    sig = make_signal(puffs([5], hold=1.5))
    c = detect_cycles(sig, CycleConfig())[0]
    ks = plan_frames(sig, c, 4)
    assert 1 <= len(ks) <= 4 and len(set(ks)) == len(ks)
    assert all(c.hold_start <= sig.t[k] <= c.hold_end for k in ks)
    assert len(plan_frames(sig, c, 100)) == np.sum((sig.t >= c.hold_start) & (sig.t <= c.hold_end))


def test_cycle_score_sums_view_maxima():
    assert cycle_score([np.array([1.0, 3.0]), np.array([-2.0, 0.5])]) == pytest.approx(3.5)


def test_cycle_score_top2_and_tight_view():
    # top2 — среднее двух лучших кадров вида: один случайно высокий кадр не решает
    assert cycle_score([np.array([1.0, 3.0, 9.0]), np.array([0.5])], "top2") == pytest.approx(6.0 + 0.5)
    with pytest.raises(ValueError):
        cycle_score([np.array([1.0])], "median")
    assert view_plan(AppearanceConfig()) == [(0.35, 0), (0.7, 1)]
    assert view_plan(AppearanceConfig(tight_scale=0.2)) == [(0.35, 0), (0.7, 1), (0.2, 0)]  # тесный — словарь узкого


def _video(path, level, seconds=12.0, fps=30.0):
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 320))
    for _ in range(int(seconds * fps)):
        writer.write(np.full((320, 320, 3), level, np.uint8))
    writer.release()
    return str(path)


class BrightnessEncoder:
    """Оценка кропа — его средняя яркость / 10; считает вызовы."""

    def __init__(self):
        self.calls = 0

    def __call__(self, crops, view):
        self.calls += 1
        assert view in (0, 1) and crops
        return np.array([c.mean() / 10 for c in crops])


def test_scorer_rejects_dark_keeps_bright_and_caches(tmp_path):
    sig = make_signal(puffs([5]))  # кадр = 3 * отсчёт, как у видео 30 к/с при сигнале 10 Гц
    cfg = AppearanceConfig(min_score=25.0)  # яркость 200 -> 20 + 20 = 40; 50 -> 5 + 5 = 10
    enc = BrightnessEncoder()
    scorer = AppearanceScorer(encoder=enc)
    bright = detect_cycles(sig, CycleConfig())
    scorer.score("b", _video(tmp_path / "b.avi", 200), {1: bright}, {1: sig}, cfg)
    assert bright[0].valid and bright[0].appearance == pytest.approx(40, abs=1)
    calls = enc.calls
    again = detect_cycles(sig, CycleConfig())
    scorer.score("b", "нет такого файла", {1: again}, {1: sig}, cfg)  # из кэша: видео не читается
    assert enc.calls == calls and again[0].appearance == bright[0].appearance

    dark = detect_cycles(sig, CycleConfig())
    scorer.score("d", _video(tmp_path / "d.avi", 50), {1: dark}, {1: sig}, cfg)
    assert not dark[0].valid and dark[0].reject_reason.startswith("appearance:")
    kept = detect_cycles(sig, CycleConfig())
    scorer.score("d", "нет такого файла", {1: kept}, {1: sig}, cfg, keep=lambda c: True)  # прямой признак
    assert kept[0].valid and kept[0].appearance == pytest.approx(10, abs=1)


def test_look_and_rescue(tmp_path):
    sig = make_signal(puffs([5]))
    video = _video(tmp_path / "b.avi", 200)  # оценка 40
    cycles = detect_cycles(sig, CycleConfig())
    assert build_events({1: cycles}, {1: sig}, EpisodeConfig()) == []  # один цикл без признака — не эпизод
    scorer = AppearanceScorer(encoder=BrightnessEncoder())
    scorer.score("b", video, {1: cycles}, {1: sig}, AppearanceConfig(min_score=25.0, direct_score=30.0))
    assert cycles[0].look
    events = build_events({1: cycles}, {1: sig}, EpisodeConfig())
    assert len(events) == 1 and events[0].passed and events[0].look  # look — прямой признак
    assert build_events({1: cycles}, {1: sig}, EpisodeConfig(appearance_as_direct=False)) == []

    for reason, score, back in (("object:bottle", 35.0, True), ("track_object:cup", 45.0, False),
                                ("track_object:lollipop", 0.0, False), ("short_hold", 0.0, False)):  # fmt: skip
        cs = detect_cycles(sig, CycleConfig())
        cs[0].valid, cs[0].reject_reason = False, reason
        scorer.score("b", video, {1: cs}, {1: sig}, AppearanceConfig(min_score=25.0, rescue_score=score))
        assert cs[0].valid == back, reason
        assert cs[0].rescued_from == (reason if back else "") and cs[0].reject_reason == ("" if back else reason)

    cfg = AppearanceConfig(min_score=25.0, rescue_score=35.0, rescue_reasons=["face_hidden", "no_approach"])
    for reason, back in (("face_hidden", True), ("face_hidden,no_approach", True), ("face_hidden,short_hold", False),
                         ("hand_low", False), ("track_object:lollipop", False)):  # fmt: skip
        cs = detect_cycles(sig, CycleConfig())
        cs[0].valid, cs[0].reject_reason = False, reason
        scorer.score("b", video, {1: cs}, {1: sig}, cfg)
        assert cs[0].valid == back, reason


def test_rejected_cycles_do_not_make_event(tmp_path):
    sig = make_signal(puffs([5, 15]))
    cycles = detect_cycles(sig, CycleConfig())
    assert len(build_events({1: cycles}, {1: sig}, EpisodeConfig())) == 1
    scorer = AppearanceScorer(encoder=BrightnessEncoder())
    scorer.score("d", _video(tmp_path / "d.avi", 50, seconds=22.0), {1: cycles}, {1: sig}, AppearanceConfig(min_score=25.0))
    assert not any(c.valid for c in cycles)
    assert build_events({1: cycles}, {1: sig}, EpisodeConfig()) == []


def test_scorer_without_model_keeps_cycles(monkeypatch, tmp_path):
    def boom(*a, **k):
        raise RuntimeError("нет весов")

    monkeypatch.setattr(appearance, "ClipEncoder", boom)
    sig = make_signal(puffs([5]))
    cycles = detect_cycles(sig, CycleConfig())
    scorer = AppearanceScorer()
    scorer.score("x", _video(tmp_path / "x.avi", 50), {1: cycles}, {1: sig}, AppearanceConfig())
    assert cycles[0].valid and np.isnan(cycles[0].appearance)
    info = scorer.describe(AppearanceConfig())
    assert not info["available"] and "нет весов" in info["error"]
