"""Тесты логики без моделей: синтетический сигнал d(t) → циклы → события; протокол оценки; ROI."""
import json

import numpy as np
import pytest

from smokedet.config import CycleConfig, EpisodeConfig, Config, FlashConfig, VerifierConfig, load_config, with_overrides
from smokedet.cycles import detect_cycles
from smokedet.episodes import build_events
from smokedet.flash import Flash, LightMeter, attach_flashes, find_flashes
from smokedet.evaluation import EvalParams, Item, evaluate
from smokedet.geometry import TrackSignal, build_signal
from smokedet.roi import Roi
from smokedet.tracking import PersonObs, stitch_tracks
from smokedet.verifier import ObjectVerifier

FPS = 10.0


def make_signal(d, track_id=1, ear_ratio=2.0, wrist_above=1.0, wrist_dy=0.5, elbow_raise=-0.8, face_visible=1.0):
    d = np.asarray(d, dtype=float)
    n = len(d)
    t = np.arange(n) / FPS
    return TrackSignal(
        track_id=track_id,
        t=t,
        frame_idx=np.arange(n) * 3,
        boxes=np.tile([100.0, 100.0, 160.0, 300.0], (n, 1)),
        valid=np.ones(n, bool),
        mouth=np.tile([130.0, 120.0], (n, 1)),
        hand=np.tile([130.0, 150.0], (n, 1)),
        scale=np.full(n, 30.0),
        d_raw=d.copy(),
        d=d,
        ear_ratio=np.full(n, ear_ratio),
        wrist_above=np.full(n, wrist_above),
        kconf=np.full(n, 0.8),
        wrist_dy=np.full(n, wrist_dy),
        elbow_raise=np.full(n, elbow_raise),
        face_visible=np.full(n, face_visible),
    )


def puffs(starts, hold=1.5, total=40.0, far=1.5, near=0.3, ramp=0.5):
    """Сигнал: кисть далеко (far), в моменты starts подходит ко рту на hold секунд."""
    t = np.arange(0, total, 1 / FPS)
    d = np.full(len(t), far)
    for s in starts:
        for i, ti in enumerate(t):
            if s <= ti < s + ramp:
                d[i] = far - (far - near) * (ti - s) / ramp
            elif s + ramp <= ti < s + ramp + hold:
                d[i] = near
            elif s + ramp + hold <= ti < s + 2 * ramp + hold:
                d[i] = near + (far - near) * (ti - s - ramp - hold) / ramp
    return d


def test_two_puffs_give_one_event():
    sig = make_signal(puffs([5, 15]))
    cycles = detect_cycles(sig, CycleConfig())
    assert [c.valid for c in cycles] == [True, True]
    c = cycles[0]
    assert 4.5 <= c.start <= 5.2 and 1.2 <= c.hold <= 1.9 and 7.0 <= c.end <= 7.8
    events = build_events({1: cycles}, {1: sig}, EpisodeConfig())
    assert len(events) == 1
    e = events[0]
    assert e.passed and e.start == cycles[0].start and e.end == cycles[1].end


def test_single_puff_without_evidence_is_not_event():
    sig = make_signal(puffs([5]))
    cycles = detect_cycles(sig, CycleConfig())
    assert len(cycles) == 1 and cycles[0].valid
    assert build_events({1: cycles}, {1: sig}, EpisodeConfig()) == []


def test_single_puff_with_direct_evidence_is_event():
    sig = make_signal(puffs([5]))
    cycles = detect_cycles(sig, CycleConfig())
    cycles[0].direct, cycles[0].pos_conf, cycles[0].pos_frames = True, 0.35, 3
    events = build_events({1: cycles}, {1: sig}, EpisodeConfig())
    assert len(events) == 1 and events[0].passed


def test_puffs_far_apart_are_not_paired():
    # циклы в 25 с друг от друга: окно 20 с не покрывает оба
    sig = make_signal(puffs([2, 27], total=45))
    cycles = detect_cycles(sig, CycleConfig())
    assert len(cycles) == 2
    assert build_events({1: cycles}, {1: sig}, EpisodeConfig()) == []


def test_merge_gap():
    # пары (2, 10) и (40, 48): между ними пауза > 15 с → два события; (2,10,22,30) → одно
    sig = make_signal(puffs([2, 10, 40, 48], total=60))
    events = build_events({1: detect_cycles(sig, CycleConfig())}, {1: sig}, EpisodeConfig())
    assert len(events) == 2
    sig = make_signal(puffs([2, 10, 22, 30], total=40))
    events = build_events({1: detect_cycles(sig, CycleConfig())}, {1: sig}, EpisodeConfig())
    assert len(events) == 1 and len(events[0].cycles) == 4


def test_long_hold_rejected():
    sig = make_signal(puffs([3, 15], hold=8.0, total=30))
    cycles = detect_cycles(sig, CycleConfig())
    assert cycles and all(not c.valid and "long_hold" in c.reject_reason for c in cycles)


def test_short_touch_rejected():
    sig = make_signal(puffs([3, 8], hold=0.1, total=20))
    cycles = detect_cycles(sig, CycleConfig())
    assert cycles and all("short_hold" in c.reject_reason for c in cycles)


def test_phone_like_rejected():
    sig = make_signal(puffs([3, 10]), ear_ratio=0.4)
    cycles = detect_cycles(sig, CycleConfig())
    assert cycles and all("phone_like" in c.reject_reason for c in cycles)


def test_hand_above_mouth_rejected():
    # кисть у лба/волос: запястье выше рта или локоть выше плеча
    for kw in ({"wrist_dy": -0.3}, {"elbow_raise": 0.25}):
        cycles = detect_cycles(make_signal(puffs([3, 10]), **kw), CycleConfig())
        assert cycles and all("hand_above_mouth" in c.reject_reason for c in cycles)


def test_face_hidden_rejected():
    cycles = detect_cycles(make_signal(puffs([3, 10]), face_visible=0.0), CycleConfig())
    assert cycles and all("face_hidden" in c.reject_reason for c in cycles)


def _verifier():
    v = object.__new__(ObjectVerifier)  # без загрузки модели: проверяется только агрегация
    cfg = VerifierConfig()
    v.positive, v.negative = set(cfg.positive_classes), set(cfg.negative_classes)
    return v, cfg


def _cycle():
    return detect_cycles(make_signal(puffs([3])), CycleConfig())[0]


def test_strong_negative_rejects_cycle():
    v, cfg = _verifier()
    c = _cycle()
    v._aggregate(c, [{"beverage can": 0.7}, {"beverage can": 0.3}, {}, {}, {}, {}], cfg)
    assert not c.valid and c.reject_reason == "object:beverage can" and c.strong_negative == "beverage can"


def test_weak_negative_and_positive_do_not_reject():
    v, cfg = _verifier()
    c = _cycle()  # «телефон» — частая ложная метка на кисти: сильным негативом не бывает
    v._aggregate(c, [{"cell phone": 0.5}, {"cell phone": 0.5}, {}, {}, {}, {}], cfg)
    assert c.valid and not c.strong_negative and c.neg_fraction == pytest.approx(2 / 6)
    c = _cycle()  # сигарета рядом с бутылкой: цикл остаётся
    v._aggregate(c, [{"bottle": 0.6, "cigarette": 0.3}, {"bottle": 0.6}, {}, {}, {}, {}], cfg)
    assert c.valid and c.strong_negative == "bottle"


def test_negatives_in_most_cycles_reject_event():
    sig = make_signal(puffs([2, 8, 14], total=25))
    cycles = detect_cycles(sig, CycleConfig())
    for c in cycles[:2]:
        c.neg_fraction = 1 / 6  # слабый предмет в одном кадре — цикл не отброшен
    events = build_events({1: cycles}, {1: sig}, EpisodeConfig())
    assert len(events) == 1 and not events[0].passed and events[0].reject_reason
    cycles[1].neg_fraction = 0.0  # 1 из 3 циклов — событие остаётся
    events = build_events({1: cycles}, {1: sig}, EpisodeConfig())
    assert events[0].passed and not events[0].reject_reason


def test_no_approach_rejected():
    # трек начинается с кисти у рта — фазы подхода не видно
    d = puffs([10], total=20)
    d[:15] = 0.3
    cycles = detect_cycles(make_signal(d), CycleConfig())
    assert cycles[0].reject_reason == "no_approach"


def test_people_not_merged():
    s1, s2 = make_signal(puffs([5, 12]), track_id=1), make_signal(puffs([6, 13]), track_id=2)
    cycles = {1: detect_cycles(s1, CycleConfig()), 2: detect_cycles(s2, CycleConfig())}
    events = build_events(cycles, {1: s1, 2: s2}, EpisodeConfig())
    assert sorted(e.track_id for e in events) == [1, 2]


def _obs(t, x, kpts=None):
    return PersonObs(int(t * 10), t, np.array([x, 100, x + 50, 250], np.float32), 0.9,
                     kpts if kpts is not None else np.zeros((17, 3), np.float32))  # fmt: skip


def test_stitch_tracks():
    cfg = Config().tracking
    tracks = {1: [_obs(t / 10, 100) for t in range(0, 30)], 2: [_obs(3.5 + t / 10, 105) for t in range(20)],
              3: [_obs(3.5 + t / 10, 900) for t in range(20)]}  # fmt: skip
    merged = stitch_tracks(tracks, cfg)
    assert sorted(merged) == [1, 3] and len(merged[1]) == 50


def test_build_signal_geometry():
    k = np.zeros((17, 3), np.float32)
    k[:, 2] = 0.9
    k[0, :2] = [125, 120]  # нос
    k[1, :2], k[2, :2] = [120, 115], [130, 115]
    k[3, :2], k[4, :2] = [112, 118], [138, 118]
    k[5, :2], k[6, :2] = [105, 150], [145, 150]  # плечи → scale = 40
    k[7, :2], k[8, :2] = [100, 190], [150, 190]
    k[9, :2], k[10, :2] = [100, 230], [128, 135]  # правое запястье у рта
    obs = [_obs(i / 10, 100, k.copy()) for i in range(5)]
    sig = build_signal(1, obs, Roi.full(), Config().geometry, Config().tracking, 10.0)
    assert sig.scale[0] == pytest.approx(40, abs=1)
    assert sig.d[2] < 0.6  # кисть у рта
    assert sig.wrist_above[2] == 1.0


def _item(start, end, box=(0, 0, 100, 200), label="POSITIVE", clip="c1", **row):
    return Item("cam", clip, start, end, box, label, row.pop("confidence", 1.0), row)


def test_evaluate_protocol():
    gts = [
        _item(10, 20, scenario="P01"),
        _item(40, 50, scenario="P02"),
        _item(60, 70, label="IGNORE"),
        _item(80, 90, label="NEGATIVE", scenario="N02"),
    ]
    preds = [
        _item(22, 25),  # пересекает расширенный эталон [7, 23] на 1 с → TP
        _item(41, 45, box=(500, 0, 600, 200)),  # IoU = 0 → не пара → FP
        _item(61, 68),  # внутри IGNORE → не FP
        _item(82, 88),  # на негативе N02 → FP
    ]
    rep = evaluate(preds, gts, {("cam", "c1"): 3600.0}, EvalParams())
    assert (rep["tp"], rep["fp"], rep["fn"], rep["ignored_preds"]) == (1, 2, 1, 1)
    assert rep["precision"] == pytest.approx(1 / 3) and rep["recall"] == pytest.approx(0.5)
    assert rep["latency_median_sec"] == pytest.approx(12.0)
    assert any(r["result"] == "FP" and r["gt_scenario"] == "N02" for r in rep["matches"])
    assert rep["negative_hours"] == pytest.approx((3600 - 30) / 3600)


def test_evaluate_one_to_one():
    gts = [_item(10, 20)]
    preds = [_item(10, 20), _item(12, 18)]
    rep = evaluate(preds, gts)
    assert (rep["tp"], rep["fp"]) == (1, 1)


def test_roi_polygons(tmp_path):
    path = tmp_path / "roi.json"
    path.write_text(json.dumps({"include": [[[0, 0], [100, 0], [100, 100], [0, 100]]],
                                "exclude": [[[40, 40], [60, 40], [60, 60], [40, 60]]]}))  # fmt: skip
    roi = Roi.load(str(path), 200, 200)
    assert roi.contains(10, 10) and not roi.contains(50, 50) and not roi.contains(150, 150)
    norm = Roi.from_polygons([[[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]], [], 200, 200)
    assert norm.contains(90, 90) and not norm.contains(110, 110)


def test_config_roundtrip():
    cfg = load_config("configs/default.yaml", ["episode.confidence_threshold=0.7"])
    assert cfg.episode.confidence_threshold == 0.7
    defaults = Config().to_dict()
    defaults["episode"]["confidence_threshold"] = 0.7
    assert cfg.to_dict() == defaults  # default.yaml не расходится с dataclass-значениями
    assert with_overrides(cfg, ["cycle.near_enter=0.5"]).cycle.near_enter == 0.5
    with pytest.raises(KeyError):
        load_config(None, ["cycle.unknown=1"])


def _face_kpts():
    """Те же точки, что в test_build_signal_geometry: scale = 40, правая кисть у рта."""
    k = np.zeros((17, 3), np.float32)
    k[:, 2] = 0.9
    k[0, :2] = [125, 120]
    k[1, :2], k[2, :2] = [120, 115], [130, 115]
    k[3, :2], k[4, :2] = [112, 118], [138, 118]
    k[5, :2], k[6, :2] = [105, 150], [145, 150]
    k[7, :2], k[8, :2] = [100, 190], [150, 190]
    k[9, :2], k[10, :2] = [100, 230], [128, 135]
    return k


def test_light_meter():
    k, box = _face_kpts()[None], np.array([[95, 100, 155, 300]], np.float32)
    bg = np.full((400, 400, 3), 90, np.uint8)
    meter = LightMeter()
    assert meter.measure(bg, 0.0, box, k) == [None]  # фона ещё нет
    for t in (1.0, 2.0, 3.0):
        meter.measure(bg, t, box, k)
    flame = bg.copy()
    flame[124:144, 118:138] = 255  # пересвет 20x20 у рта
    zone, ring, scale = meter.measure(flame, 3.1, box, k)[0]
    assert zone == pytest.approx(400, rel=0.15) and ring == 0 and scale == pytest.approx(40, abs=1)
    door = bg.copy()
    door[100:300, 156:186] = 255  # светлая дверь рядом с человеком, но не у лица
    zone, ring, _ = meter.measure(door, 3.2, box, k)[0]
    assert zone == 0 and ring == pytest.approx(6000, rel=0.15)


def test_build_signal_light():
    obs = [_obs(i / 10, 100, _face_kpts()) for i in range(5)]
    obs[2].light = np.array([3200.0, 800.0, 40.0], np.float32)
    sig = build_signal(1, obs, Roi.full(), Config().geometry, Config().tracking, 10.0)
    assert sig.light_area[2] == pytest.approx(2.0) and sig.light_ring[2] == pytest.approx(0.5)
    assert np.isnan(sig.light_area[0])


def _lit(area_at, ring=0.0, d=0.3, n=100):
    sig = make_signal(np.full(n, d))
    sig.light_area, sig.light_ring = np.zeros(n), np.zeros(n)
    for i, a in area_at.items():
        sig.light_area[i], sig.light_ring[i] = a, ring
    return sig


def test_find_flashes():
    cfg = FlashConfig()
    flashes = find_flashes(_lit({50: 1.5, 51: 2.0, 52: 1.2}), cfg)
    assert len(flashes) == 1
    f = flashes[0]
    assert (f.start, f.end, f.area) == (pytest.approx(5.0), pytest.approx(5.2), pytest.approx(2.0))
    assert find_flashes(_lit({50: 2.0}), cfg) == []  # один отсчёт — блик
    assert find_flashes(_lit({50: 2.0, 51: 2.0}, ring=1.5), cfg) == []  # светло и вокруг: дверь, фары
    assert find_flashes(_lit({50: 2.0, 51: 2.0}, d=3.0), cfg) == []  # руки далеко от лица


def test_flash_makes_single_puff_an_event():
    sig = make_signal(puffs([5]))
    cycles = detect_cycles(sig, CycleConfig())
    attach_flashes(cycles, [Flash(1, 4.6, 4.9, 1.8, 0.1)], FlashConfig())  # прикуривание перед затяжкой
    assert cycles[0].flash and not cycles[0].direct
    events = build_events({1: cycles}, {1: sig}, EpisodeConfig())
    assert len(events) == 1 and events[0].passed and events[0].flash
    assert build_events({1: cycles}, {1: sig}, EpisodeConfig(flash_as_direct=False)) == []
    late = detect_cycles(sig, CycleConfig())
    attach_flashes(late, [Flash(1, 20.0, 20.3, 1.8, 0.1)], FlashConfig())  # вспышка намного позже цикла
    assert not late[0].flash


def test_flash_raises_confidence():
    sig = make_signal(puffs([5, 12]))
    plain = detect_cycles(sig, CycleConfig())
    lit = detect_cycles(sig, CycleConfig())
    attach_flashes(lit, [Flash(1, 4.6, 4.9, 1.8, 0.1)], FlashConfig())
    cfg = EpisodeConfig()
    assert build_events({1: lit}, {1: sig}, cfg)[0].confidence > build_events({1: plain}, {1: sig}, cfg)[0].confidence
