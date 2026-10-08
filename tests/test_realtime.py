"""Потоковый режим (smokedet/realtime.py) без моделей: отбор кадров, кольцевой буфер, трекер 1 Гц,
фиксация циклов по ходу потока и повтор из кэша, правило эпизода, конфиг режима."""
import numpy as np
import pytest

from smokedet.appearance import square_crop
from smokedet.config import Config, with_overrides
from smokedet.episodes import build_events
from smokedet.realtime import (
    FrameRing,
    PersonJob,
    SparseTracker,
    StreamFrame,
    Ticker,
    letterbox,
    load_rt_config,
    person_cycles,
    person_signals,
    replay_job,
    sample_features,
    store_crop,
)
from smokedet.roi import Roi
from smokedet.tracking import PersonObs

BOX = np.array([100.0, 100.0, 150.0, 300.0])


def pose(alpha: float) -> np.ndarray:
    """17 точек человека в BOX; правое запястье от бедра (alpha=0, кисть далеко) ко рту (alpha=1)."""
    k = np.zeros((17, 3), np.float32)
    k[:, 2] = 0.9
    k[0, :2] = [125, 120]  # нос
    k[1, :2], k[2, :2] = [120, 115], [130, 115]
    k[3, :2], k[4, :2] = [112, 118], [138, 118]
    k[5, :2], k[6, :2] = [105, 150], [145, 150]  # плечи: scale = 40
    k[7, :2], k[8, :2] = [100, 190], [150, 190]  # локти
    k[9, :2] = [100, 230]  # левое запястье внизу
    far, near = np.array([150.0, 250.0]), np.array([128.0, 135.0])
    k[10, :2] = far + alpha * (near - far)
    return k


def puff_alpha(t: float, starts, ramp: float = 0.6, hold: float = 1.5) -> float:
    for s in starts:
        if s <= t < s + ramp:
            return (t - s) / ramp
        if s + ramp <= t < s + ramp + hold:
            return 1.0
        if s + ramp + hold <= t < s + 2 * ramp + hold:
            return 1.0 - (t - s - ramp - hold) / ramp
    return 0.0


def stream_job(starts, total=20.0, fps=5.0, cfg=None):
    """Задание плотной проверки на синтетических отсчётах: шаги фиксации раз в секунду, как в потоке."""
    cfg = cfg or Config()
    job = PersonJob(1, BOX, total)
    finalized = []  # (момент шага, цикл)
    for i in range(int(total * fps)):
        t = i / fps
        job.add(PersonObs(i * 6, t, BOX.copy(), 0.9, pose(puff_alpha(t, starts))))
        if i % int(fps) == 0:
            new, _ = job.finalize(t, False, cfg, Roi.full())
            finalized += [(t, c) for c in new]
    new, _ = job.finalize(total, True, cfg, Roi.full())
    finalized += [(total, c) for c in new]
    return job, finalized


def test_ticker_by_time():
    sparse, dense = Ticker(1.0), Ticker(0.2)
    ts = [i / 30 for i in range(90)]
    assert [i for i, t in enumerate(ts) if sparse(t)] == [0, 30, 60]
    assert [i for i, t in enumerate(ts) if dense(t)] == list(range(0, 90, 6))
    # пропуск кадров (поток): после разрыва — одно срабатывание, без серии догоняющих
    tk = Ticker(0.2)
    assert [tk(t) for t in (0.0, 0.1, 1.05, 1.1, 1.2)] == [True, False, True, False, True]


def test_frame_ring_keeps_last_seconds():
    ring = FrameRing(3.0)
    frame = np.zeros((2, 2, 3), np.uint8)
    for i in range(50):
        ring.push(StreamFrame(i, i * 0.2, frame, i % 5 == 0))
    ts = [f.t for f in ring.items]
    assert ts[-1] == pytest.approx(9.8) and ts[0] == pytest.approx(6.8)
    got = [f.t for f in ring.between(7.6, 8.4)]
    assert got == pytest.approx([7.8, 8.0, 8.2])  # границы не входят


def test_sparse_tracker_keeps_ids():
    tr = SparseTracker(iou_min=0.1, dist_max=0.8, max_missed=3.0)
    a, b = np.array([100, 100, 150, 300.0]), np.array([400, 100, 450, 300.0])
    assert tr.update(np.stack([a, b]), 0.0) == [1, 2]
    # порядок рамок другой, небольшое смещение — те же люди
    assert tr.update(np.stack([b + 5, a - 4]), 1.0) == [2, 1]
    # человек 1 пропал на секунду, затем снова в том же месте; новый человек — новый id
    assert tr.update(np.stack([b]), 2.0) == [2]
    c = np.array([700, 100, 750, 300.0])
    assert tr.update(np.stack([a, b, c]), 3.0) == [1, 2, 3]
    # дольше max_missed без рамки — трек закрыт, рамка на том же месте открывает новый
    for t in (4.0, 5.5, 7.0):
        tr.update(np.stack([b]), t)
    assert tr.update(np.stack([a, b]), 8.0) == [4, 2]
    # рамка вдвое выше — другой человек (перекрытие ближнего и дальнего)
    assert tr.update(np.stack([[90, 0, 160, 450.0], b]), 9.0)[0] == 5


def test_sample_features_trigger_distance():
    d_near, mouth, hand = sample_features(pose(1.0), BOX, Config().geometry)
    d_far, _, _ = sample_features(pose(0.0), BOX, Config().geometry)
    assert d_near < 0.6 < 1.2 < d_far
    assert mouth is not None and hand is not None


def test_job_finalizes_cycles_once_after_settle():
    cfg = Config()
    job, fin = stream_job([3.0, 10.0])
    assert len(job.cycles) == 2 and all(c.valid for c in job.cycles)
    settle = cfg.cycle.lookaround + 1.0
    for now, c in fin:
        assert now - c.end >= settle - 1e-6  # зафиксирован только после конца отвода и сглаживания
        assert now - c.end <= settle + 1.0  # но не позже следующего шага 1 Гц
    assert job.cycles[0].hold_start == pytest.approx(3.6, abs=0.25)


def test_job_final_step_flushes_unsettled_cycle():
    job, fin = stream_job([16.0], total=19.0)
    assert len(job.cycles) == 1 and fin[0][0] == 19.0  # отвод в 18.7 с, фиксация на закрытии задания


def test_replay_matches_stream():
    job, _ = stream_job([3.0, 10.0, 25.0], total=32.0)
    rep = replay_job(job.record(), Config(), Roi.full())
    assert [(c.start, c.hold_start, c.end) for c in rep.cycles] == [(c.start, c.hold_start, c.end) for c in job.cycles]
    assert len(rep.obs) == len(job.obs)


def test_episode_rule_two_cycles_or_override():
    cfg = Config()
    job, _ = stream_job([3.0, 10.0])
    events = build_events(person_cycles([job]), person_signals([job], cfg, Roi.full()), cfg.episode)
    assert len(events) == 1 and events[0].passed and len(events[0].cycles) == 2
    assert events[0].start < 3.0 + 0.6 and events[0].end > 10.0 + 2.1
    # один цикл без прямого признака — не эпизод по правилу кейса; min_cycles=1 в режиме — эпизод
    single, _ = stream_job([3.0])
    assert not [e for e in build_events(person_cycles([single]), person_signals([single], cfg, Roi.full()), cfg.episode) if e.passed]
    cfg1 = with_overrides(cfg, ["episode.min_cycles=1"])
    assert len(build_events(person_cycles([single]), person_signals([single], cfg1, Roi.full()), cfg1.episode)) == 1


def test_people_cycles_not_merged():
    cfg = Config()
    a, _ = stream_job([3.0])
    b, _ = stream_job([10.0])
    b.tid = 2
    for c in b.cycles:
        c.track_id = 2
    assert not [e for e in build_events(person_cycles([a, b]), person_signals([a, b], cfg, Roi.full()), cfg.episode) if e.passed]


def test_rt_config_rule_overrides(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text('realtime:\n  rule_overrides: ["episode.min_cycles=1", "cycle.near_enter=0.7"]\n', encoding="utf-8")
    cfg = load_rt_config(str(path))
    assert cfg.episode.min_cycles == 1 and cfg.cycle.near_enter == pytest.approx(0.7)
    cfg = load_rt_config(str(path), ["cycle.near_enter=0.8"])  # --set перекрывает переопределения режима
    assert cfg.cycle.near_enter == pytest.approx(0.8) and cfg.episode.min_cycles == 1


def test_letterbox_maps_back():
    img = np.random.default_rng(0).integers(0, 255, (200, 80, 3), dtype=np.uint8)
    sq, r, px, py = letterbox(img, 320)
    assert sq.shape == (320, 320, 3) and r == pytest.approx(1.6)
    x, y = 40, 150  # точка кропа -> квадрат -> обратно
    assert ((x * r + px - px) / r, (y * r + py - py) / r) == pytest.approx((x, y))
    assert px == (320 - 128) // 2 and py == 0


def test_stored_crop_equals_frame_crop():
    """Стадия 4: кроп из сохранённого квадрата совпадает с кропом из кадра (как у AppearanceScorer)."""
    frame = np.random.default_rng(1).integers(0, 255, (400, 600, 3), dtype=np.uint8)
    o = PersonObs(0, 0.0, np.array([300.0, 50.0, 360.0, 250.0]), 0.9, pose(1.0))
    mouth, hand = np.array([330.0, 80.0]), np.array([320.0, 90.0])
    sc = store_crop(frame, o, mouth, hand)
    cx, cy, side = 326.0, 84.0, 0.7 * 200
    direct = square_crop(frame, cx, cy, side)
    via = square_crop(sc.img, (cx - sc.x0) * sc.k, (cy - sc.y0) * sc.k, side * sc.k)
    assert np.array_equal(direct, via)
