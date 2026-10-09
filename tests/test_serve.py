"""Сервер на много камер (smokedet/serve) без моделей: регулятор, ступени, исполнитель батчей, источник."""
import numpy as np
import pytest

from smokedet.config import Config, ServeConfig, with_overrides
from smokedet.serve.control import PRIO_ALERT, PRIO_EMPTY, PRIO_JOBS, Controller, Ladder, parse_step


def run_ctl(ctl, lags, util=0.5, t0=0.0):
    """Шаги регулятора раз в секунду; lags — отставание на каждом шаге."""
    out = []
    for i, lag in enumerate(lags):
        out.append(ctl.update(t0 + i, lag, util))
    return out


def test_pressure_rises_with_lag_and_waits_between_raises():
    sc = ServeConfig()
    ctl = Controller(sc, steps=4)
    g = run_ctl(ctl, [3.0] * 7)  # отставание 3 с (выше lag_high 2, ниже 2·lag_high)
    assert g == [1, 1, 1, 2, 2, 2, 3]  # повышение не чаще raise_hold 3 с
    ctl = Controller(sc, steps=4)
    assert run_ctl(ctl, [5.0])[0] == 2  # сильное отставание — сразу +2
    ctl.G = ctl.top
    assert run_ctl(ctl, [9.0] * 5, t0=10)[-1] == ctl.top  # не выше K + 3


def test_pressure_falls_after_calm_and_boost_needs_headroom():
    sc = ServeConfig()
    ctl = Controller(sc, steps=4)
    ctl.G = 2
    g = run_ctl(ctl, [0.2] * 30, util=0.7)
    assert g[8] == 2 and g[9] == 1 and g[19] == 0  # по ступени за calm_hold 10 с
    assert g[-1] == 0  # util 0.7 ≥ util_boost 0.6: без буста
    g = run_ctl(ctl, [0.2] * 12, util=0.3, t0=30)
    assert g[-1] == -1
    assert ctl.update(43, 0.2, 0.9) == 0  # буст съел запас — сразу назад
    ctl = Controller(sc, steps=4)
    ctl.G = 2
    assert run_ctl(ctl, [0.2] * 30, util=0.9)[-1] == 2  # загрузка без запаса — не снижать
    assert run_ctl(ctl, [1.0] * 30, util=0.5, t0=30)[-1] == 2  # отставание между порогами — держать


def test_failed_lowering_doubles_calm_hold():
    sc = ServeConfig()
    ctl = Controller(sc, steps=4)
    ctl.G = 2
    run_ctl(ctl, [0.2] * 10)  # снижение на 10 с
    assert ctl.G == 1
    run_ctl(ctl, [3.0], t0=12)  # и тут же отставание
    assert ctl.G == 2 and ctl.hold == 2 * sc.calm_hold
    g = run_ctl(ctl, [0.2] * 25, t0=13)
    assert g[18] == 2 and g[19] == 1  # теперь 20 с спокойствия


def test_levels_by_priority():
    ctl = Controller(ServeConfig(), steps=4)
    ctl.G = 2
    assert ctl.level(PRIO_EMPTY) == 2 and ctl.level(PRIO_JOBS) == 0 and ctl.level(PRIO_ALERT) == 0
    ctl.G = 7
    assert ctl.level(PRIO_EMPTY) == 4 and ctl.level(PRIO_ALERT) == 4
    ctl.G = -1
    assert ctl.level(PRIO_ALERT) == -1


def test_ladder_applies_cumulative_steps_and_resets():
    base = with_overrides(Config(), ["serve.ladder=['realtime.dense_fps=2.5', 'realtime.max_jobs=4,appearance.max_frames=4']",
                                     "realtime.max_jobs=10"])  # fmt: skip
    lad = Ladder(base)
    cfg = with_overrides(base, [])
    lad.apply(cfg, 2)
    assert (cfg.realtime.dense_fps, cfg.realtime.max_jobs, cfg.appearance.max_frames) == (2.5, 4, 4)
    lad.apply(cfg, 1)
    assert (cfg.realtime.dense_fps, cfg.realtime.max_jobs, cfg.appearance.max_frames) == (2.5, 10, 8)
    lad.apply(cfg, -1)
    assert cfg.realtime.fill_budget and cfg.realtime.dense_fps == base.realtime.dense_fps
    lad.apply(cfg, 0)
    assert not cfg.realtime.fill_budget
    assert isinstance(parse_step("realtime.dense_fps=3")[0][2], int)
    lad.apply(cfg, 0)
    with pytest.raises(KeyError):
        parse_step("realtime.pose_model=x")  # модель посреди потока не меняется


def test_batch_executor_merges_requests_of_same_key():
    import threading

    from smokedet.serve.hub import BatchExecutor

    calls = []
    gate = threading.Event()

    def fn(items, key):
        calls.append((key, len(items)))
        gate.wait(2)  # первый батч держит исполнитель: остальные запросы копятся в очереди
        return [(key, x * 10) for x in items]

    ex = BatchExecutor(fn, "t", max_items=5, wait_ms=0.0)
    res = {}

    def worker(name, items, key):
        res[name] = ex(items, key)

    ts = [threading.Thread(target=worker, args=("a", [1], 0))]
    ts[0].start()
    while not calls:
        pass
    for name, items, key in [("b", [2, 3], 0), ("c", [4], 1), ("d", [5, 6], 0), ("e", [7, 8], 0)]:
        t = threading.Thread(target=worker, args=(name, items, key))
        t.start()
        ts.append(t)
    import time

    time.sleep(0.2)
    gate.set()
    for t in ts:
        t.join(5)
    assert res["a"] == [(0, 10)] and res["b"] == [(0, 20), (0, 30)] and res["c"] == [(1, 40)]
    assert res["e"] == [(0, 70), (0, 80)]
    assert calls[0] == (0, 1) and (0, 4) in calls  # b и d — одним батчем (e не влез в max_items 5)
    st = ex.snapshot()
    assert st.items == 8 and st.requests == 5 and st.batches == len(calls)
    ex.close()


def test_batch_executor_error_reaches_every_request():
    from smokedet.serve.hub import BatchExecutor

    def fn(items, key):
        raise ValueError("boom")

    ex = BatchExecutor(fn, "t", workers=2)
    with pytest.raises(ValueError):
        ex([1, 2])
    assert ex([]) == []
    ex.close()
    with pytest.raises(RuntimeError):
        ex([1])


class FakeCap:
    """cv2.VideoCapture без видео: n кадров с частотой fps, в кадре записано его время (pts)."""

    def __init__(self, n, fps=25.0):
        self.n, self.fps, self.i = n, fps, -1

    def grab(self):
        self.i += 1
        return self.i < self.n

    def retrieve(self):
        return True, np.full((4, 4, 3), self.i / self.fps)

    def get(self, prop):
        import cv2

        return {cv2.CAP_PROP_POS_MSEC: self.i * 1000.0 / self.fps, cv2.CAP_PROP_FPS: self.fps,
                cv2.CAP_PROP_FRAME_COUNT: self.n, cv2.CAP_PROP_FRAME_WIDTH: 640, cv2.CAP_PROP_FRAME_HEIGHT: 480}[prop]  # fmt: skip

    def release(self):
        pass


def drain(src):
    out = []
    while True:
        a = src.get(timeout=1.0)
        if a is None:
            if src.done:
                return out
            continue
        out.append(a)


def test_source_grids_and_live_sparse_rate():
    import time

    from smokedet.serve.source import CameraSource

    cap = FakeCap(250)  # 10 с
    rate = lambda: 1.0 if cap.i < 100 else 0.5  # noqa: E731 — с 4 с ступень камеры: детектор раз в 2 с
    src = CameraSource("clip.mp4", 5.0, rate, queue_sec=1.0, cap=cap).start()
    got = []
    while True:  # читатель медленнее источника: файл без темпа ждёт места в очереди, кадры не теряются
        a = src.get(timeout=1.0)
        if a is None and src.done:
            break
        if a is not None:
            got.append(a)
            time.sleep(0.001)
    ts = [a.f.t for a in got]
    assert np.allclose(np.diff(ts), 0.2) and len(ts) == 50 and src.stats["dropped_queue"] == 0
    sparse = [a.f.t for a in got if a.f.sparse]
    assert np.allclose(sparse, [0, 1, 2, 3, 4, 6, 8])
    assert src.stats["decoded_frames"] == 250


def test_stream_source_drops_old_dense_frames_first():
    from smokedet.serve.source import CameraSource

    src = CameraSource("rtsp://cam", 5.0, lambda: 1.0, queue_sec=2.0, cap=FakeCap(250)).start()  # 11 кадров в очереди
    while not src.done:
        src.cv.acquire()
        src.cv.wait(0.1)
        src.cv.release()
    got = drain(src)
    assert len(got) == 11 and src.stats["dropped_queue"] == 50 - 11
    assert sum(a.f.sparse for a in got) >= 9  # старые кадры детектора остаются, выброшены кадры плотной сетки


def test_source_pace_arrival_follows_pts():
    import time

    from smokedet.serve.source import CameraSource

    t0 = time.monotonic()
    got = drain(CameraSource("clip.mp4", 5.0, lambda: 1.0, pace=True, cap=FakeCap(15)).start())  # 0.6 с
    arr = np.array([a.arrival for a in got]) - got[0].arrival
    assert np.allclose(arr, [a.f.t for a in got])
    assert time.monotonic() - t0 >= 0.39  # отдаётся в темпе реального времени, не быстрее


def test_assign_clips_longest_first():
    from smokedet.pipeline import ClipInput
    from smokedet.serve.server import emulated_specs

    clips = [ClipInput(f"{k}.mp4", "c", k) for k in "abcde"]
    dur = {"a": 100, "b": 60, "c": 50, "d": 40, "e": 10}
    specs = emulated_specs(clips, dur, 3)
    assert [[c.clip_id for c in s.playlist] for s in specs] == [["a"], ["b", "e"], ["c", "d"]]
    assert all(s.filler == s.playlist for s in specs)
    extra = emulated_specs(clips[:2], dur, 3)
    assert not extra[2].playlist and extra[2].filler


class HubModels:
    """Модели для Hub без сетей (FakeModels из test_realtime) + поля RealtimeDetector."""

    device = "cpu"

    def __init__(self, starts):
        from test_realtime import FakeModels

        fm = FakeModels(starts)
        self.detect, self.pose = fm.detect, fm.pose

    def encoder(self, cfg):
        return None


def run_server(tmp_path, monkeypatch, level, starts=(3.0, 10.0), n=2):
    from smokedet import ov_yolo
    from smokedet.pipeline import ClipInput
    from smokedet.serve.server import Server, emulated_specs
    from smokedet.serve.source import CameraSource

    monkeypatch.setattr(ov_yolo, "STREAMS", 0)
    cfg = with_overrides(Config(), ["realtime.objects=false", "serve.workers=2", "serve.tick=0.2"])
    clips = [ClipInput(f"v{i}.mp4", f"cam{i}", f"clip{i}") for i in range(n)]
    specs = emulated_specs(clips, {}, n)

    def source(url, *args, **kw):
        return CameraSource(url, *args, cap=FakeCap(750), **kw)  # 30 с

    srv = Server(cfg, specs, tmp_path, HubModels(list(starts)), pace=False, fixed_level=level, source=source)
    return srv, srv.run()


def test_server_segments_match_single_session(tmp_path, monkeypatch):
    import csv
    import json

    from test_realtime import run_session

    srv, summary = run_server(tmp_path, monkeypatch, level=0)
    assert summary["segments"] == 2 and summary["errors"] == 0
    with open(tmp_path / "predictions.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ref = run_session([3.0, 10.0])  # та же сессия без сервера
    from smokedet.realtime import person_cycles

    n_ref = len(person_cycles(ref.jobs)[1])
    assert {r["clip_id"] for r in rows} == {"clip0", "clip1"}
    dbg = json.loads((tmp_path / "debug" / "clip0.json").read_text(encoding="utf-8"))
    assert len(dbg["cycles"]) == n_ref
    alerts = [json.loads(x) for x in (tmp_path / "alerts.jsonl").read_text(encoding="utf-8").splitlines()]
    assert alerts and all("latency_wall" in a for a in alerts if not a.get("withdrawn"))
    assert (tmp_path / "serve_log.csv").exists() and summary["level_share"].get("0", 0) > 0.99


def test_server_fixed_level_applies_ladder(tmp_path, monkeypatch):
    srv, summary = run_server(tmp_path, monkeypatch, level=1)
    assert all(c.cfg.realtime.dense_fps == 2.5 for c in srv.cams)  # ступень 1 по умолчанию — dense_fps 2.5
    assert srv.base.realtime.dense_fps == 5.0
    assert summary["level_share"] == {"1": 1.0}
