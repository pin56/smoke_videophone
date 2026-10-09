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
