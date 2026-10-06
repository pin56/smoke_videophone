"""Тесты подбора порогов без моделей: синтетические счётчики по точкам сетки и клипам."""
import pytest

from smokedet.config import Config
from smokedet.evaluation import Item, evaluate
from smokedet.tuning import (
    Counts, base_point, clip_counts, cross_validate, cv_modes, explain_choice, format_cv, grid_points, make_folds,
    neighbours, parse_grid, point_overrides, pool, rank_points, result_rows, smooth, summarize_cv,
)  # fmt: skip


def cnt(tp=0, fp=0, fn=0, hours=None):
    return Counts(tp=tp, fp=fp, fn=fn, n_pred=tp + fp, negative_hours=hours)


def test_parse_grid_and_points():
    axes = parse_grid(["episode.confidence_threshold=0.4,0.5,0.6", "cycle.near_enter=0.5, 0.6"])
    assert [a.key for a in axes] == ["episode.confidence_threshold", "cycle.near_enter"]
    assert axes[1].values == ("0.5", "0.6")
    points = grid_points(axes)
    assert len(points) == 6 and points[0] == (0, 0) and points[-1] == (2, 1)
    assert point_overrides(axes, (1, 0)) == ["episode.confidence_threshold=0.5", "cycle.near_enter=0.5"]
    assert grid_points([]) == [()]
    for bad in (["episode.window=1", "episode.window=2"], ["detector.conf=0.1,0.2"], ["episode.window=1,,2"],
                ["episode.window=1,1"], ["window=1"]):  # fmt: skip
        with pytest.raises(ValueError):
            parse_grid(bad)


def test_base_point():
    axes = parse_grid(["episode.confidence_threshold=0.4,0.5,0.6", "cycle.near_enter=0.5,0.6", "cycle.min_hold=0.3"])
    assert base_point(axes, Config()) == (1, 1, None)


def test_none_fp_per_hour_ranks_last():
    # F1 одинаковы (TP=0), у второй точки FP/час известен, хоть FP и больше; раньше None превращался в 0 и шёл первым
    axes = parse_grid(["episode.confidence_threshold=0.4,0.5"])
    counts = {(0,): cnt(fn=1), (1,): cnt(fp=3, fn=1, hours=1.0)}
    ranked = rank_points(axes, counts, (None,))
    assert [r.point for r in ranked] == [(1,), (0,)]
    assert "FP/час не определён" in explain_choice(axes, ranked)
    rows = result_rows(axes, ranked)
    assert rows[1]["fp_per_hour"] == "" and rows[0]["fp_per_hour"] == 3.0


def test_lower_fp_per_hour_wins():
    axes = parse_grid(["episode.confidence_threshold=0.4,0.5"])
    counts = {(0,): cnt(fp=2, fn=1, hours=1.0), (1,): cnt(fp=1, fn=1, hours=1.0)}
    ranked = rank_points(axes, counts, (0,))  # близость к базе проигрывает FP/час
    assert ranked[0].point == (1,)


def f1_counts(f1):
    """Счётчики с заданным F1 = 2TP / (2TP + FP + FN)."""
    return {0.2: cnt(tp=1, fp=4, fn=4), 0.7: cnt(tp=7, fp=3, fn=3), 0.75: cnt(tp=3, fp=1, fn=1), 1.0: cnt(tp=1)}[f1]


def test_smoothing_prefers_plateau_over_spike():
    f1s = [0.2, 0.2, 1.0, 0.2, 0.7, 0.75, 0.7, 0.2]
    axes = parse_grid(["episode.confidence_threshold=" + ",".join(f"0.{i}" for i in range(1, 9))])
    counts = {(i,): f1_counts(v) for i, v in enumerate(f1s)}
    assert [round(counts[(i,)].f1, 3) for i in range(8)] == f1s
    ranked = rank_points(axes, counts, (None,))
    chosen = ranked[0]
    assert chosen.point == (5,) and chosen.n_smooth == 3
    assert chosen.f1_smooth == pytest.approx((0.7 + 0.75 + 0.7) / 3)
    spike = next(r for r in ranked if r.point == (2,))
    assert spike.f1_smooth == pytest.approx(1.4 / 3)
    text = explain_choice(axes, ranked)
    assert "episode.confidence_threshold=0.6" in text.splitlines()[0]
    assert "одиночный пик" in text and "episode.confidence_threshold=0.3" in text


def test_neighbours_and_smooth_2d():
    sizes = [3, 2]
    assert sorted(neighbours((1, 0), sizes)) == [(0, 0), (1, 1), (2, 0)]
    assert sorted(neighbours((0, 1), sizes)) == [(0, 0), (1, 1)]
    values = {p: float(p[0] + p[1]) for p in grid_points(parse_grid(["a.x=1,2,3", "a.y=1,2"]))}
    sm = smooth(values, sizes)
    assert sm[(1, 0)] == (pytest.approx((1 + 0 + 2 + 2) / 4), 4)
    assert sm[(0, 1)] == (pytest.approx((1 + 0 + 2) / 3), 3)


def test_tie_breaks():
    axes = parse_grid(["episode.confidence_threshold=0.4,0.5"])
    # у двух значений одной оси сглаженный F1 одинаков → решает F1 самой точки
    ranked = rank_points(axes, {(0,): cnt(tp=1, fn=1), (1,): cnt(tp=1)}, (0,))
    assert ranked[0].point == (1,) and ranked[0].f1_smooth == ranked[1].f1_smooth
    assert "выше F1 самой точки" in explain_choice(axes, ranked)

    # метрики равны → меньше изменённых параметров, затем расстояние до базы, затем порядок сетки
    axes = parse_grid(["a.x=1,2,3,4", "a.y=1,2"])
    same = {p: cnt(tp=1, fn=1, hours=1.0) for p in grid_points(axes)}
    ranked = rank_points(axes, same, (2, 1))
    assert [r.point for r in ranked[:5]] == [(2, 1), (1, 1), (2, 0), (3, 1), (0, 1)]
    assert "меньше параметров изменено" in explain_choice(axes, ranked)
    ranked = rank_points(axes, same, (2, None))
    assert [r.point for r in ranked[:3]] == [(2, 0), (2, 1), (1, 0)]
    ranked = rank_points(axes, same, (None, None))
    assert [r.point for r in ranked] == grid_points(axes)  # полное равенство — порядок сетки, детерминированно
    assert "первая по порядку сетки" in explain_choice(axes, ranked)


def test_clip_counts_are_additive():
    def item(cam, clip, s, e, label="POSITIVE", box=(0, 0, 10, 10)):
        return Item(cam, clip, s, e, box, label)

    gts = [item("c1", "a", 10, 20), item("c1", "a", 50, 55, "IGNORE"), item("c2", "b", 5, 9)]
    preds = [item("c1", "a", 12, 18), item("c1", "a", 50, 54), item("c1", "a", 80, 90), item("c2", "b", 30, 40)]
    durations = {("c1", "a"): 120.0, ("c2", "b"): 60.0}
    total = pool(clip_counts(preds, gts, k, d) for k, d in durations.items())
    rep = evaluate(preds, gts, durations)
    assert (total.tp, total.fp, total.fn, total.ignored, total.n_pred) == (1, 2, 1, 1, 4)
    assert (rep["tp"], rep["fp"], rep["fn"], rep["ignored_preds"]) == (total.tp, total.fp, total.fn, total.ignored)
    assert total.f1 == pytest.approx(rep["f1"]) and total.fp_per_hour == pytest.approx(rep["fp_per_hour"])
    assert pool([cnt(tp=1), cnt(fp=1)]).negative_hours is None


def cv_setup():
    """Клипы A, B (камера c1) и C (c2), в каждом один эталонный эпизод. Т — найден, F — пропущен.
    точка:  0.3  0.4  0.5
      A      F    T    T
      B      F    T    T
      C      T    T    F
    На всех клипах F1 = 0.5, 1.0, 0.8; сглаженный 0.75, 0.767, 0.9 → выбор 0.5 (база — 0.4)."""
    axes = parse_grid(["episode.confidence_threshold=0.3,0.4,0.5"])
    A, B, C = ("c1", "A"), ("c1", "B"), ("c2", "C")
    hit, miss = cnt(tp=1), cnt(fn=1)
    per_clip = {(0,): {A: miss, B: miss, C: hit}, (1,): {A: hit, B: hit, C: hit}, (2,): {A: hit, B: hit, C: miss}}
    return axes, per_clip, [A, B, C]


def test_cross_validation_folds():
    axes, per_clip, keys = cv_setup()
    full = rank_points(axes, {p: pool(c.values()) for p, c in per_clip.items()}, (1,))[0]
    assert full.point == (2,)

    # без A: F1 = 0.667, 1.0, 0.667, сглаженный 0.833, 0.778, 0.833 → ничья 0.3/0.5 по всем метрикам и по
    # расстоянию до базы → первая по порядку 0.3; на A она пропускает. Без B — так же. Без C: 0.0, 1.0, 1.0 → 0.5.
    s = summarize_cv(axes, "clip", cross_validate(axes, per_clip, make_folds(keys, "clip"), (1,)), full.point)
    assert [r.fold.name for r in s.folds] == ["c1/A", "c1/B", "c2/C"]
    assert [r.chosen.point for r in s.folds] == [(0,), (0,), (2,)]
    assert [(r.test.tp, r.test.fn) for r in s.folds] == [(0, 1), (0, 1), (0, 1)]
    assert (s.pooled.tp, s.pooled.fp, s.pooled.fn, s.pooled.f1) == (0, 0, 3, 0.0)
    assert dict(s.value_counts["episode.confidence_threshold"]) == {"0.3": 2, "0.5": 1}
    assert s.same_as_full == 1
    text = format_cv(axes, s, full.counts, cnt(tp=3))
    assert "F1=0.000" in text and "переобучены" in text and "0.3 (2 из 3)" in text
    # вывод tune.py на Windows перенаправляется в cp1251: печатаемые тексты должны в неё кодироваться
    text.encode("cp1251")
    explain_choice(axes, rank_points(axes, {p: pool(c.values()) for p, c in per_clip.items()}, (1,))).encode("cp1251")

    # по камерам: без c1 обучение только на C (F1 1, 1, 0 → 0.3), без c2 — как без C
    s = summarize_cv(axes, "camera", cross_validate(axes, per_clip, make_folds(keys, "camera"), (1,)), full.point)
    assert [r.fold.name for r in s.folds] == ["c1", "c2"]
    assert [r.fold.test for r in s.folds] == [(keys[0], keys[1]), (keys[2],)]
    assert [r.chosen.point for r in s.folds] == [(0,), (2,)]
    assert (s.pooled.tp, s.pooled.fn) == (0, 3)


def test_cv_modes_and_folds():
    keys = [("c1", "A"), ("c1", "B"), ("c2", "C")]
    assert cv_modes(keys, "auto") == ["clip", "camera"]
    assert cv_modes([("c1", "A"), ("c2", "B")], "auto") == ["clip"]  # камера = клип, повтор не нужен
    assert cv_modes([("c1", "A")], "auto") == []
    assert cv_modes(keys, "none") == [] and cv_modes(keys, "camera") == ["camera"]
    assert make_folds([("c1", "A"), ("c1", "B")], "camera") == []
    assert len(make_folds(keys, "clip")) == 3
