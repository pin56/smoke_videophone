"""Подбор порогов по сетке: устойчивый выбор точки и перекрёстная проверка по клипам и камерам.

Чистые функции без моделей. На вход — счётчики протокола (TP/FP/FN, часы отрицательного видео) для каждой
точки сетки и каждого клипа, на выход — ранжирование точек и оценка переобучения.

Метрики протокола аддитивны по клипам: сопоставление идёт только внутри пары (camera_id, clip_id), часы
отрицательного видео тоже суммируются по клипам. Поэтому оценка любого подмножества клипов — сумма их
счётчиков, и перекрёстная проверка не требует повторного анализа.

Выбор точки. На эталонном наборе из нескольких клипов лучшая точка сетки часто оказывается одиночным пиком.
Поэтому точка оценивается сглаженным F1: средним F1 самой точки и её соседей по каждой оси (значения оси
соседствуют в том порядке, в котором заданы в --grid). Равенство разрешается по порядку: F1 самой точки,
меньше FP/час (точка без FP/час идёт после любой с ним), меньше FP, меньше изменённых относительно
базового конфига параметров, ближе к базовым значениям по сетке, раньше по порядку сетки.
"""
from __future__ import annotations

import itertools
import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from .config import Config
from .evaluation import EvalParams, Item, evaluate

ClipKey = Tuple[str, str]  # (camera_id, clip_id)
Point = Tuple[int, ...]  # индексы значений по осям сетки

# от этих секций зависят треки, а трекинг в tune.py считается один раз базовым конфигом
TRACKING_SECTIONS = ("detector", "video")

CV_MODES = ("auto", "clip", "camera", "none")


@dataclass(frozen=True)
class Axis:
    key: str  # секция.параметр
    values: Tuple[str, ...]  # значения в порядке --grid, как строки YAML


def parse_grid(items: Sequence[str]) -> List[Axis]:
    """``--grid секция.параметр=V1,V2,...`` → оси сетки."""
    axes: List[Axis] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Ожидается секция.параметр=V1,V2,...: {item}")
        key, raw = item.split("=", 1)
        key = key.strip()
        values = tuple(v.strip() for v in raw.split(","))
        if len(key.split(".")) != 2:
            raise ValueError(f"Ключ должен быть вида секция.параметр: {key}")
        if key.split(".")[0] in TRACKING_SECTIONS:
            raise ValueError(f"{key}: секции {'/'.join(TRACKING_SECTIONS)} меняют треки, перебирать их нельзя")
        if any(a.key == key for a in axes):
            raise ValueError(f"Параметр {key} указан в --grid дважды")
        if any(v == "" for v in values):
            raise ValueError(f"Пустое значение в --grid: {item}")
        if len(set(values)) != len(values):
            raise ValueError(f"Повторяющиеся значения в --grid: {item}")
        axes.append(Axis(key, values))
    return axes


def grid_points(axes: Sequence[Axis]) -> List[Point]:
    """Все точки сетки в лексикографическом порядке; без осей — одна пустая точка (базовый конфиг)."""
    return list(itertools.product(*(range(len(a.values)) for a in axes)))


def point_overrides(axes: Sequence[Axis], point: Point) -> List[str]:
    return [f"{a.key}={a.values[i]}" for a, i in zip(axes, point)]


def point_label(axes: Sequence[Axis], point: Point) -> str:
    return " ".join(point_overrides(axes, point)) or "(база)"


def _same(a, b) -> bool:
    numbers = (int, float)
    if isinstance(a, numbers) and isinstance(b, numbers) and not isinstance(a, bool) and not isinstance(b, bool):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def base_point(axes: Sequence[Axis], cfg: Config) -> Tuple[Optional[int], ...]:
    """Индекс значения базового конфига на каждой оси; None, если этого значения в сетке нет."""
    data = cfg.to_dict()
    out: List[Optional[int]] = []
    for a in axes:
        section, name = a.key.split(".")
        current = data[section][name]
        out.append(next((i for i, v in enumerate(a.values) if _same(yaml.safe_load(v), current)), None))
    return tuple(out)


@dataclass
class Counts:
    """Счётчики протокола для клипа или набора клипов."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    ignored: int = 0  # предсказания, попавшие на IGNORE (не FP)
    n_pred: int = 0
    negative_hours: Optional[float] = None  # None — длительности клипов неизвестны

    @classmethod
    def from_report(cls, rep: dict) -> "Counts":
        return cls(tp=rep["tp"], fp=rep["fp"], fn=rep["fn"], ignored=rep["ignored_preds"], n_pred=rep["n_pred"],
                   negative_hours=rep["negative_hours"])  # fmt: skip

    def __add__(self, other: "Counts") -> "Counts":
        if self.negative_hours is None and other.negative_hours is None:
            hours = None
        else:
            hours = (self.negative_hours or 0.0) + (other.negative_hours or 0.0)
        return Counts(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn, self.ignored + other.ignored,
                      self.n_pred + other.n_pred, hours)  # fmt: skip

    # формулы — как в evaluation.evaluate
    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def fp_per_hour(self) -> Optional[float]:
        return self.fp / self.negative_hours if self.negative_hours else None

    def to_dict(self) -> dict:
        fph = self.fp_per_hour
        return {"f1": round(self.f1, 4), "precision": round(self.precision, 4), "recall": round(self.recall, 4),
                "tp": self.tp, "fp": self.fp, "fn": self.fn, "ignored": self.ignored, "n_pred": self.n_pred,
                "fp_per_hour": None if fph is None else round(fph, 3),
                "negative_hours": None if self.negative_hours is None else round(self.negative_hours, 4)}  # fmt: skip


def pool(counts: Iterable[Counts]) -> Counts:
    return sum(counts, Counts())


def clip_counts(
    preds: List[Item], gts: List[Item], key: ClipKey, duration: Optional[float], params: EvalParams = EvalParams()
) -> Counts:
    """Счётчики одного клипа по протоколу (evaluation.evaluate) — только его предсказания, эталон и длительность."""
    own_preds = [p for p in preds if p.key == key]
    own_gts = [g for g in gts if g.key == key]
    rep = evaluate(own_preds, own_gts, {key: duration} if duration is not None else None, params)
    return Counts.from_report(rep)


# ---------- устойчивый выбор точки ----------


def neighbours(point: Point, sizes: Sequence[int]) -> List[Point]:
    """Соседи точки: ±1 шаг по каждой оси в пределах сетки."""
    out = []
    for axis, size in enumerate(sizes):
        for step in (-1, 1):
            j = point[axis] + step
            if 0 <= j < size:
                out.append(point[:axis] + (j,) + point[axis + 1 :])
    return out


def smooth(values: Dict[Point, float], sizes: Sequence[int]) -> Dict[Point, Tuple[float, int]]:
    """Среднее значения точки и её соседей → (среднее, число усреднённых точек)."""
    out = {}
    for p, v in values.items():
        group = [v] + [values[q] for q in neighbours(p, sizes) if q in values]
        out[p] = (sum(group) / len(group), len(group))
    return out


@dataclass
class Ranked:
    point: Point
    counts: Counts
    f1_smooth: float
    n_smooth: int  # сколько точек усреднено
    changed: int  # сколько параметров отличается от базового конфига
    distance: int  # сумма расстояний в шагах сетки до базовых значений (по осям, где они есть)

    def key(self) -> tuple:
        fph = self.counts.fp_per_hour
        return (
            -round(self.f1_smooth, 9),
            -round(self.counts.f1, 9),
            fph is None,  # точка без FP/час — после любой с известным FP/час
            round(fph, 9) if fph is not None else 0.0,
            self.counts.fp,
            self.changed,
            self.distance,
            self.point,
        )


# причины выбора по компоненту ключа Ranked.key, в котором первая точка выиграла у второй
REASONS = (
    "у неё больше сглаженный F1",
    "сглаженный F1 равен, но выше F1 самой точки",
    "F1 равны, но у следующей точки FP/час не определён",
    "F1 равны, но меньше FP/час",
    "F1 и FP/час равны, но меньше FP",
    "метрики равны, но меньше параметров изменено относительно базового конфига",
    "метрики равны, но значения ближе к базовым по сетке",
    "всё равно, первая по порядку сетки",
)


def closeness(point: Point, base: Sequence[Optional[int]]) -> Tuple[int, int]:
    changed = sum(b is None or i != b for i, b in zip(point, base))
    distance = sum(abs(i - b) for i, b in zip(point, base) if b is not None)
    return changed, distance


def rank_points(axes: Sequence[Axis], counts: Dict[Point, Counts], base: Sequence[Optional[int]]) -> List[Ranked]:
    """Точки сетки от лучшей к худшей по устойчивому правилу (см. описание модуля)."""
    sizes = [len(a.values) for a in axes]
    sm = smooth({p: c.f1 for p, c in counts.items()}, sizes)
    ranked = [Ranked(p, c, sm[p][0], sm[p][1], *closeness(p, base)) for p, c in counts.items()]
    ranked.sort(key=Ranked.key)
    return ranked


def decisive(a: Ranked, b: Ranked) -> int:
    """Номер компонента ключа, по которому a выигрывает у b."""
    return next(i for i, (x, y) in enumerate(zip(a.key(), b.key())) if x != y)


def _fmt_fph(v: Optional[float]) -> str:
    return "n/a" if v is None else f"{v:.2f}"


def explain_choice(axes: Sequence[Axis], ranked: List[Ranked]) -> str:
    """Почему выбрана первая точка: её окрестность, решающий критерий и отвергнутый пик сырого F1."""
    chosen = ranked[0]
    c = chosen.counts
    by_point = {r.point: r for r in ranked}
    lines = [f"Выбрана точка: {point_label(axes, chosen.point)}"]
    near = []
    for q in neighbours(chosen.point, [len(a.values) for a in axes]):
        if q in by_point:
            axis = next(i for i, (x, y) in enumerate(zip(q, chosen.point)) if x != y)
            near.append(f"{axes[axis].key}={axes[axis].values[q[axis]]}: {by_point[q].counts.f1:.3f}")
    lines.append(f"  сглаженный F1={chosen.f1_smooth:.3f} — среднее по {chosen.n_smooth} точкам: сама {c.f1:.3f}"
                 + (f"; соседи {', '.join(near)}" if near else ""))  # fmt: skip
    lines.append(f"  P={c.precision:.3f} R={c.recall:.3f} TP={c.tp} FP={c.fp} FN={c.fn} FP/час={_fmt_fph(c.fp_per_hour)}")
    if len(ranked) == 1:
        lines.append("  в сетке одна точка")
        return "\n".join(lines)
    second = ranked[1]
    lines.append(f"  против следующей ({point_label(axes, second.point)}, сглаженный F1={second.f1_smooth:.3f}, "
                 f"F1={second.counts.f1:.3f}): {REASONS[decisive(chosen, second)]}")  # fmt: skip
    peak = min(ranked, key=lambda r: (-round(r.counts.f1, 9), r.key()))
    if peak is not chosen and round(peak.counts.f1, 9) > round(c.f1, 9):
        lines.append(f"  наибольший F1={peak.counts.f1:.3f} у точки {point_label(axes, peak.point)} не выбран: "
                     f"сглаженный F1 там {peak.f1_smooth:.3f}, соседи хуже — похоже на одиночный пик")  # fmt: skip
    return "\n".join(lines)


def result_rows(axes: Sequence[Axis], ranked: List[Ranked]) -> List[dict]:
    """Строки tune_results.csv в порядке ранжирования."""
    rows = []
    for n, r in enumerate(ranked, 1):
        c = r.counts
        row = {"rank": n, "params": point_label(axes, r.point)}
        row.update({a.key: a.values[i] for a, i in zip(axes, r.point)})
        row.update({"f1_smooth": round(r.f1_smooth, 4), "n_smooth": r.n_smooth, "f1": round(c.f1, 4),
                    "precision": round(c.precision, 4), "recall": round(c.recall, 4), "tp": c.tp, "fp": c.fp,
                    "fn": c.fn, "ignored": c.ignored, "n_pred": c.n_pred,
                    "fp_per_hour": "" if c.fp_per_hour is None else round(c.fp_per_hour, 3),
                    "changed_from_base": r.changed, "base_distance": r.distance})  # fmt: skip
        rows.append(row)
    return rows


# ---------- перекрёстная проверка ----------


@dataclass(frozen=True)
class Fold:
    mode: str  # clip | camera
    name: str  # что отложено
    test: Tuple[ClipKey, ...]


def cv_modes(keys: Sequence[ClipKey], requested: str) -> List[str]:
    """auto: по клипам, если их >= 2; плюс по камерам, если камер >= 2 и это не то же самое, что по клипам."""
    if requested not in CV_MODES:
        raise ValueError(f"Неизвестный режим перекрёстной проверки: {requested}")
    if requested == "none":
        return []
    if requested != "auto":
        return [requested]
    n_clips, n_cams = len(set(keys)), len({k[0] for k in keys})
    return (["clip"] if n_clips >= 2 else []) + (["camera"] if 2 <= n_cams < n_clips else [])


def make_folds(keys: Sequence[ClipKey], mode: str) -> List[Fold]:
    """Фолды «оставить один клип» или «оставить одну камеру»; меньше двух групп — фолдов нет."""
    groups: Dict[str, List[ClipKey]] = {}
    for k in keys:
        name = f"{k[0]}/{k[1]}" if mode == "clip" else k[0]
        groups.setdefault(name, []).append(k)
    if len(groups) < 2:
        return []
    return [Fold(mode, name, tuple(ks)) for name, ks in groups.items()]


@dataclass
class FoldResult:
    fold: Fold
    chosen: Ranked  # точка, выбранная на остальных клипах (counts — по ним)
    test: Counts  # результат этой точки на отложенных клипах


def cross_validate(
    axes: Sequence[Axis],
    per_clip: Dict[Point, Dict[ClipKey, Counts]],
    folds: Sequence[Fold],
    base: Sequence[Optional[int]],
) -> List[FoldResult]:
    """Для каждого фолда: выбор точки на остальных клипах тем же правилом, оценка на отложенных."""
    keys = list(next(iter(per_clip.values())))
    results = []
    for fold in folds:
        test = set(fold.test)
        train = [k for k in keys if k not in test]
        if not train:
            continue
        counts = {p: pool(by_clip[k] for k in train) for p, by_clip in per_clip.items()}
        chosen = rank_points(axes, counts, base)[0]
        results.append(FoldResult(fold, chosen, pool(per_clip[chosen.point][k] for k in fold.test)))
    return results


@dataclass
class CvSummary:
    mode: str
    folds: List[FoldResult]
    pooled: Counts  # сумма счётчиков отложенных клипов по всем фолдам
    value_counts: Dict[str, Counter]  # параметр → сколько раз выбрано каждое значение
    same_as_full: int  # в скольких фолдах выбрана та же точка, что на всех клипах


def summarize_cv(axes: Sequence[Axis], mode: str, results: List[FoldResult], full_choice: Point) -> CvSummary:
    value_counts = {a.key: Counter(a.values[r.chosen.point[i]] for r in results) for i, a in enumerate(axes)}
    same = sum(r.chosen.point == full_choice for r in results)
    return CvSummary(mode, results, pool(r.test for r in results), value_counts, same)


CV_TITLES = {"clip": "оставить один клип", "camera": "оставить одну камеру"}


def format_cv(axes: Sequence[Axis], s: CvSummary, chosen: Counts, base: Optional[Counts]) -> str:
    """Сводка перекрёстной проверки и сравнение с выбранной точкой и базовым конфигом на всех клипах."""
    c = s.pooled
    lines = [
        f"Перекрёстная проверка ({CV_TITLES[s.mode]}, фолдов {len(s.folds)}):",
        f"  на отложенных: TP={c.tp} FP={c.fp} FN={c.fn}  P={c.precision:.3f} R={c.recall:.3f} F1={c.f1:.3f} "
        f"FP/час={_fmt_fph(c.fp_per_hour)}",
        f"  выбранная точка на всех клипах: F1={chosen.f1:.3f} (оптимизм подбора {chosen.f1 - c.f1:+.3f})",
    ]
    if base is not None:
        verdict = ""
        if round(c.f1, 9) < round(base.f1, 9):
            verdict = " — подбор на отложенных клипах хуже базового конфига, значения сетки переобучены"
        lines.append(f"  базовый конфиг без подбора: F1={base.f1:.3f}{verdict}")
    lines.append(f"  та же точка, что на всех клипах: в {s.same_as_full} из {len(s.folds)} фолдов")
    for a in axes:
        counts = s.value_counts[a.key]
        lines.append(f"  {a.key}: " + ", ".join(f"{v} ({counts[v]} из {len(s.folds)})" for v in a.values if counts[v]))
    return "\n".join(lines)


def cv_rows(axes: Sequence[Axis], s: CvSummary) -> List[dict]:
    """Строки tune_cv.csv: по одной на фолд."""
    rows = []
    for r in s.folds:
        t = r.test
        rows.append({"mode": s.mode, "held_out": r.fold.name, "params": point_label(axes, r.chosen.point),
                     "train_f1_smooth": round(r.chosen.f1_smooth, 4), "train_f1": round(r.chosen.counts.f1, 4),
                     "test_tp": t.tp, "test_fp": t.fp, "test_fn": t.fn, "test_ignored": t.ignored,
                     "test_fp_per_hour": "" if t.fp_per_hour is None else round(t.fp_per_hour, 3)})  # fmt: skip
    return rows
