"""Регулятор нагрузки сервера: одно «давление» G на сервер, ступень деградации камеры — G минус её приоритет.

Сигнал — отставание обработки (сколько секунд ждёт самый старый необработанный кадр камеры): от него напрямую
зависит задержка тревоги, и он одинаково измерим на GPU и CPU, без чисел под конкретное железо. Отставание растёт —
давление растёт (камеры, где ничего не происходит, деградируют первыми, камеры с курильщиком — последними);
долго спокойно и загрузка с запасом — давление снижается, ниже нуля — буст (запас тратится на точность).
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import yaml

from ..config import SECTIONS, Config, ServeConfig, _coerce

# поля, которые RealtimeSession и score_from_memory читают на каждом шаге: их можно менять посреди потока
LIVE_KEYS = {
    "realtime.skip_confirmed",
    "realtime.dense_fps",
    "realtime.max_jobs",
    "realtime.sparse_fps",
    "realtime.fill_budget",
    "realtime.trigger_dist",
    "realtime.alert_hold",
    "appearance.max_frames",
}
PRIO_EMPTY, PRIO_PEOPLE, PRIO_JOBS, PRIO_ALERT = 0, 1, 2, 3
HOLD_MAX = 8  # calm_hold растёт не больше чем во столько раз
HOLD_RESET = 300.0  # с без повышений давления — calm_hold возвращается к исходному


def parse_step(step: str) -> List[Tuple[str, str, object]]:
    """«realtime.dense_fps=2.5,realtime.max_jobs=4» -> [(секция, ключ, значение)]."""
    out = []
    for item in step.split(","):
        item = item.strip()
        if not item:
            continue
        key, _, raw = item.partition("=")
        key = key.strip()
        if key not in LIVE_KEYS:
            raise KeyError(f"Ступень регулятора: {key} нельзя менять посреди потока (можно: {sorted(LIVE_KEYS)})")
        section, name = key.split(".")
        out.append((section, name, yaml.safe_load(raw)))
    return out


class Ladder:
    """Ступени деградации (serve.ladder, накопительно) и буст (serve.boost, ступень −1) поверх базового конфига."""

    def __init__(self, base: Config):
        self.base = base
        self.steps = [parse_step(s) for s in base.serve.ladder]
        self.boost = [kv for s in base.serve.boost for kv in parse_step(s)]
        self.keys = sorted({(sec, name) for st in self.steps + [self.boost] for sec, name, _ in st})

    @property
    def size(self) -> int:
        return len(self.steps)

    def apply(self, cfg: Config, level: int) -> None:
        """Конфиг камеры на ступени level: затронутые ступенями поля — к базе, затем ступени 1..level (или буст)."""
        for sec, name in self.keys:
            setattr(getattr(cfg, sec), name, getattr(getattr(self.base, sec), name))
        todo = self.boost if level < 0 else [kv for st in self.steps[:level] for kv in st]
        for sec, name, value in todo:
            default = getattr(SECTIONS[sec](), name)
            setattr(getattr(cfg, sec), name, _coerce(default, value))


class Controller:
    """Давление G ∈ [−1, K + 3] (K — число ступеней); шаг — update раз в serve.tick с."""

    def __init__(self, sc: ServeConfig, steps: int, boost: bool = True):
        self.sc, self.K, self.has_boost = sc, steps, boost
        self.G = 0
        self.calm = 0.0  # с подряд со спокойным отставанием и загрузкой
        self.hold = sc.calm_hold
        self.now: Optional[float] = None
        self._last_raise = -math.inf
        self._last_lower = -math.inf

    @property
    def top(self) -> int:
        return self.K + PRIO_ALERT

    def update(self, now: float, lag: float, util: float) -> int:
        """lag — наибольшее отставание камер, с; util — наибольшая загрузка исполнителей моделей (0..1)."""
        sc = self.sc
        dt = sc.tick if self.now is None else max(now - self.now, 0.0)
        self.now = now
        if now - self._last_raise > HOLD_RESET:
            self.hold = sc.calm_hold
        if lag > sc.lag_high:
            self.calm = 0.0
            if now - self._last_raise >= sc.raise_hold and self.G < self.top:
                if now - self._last_lower < 2 * self.hold:  # снижение не удержалось — дольше ждать следующего
                    self.hold = min(self.hold * 2, sc.calm_hold * HOLD_MAX)
                self.G = min(self.G + (2 if lag > 2 * sc.lag_high else 1), self.top)
                self._last_raise = now
        elif lag < sc.lag_low and util < sc.util_low:
            self.calm += dt
            floor = -1 if self.has_boost and util < sc.util_boost else 0
            if self.calm >= self.hold - 1e-9 and self.G > floor:
                self.G -= 1
                self.calm = 0.0
                self._last_lower = now
        else:
            self.calm = 0.0
        if self.G < 0 and (util >= sc.util_low or lag > sc.lag_low):
            self.G = 0  # буст съел запас — сразу назад
            self._last_lower = now
        return self.G

    def level(self, prio: int) -> int:
        if self.G < 0:
            return -1
        return min(max(self.G - prio, 0), self.K)

    def levels(self, prios: Dict[str, int]) -> Dict[str, int]:
        return {cam: self.level(p) for cam, p in prios.items()}
