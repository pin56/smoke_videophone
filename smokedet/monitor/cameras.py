"""Потоки камер для стены мониторинга — симуляция.

Видео объекта нельзя публиковать, поэтому камеры здесь — сцены с людьми (рамка, поза, движение руки), а не RTSP.
Симуляция идёт на сервере с шагом 0.1 с (10 к/с), как поток после детектора: одна хронология даёт эфир, архив
(кольцевой буфер последних `archive_sec` секунд), эпизоды журнала и фрагменты записи.

Правило эпизода — как у пайплайна (`episodes.build_events`), в упрощённом виде: ≥ 2 цикла «рука ко рту» за 20 с
открывают эпизод, циклы одного человека объединяются, пока пауза ≤ 15 с. Один цикл — подозрение, в журнал не идёт.

Состояние камеры (`health`) не зависит от тревог: анализ идёт только при «В сети» и «Задержка»; при «Нет сигнала»
и «Ошибка» кадры не пишутся в архив, а при «Анализируется» кадры есть, но вывода по ним ещё нет.
"""
from __future__ import annotations

import json
import logging
import math
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

from .journal import Journal

log = logging.getLogger(__name__)

FPS = 10
DT = 1.0 / FPS
WINDOW = 20.0  # окно правила, с
MERGE_GAP = 15.0  # пауза, после которой эпизод закрыт, с
PRE_ROLL = 10.0  # запас фрагмента до эпизода, с
POST_ROLL = 10.0  # и после последнего цикла, с

ROLES = ("walker", "idle", "smoker", "sipper")
STATES = ("normal", "suspect", "smoking")
HEALTH = ("online", "no_signal", "delay", "analyzing", "error")

# Сценарии: walker — проходит мимо, idle — стоит, smoker — курит сериями, sipper — изредка пьёт (один цикл).
# smoke_at — через сколько секунд от начала симуляции первая сигарета.
CAMS = [
    ("CAM-01", "Вокзал Гатчина-Балтийская", "Перрон 1, северный торец", "platform",
     [("walker",), ("idle",), ("walker",)]),
    ("CAM-02", "Вокзал, зал ожидания", "Вход с привокзальной площади", "indoor",
     [("sipper", 8), ("idle",), ("walker",)]),
    ("CAM-03", "Остановка «ул. Соборная»", "Павильон, сторона к центру", "street",
     [("smoker", 20), ("walker",), ("idle",)]),
    ("CAM-04", "Остановка «пл. Варшавская»", "Посадочная площадка", "street",
     [("walker",), ("walker",), ("idle",)]),
    ("CAM-05", "ТЦ «Пилот»", "Главный вход, козырёк", "street",
     [("idle",), ("walker",), ("smoker", 260)]),
    ("CAM-06", "Автостанция", "Посадка, платформа 3", "platform",
     [("smoker", 120), ("walker",), ("idle",), ("walker",)]),
    ("CAM-07", "Пр. 25 Октября", "Пешеходный переход у рынка", "street",
     [("walker",), ("walker",)]),
    ("CAM-08", "Остановка «Въезд в парк»", "Павильон", "street",
     [("idle",), ("smoker", 400), ("walker",)]),
    ("CAM-09", "Вокзал Гатчина-Варшавская", "Вестибюль, турникеты", "indoor",
     [("walker",), ("walker",), ("idle",)]),
    ("CAM-10", "Остановка «Больница»", "Посадочная площадка", "street",
     [("sipper", 27), ("walker",)]),
    ("CAM-11", "Перехватывающая парковка", "Въезд, шлагбаум", "street",
     [("walker",), ("idle",), ("smoker", 330)]),
    ("CAM-12", "Вокзал Гатчина-Балтийская", "Перрон 2, навес", "platform",
     [("idle",), ("walker",), ("sipper", 55)]),
]  # fmt: skip


def health_script(cam_id: str, t: float, t_start: float) -> Tuple[str, str, float, Optional[float]]:
    """Состояние камеры в момент t: (статус, причина, задержка с, начало сбоя или None).

    t_start — запуск сервера: сбои отсчитываются от него, чтобы при показе они были видны сразу.
    """
    if cam_id == "CAM-07":
        since = t_start - 252
        if t >= since:
            return "no_signal", "RTSP: камера не отвечает, таймаут 10 с", 0.0, since
    elif cam_id == "CAM-12":
        since = t_start - 540
        if t >= since:
            return "error", "Анализ остановлен: поток H.265 не декодируется", 0.0, since
    elif cam_id == "CAM-04":
        period = 240.0
        k = (t - t_start) % period
        base = t - k
        if 150 <= k < 175:
            return "no_signal", "Обрыв RTSP-сессии, переподключение", 0.0, base + 150
        if 175 <= k < 205:
            return "analyzing", "Переподключено, анализ догоняет поток", 0.0, base + 175
    elif cam_id == "CAM-11":
        return "delay", "Канал перегружен", 6.0 + 3.0 * math.sin(t / 20.0), None
    return "online", "", 0.0, None


@dataclass
class Cycle:
    start: float
    end: float


@dataclass
class Person:
    role: str
    track: int
    x: float
    y: float
    dir: int
    speed: float
    tone: int
    sway: float
    bias: float
    phase: float = 0.0
    a: float = 0.0  # 0 — рука опущена, 1 — кисть у рта
    cycle: Optional[Tuple[float, float, float, float]] = None  # начало, подъём, пауза, отвод
    cycles: List[Cycle] = field(default_factory=list)
    floor: float = -1e18  # циклы не позже этого момента ушли в закрытый эпизод
    left: int = 0
    next: float = math.inf
    lit_until: float = -1.0
    state: str = "normal"
    episode_start: Optional[float] = None
    event_id: Optional[int] = None

    def recent(self, t: float) -> List[Cycle]:
        return [c for c in self.cycles if c.end > self.floor and t - c.end <= WINDOW]

    def encode(self, t: float) -> str:
        lit = 1 if self.role == "smoker" and t < self.lit_until else 0
        return (f"[{self.track},{self.x:.4f},{self.y:.3f},{self.dir},{self.phase:.2f},{self.a:.3f},"
                f"{ROLES.index(self.role)},{STATES.index(self.state)},{lit},{self.tone},{self.sway:.2f}]")  # fmt: skip


class Camera:
    def __init__(self, idx: int, spec, t0: float, tracks: "TrackIds"):
        self.idx = idx
        self.id, self.name, self.zone, self.kind, people = spec
        self.rnd = random.Random(1000 + idx * 7919)
        self.tracks = tracks
        self.persons = [self._person(p, t0) for p in people]
        self.frames: Deque[Tuple[float, str]] = deque()
        self.health = ("online", "", 0.0, None)
        self.last_frame: Optional[float] = None

    def _person(self, spec, t0: float) -> Person:
        r = self.rnd
        p = Person(
            role=spec[0], track=self.tracks.next(), x=0.12 + r.random() * 0.76, y=0.6 + r.random() * 0.32,
            dir=-1 if r.random() < 0.5 else 1, speed=0.035 + r.random() * 0.03, tone=34 + int(r.random() * 16),
            sway=r.random() * 6, bias=(r.random() - 0.5) * 0.08,
        )  # fmt: skip
        if p.role == "smoker":
            p.next = t0 + spec[1]
            p.left = 4 + int(r.random() * 3)
        elif p.role == "sipper":
            p.next = t0 + spec[1]
        return p

    @property
    def observing(self) -> bool:
        return self.health[0] in ("online", "delay")

    @property
    def recording(self) -> bool:
        return self.health[0] in ("online", "delay", "analyzing")

    def status(self) -> str:
        if not self.observing:
            return "unknown"
        states = [p.state for p in self.persons]
        return "smoking" if "smoking" in states else "suspect" if "suspect" in states else "normal"

    def step(self, t: float, t_start: float, sink: "EpisodeSink") -> None:
        self.health = health_script(self.id, t, t_start)
        for p in self.persons:
            self._move(p, t)
        for p in self.persons:
            if self.observing:
                self._evaluate(p, t, sink)
            else:
                self._blind(p, t, sink)
        if self.recording:
            self.frames.append((t, "[" + ",".join(p.encode(t) for p in self.persons) + "]"))
            self.last_frame = t

    def _move(self, p: Person, t: float) -> None:
        r = self.rnd
        if p.role == "walker":
            p.x += p.dir * p.speed * DT
            p.phase = (p.phase + DT * p.speed * 70) % (2 * math.pi)
            if p.x < -0.1 or p.x > 1.1:  # ушёл из кадра — входит новый человек
                p.dir = -1 if r.random() < 0.5 else 1
                p.x = -0.08 if p.dir > 0 else 1.08
                p.y = 0.6 + r.random() * 0.32
                p.speed = 0.035 + r.random() * 0.03
                p.track = self.tracks.next()
        if p.cycle is None and t >= p.next:
            if p.role == "smoker":
                if p.left <= 0:
                    p.left = 4 + int(r.random() * 3)
                if p.lit_until < t:
                    p.lit_until = math.inf
                p.cycle = (t, 0.7, 1.1 + r.random() * 1.4, 0.7)
                p.left -= 1
                p.next = t + 7 + r.random() * 4
                if p.left == 0:
                    p.next = t + 240 + r.random() * 240  # следующая сигарета через 4–8 мин
            elif p.role == "sipper":
                p.cycle = (t, 0.7, 1.6 + r.random() * 0.8, 0.7)
                p.next = t + 50 + r.random() * 30
        if p.cycle is not None:
            start, up, hold, down = p.cycle
            e = t - start
            if e < up:
                p.a = _ease(e / up)
            elif e < up + hold:
                p.a = 1.0
            elif e < up + hold + down:
                p.a = 1.0 - _ease((e - up - hold) / down)
            else:
                p.a = 0.0
                p.cycle = None
                p.cycles.append(Cycle(start, t))
                if p.role == "smoker" and p.left == 0:
                    p.lit_until = t + 4
        p.cycles = [c for c in p.cycles if t - c.end <= 120]

    def _evaluate(self, p: Person, t: float, sink: "EpisodeSink") -> None:
        last = p.cycles[-1].end if p.cycles else -1e18
        if p.episode_start is not None and t - last > MERGE_GAP:
            self._close(p, t, last, sink)
        recent = p.recent(t)
        if p.episode_start is None and len(recent) >= 2:
            p.episode_start = recent[0].start
            n = len(recent)
            p.event_id = sink.open(self, p, p.episode_start, last, n, _confidence(p, n))
        elif p.episode_start is not None and p.event_id is not None:
            n = sum(1 for c in p.cycles if c.end > p.floor and c.start >= p.episode_start)
            sink.update(p.event_id, p.episode_start, last, n, _confidence(p, n))
        p.state = "smoking" if p.episode_start is not None else "suspect" if recent else "normal"

    def _blind(self, p: Person, t: float, sink: "EpisodeSink") -> None:
        """Анализа нет: открытый эпизод закрывается на последнем виденном цикле, выводов по кадрам нет."""
        if p.episode_start is not None:
            last = p.cycles[-1].end if p.cycles else t
            self._close(p, t, last, sink)
        p.floor = t
        p.state = "normal"

    def _close(self, p: Person, t: float, last: float, sink: "EpisodeSink") -> None:
        if p.event_id is not None:
            sink.close(self, p.event_id, p.episode_start, last, min(last + POST_ROLL, t))
        p.floor = last
        p.episode_start = None
        p.event_id = None

    def window(self, a: float, b: float) -> List[Tuple[float, str]]:
        return [f for f in self.frames if a <= f[0] <= b]

    def info(self) -> dict:
        return {"id": self.id, "name": self.name, "zone": self.zone, "kind": self.kind, "scene": self.idx}


class TrackIds:
    def __init__(self) -> None:
        self._n = 0

    def next(self) -> int:
        self._n += 1
        return self._n


def _ease(k: float) -> float:
    return k * k * (3 - 2 * k)


def _confidence(p: Person, n: int) -> float:
    return max(0.3, min(0.97, 0.58 + 0.07 * n + p.bias))


class EpisodeSink:
    """Эпизоды камер → журнал; при закрытии эпизода сохраняет фрагмент записи с запасом до и после."""

    def __init__(self, journal: Journal, clips_dir: Path):
        self.journal = journal
        self.clips_dir = Path(clips_dir)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self._cycles: Dict[int, int] = {}  # эпизод → число циклов в журнале, чтобы не писать без изменений

    def open(self, cam: Camera, p: Person, start: float, last: float, cycles: int, conf: float) -> int:
        event_id = self.journal.add_event(
            source_type="camera", source_id=cam.id, source_name=cam.name, track=p.track, started_at=start,
            duration=max(0.0, last - start), cycles=cycles, confidence=round(conf, 3), active=1,
        )  # fmt: skip
        self._cycles[event_id] = cycles
        return event_id

    def update(self, event_id: int, start: float, last: float, cycles: int, conf: float) -> None:
        if self._cycles.get(event_id) != cycles:
            self._cycles[event_id] = cycles
            self.journal.update_event(event_id, duration=max(0.0, last - start), cycles=cycles,
                                      confidence=round(conf, 3))  # fmt: skip

    def close(self, cam: Camera, event_id: int, start: float, last: float, until: float) -> None:
        self._cycles.pop(event_id, None)
        self.journal.update_event(event_id, duration=max(0.0, last - start), active=0)
        a, b = start - PRE_ROLL, until
        frames = cam.window(a, b)
        if not frames:
            return
        path = self.clips_dir / f"{cam.id}_{event_id}.json"
        body = {"kind": "sim", "camera": cam.info(), "fps": FPS, "event": [start, last]}
        text = json.dumps(body, ensure_ascii=False)[:-1] + ',"frames":[' + ",".join(
            f"[{t:.2f},{persons}]" for t, persons in frames
        ) + "]}"
        path.write_text(text, encoding="utf-8")
        # срок хранения отсчитывается от момента сохранения в шкале часов журнала
        self.journal.attach_clip(event_id, "sim", str(path), frames[0][0], frames[-1][0])


class CameraWall:
    """Все камеры, их кольцевые буферы и фоновый шаг симуляции в реальном времени."""

    def __init__(self, journal: Journal, clips_dir: Path, archive_sec: float = 1800.0, prefill_sec: float = 600.0,
                 clock=time.time):  # fmt: skip
        self.clock = clock
        self.archive_sec = archive_sec
        self.t_start = clock()
        self.t = self.t_start - prefill_sec
        tracks = TrackIds()
        self.cams = [Camera(i, spec, self.t, tracks) for i, spec in enumerate(CAMS)]
        self.by_id = {c.id: c for c in self.cams}
        self.sink = EpisodeSink(journal, clips_dir)
        self.cond = threading.Condition()
        self.seq = 0

    def advance(self, until: float) -> None:
        while self.t + DT <= until:
            self.t += DT
            for cam in self.cams:
                cam.step(self.t, self.t_start, self.sink)
                while cam.frames and cam.frames[0][0] < self.t - self.archive_sec:
                    cam.frames.popleft()
        with self.cond:
            self.seq += 1
            self.cond.notify_all()

    def start(self) -> threading.Thread:
        started, t0 = time.perf_counter(), self.t
        self.advance(self.clock())  # история до запуска: архив и журнал не пустые сразу
        log.info("камеры: подготовлено %.0f с истории за %.1f с", self.t - t0, time.perf_counter() - started)

        def loop() -> None:
            while True:
                try:
                    self.advance(self.clock())
                except Exception:
                    log.exception("ошибка симуляции камер")
                time.sleep(DT / 2)

        thread = threading.Thread(target=loop, name="monitor-cameras", daemon=True)
        thread.start()
        return thread

    # ---------- данные для клиента ----------

    def live_frame(self) -> str:
        """Последний кадр каждой камеры (у камеры с задержкой — кадр с опозданием), JSON-строкой."""
        parts = []
        for cam in self.cams:
            status, _, lag, _ = cam.health
            frame = None
            if cam.recording and cam.frames:
                if lag > 0:
                    target = self.t - lag
                    frame = next((f for f in reversed(cam.frames) if f[0] <= target), cam.frames[0])
                else:
                    frame = cam.frames[-1]
            parts.append(f'["{cam.id}",{frame[0]:.2f},{frame[1]}]' if frame else f'["{cam.id}",null,null]')
        return f'{{"t":{self.t:.2f},"cams":[{",".join(parts)}]}}'

    def state(self) -> dict:
        """Состояние камер раз в секунду: здоровье, вывод анализа, циклы людей за последнюю минуту."""
        out = []
        for cam in self.cams:
            status, reason, lag, since = cam.health
            last = cam.last_frame
            if status == "delay" and last is not None:
                last -= lag
            out.append({
                **cam.info(),
                "health": status,
                "reason": reason,
                "lag": round(lag, 1),
                "since": since,
                "last_frame": last,
                "detection": cam.status(),
                "recent": max((len(p.recent(self.t)) for p in cam.persons), default=0) if cam.observing else 0,
                "persons": [
                    {"track": p.track, "role": p.role, "state": p.state if cam.observing else "unknown",
                     "event_id": p.event_id, "cycles": [round(c.end, 2) for c in p.cycles if self.t - c.end <= 60]}
                    for p in cam.persons
                ],
            })  # fmt: skip
        return {"t": self.t, "cameras": out}

    def archive(self, a: float, b: float, step: int = 1) -> str:
        """Кадры всех камер в [a, b] (каждый step-й), JSON-строкой: {"from","to","cams":{id:[[t,persons],…]}}."""
        parts = []
        for cam in self.cams:
            frames = cam.window(a, b)[:: max(1, step)]
            parts.append(f'"{cam.id}":[' + ",".join(f"[{t:.2f},{s}]" for t, s in frames) + "]")
        return f'{{"from":{a:.2f},"to":{b:.2f},"cams":{{{",".join(parts)}}}}}'

    def archive_range(self) -> Tuple[float, float]:
        firsts = [c.frames[0][0] for c in self.cams if c.frames]
        return (min(firsts) if firsts else self.t), self.t
