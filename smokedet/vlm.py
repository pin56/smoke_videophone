"""VLM-арбитр тревог потокового режима. Тревога уходит мультимодальной модели (Qwen3-VL) вместе с кропами «голова и
кисти» лучших по CLIP циклов события. Модель выбирает, что делает человек; отказ («пьёт», «телефон», …) снимает событие.
Вопрос, кропы и пороги подобраны в tools/vlm_lab.py.

Модель считает в своём потоке (ThreadPoolExecutor на один поток — очередь запросов), поток видео её не ждёт. Кропы
режутся из кропов в памяти задания (StoredCrop) сразу после оценки цикла CLIP, поэтому модель видит только кадры до
тревоги. Время в журнале тревог — время потока: отправка — в момент тревоги, начало работы — когда модель свободна,
ответ — через измеренное время модели. Так журнал один и тот же в реальном времени и при обработке файла быстрее
реального времени (ответ применяется, когда время потока до него дошло).
"""
from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .appearance import crop_geometry, square_crop
from .config import VlmConfig
from .cycles import Cycle
from .episodes import Event
from .geometry import TrackSignal

log = logging.getLogger(__name__)

EPS = 0.05
CLASSES = (
    ("smoke", "smoking a cigarette"),
    ("vape", "vaping (an electronic cigarette)"),
    ("drink", "drinking from a bottle, can or cup"),
    ("eat", "eating food"),
    ("phone", "using a mobile phone"),
    ("other", "touching the face (scratching, wiping the mouth, adjusting glasses or mask) or something else"),
)  # fmt: skip
POSITIVE = ("smoke", "vape")
LETTERS = "ABCDEF"
SCENE = ("The {n} images are crops from a low-resolution outdoor CCTV camera. They show the same person at {n} "
         "different moments when the person's hand is raised to the face. ")  # fmt: skip
PROMPT = SCENE + "What is the person most likely doing?\n{options}\nAnswer with a single letter."
KEEP_SEC = 120.0  # с: кропы циклов старше не хранятся (событие длиннее — к модели уходят его последние циклы)
SHOW = 5.0  # с: вердикт у рамки в ролике
MIN_GPU_GIB = 10.0  # GPU меньше — модель на CPU: рядом с детектором, позой и CLIP не помещается (6 ГБ: загрузка 3.6)


def torch_device(device: str) -> str:
    return f"cuda:{device}" if device.isdigit() else device


def pick_device(cfg: VlmConfig, device: str) -> str:
    """vlm.device, а если пусто — устройство детектора; на GPU меньше MIN_GPU_GIB — CPU."""
    if cfg.device:
        return torch_device(cfg.device)
    dev = torch_device(device or "cpu")
    if dev.startswith("cuda"):
        import torch

        props = torch.cuda.get_device_properties(torch.device(dev).index or 0)
        gib = props.total_memory / 2**30
        if gib < MIN_GPU_GIB:
            log.info("VLM на CPU: у %s %.1f ГиБ, рядом с моделями потока не поместится (vlm.device — задать явно)",
                     props.name, gib)  # fmt: skip
            return "cpu"
    return dev


def cycle_crop(crops: dict, sig: TrackSignal, c: Cycle, cfg: VlmConfig) -> Optional[np.ndarray]:
    """Кроп цикла для модели из кропов в памяти задания (кадр -> StoredCrop): кадр паузы, где кисть ближе всего ко рту,
    квадрат cfg.scale роста вокруг «рот + кисть», cfg.px. Как vlm_lab.py extract, только без перечитывания видео."""
    idx = np.nonzero((sig.t >= c.hold_start) & (sig.t <= c.hold_end))[0]
    if not len(idx):
        idx = np.array([sig.index_at((c.hold_start + c.hold_end) / 2)])
    d = np.where(np.isfinite(sig.d[idx]), sig.d[idx], np.inf)
    for k in idx[np.argsort(d, kind="stable")]:
        sc = crops.get(int(sig.frame_idx[k]))
        if sc is None:
            continue
        centers, height = crop_geometry(sig, c, np.array([k]))
        cx, cy = centers[0]
        img = square_crop(sc.img, (cx - sc.x0) * sc.k, (cy - sc.y0) * sc.k, cfg.scale * height * sc.k)
        return cv2.resize(img, (cfg.px, cfg.px), interpolation=cv2.INTER_CUBIC)
    return None


class VlmModel:
    """Вопрос «что делает человек» с шестью вариантами: вероятности букв ответа на последней позиции, усреднённые по
    порядкам вариантов. Результат — P(курит) + P(вейп) и вероятности классов."""

    def __init__(self, cfg: VlmConfig, device: str):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch, self.device = torch, pick_device(cfg, device)
        dtype = torch.float16 if self.device.startswith("cuda") else torch.bfloat16
        self.proc = AutoProcessor.from_pretrained(cfg.model)
        self.model = AutoModelForImageTextToText.from_pretrained(cfg.model, dtype=dtype).to(self.device).eval()
        self.ids = [self.proc.tokenizer.encode(x, add_special_tokens=False)[0] for x in LETTERS[: len(CLASSES)]]
        order = list(range(len(CLASSES)))
        self.orders = [order, order[::-1]] if cfg.orders > 1 else [order]

    def __call__(self, imgs: List[np.ndarray]) -> Tuple[float, Dict[str, float]]:
        from PIL import Image

        torch = self.torch
        pil = [Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in imgs]
        probs = np.zeros(len(CLASSES))
        for order in self.orders:
            options = "\n".join(f"{LETTERS[i]}. {CLASSES[c][1]}" for i, c in enumerate(order))
            msg = [{"role": "user", "content": [*({"type": "image", "image": im} for im in pil),
                                                {"type": "text", "text": PROMPT.format(n=len(pil), options=options)}]}]  # fmt: skip
            inputs = self.proc.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, return_dict=True,
                                                   return_tensors="pt").to(self.device)  # fmt: skip
            with torch.inference_mode():
                logits = self.model(**inputs, logits_to_keep=1).logits[0, -1].float()
            p = torch.softmax(logits[self.ids], 0).cpu().numpy()
            for i, c in enumerate(order):
                probs[c] += p[i] / len(self.orders)
        smoke = float(sum(probs[i] for i, (c, _) in enumerate(CLASSES) if c in POSITIVE))
        return smoke, {c: round(float(probs[i]), 4) for i, (c, _) in enumerate(CLASSES)}


class VlmArbiter:
    """Модель и очередь запросов к ней (один поток: запросы всех сессий ждут друг друга)."""

    def __init__(self, cfg: VlmConfig, device: str):
        t0 = time.perf_counter()
        self.cfg = cfg
        self.model = VlmModel(cfg, device)
        self.model([np.zeros((cfg.px, cfg.px, 3), dtype=np.uint8)])  # первый вызов CUDA долгий
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vlm")
        self.name = cfg.model.rstrip("/").split("/")[-1].replace("-Instruct", "")
        log.info("VLM %s на %s загружена и прогрета за %.1f с", self.name, self.model.device, time.perf_counter() - t0)

    def submit(self, imgs: List[np.ndarray]) -> Future:
        def run():
            t0 = time.perf_counter()
            smoke, probs = self.model(imgs)
            return smoke, probs, time.perf_counter() - t0

        return self.pool.submit(run)


@dataclass
class Ask:
    tid: int
    start: float  # событие на момент запроса
    end: float
    t_send: float
    n: int  # кропов
    best: float  # лучший CLIP события
    kind: str  # alert | recheck
    future: Future
    cycles: Tuple[float, ...]  # начала отправленных циклов


class SessionArbiter:
    """Запросы одной сессии (камеры): кропы циклов, запрос по тревоге и перепроверка отказа, ответы в журнал тревог
    (alerts) по времени потока."""

    def __init__(self, arbiter: VlmArbiter, cfg: VlmConfig, alerts: List[dict], stats):
        self.a, self.cfg, self.alerts, self.stats = arbiter, cfg, alerts, stats
        # повтор с файла идёт быстрее реального времени: поток ждёт модель, чтобы узнать ответ в тот же момент
        # времени потока, что живая камера; ожидание (wait, с) не входит в нагрузку
        self.sync = False
        self.wait = 0.0
        self.crops: Dict[int, List[Tuple[Cycle, np.ndarray]]] = defaultdict(list)
        self.pending: List[Ask] = []
        self.busy = -math.inf  # время потока, когда модель освободится (по полученным ответам)
        self.last: Dict[int, dict] = {}  # человек -> последний ответ (запись журнала)
        self.asks: Dict[Tuple[int, float], int] = defaultdict(int)  # (человек, начало события) -> запросов
        self.sent: Dict[int, Tuple[float, ...]] = {}  # человек -> циклы последнего запроса
        self.state: Dict[int, tuple] = {}  # человек -> событие (start, end, ...) на последней тревоге или продлении

    def add(self, tid: int, crops: dict, sig: TrackSignal, cycles: List[Cycle], now: float) -> None:
        """Кропы валидных циклов (после оценки CLIP): задание их скоро удалит, а событие может длиться дольше."""
        for c in cycles:
            if c.valid:
                img = cycle_crop(crops, sig, c, self.cfg)
                if img is not None:
                    self.crops[tid].append((c, img))
        for t in list(self.crops):
            self.crops[t] = [(c, img) for c, img in self.crops[t] if now - c.end <= KEEP_SEC]

    def rejected(self, tid: int) -> bool:
        r = self.last.get(tid)
        return r is not None and not r["notify"]

    def _pool(self, tid: int, st) -> List[Tuple[Cycle, np.ndarray]]:
        pool = [(c, img) for c, img in self.crops.get(tid, []) if c.valid and st[0] - EPS <= c.start and c.end <= st[1] + EPS]
        return sorted(pool, key=lambda x: -x[0].appearance if np.isfinite(x[0].appearance) else math.inf)

    def ask(self, tid: int, st, now: float, kind: str) -> None:
        ranked = self._pool(tid, st)
        pool = ranked[: self.cfg.top]
        best = max((c.appearance for c, _ in ranked if np.isfinite(c.appearance)), default=-math.inf)
        if not pool or best >= self.cfg.hi:
            # кропов нет (дубль, слитый с другим заданием) или CLIP уверен и ответ модели не нужен — без очереди
            rec = {"t": round(now, 2), "track_id": tid, "start": round(st[0], 2), "end": round(st[1], 2), "vlm": None,
                   "notify": True, "send": round(now, 2), "work": round(now, 2), "sec": 0.0, "crops": 0, "kind": kind,
                   "skip": "clip" if pool else "crops", "clip_best": round(best, 2) if np.isfinite(best) else None}  # fmt: skip
            self.alerts.append(rec)
            self.last[tid] = rec
            log.info("t=%.1f с: VLM, человек %d — без модели (%s)", now, tid,
                     f"CLIP {best:.1f}" if pool else "нет кропов")  # fmt: skip
            return
        self.asks[(tid, round(st[0], 2))] += 1
        self.sent[tid] = tuple(c.start for c, _ in pool)
        fut = self.a.submit([img for _, img in pool])
        self.pending.append(Ask(tid, st[0], st[1], now, len(pool), best, kind, fut, self.sent[tid]))
        self.stats["vlm_asks"] += 1
        log.info("t=%.1f с: VLM, человек %d — запрос (%s, кропов %d)", now, tid, kind, len(pool))

    def on_alert(self, tid: int, st, now: float) -> None:
        self.state[tid] = st
        self.ask(tid, st, now, "alert")

    def on_extend(self, tid: int, st, now: float) -> None:
        self.state[tid] = st
        self._recheck(tid, now)

    def _recheck(self, tid: int, now: float) -> None:
        """Событие с отказом модели продолжилось: перепросить, если появились новые циклы и запросы не исчерпаны
        (или сразу уведомить, если CLIP нового цикла уверен)."""
        st = self.state.get(tid)
        if st is None or not self.rejected(tid) or any(q.tid == tid for q in self.pending):
            return
        if self.asks[(tid, round(st[0], 2))] >= self.cfg.max_asks:
            return
        top = tuple(c.start for c, _ in self._pool(tid, st)[: self.cfg.top])
        if top and top != self.sent.get(tid):
            self.ask(tid, st, now, "recheck")

    def poll(self, now: float, final: bool = False) -> None:
        """Ответы, до которых дошло время потока (final — все, с ожиданием модели)."""
        while self.pending:
            q = self.pending[0]
            if not q.future.done():
                if not (final or self.sync):
                    return
                t0 = time.perf_counter()
                q.future.result()
                self.wait += time.perf_counter() - t0
            smoke, probs, sec = q.future.result()
            work = max(q.t_send, self.busy)
            t = work + sec
            if not (final or self.sync):  # живой источник: ответ доступен потоку не раньше, чем модель его вернула
                t = max(t, now)
            if t > now and not final:
                return
            self.pending.pop(0)
            self.busy = t
            notify = smoke >= self.cfg.thr or q.best >= self.cfg.hi
            rec = {"t": round(t, 2), "track_id": q.tid, "start": round(q.start, 2), "end": round(q.end, 2),
                   "vlm": round(smoke, 4), "answer": smoke >= self.cfg.thr, "notify": notify, "send": round(q.t_send, 2), "work": round(work, 2),
                   "sec": round(sec, 3), "crops": q.n, "kind": q.kind,
                   "clip_best": round(q.best, 2) if np.isfinite(q.best) else None, "probs": probs}  # fmt: skip
            self.alerts.append(rec)
            self.last[q.tid] = rec
            self.stats["vlm_sec"] += sec
            self.stats["vlm_rejects"] += not notify
            log.info("t=%.1f с: VLM, человек %d — %s (P(курит)=%.2f, %.2f с)", t, q.tid,
                     "подтверждено" if notify else "отказ", smoke, sec)  # fmt: skip
            if not notify:  # пока модель думала, событие могло вырасти — перепроверка в момент ответа
                self._recheck(q.tid, t)


def apply_veto(events: List[Event], alerts: List[dict]) -> int:
    """Снять события, по которым последний ответ модели — отказ. Возвращает число снятых."""
    answers = [r for r in alerts if "vlm" in r]
    n = 0
    for e in events:
        if not e.passed:
            continue
        rs = [r for r in answers if r["track_id"] == e.track_id and r["start"] <= e.end + EPS and r["end"] >= e.start - EPS]
        last = max(reversed(rs), key=lambda r: r["t"], default=None)  # при равном t — последняя запись
        if last is not None and not last["notify"]:
            e.passed = False
            e.reject_reason = f"vlm:{last['vlm']:.3f}"
            n += 1
    return n


def overlay(alerts: List[dict], events: List[Event], model: str = "VLM", lines: int = 6):
    """Журнал тревог -> состояния людей для render_video (marks) и панель арбитра panel(t): тревога, отправка,
    очередь, работа модели, ответ, уведомление или отказ, снятие тревоги."""
    import bisect

    from .render import GRAY, ORANGE, RED, WHITE, Mark

    marks: Dict[int, List[Mark]] = defaultdict(list)
    journal: List[Tuple[float, int, str, tuple]] = []  # (t, порядок, текст, цвет)
    work: List[Tuple[float, float, float, int]] = []
    for k, r in enumerate(alerts):
        tid, t = r["track_id"], r["t"]
        if r.get("withdrawn"):
            marks[tid].append(Mark(t, t + SHOW, "ALERT WITHDRAWN", GRAY))
            journal.append((t, 4 * k, f"id{tid} alert withdrawn ({r['reason'].split(':')[0]})", GRAY))
        elif "vlm" not in r:
            marks[tid].append(Mark(t, t + 0.3, "ALERT", ORANGE))
            journal.append((t, 4 * k, f"id{tid} ALERT event {r['start']:.1f}-{r['end']:.1f}s", ORANGE))
        elif r["vlm"] is None:
            why = f"CLIP {r['clip_best']:.1f} high" if r.get("skip") == "clip" else "no crops"
            end = max((e.end for e in events if e.track_id == tid and e.start <= r["end"] + EPS and e.end >= r["start"] - EPS),
                      default=r["end"])  # fmt: skip
            marks[tid].append(Mark(t, max(t + SHOW, end), f"SMOKING  {why}, VLM skipped", RED, alarm=True))
            journal.append((t, 4 * k + 3, f"id{tid} {why}, VLM skipped -> NOTIFY", RED))
        else:
            send, w = r["send"], r["work"]
            work.append((send, w, t, tid))
            again = " (recheck)" if r["kind"] == "recheck" else ""
            journal.append((send, 4 * k + 1, f"id{tid} VLM send{again}: {r['crops']} crop(s)", ORANGE))
            journal.append((w, 4 * k + 2, f"id{tid} VLM work", ORANGE))
            if w > send:
                marks[tid].append(Mark(send, w, "ALERT  VLM queued {dt:.1f}s", ORANGE))
            marks[tid].append(Mark(w, t, "ALERT  VLM work {dt:.1f}s", ORANGE))
            text = f"VLM answer {r['answer']} p={r['vlm']:.2f}"
            if r["notify"] and not r["answer"]:
                text += f", CLIP {r['clip_best']:.1f} high"
            end = max((e.end for e in events if e.track_id == tid and e.start <= r["end"] + EPS and e.end >= r["start"] - EPS),
                      default=r["end"])  # fmt: skip
            if r["notify"]:
                marks[tid].append(Mark(t, max(t + SHOW, end), f"SMOKING  VLM {r['vlm']:.2f}", RED, alarm=True))
                journal.append((t, 4 * k + 3, f"id{tid} {text} -> NOTIFY", RED))
            else:
                marks[tid].append(Mark(t, t + SHOW, f"REJECTED  VLM {r['vlm']:.2f}", GRAY))
                journal.append((t, 4 * k + 3, f"id{tid} {text} -> REJECT", GRAY))
    journal.sort(key=lambda x: (x[0], x[1]))
    times = [x[0] for x in journal]

    def panel(t: float) -> List[Tuple[str, tuple]]:
        busy = [(w, tid) for _, w, end, tid in work if w <= t < end]
        queued = sum(1 for s, w, _, _ in work if s <= t < w)
        state = f"work id{busy[0][1]} {t - busy[0][0]:.1f}s" if busy else "idle"
        head = (f"{model} arbiter: {state}" + (f", queue {queued}" if queued else ""), ORANGE if busy else WHITE)
        n = bisect.bisect_right(times, t)
        return [head] + [(f"{ts:7.2f}s  {text}", color) for ts, _, text, color in journal[max(0, n - lines): n]]

    return marks, panel
