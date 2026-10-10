"""VLM-арбитр потока без модели: очередь ответов по времени потока, вето событий, перепроверка отказа."""
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace

import numpy as np

from smokedet.config import VlmConfig
from smokedet.vlm import SessionArbiter, apply_veto, overlay


class FakeArbiter:
    """Ответы (P(курит), время модели) по порядку запросов."""

    def __init__(self, answers):
        self.answers, self.calls = list(answers), []

    def submit(self, imgs):
        smoke, sec = self.answers.pop(0)
        self.calls.append(len(imgs))
        f = Future()
        f.set_result((smoke, {}, sec))
        return f


def cyc(start, end, score):
    return SimpleNamespace(start=start, end=end, valid=True, appearance=score)


def session(answers, **kw):
    alerts = []
    sa = SessionArbiter(FakeArbiter(answers), VlmConfig(enabled=True, **kw), alerts, Counter())
    sa.sync = True  # повтор с файла: время ответа — виртуальное (отправка + очередь + время модели)
    return sa, alerts


def test_answers_follow_stream_time_and_queue():
    sa, alerts = session([(0.9, 0.5), (0.02, 0.4)])
    img = np.zeros((4, 4, 3), np.uint8)
    sa.crops[1] = [(cyc(1, 3, 5.0), img)]
    sa.crops[2] = [(cyc(2, 4, 4.0), img), (cyc(5, 7, 6.0), img)]
    sa.on_alert(1, (1, 3, True), 10.0)
    sa.on_alert(2, (2, 7, True), 10.2)
    sa.poll(10.4)
    assert alerts == []  # ответ ещё «в пути»: модель считает 0.5 с
    sa.poll(10.6)
    assert [(a["track_id"], a["t"], a["notify"]) for a in alerts] == [(1, 10.5, True)]
    sa.poll(11.0)
    second = alerts[1]
    assert second["work"] == 10.5 and second["t"] == 10.9 and not second["notify"]  # ждал, пока модель занята
    assert sa.a.calls == [1, 2] and sa.rejected(2) and not sa.rejected(1)


def test_high_clip_score_skips_model():
    sa, alerts = session([], hi=9.0)
    sa.crops[1] = [(cyc(1, 3, 12.0), np.zeros((4, 4, 3), np.uint8))]
    sa.on_alert(1, (1, 3, True), 5.0)
    assert alerts[0]["notify"] and alerts[0]["vlm"] is None and alerts[0]["skip"] == "clip" and not sa.pending


def test_recheck_when_event_grew_during_answer():
    sa, alerts = session([(0.02, 2.0), (0.9, 0.5)])
    img = np.zeros((4, 4, 3), np.uint8)
    sa.crops[1] = [(cyc(1, 3, 4.0), img)]
    sa.on_alert(1, (1, 3, True), 5.0)
    sa.crops[1].append((cyc(5, 6, 5.0), img))
    sa.on_extend(1, (1, 6, True), 6.0)  # запрос ещё в работе — перепроверка ждёт ответа
    assert len(sa.pending) == 1
    sa.poll(7.1)
    assert [a["kind"] for a in alerts] == ["alert"] and sa.pending[0].t_send == 7.0
    sa.poll(0.0, final=True)
    assert alerts[-1]["kind"] == "recheck" and alerts[-1]["notify"] and alerts[-1]["t"] == 7.5


def test_recheck_after_refusal_only_with_new_cycles():
    sa, alerts = session([(0.02, 0.1), (0.8, 0.1)], max_asks=2)
    img = np.zeros((4, 4, 3), np.uint8)
    sa.crops[1] = [(cyc(1, 3, 4.0), img)]
    sa.on_alert(1, (1, 3, True), 5.0)
    sa.poll(6.0)
    sa.on_extend(1, (1, 3, True), 6.5)  # новых циклов нет
    assert len(sa.pending) == 0
    sa.crops[1].append((cyc(8, 9, 5.0), img))
    sa.on_extend(1, (1, 9, True), 10.0)
    sa.poll(11.0)
    assert [a["kind"] for a in alerts] == ["alert", "recheck"] and alerts[-1]["notify"]
    sa.crops[1].append((cyc(12, 13, 7.0), img))
    sa.last[1]["notify"] = False
    sa.on_extend(1, (1, 13, True), 14.0)  # запросы на событие исчерпаны
    assert len(sa.pending) == 0


def test_veto_uses_last_answer_of_event():
    ev = lambda tid, s, e: SimpleNamespace(track_id=tid, start=s, end=e, passed=True, reject_reason="")  # noqa: E731
    events = [ev(1, 0, 10), ev(2, 0, 10), ev(3, 0, 10)]
    alerts = [
        {"t": 5, "track_id": 1, "start": 0, "end": 3, "latency": 2},
        {"t": 5.4, "track_id": 1, "start": 0, "end": 3, "vlm": 0.01, "answer": False, "notify": False},
        {"t": 9.4, "track_id": 1, "start": 0, "end": 8, "vlm": 0.7, "answer": True, "notify": True},
        {"t": 5.4, "track_id": 2, "start": 0, "end": 3, "vlm": 0.05, "answer": False, "notify": False},
    ]
    assert apply_veto(events, alerts) == 1
    assert [e.passed for e in events] == [True, False, True]
    assert events[1].reject_reason == "vlm:0.050"


def test_overlay_states():
    alerts = [
        {"t": 5.0, "track_id": 1, "start": 0, "end": 3, "latency": 2},
        {"t": 5.4, "track_id": 1, "start": 0, "end": 3, "vlm": 0.9, "answer": True, "notify": True, "send": 5.0,
         "work": 5.0, "sec": 0.4, "crops": 2, "kind": "alert", "clip_best": 4.0},
    ]  # fmt: skip
    marks, panel = overlay(alerts, [], "Qwen3-VL-2B")
    assert panel(5.2)[0][0].startswith("Qwen3-VL-2B arbiter: work id1")
    assert panel(6.0)[0][0] == "Qwen3-VL-2B arbiter: idle"
    assert "NOTIFY" in panel(6.0)[-1][0]
    alarm = [m for m in marks[1] if m.alarm]
    assert alarm and alarm[0].t0 == 5.4


def test_file_replay_waits_for_answer():
    pool = ThreadPoolExecutor(1)
    sa, alerts = session([])
    sa.a.submit = lambda imgs: pool.submit(lambda: (time.sleep(0.05), (0.9, {}, 0.3))[1])
    sa.sync = True  # повтор с файла: поток ждёт модель, ожидание — в wait
    sa.crops[1] = [(cyc(1, 3, 4.0), np.zeros((4, 4, 3), np.uint8))]
    sa.on_alert(1, (1, 3, True), 5.0)
    sa.poll(5.1)
    assert alerts == [] and sa.pending[0].future.done() and sa.wait > 0
    sa.poll(5.3)
    assert alerts[0]["t"] == 5.3 and alerts[0]["notify"]


def test_live_answer_not_before_model_returned():
    sa, alerts = session([(0.9, 0.5)])
    sa.sync = False  # живой источник: модель общая для камер, ответ виден потоку, когда она его вернула
    sa.crops[1] = [(cyc(1, 3, 5.0), np.zeros((4, 4, 3), np.uint8))]
    sa.on_alert(1, (1, 3, True), 10.0)
    sa.poll(12.0)
    assert alerts[0]["t"] == 12.0 and alerts[0]["work"] == 10.0
