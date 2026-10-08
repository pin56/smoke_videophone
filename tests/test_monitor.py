"""Мониторинг (smokedet/monitor): журнал и срок хранения, эпизоды камер и фрагменты, состояние камер, очередь
проверки видео и HTTP-сервер — без моделей и без браузера."""
import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from smokedet.monitor.analysis import AnalysisQueue, DemoAnalyzer, PipelineAnalyzer, video_sources
from smokedet.monitor.cameras import PRE_ROLL, CameraWall, health_script
from smokedet.monitor.journal import DAY, Journal
from smokedet.monitor.server import MonitorServer, parse_range

T0 = 1_800_000_000.0


class Clock:
    def __init__(self, t=T0):
        self.t = t

    def __call__(self):
        return self.t


def _journal(tmp_path, clock=None, retention=7 * DAY):
    return Journal(tmp_path / "journal.sqlite", retention=retention, clock=clock or Clock())


def test_review_is_separate_from_detection(tmp_path):
    j = _journal(tmp_path)
    eid = j.add_event(source_type="camera", source_id="CAM-01", source_name="x", started_at=T0, duration=30,
                      cycles=3, confidence=0.8)  # fmt: skip
    ev = j.event(eid)
    assert ev["detection"] == "auto" and ev["review"] == "pending" and ev["reviewed_at"] is None
    assert j.review(eid, "confirmed")
    ev = j.event(eid)
    assert ev["detection"] == "auto" and ev["review"] == "confirmed" and ev["reviewed_at"] == T0
    assert j.review(eid, "pending") and j.event(eid)["reviewed_at"] is None
    assert not j.review(9999, "rejected")
    try:
        j.review(eid, "smoking")
    except ValueError:
        pass
    else:
        raise AssertionError("неизвестное решение должно отклоняться")


def test_purge_deletes_files_after_retention_but_keeps_records(tmp_path):
    clock = Clock()
    j = _journal(tmp_path, clock)
    clip = tmp_path / "clip.json"
    clip.write_text("{}")
    upload = tmp_path / "up.mp4"
    upload.write_bytes(b"x")
    eid = j.add_event(source_type="camera", source_id="CAM-01", source_name="x", started_at=T0)
    j.attach_clip(eid, "sim", str(clip), T0 - 10, T0 + 40)
    j.add_job("job1", "a.mp4", str(upload), 1, 10.0, "demo")
    j.update_job("job1", state="done", finished_at=T0, expires_at=T0 + 7 * DAY)
    assert j.event(eid)["clip_expires_at"] == T0 + 7 * DAY

    assert j.purge(T0 + 7 * DAY - 1) == {"clips": 0, "uploads": 0}
    assert clip.exists() and upload.exists()

    clock.t = T0 + 7 * DAY
    assert j.purge() == {"clips": 1, "uploads": 1}
    assert not clip.exists() and not upload.exists()
    ev = j.event(eid)
    assert ev is not None and ev["clip_deleted_at"] == T0 + 7 * DAY
    assert j.job("job1")["deleted_at"] == T0 + 7 * DAY
    assert j.purge() == {"clips": 0, "uploads": 0}


def test_restart_marks_interrupted_jobs_and_closes_active_events(tmp_path):
    j = _journal(tmp_path)
    j.add_job("job1", "a.mp4", "x", 1, None, "demo")
    j.update_job("job1", state="analyzing")
    eid = j.add_event(source_type="camera", source_id="CAM-01", source_name="x", started_at=T0, active=1)
    j.close()
    j2 = _journal(tmp_path)
    assert j2.job("job1")["state"] == "error" and "прерван" in j2.job("job1")["error"]
    assert j2.event(eid)["active"] == 0


def test_health_does_not_depend_on_alarms():
    start = T0
    assert health_script("CAM-07", start - 300, start)[0] == "online"
    status, reason, _, since = health_script("CAM-07", start, start)
    assert status == "no_signal" and reason and since == start - 252
    assert health_script("CAM-12", start, start)[0] == "error"
    status, _, lag, _ = health_script("CAM-11", start, start)
    assert status == "delay" and 3 <= lag <= 9
    seen = {health_script("CAM-04", start + k, start)[0] for k in range(0, 240, 5)}
    assert seen == {"online", "no_signal", "analyzing"}
    assert health_script("CAM-01", start, start)[0] == "online"


def _wall(tmp_path, prefill=600.0):
    j = _journal(tmp_path)
    wall = CameraWall(j, tmp_path / "clips", archive_sec=900, prefill_sec=prefill, clock=Clock())
    wall.advance(T0)
    return j, wall


def test_camera_episodes_have_clips_with_pre_roll(tmp_path):
    j, wall = _wall(tmp_path)
    events = j.events(source_type="camera")
    closed = [e for e in events if not e["active"]]
    assert closed, "за 10 минут истории курящие должны дать эпизоды"
    for ev in closed:
        assert ev["cycles"] >= 2 and ev["detection"] == "auto" and ev["review"] == "pending"
        assert ev["clip_kind"] == "sim" and Path(ev["clip_path"]).is_file()
        clip = json.loads(Path(ev["clip_path"]).read_text(encoding="utf-8"))
        assert clip["camera"]["id"] == ev["source_id"] and clip["frames"]
        if ev["started_at"] - PRE_ROLL > T0 - 600 + 1:  # запас до эпизода, если запись уже была
            assert clip["frames"][0][0] <= ev["started_at"] - PRE_ROLL + 0.2
        assert clip["frames"][-1][0] >= ev["started_at"] + ev["duration"]
        assert ev["clip_expires_at"] == ev["clip_saved_at"] + 7 * DAY


def test_no_frames_and_no_analysis_without_signal(tmp_path):
    _, wall = _wall(tmp_path)
    cam = wall.by_id["CAM-07"]
    assert cam.frames[-1][0] < T0 - 252 + 0.2
    st = next(c for c in wall.state()["cameras"] if c["id"] == "CAM-07")
    assert st["health"] == "no_signal" and st["detection"] == "unknown"
    assert abs(st["last_frame"] - (T0 - 252)) < 0.2 and st["reason"]
    delayed = next(c for c in wall.state()["cameras"] if c["id"] == "CAM-11")
    assert delayed["health"] == "delay" and delayed["last_frame"] < wall.t - 2
    live = json.loads(wall.live_frame())
    row = next(r for r in live["cams"] if r[0] == "CAM-07")
    assert row[1] is None


def test_archive_returns_frames_in_window(tmp_path):
    _, wall = _wall(tmp_path, prefill=120.0)
    data = json.loads(wall.archive(T0 - 60, T0 - 50, step=2))
    frames = data["cams"]["CAM-01"]
    assert 48 <= len(frames) <= 51 and all(T0 - 60 <= f[0] <= T0 - 50 for f in frames)
    a, b = wall.archive_range()
    assert a <= T0 - 119 and b == wall.t


def test_parse_range():
    assert parse_range(None, 100) is None
    assert parse_range("bytes=10-19", 100) == (10, 19)
    assert parse_range("bytes=-10", 100) == (90, 99)
    assert parse_range("bytes=90-", 100) == (90, 99)
    try:
        parse_range("bytes=200-", 100)
    except ValueError:
        pass
    else:
        raise AssertionError


class _NoModels(PipelineAnalyzer):
    def available(self) -> str:
        return "не установлены torch: анализ видео недоступен"


def test_pipeline_without_models_fails_job_with_reason(tmp_path):
    j = _journal(tmp_path, clock=time.time)
    q = AnalysisQueue(j, _NoModels("cfg", [], tmp_path), tmp_path / "up", tmp_path / "clips")
    (tmp_path / "up" / "a.mp4").write_bytes(b"x")
    j.add_job("job1", "a.mp4", str(tmp_path / "up" / "a.mp4"), 1, 5.0, "pipeline")
    q.start()
    q.submit("job1")
    _wait(lambda: j.job("job1")["state"] == "error")
    assert "torch" in j.job("job1")["error"]


def test_video_sources_from_files_folders_and_csv(tmp_path):
    for d in ("true", "false"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "1.mp4").write_bytes(b"x")
    (tmp_path / "true" / "notes.txt").write_text("-")
    rows = ["video_path,camera_id,clip_id", "true/1.mp4,cam1,true_1", "false/1.mp4,cam2,"]
    (tmp_path / "list.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    got = video_sources([str(tmp_path / "true"), str(tmp_path / "list.csv"), str(tmp_path / "false" / "1.mp4")])
    assert [name for _, name in got] == ["true/1.mp4", "true_1.mp4", "1.mp4", "1.mp4"]
    assert got[1][0] == (tmp_path / "true" / "1.mp4").resolve()
    with pytest.raises(ValueError):
        video_sources([str(tmp_path / "missing.mp4")])
    with pytest.raises(ValueError):
        video_sources([str(tmp_path / "true" / "notes.txt")])


class _NoEpisodes:
    name = "realtime"

    def available(self) -> str:
        return ""

    def run(self, path, job_id, on_progress):
        assert Path(path).read_bytes() == b"video"
        return [], 12.0


def test_enqueued_record_survives_purge(tmp_path):
    clock = Clock()
    j = _journal(tmp_path, clock=clock, retention=DAY)
    q = AnalysisQueue(j, _NoEpisodes(), tmp_path / "up", tmp_path / "clips")
    src = tmp_path / "rec" / "cam.mp4"
    src.parent.mkdir()
    src.write_bytes(b"video")
    job_id = q.enqueue_file(src, "rec/cam.mp4")
    job = j.job(job_id)
    assert (job["filename"], job["analyzer"], job["state"]) == ("rec/cam.mp4", "realtime", "queued")
    q.process(job_id)
    assert j.job(job_id)["state"] == "done" and j.job(job_id)["duration"] == 12.0
    clock.t += 2 * DAY
    assert j.purge()["uploads"] == 1
    assert not Path(job["path"]).exists() and src.read_bytes() == b"video"


def _wait(cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return
        time.sleep(0.05)
    raise AssertionError("не дождались")


def _request(url, data=None, headers=None, method=None):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def test_http_upload_analysis_review_and_expiry(tmp_path):
    clock = Clock(time.time())
    j = Journal(tmp_path / "journal.sqlite", clock=clock)
    q = AnalysisQueue(j, DemoAnalyzer(min_work=0, max_work=0), tmp_path / "up", tmp_path / "clips")
    q.ffmpeg = None  # фрагмент — окно в исходном файле, без внешних программ
    wall = CameraWall(j, tmp_path / "clips", archive_sec=60, prefill_sec=30, clock=clock)
    wall.advance(clock())
    server = MonitorServer(("127.0.0.1", 0), j, wall, q)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    q.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        status, _, body = _request(f"{base}/api/jobs", b"abc", {"X-Filename": "notes.txt"}, "POST")
        assert status == 415 and "видеофайл" in json.loads(body)["error"]

        payload = bytes(range(256)) * 400
        status, _, body = _request(f"{base}/api/jobs", payload, {"X-Filename": "%D0%BA%D0%B0%D0%BC%D0%B5%D1%80%D0%B0.mp4",
                                                                 "X-Duration": "600"}, "POST")  # fmt: skip
        assert status == 201
        job = json.loads(body)
        assert job["filename"] == "камера.mp4" and job["state"] in ("queued", "analyzing")
        _wait(lambda: j.job(job["id"])["state"] == "done")
        status, _, body = _request(f"{base}/api/jobs/{job['id']}")
        detail = json.loads(body)
        assert detail["state"] == "done" and detail["episodes"] == len(detail["events"]) > 0
        ev = detail["events"][0]
        assert ev["source_type"] == "file" and ev["demo"] == 1 and ev["clip_kind"] == "range"
        assert ev["clip_start"] <= ev["offset_sec"] and ev["clip_status"] == "saved"

        status, headers, body = _request(f"{base}/files/{job['id']}", headers={"Range": "bytes=100-199"})
        assert status == 206 and body == payload[100:200] and headers["Content-Range"] == f"bytes 100-199/{len(payload)}"
        status, headers, _ = _request(f"{base}/clips/{ev['id']}?download=1")
        assert status == 200 and "attachment" in headers["Content-Disposition"]

        status, _, _ = _request(f"{base}/api/events/{ev['id']}/review", b'{"decision":"smoking"}', method="POST")
        assert status == 400
        status, _, body = _request(f"{base}/api/events/{ev['id']}/review", b'{"decision":"rejected"}', method="POST")
        assert status == 200 and json.loads(body)["review"] == "rejected"

        clock.t += 7 * DAY + 1
        j.purge()
        assert _request(f"{base}/clips/{ev['id']}")[0] == 410
        assert _request(f"{base}/files/{job['id']}")[0] == 410
        status, _, body = _request(f"{base}/api/events/{ev['id']}")
        assert json.loads(body)["clip_status"] == "deleted"

        status, _, body = _request(f"{base}/api/state")
        assert status == 200 and len(json.loads(body)["cameras"]) == 12
        status, _, body = _request(f"{base}/")
        assert status == 200 and "Проверка видео".encode() in body
    finally:
        server.shutdown()
        server.server_close()
