"""Веб-интерфейс: данные для повтора (payload) и HTTP-сервер — без моделей."""
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from smokedet.config import Config
from smokedet.cycles import detect_cycles
from smokedet.episodes import build_events
from smokedet.io_video import VideoMeta
from smokedet.pipeline import Analysis, ClipInput
from smokedet.roi import Roi
from smokedet.tracking import TrackSet
from smokedet.ui import server as server_mod
from smokedet.ui.payload import build_payload, plural, reason_text, roi_contours
from smokedet.ui.server import ClipJob, ClipStore, UiServer, parse_range
from test_logic import _obs, make_signal, puffs


def _analysis(starts, direct=False):
    cfg = Config()
    sig = make_signal(puffs(starts))
    cycles = {1: detect_cycles(sig, cfg.cycle)}
    if direct:
        for c in cycles[1]:
            c.direct, c.pos_conf, c.objects = True, 0.71, {"cigarette": 0.71, "cup": 0.2}
    events = build_events(cycles, {1: sig}, cfg.episode)
    tracks = {1: [_obs(float(t), 100) for t in sig.t]}
    return cfg, Analysis(signals={1: sig}, cycles=cycles, events=events), tracks


def _payload(cfg, analysis, tracks):
    clip = ClipInput("x/clip.mp4", "cam1", "c1")
    meta = VideoMeta("x/clip.mp4", 30.0, 1200, 640, 480)
    return build_payload(clip, meta, tracks, analysis, cfg, "/video/c1")


def test_payload_event_revealed_after_second_cycle():
    p = _payload(*_analysis([2, 8]))
    json.dumps(p, allow_nan=False)  # NaN в JSON недопустим
    valid = [c for c in p["cycles"] if c["valid"]]
    assert len(valid) == 2 and p["stats"] == {"tracks": 1, "cycles": 2, "events": 1}
    ev = p["events"][0]
    assert ev["passed"] and ev["reveal_t"] == valid[1]["end"] and ev["cycle_ids"] == [c["id"] for c in valid]
    tr = p["tracks"][0]
    assert len(tr["kp"][0]) == 34 and len(tr["kc"][0]) == 17 and len(tr["t"]) == len(tr["d"])
    kinds = [r["kind"] for r in p["feed"]]
    assert kinds.count("cycle") == 2 and kinds[-2:] == ["event", "event_end"]
    assert p["feed"][-2]["text"] == "ЭПИЗОД: 2 цикла" and ev["reveal_basis"] == ev["basis"] == "2 цикла"
    assert "уверенность" in p["feed"][-1]["text"]
    assert [r["t"] for r in p["feed"]] == sorted(r["t"] for r in p["feed"])


def test_payload_direct_evidence_reveals_after_first_cycle():
    p = _payload(*_analysis([2], direct=True))
    ev = p["events"][0]
    assert ev["reveal_t"] == p["cycles"][0]["end"]
    assert ev["evidence"] == ["сигарета 0.71"] and p["cycles"][0]["direct"]
    assert ev["reveal_basis"] == "1 цикл + сигарета 0.71"


def test_reason_text_and_plural():
    assert "телефон" in reason_text("phone_like")
    assert "лицо не видно" in reason_text("phone_like,face_hidden")
    assert reason_text("track_object:lollipop") == "в руке чупа-чупс"
    assert reason_text("object:bottle") == "у рта бутылка"
    assert "67%" in reason_text("objects_in_67%_cycles")
    assert reason_text("unknown_code") == "unknown_code"
    forms = ("цикл", "цикла", "циклов")
    assert [plural(n, forms) for n in (1, 2, 5, 11, 21, 22)] == ["цикл", "цикла", "циклов", "циклов", "цикл", "цикла"]


def test_roi_contours_with_exclusion():
    assert roi_contours(Roi.full()) is None
    mask = np.zeros((100, 100), bool)
    mask[10:90, 10:90] = True
    mask[40:60, 40:60] = False  # исключённая зона — дыра
    polys = roi_contours(Roi(mask))
    assert len(polys) == 2 and all(len(p) >= 4 for p in polys)


def test_parse_range():
    assert parse_range(None, 100) is None
    assert parse_range("bytes=0-9", 100) == (0, 9)
    assert parse_range("bytes=90-", 100) == (90, 99)
    assert parse_range("bytes=-10", 100) == (90, 99)
    assert parse_range("bytes=50-500", 100) == (50, 99)
    assert parse_range("items=0-1", 100) is None
    with pytest.raises(ValueError):
        parse_range("bytes=100-", 100)


@pytest.fixture
def base_url(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(bytes(range(256)) * 4)
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_text("<html>ui</html>", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
    store = ClipStore([], tmp_path, pipeline_factory=None)
    store.jobs = {
        "c1": ClipJob(ClipInput(str(video), "cam1", "c1"), state="ready", payload=b'{"ok":1}', video=video),
        "c2": ClipJob(ClipInput(str(video), "cam1", "c2")),
    }
    srv = UiServer(("127.0.0.1", 0), store, static_dir=static)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def _get(url, headers=None):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers or {}), timeout=5) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def test_server_routes(base_url):
    status, _, body = _get(base_url + "/")
    assert status == 200 and b"ui" in body
    status, _, body = _get(base_url + "/api/clips")
    assert status == 200 and [c["state"] for c in json.loads(body)] == ["ready", "queued"]
    assert _get(base_url + "/api/clips/c1")[2] == b'{"ok":1}'
    assert _get(base_url + "/api/clips/c2")[0] == 409  # ещё не готов
    assert _get(base_url + "/api/clips/nope")[0] == 404
    assert _get(base_url + "/static/%2e%2e/secret.txt")[0] == 404  # за пределы папки статики не выйти
    assert _get(base_url + "/video/c2")[0] == 404


def test_server_video_range(base_url):
    status, headers, body = _get(base_url + "/video/c1", {"Range": "bytes=10-19"})
    assert status == 206 and body == bytes(range(10, 20)) and headers["Content-Range"] == "bytes 10-19/1024"
    status, headers, body = _get(base_url + "/video/c1")
    assert status == 200 and len(body) == 1024 and headers["Accept-Ranges"] == "bytes"
    assert _get(base_url + "/video/c1", {"Range": "bytes=2000-"})[0] == 416


def test_clip_store_builds_payload(tmp_path, monkeypatch):
    cfg, analysis, tracks = _analysis([2, 8])
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"0")
    progress = []

    class FakePipeline:
        def __init__(self):
            self.cfg = cfg

        def track(self, clip, cache_dir, on_progress=None):
            on_progress(5, 10)
            progress.append(store.jobs["c1"].progress)
            return TrackSet(meta=VideoMeta(str(video), 30.0, 1200, 640, 480), stride=3, tracks=tracks)

        def analyze(self, clip, ts):
            return analysis

    monkeypatch.setattr(server_mod, "browser_video", lambda src, proxy_dir, clip_id: src)
    store = ClipStore([ClipInput(str(video), "cam1", "c1")], tmp_path, FakePipeline)
    store.run()
    job = store.jobs["c1"]
    assert progress == [0.5] and job.state == "ready" and job.events == 1 and job.video == video
    assert json.loads(job.payload)["clip"]["video_url"] == "/video/c1"


def test_clip_store_reports_errors(tmp_path):
    store = ClipStore([ClipInput("rtsp://cam", "cam1", "s1")], tmp_path, lambda: object())
    store.run()
    assert store.jobs["s1"].state == "error" and "видеофайл" in store.jobs["s1"].error


def test_lazy_store_waits_for_request(tmp_path):
    store = ClipStore([ClipInput("a.mp4", "cam", "a")], tmp_path, lambda: object(), lazy=True)
    assert store.jobs["a"].state == "idle"
    store.request("a")
    assert store.jobs["a"].state == "queued"
    store.request("missing")  # неизвестный клип игнорируется


def test_scan_videos(tmp_path):
    import ui as ui_mod

    (tmp_path / "true").mkdir()
    (tmp_path / "true" / "1.mp4").write_bytes(b"0")
    (tmp_path / "2.MOV").write_bytes(b"0")
    (tmp_path / "notes.txt").write_text("x")
    clips = ui_mod.scan_videos(tmp_path)
    assert [c.clip_id for c in clips] == ["2", "true__1"]
