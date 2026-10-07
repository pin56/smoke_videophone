import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import label_server  # noqa: E402


def _dataset(tmp_path):
    rec = {"id": "c__t1_0", "clip_id": "c", "track_id": 1, "start": 1.0, "end": 3.0, "times": [1.0, 1.1], "valid": True,
           "reject_reason": "", "in_event": False, "auto": {}}
    (tmp_path / "cycles.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    (tmp_path / "frames" / "c__t1_0").mkdir(parents=True)
    (tmp_path / "frames" / "c__t1_0" / "000.jpg").write_bytes(b"\xff\xd8jpg")
    (tmp_path / "secret.jpg").write_bytes(b"x")
    return label_server.Store(tmp_path)


def test_store_validation_and_last_label_wins(tmp_path):
    store = _dataset(tmp_path)
    ids = {"c__t1_0"}
    assert store.add({"id": "nope", "label": "smoking"}, ids)
    assert store.add({"id": "c__t1_0", "label": "x"}, ids)
    assert store.add({"id": "c__t1_0", "label": "not_smoking"}, ids)  # без категории
    assert not store.add({"id": "c__t1_0", "label": "not_smoking", "category": "drink", "vapor": True}, ids)
    assert not store.add({"id": "c__t1_0", "label": "smoking", "category": "drink", "vapor": True}, ids)
    lab = store.labels()["c__t1_0"]
    assert lab["label"] == "smoking" and lab["category"] == "" and lab["vapor"] is True
    assert "c__t1_0,c,1,1.0,3.0,smoking" in store.csv()


def test_http_auth_label_and_traversal(tmp_path):
    store = _dataset(tmp_path)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), label_server.make_handler(store, "tok"))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        def status(url, data=None, cookie=None):
            req = urllib.request.Request(url, data=data, headers={"Cookie": cookie or "", "Content-Type": "application/json"})
            try:
                return urllib.request.urlopen(req).status
            except urllib.error.HTTPError as e:
                return e.code

        assert status(base + "/") == 403
        assert status(base + "/?token=tok") == 200
        assert status(base + "/frames/c__t1_0/000.jpg", cookie="token=tok") == 200
        assert status(base + "/frames/../secret.jpg", cookie="token=tok") == 404
        body = json.dumps({"id": "c__t1_0", "label": "unsure"}).encode()
        assert status(base + "/api/label", body) == 403
        assert status(base + "/api/label", body, cookie="token=tok") == 200
        assert store.labels()["c__t1_0"]["label"] == "unsure"
    finally:
        srv.shutdown()
