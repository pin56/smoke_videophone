import http.client
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


def _serve(store, token, auth=None):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), label_server.make_handler(store, token, auth=auth))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _req(srv, method, path, body=None, cookie="", ctype="application/x-www-form-urlencoded"):
    """Один запрос без автоперехода по редиректам: (код, заголовки, тело)."""
    conn = http.client.HTTPConnection("127.0.0.1", srv.server_address[1], timeout=10)
    conn.request(method, path, body=body, headers={"Cookie": cookie, "Content-Type": ctype})
    r = conn.getresponse()
    out = (r.status, dict(r.getheaders()), r.read())
    conn.close()
    return out


def test_login_flow(tmp_path, monkeypatch):
    monkeypatch.setattr(label_server.time, "sleep", lambda s: None)
    store = _dataset(tmp_path)
    auth = label_server.Auth(tmp_path / "users")
    auth.add_user("alice", "correct-horse")
    srv = _serve(store, "", auth)
    try:
        code, hdr, _ = _req(srv, "GET", "/")
        assert code == 303 or code == 302
        assert hdr["Location"] == "/login"
        assert _req(srv, "GET", "/api/cycles")[0] == 401
        assert _req(srv, "GET", "/login")[0] == 200

        assert _req(srv, "POST", "/login", "user=alice&password=wrong")[0] == 401
        assert _req(srv, "POST", "/login", "user=nobody&password=correct-horse")[0] == 401
        code, hdr, _ = _req(srv, "POST", "/login", "user=alice&password=correct-horse")
        assert code == 303 and hdr["Location"] == "/"
        cookie = hdr["Set-Cookie"].split(";")[0]
        assert "HttpOnly" in hdr["Set-Cookie"] and "SameSite=Strict" in hdr["Set-Cookie"]

        code, _, body = _req(srv, "GET", "/api/cycles", cookie=cookie)
        assert code == 200 and json.loads(body)["me"] == "alice"
        forged = cookie[:-4] + "0000"
        assert _req(srv, "GET", "/api/cycles", cookie=forged)[0] == 401
        assert _req(srv, "GET", "/api/cycles", cookie=cookie.replace("alice", "bob"))[0] == 401

        # автор метки — вошедший пользователь, а не то, что прислал браузер
        rec = json.dumps({"id": "c__t1_0", "label": "smoking", "who": "mallory"})
        assert _req(srv, "POST", "/api/label", rec, cookie, "application/json")[0] == 200
        assert store.labels()["c__t1_0"]["who"] == "alice"

        code, hdr, _ = _req(srv, "GET", "/logout", cookie=cookie)
        assert "Max-Age=0" in hdr["Set-Cookie"]
    finally:
        srv.shutdown()


def test_login_lockout_and_session_expiry(tmp_path, monkeypatch):
    monkeypatch.setattr(label_server.time, "sleep", lambda s: None)
    auth = label_server.Auth(tmp_path / "users")
    auth.add_user("bob", "long-enough-pw")
    srv = _serve(_dataset(tmp_path), "", auth)
    try:
        for _ in range(label_server.MAX_FAILS):
            assert _req(srv, "POST", "/login", "user=bob&password=nope")[0] == 401
        # после лимита блокируется и верный пароль
        assert _req(srv, "POST", "/login", "user=bob&password=long-enough-pw")[0] == 429
    finally:
        srv.shutdown()
    now = 1_000_000.0
    value = auth.make_session("bob", now)
    assert auth.session_user(value, now + 10) == "bob"
    assert auth.session_user(value, now + label_server.SESSION_TTL + 1) == ""


def test_add_user_validation(tmp_path):
    auth = label_server.Auth(tmp_path / "users")
    for name, pw in [("bad name", "long-enough-pw"), ("a|b", "long-enough-pw"), ("ok", "short")]:
        try:
            auth.add_user(name, pw)
        except ValueError:
            continue
        raise AssertionError(f"{name!r} должен отклоняться")
    auth.add_user("ok", "long-enough-pw")
    assert "long-enough-pw" not in (tmp_path / "users").read_text()  # хранится только хэш


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
