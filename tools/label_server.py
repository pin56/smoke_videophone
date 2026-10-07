#!/usr/bin/env python
"""Веб-разметчик циклов, выгруженных tools/export_cycles.py. Только стандартная библиотека.

    python tools/label_server.py --data data/labeling/set1 --host 0.0.0.0 --port 8765

Метки дописываются в <data>/labels.jsonl (последняя метка цикла побеждает), сводка — по /labels.csv.
Доступ — либо вход по логину и паролю (`--users users.txt`, пользователей добавляет `--add-user ИМЯ`; пароли
хранятся как scrypt, сессия — подписанная cookie на 12 часов, 5 неудачных входов с адреса блокируют его на 5 минут),
либо по токену (печатается при запуске; передаётся в ссылке один раз и сохраняется в cookie).
Для доступа из интернета включайте HTTPS (`--cert/--key`): иначе пароль и кадры идут открытым текстом.
Сервер ничего не отправляет наружу: кадры отдаются только с диска набора. Держите его во внутренней
сети, не пробрасывайте порт в интернет — это записи объекта.
"""
from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import hmac
import html
import io
import json
import os
import re
import secrets
import ssl
import sys
import threading
import time
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

LABELS = {"smoking", "not_smoking", "unsure"}
CATEGORIES = {"face_touch", "drink", "phone", "eat", "candy", "other_gesture"}
CSV_FIELDS = ["id", "clip_id", "track_id", "start", "end", "label", "category", "vapor", "note", "who", "ts"]


class Store:
    def __init__(self, data: Path):
        self.data = data
        self.labels_path = data / "labels.jsonl"
        self.lock = threading.Lock()

    def cycles(self):
        path = self.data / "cycles.jsonl"
        if not path.exists():
            return []
        return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]

    def labels(self):
        out = {}
        if self.labels_path.exists():
            for line in self.labels_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    out[r["id"]] = r
        return out

    def add(self, rec: dict, known: set) -> str:
        """Проверяет метку и дописывает её. Возвращает текст ошибки или ''."""
        if not isinstance(rec, dict) or rec.get("id") not in known:
            return "неизвестный цикл"
        if rec.get("label") not in LABELS:
            return "неверная метка"
        if rec["label"] == "not_smoking" and rec.get("category") not in CATEGORIES:
            return "нужна категория"
        row = {
            "id": rec["id"], "label": rec["label"],
            "category": rec.get("category", "") if rec["label"] == "not_smoking" else "",
            "vapor": bool(rec.get("vapor")) if rec["label"] == "smoking" else False,
            "note": str(rec.get("note", ""))[:500], "who": str(rec.get("who", ""))[:40],
            "ts": rec.get("ts", ""),
        }  # fmt: skip
        with self.lock, open(self.labels_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        return ""

    def csv(self) -> str:
        labels = self.labels()
        buf = io.StringIO()
        w = csv.DictWriter(buf, CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for c in self.cycles():
            if c["id"] in labels:
                w.writerow({**c, **labels[c["id"]]})
        return buf.getvalue()


PAGE = r"""<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Разметка циклов</title>
<style>
body{font:15px system-ui,sans-serif;margin:0;background:#16181c;color:#e6e6e6;display:flex;flex-direction:column;min-height:100vh}
header{padding:8px 14px;background:#20242b;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
main{flex:1;display:flex;gap:16px;padding:14px;flex-wrap:wrap;justify-content:center}
#view{background:#000;max-width:100%;border-radius:6px;min-width:260px;min-height:200px;object-fit:contain}
.side{width:340px;max-width:100%}
button{font:inherit;padding:8px 12px;margin:3px 3px 3px 0;border:0;border-radius:6px;background:#343a45;color:#eee;cursor:pointer}
button:hover{background:#454d5c}.smk{background:#b3382c}.no{background:#2b6e45}.un{background:#7a6a1f}
kbd{background:#0006;border-radius:3px;padding:0 5px;font-size:12px}
table{border-collapse:collapse;font-size:13px}td{padding:2px 8px 2px 0;color:#aab}td:nth-child(2){color:#fff}
.tag{display:inline-block;padding:1px 7px;border-radius:9px;background:#343a45;font-size:12px;margin-right:4px}
#cats button.on,#vap.on{outline:2px solid #fff}input[type=text]{width:100%;padding:6px;box-sizing:border-box;background:#0006;color:#eee;border:1px solid #444;border-radius:5px}
.hint{color:#889;font-size:12px;margin-top:8px}
</style>
<header><b>Разметка циклов</b><span id="prog"></span>
<label>Фильтр <select id="flt"><option value="todo">неразмеченные</option><option value="all">все</option>
<option value="valid">валидные</option><option value="rej">отброшенные</option><option value="ev">в событиях</option></select></label>
<label>Скорость <select id="spd"><option value="1">1×</option><option value="0.5" selected>0.5×</option><option value="0.25">0.25×</option></select></label>
<label>Вас зовут <input id="who" size="8" style="background:#0006;color:#eee;border:1px solid #444;border-radius:4px"></label>
<a href="/labels.csv" style="color:#8ab4f8">labels.csv</a>
<a id="out" href="/logout" hidden style="color:#8ab4f8">выйти (<span id="meName"></span>)</a></header>
<main><img id="view" alt="">
<div class="side"><div id="info"></div>
<div style="margin-top:10px"><button class="smk" onclick="lab('smoking')"><kbd>S</kbd> курит</button>
<button class="no" onclick="lab('not_smoking')"><kbd>N</kbd> не курит</button>
<button class="un" onclick="lab('unsure')"><kbd>U</kbd> неясно</button></div>
<div id="cats"></div>
<div><button id="vap" onclick="togVap()"><kbd>V</kbd> выдох/дым виден</button></div>
<input type="text" id="note" placeholder="заметка (необязательно)">
<div style="margin-top:8px"><button onclick="go(-1)"><kbd>←</kbd></button><button onclick="go(1)"><kbd>→</kbd> пропустить</button>
<button onclick="play()"><kbd>пробел</kbd> ещё раз</button></div>
<div class="hint">Красная кайма — пауза у рта, зелёная рамка — размечаемый человек. «Не курит»: затем клавиша категории
1 лицо, 2 питьё, 3 телефон, 4 еда, 5 чупа-чупс, 6 другое. Метка сохраняется сразу и можно менять позже.</div></div></main>
<script>
const CATS=[["face_touch","трогает лицо"],["drink","питьё"],["phone","телефон"],["eat","еда"],["candy","чупа-чупс"],["other_gesture","другой жест"]];
let all=[],labels={},list=[],pos=0,cat="",vap=false,timer=null,imgs=[];
const $=id=>document.getElementById(id);
$("who").value=localStorage.who||"";$("who").onchange=()=>localStorage.who=$("who").value;
$("cats").innerHTML=CATS.map((c,i)=>`<button data-c="${c[0]}" onclick="pickCat('${c[0]}')"><kbd>${i+1}</kbd> ${c[1]}</button>`).join("");
async function load(){const r=await fetch("/api/cycles");if(r.status==401){location.href="/login";return}
 const d=await r.json();all=d.cycles;labels=d.labels;
 if(d.me){$("who").value=d.me;$("who").readOnly=true;$("meName").textContent=d.me;$("out").hidden=false}refilter(true)}
function filtered(){const f=$("flt").value;
 return all.filter(c=>f=="all"||(f=="todo"&&!labels[c.id])||(f=="valid"&&c.valid)||(f=="rej"&&!c.valid)||(f=="ev"&&c.in_event))}
function refilter(keep){const cur=list[pos]&&list[pos].id;list=filtered();
 pos=keep&&cur?Math.max(0,list.findIndex(c=>c.id==cur)):0;show()}
// циклы выгружаются постепенно: раз в 20 с подтягиваем новые, не прерывая текущий ролик
async function poll(){try{const r=await fetch("/api/cycles");if(r.status==401){location.href="/login";return}const d=await r.json();
 if(d.cycles.length==all.length)return;const wasEmpty=!list.length;all=d.cycles;labels={...d.labels,...labels};
 const cur=list[pos]&&list[pos].id;list=filtered();const i=list.findIndex(c=>c.id==cur);
 if(wasEmpty||i<0&&!cur){pos=0;show()}else{pos=Math.max(0,i);
  $("prog").textContent=`размечено ${Object.keys(labels).length} из ${all.length}; в списке ${list.length}`}}catch(e){}}
setInterval(poll,20000);
$("flt").onchange=()=>refilter(false);
function show(){clearTimeout(timer);const n=Object.keys(labels).length;$("prog").textContent=`размечено ${n} из ${all.length}; в списке ${list.length}`;
 const c=list[pos];if(!c){$("info").innerHTML="Нет циклов в этом фильтре.";$("view").removeAttribute("src");return}
 const L=labels[c.id]||{};cat=L.category||"";vap=!!L.vapor;$("note").value=L.note||"";markBtns();
 const a=c.auto;$("info").innerHTML=`<div><b>${c.id}</b> (${pos+1}/${list.length})</div>
 <div style="margin:6px 0">${L.label?`<span class="tag">метка: ${L.label}${L.category?" / "+L.category:""}${L.vapor?" / пар":""}</span>`:""}
 <span class="tag">${c.valid?"валидный":"отброшен: "+c.reject_reason}</span>${c.in_event?'<span class="tag">в событии</span>':""}</div>
 <table><tr><td>время</td><td>${c.start}–${c.end} с</td></tr><tr><td>прямой признак</td><td>${a.direct?"да":"нет"}</td></tr>
 <tr><td>вспышка / пар</td><td>${a.flash?"да":"нет"} / ${a.vapor?"да":"нет"} (${a.vapor_net??"—"})</td></tr>
 <tr><td>чужие предметы</td><td>${a.strong_negative||"—"} (${a.neg_fraction})</td></tr><tr><td>мин. расстояние</td><td>${a.min_dist}</td></tr></table>`;
 imgs=c.times.map((t,i)=>{const im=new Image();im.src=`/frames/${c.id}/${String(i).padStart(3,"0")}.jpg`;return im});play()}
function play(){clearTimeout(timer);const c=list[pos];if(!c)return;let i=0;const sp=parseFloat($("spd").value);
 (function step(){$("view").src=imgs[i].src;const dt=i+1<c.times.length?(c.times[i+1]-c.times[i])*1000/sp:700;i=(i+1)%c.times.length;
 timer=setTimeout(step,i==0?Math.max(dt,600):Math.min(Math.max(dt,40),600))})()}
function markBtns(){document.querySelectorAll("#cats button").forEach(b=>b.classList.toggle("on",b.dataset.c==cat));$("vap").classList.toggle("on",vap)}
function pickCat(c){cat=c;markBtns();lab("not_smoking")}
function togVap(){vap=!vap;markBtns();const L=labels[list[pos]?.id];if(L&&L.label=="smoking")lab("smoking")}
async function lab(label){const c=list[pos];if(!c)return;if(label=="not_smoking"&&!cat){alert("выберите категорию 1–6");return}
 const rec={id:c.id,label,category:cat,vapor:vap,note:$("note").value,who:$("who").value,ts:new Date().toISOString()};
 const r=await fetch("/api/label",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(rec)});
 if(!r.ok){alert(await r.text());return}labels[c.id]=rec;go(1,true)}
function go(d,afterLabel){const f=$("flt").value;if(afterLabel&&f=="todo"){list.splice(pos,1);if(pos>=list.length)pos=Math.max(0,list.length-1)}else pos=Math.min(Math.max(0,pos+d),list.length-1);show()}
document.onkeydown=e=>{if(e.target.tagName=="INPUT")return;const k=e.key.toLowerCase();
 if(k=="s")lab("smoking");else if(k=="n"){if(cat)lab("not_smoking");else alert("сначала категория 1–6")}else if(k=="u")lab("unsure");else if(k=="v")togVap();
 else if(k>="1"&&k<="6")pickCat(CATS[+k-1][0]);else if(k=="arrowright")go(1);else if(k=="arrowleft")go(-1);else if(k==" "){e.preventDefault();play()}};
load();
</script></html>"""


SESSION_TTL = 12 * 3600  # с: сессия живёт 12 часов, потом нужен новый вход
USER_RE = re.compile(r"^[A-Za-z0-9_.-]{1,32}$")
SCRYPT = {"n": 2**14, "r": 8, "p": 1}
MAX_FAILS, LOCK_SEC = 5, 300  # 5 неудачных входов с одного адреса — блокировка на 5 минут


def hash_password(password: str, salt: bytes | None = None) -> str:
    salt = salt or os.urandom(16)
    return f"scrypt${salt.hex()}${hashlib.scrypt(password.encode(), salt=salt, **SCRYPT).hex()}"


def _check_hash(password: str, stored: str) -> bool:
    try:
        _, salt, digest = stored.split("$")
        got = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt), **SCRYPT)
        return hmac.compare_digest(got, bytes.fromhex(digest))
    except ValueError:
        return False


class Auth:
    """Вход по логину и паролю. Пользователи — строки «имя:scrypt$соль$хэш» в файле (читается при каждом входе,
    поэтому нового пользователя можно добавить, не перезапуская сервер). Сессия — подписанная cookie без состояния
    на сервере (ключ лежит рядом с файлом пользователей), переживает перезапуск."""

    def __init__(self, users_path: Path):
        self.path = users_path
        secret_path = users_path.with_name(users_path.name + ".secret")
        if not secret_path.exists():
            secret_path.write_text(secrets.token_hex(32))
            try:
                secret_path.chmod(0o600)
            except OSError:
                pass
        self.secret = bytes.fromhex(secret_path.read_text().strip())
        self._dummy = hash_password("dummy")  # для неизвестных имён считаем хэш так же долго: имена не угадать по времени
        self._fails: dict = {}
        self._lock = threading.Lock()

    def users(self) -> dict:
        out = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                name, _, h = line.strip().partition(":")
                if name and h:
                    out[name] = h
        return out

    def add_user(self, name: str, password: str) -> None:
        if not USER_RE.match(name):
            raise ValueError("имя: латиница, цифры, _.- до 32 символов")
        if len(password) < 8:
            raise ValueError("пароль короче 8 символов")
        users = self.users()
        users[name] = hash_password(password)
        self.path.write_text("".join(f"{n}:{h}\n" for n, h in users.items()), encoding="utf-8")
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    def check_password(self, name: str, password: str) -> bool:
        stored = self.users().get(name)
        ok = _check_hash(password, stored or self._dummy)
        return ok and stored is not None

    def locked_for(self, ip: str, now: float) -> int:
        with self._lock:
            n, until = self._fails.get(ip, (0, 0.0))
            return int(until - now) + 1 if until > now else 0

    def register(self, ip: str, ok: bool, now: float) -> None:
        with self._lock:
            if ok:
                self._fails.pop(ip, None)
                return
            n, until = self._fails.get(ip, (0, 0.0))
            n = 0 if until and until <= now else n  # блокировка истекла — счёт заново
            n += 1
            self._fails[ip] = (n, now + LOCK_SEC if n >= MAX_FAILS else 0.0)

    def _sign(self, msg: str) -> str:
        return hmac.new(self.secret, msg.encode(), hashlib.sha256).hexdigest()

    def make_session(self, user: str, now: float) -> str:
        body = f"{user}|{int(now) + SESSION_TTL}"
        return f"{body}|{self._sign(body)}"

    def session_user(self, value: str, now: float) -> str:
        """Имя пользователя из cookie сессии или '' (подпись неверна, срок вышел, пользователь удалён)."""
        user, _, rest = value.partition("|")
        exp, _, sig = rest.partition("|")
        if not exp.isdigit() or not hmac.compare_digest(sig, self._sign(f"{user}|{exp}")):
            return ""
        return user if int(exp) > now and user in self.users() else ""


LOGIN_PAGE = """<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Вход</title><style>
body{font:15px system-ui,sans-serif;background:#16181c;color:#e6e6e6;display:flex;min-height:100vh;align-items:center;justify-content:center;margin:0}
form{background:#20242b;padding:26px 28px;border-radius:10px;width:290px;box-shadow:0 6px 30px #0008}
h1{font-size:18px;margin:0 0 16px}input{width:100%;box-sizing:border-box;padding:9px;margin:5px 0 12px;background:#0007;color:#eee;border:1px solid #444;border-radius:6px;font:inherit}
button{width:100%;padding:10px;border:0;border-radius:6px;background:#2d6cdf;color:#fff;font:inherit;cursor:pointer}.err{color:#ff8a80;margin-bottom:10px}
</style><form method="post" action="/login"><h1>Разметка циклов</h1>__ERR__
<label>Логин<input name="user" autocomplete="username" autofocus required></label>
<label>Пароль<input name="password" type="password" autocomplete="current-password" required></label>
<button>Войти</button></form></html>"""


def make_handler(store: Store, token: str, secure: bool = False, auth: Auth | None = None):
    cycle_ids = lambda: {c["id"] for c in store.cycles()}  # noqa: E731
    cookie_flags = "HttpOnly; SameSite=Strict; Path=/" + ("; Secure" if secure else "")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):  # короткий журнал в stderr
            sys.stderr.write("%s %s\n" % (self.address_string(), fmt % a))

        def _send(self, code: int, body: bytes, ctype: str, extra=()):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if ctype.startswith(("text", "application")) else "max-age=3600")
            for k, v in extra:
                self.send_header(k, v)
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            self.wfile.write(body)

        def _redirect(self, where: str, code: int = 303, extra=()):
            self._send(code, b"", "text/plain", [("Location", where), *extra])

        def _identity(self, query) -> str:
            """'' — не вошёл; 'token' — вошёл по токену; иначе имя пользователя. Без защиты — 'anon'."""
            if not token and auth is None:
                return "anon"
            c = cookies.SimpleCookie(self.headers.get("Cookie", ""))
            if auth is not None and "session" in c:
                user = auth.session_user(c["session"].value, time.time())
                if user:
                    return user
            if token:
                got = query.get("token", [""])[0] or (c["token"].value if "token" in c else "")
                if hmac.compare_digest(got, token):
                    return "token"
            return ""

        def _login_page(self, error: str = "", code: int = 200):
            msg = f'<div class="err">{html.escape(error)}</div>' if error else ""
            self._send(code, LOGIN_PAGE.replace("__ERR__", msg).encode(), "text/html; charset=utf-8")

        def _do_login(self):
            ip = self.client_address[0]
            wait = auth.locked_for(ip, time.time())
            if wait:
                return self._login_page(f"Слишком много попыток. Подождите {wait} с.", 429)
            n = min(int(self.headers.get("Content-Length", "0") or 0), 4096)
            form = parse_qs(self.rfile.read(n).decode("utf-8", "replace"))
            user, password = form.get("user", [""])[0], form.get("password", [""])[0]
            ok = auth.check_password(user, password)
            auth.register(ip, ok, time.time())
            if not ok:
                time.sleep(0.5)
                return self._login_page("Неверный логин или пароль.", 401)
            cookie = f"session={auth.make_session(user, time.time())}; {cookie_flags}; Max-Age={SESSION_TTL}"
            self._redirect("/", extra=[("Set-Cookie", cookie)])

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            path = unquote(u.path)
            if path == "/login" and auth is not None:
                return self._login_page()
            if path == "/logout":
                return self._redirect("/login" if auth else "/", extra=[("Set-Cookie", f"session=; {cookie_flags}; Max-Age=0")])
            who = self._identity(q)
            if not who:
                if auth is not None:
                    return self._redirect("/login") if path == "/" else self._send(401, b"unauthorized", "text/plain")
                return self._send(403, "нужен токен: откройте ссылку с ?token=...".encode(), "text/plain; charset=utf-8")
            extra = [("Set-Cookie", f"token={token}; {cookie_flags}")] if who == "token" and "token" in q else []
            if path == "/":
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8", extra)
            if path == "/api/cycles":
                me = who if who not in ("token", "anon") else ""
                body = json.dumps({"cycles": store.cycles(), "labels": store.labels(), "me": me}, ensure_ascii=False)
                return self._send(200, body.encode(), "application/json")
            if path == "/labels.csv":
                return self._send(200, store.csv().encode("utf-8-sig"), "text/csv; charset=utf-8",
                                  [("Content-Disposition", 'attachment; filename="labels.csv"')])  # fmt: skip
            if path.startswith("/frames/"):
                f = (store.data / path.lstrip("/")).resolve()
                if (store.data / "frames").resolve() in f.parents and f.suffix == ".jpg" and f.is_file():
                    return self._send(200, f.read_bytes(), "image/jpeg")
            return self._send(404, b"not found", "text/plain")

        def do_POST(self):
            path = urlparse(self.path).path
            if path == "/login" and auth is not None:
                return self._do_login()
            if path != "/api/label":
                return self._send(404, b"not found", "text/plain")
            who = self._identity({})
            if not who:
                return self._send(401 if auth is not None else 403, b"forbidden", "text/plain")
            try:
                n = int(self.headers.get("Content-Length", "0"))
                rec = json.loads(self.rfile.read(min(n, 20000)))
            except (ValueError, json.JSONDecodeError):
                return self._send(400, "плохой запрос".encode(), "text/plain; charset=utf-8")
            if who not in ("token", "anon") and isinstance(rec, dict):
                rec["who"] = who  # автор — вошедший пользователь, а не то, что прислал браузер
            err = store.add(rec, cycle_ids())
            if err:
                return self._send(400, err.encode(), "text/plain; charset=utf-8")
            return self._send(200, b"{}", "application/json")

    return Handler


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Веб-разметчик циклов")
    p.add_argument("--data", help="папка набора (результат export_cycles.py)")
    p.add_argument("--users", help="файл пользователей: включает вход по логину и паролю (страница /login)")
    p.add_argument("--add-user", metavar="ИМЯ", help="добавить или сменить пользователя в --users (пароль спросят) и выйти")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 — слушать всю внутреннюю сеть")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--token", help="токен доступа (по умолчанию случайный); пустая строка не допускается")
    p.add_argument("--no-auth", action="store_true", help="без токена (только для localhost)")
    p.add_argument("--cert", help="сертификат TLS (PEM): включает HTTPS; обязателен, если порт виден из интернета")
    p.add_argument("--key", help="закрытый ключ TLS (PEM)")
    args = p.parse_args(argv)
    auth = Auth(Path(args.users)) if args.users else None
    if args.add_user:
        if auth is None:
            sys.exit("--add-user требует --users")
        pw = getpass.getpass("Пароль (не короче 8 символов): ")
        if pw != getpass.getpass("Ещё раз: "):
            sys.exit("пароли не совпали")
        try:
            auth.add_user(args.add_user, pw)
        except ValueError as e:
            sys.exit(str(e))
        print(f"пользователь {args.add_user} записан в {args.users}")
        return 0
    if not args.data:
        sys.exit("нужен --data")
    data = Path(args.data)
    if not (data / "cycles.jsonl").exists():
        sys.exit(f"нет {data / 'cycles.jsonl'}: сначала tools/export_cycles.py")
    if auth is not None and not auth.users():
        sys.exit("в файле пользователей никого нет: добавьте через --add-user")
    if args.no_auth and (args.host != "127.0.0.1" or auth is not None):
        sys.exit("--no-auth допустим только на 127.0.0.1 и без --users")
    # токен лучше передавать через LABEL_TOKEN: аргументы командной строки видны другим пользователям в ps.
    # С входом по паролю токен — только если задан явно (удобно для скриптов)
    token = args.token or os.environ.get("LABEL_TOKEN") or ("" if args.no_auth or auth is not None else secrets.token_urlsafe(12))
    if bool(args.cert) != bool(args.key):
        sys.exit("--cert и --key нужны вместе")
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(Store(data), token, secure=bool(args.cert), auth=auth))
    if args.cert:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.load_cert_chain(args.cert, args.key)
        srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
    shown = "localhost" if args.host in ("0.0.0.0", "127.0.0.1") else args.host
    print(f"Разметчик: {'https' if args.cert else 'http'}://{shown}:{args.port}/" + (f"?token={token}" if token else ""))
    print("Данные:", data.resolve(), "| Ctrl+C — остановить")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
