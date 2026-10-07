#!/usr/bin/env python
"""Веб-разметчик циклов, выгруженных tools/export_cycles.py. Только стандартная библиотека.

    python tools/label_server.py --data data/labeling/set1 --host 0.0.0.0 --port 8765

Метки дописываются в <data>/labels.jsonl (последняя метка цикла побеждает), сводка — по /labels.csv.
Доступ — по токену (печатается при запуске; передаётся в ссылке один раз и сохраняется в cookie).
Сервер ничего не отправляет наружу: кадры отдаются только с диска набора. Держите его во внутренней
сети, не пробрасывайте порт в интернет — это записи объекта.
"""
from __future__ import annotations

import argparse
import csv
import hmac
import io
import json
import secrets
import sys
import threading
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
        if rec.get("id") not in known:
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
<a href="/labels.csv" style="color:#8ab4f8">labels.csv</a></header>
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
async function load(){const r=await fetch("/api/cycles");const d=await r.json();all=d.cycles;labels=d.labels;refilter(true)}
function refilter(keep){const f=$("flt").value,cur=list[pos]&&list[pos].id;
 list=all.filter(c=>f=="all"||(f=="todo"&&!labels[c.id])||(f=="valid"&&c.valid)||(f=="rej"&&!c.valid)||(f=="ev"&&c.in_event));
 pos=keep&&cur?Math.max(0,list.findIndex(c=>c.id==cur)):0;show()}
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


def make_handler(store: Store, token: str):
    cycle_ids = lambda: {c["id"] for c in store.cycles()}  # noqa: E731

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
            self.end_headers()
            self.wfile.write(body)

        def _authed(self, query) -> bool:
            if not token:
                return True
            got = query.get("token", [""])[0]
            if not got:
                c = cookies.SimpleCookie(self.headers.get("Cookie", ""))
                got = c["token"].value if "token" in c else ""
            return hmac.compare_digest(got, token)

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if not self._authed(q):
                return self._send(403, "нужен токен: откройте ссылку с ?token=...".encode(), "text/plain; charset=utf-8")
            extra = [("Set-Cookie", f"token={token}; HttpOnly; SameSite=Strict; Path=/")] if token and "token" in q else []
            path = unquote(u.path)
            if path == "/":
                return self._send(200, PAGE.encode(), "text/html; charset=utf-8", extra)
            if path == "/api/cycles":
                body = json.dumps({"cycles": store.cycles(), "labels": store.labels()}, ensure_ascii=False)
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
            if urlparse(self.path).path != "/api/label":
                return self._send(404, b"not found", "text/plain")
            if not self._authed({}):
                return self._send(403, b"forbidden", "text/plain")
            try:
                n = int(self.headers.get("Content-Length", "0"))
                rec = json.loads(self.rfile.read(min(n, 20000)))
            except (ValueError, json.JSONDecodeError):
                return self._send(400, "плохой запрос".encode(), "text/plain; charset=utf-8")
            err = store.add(rec, cycle_ids())
            if err:
                return self._send(400, err.encode(), "text/plain; charset=utf-8")
            return self._send(200, b"{}", "application/json")

    return Handler


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Веб-разметчик циклов")
    p.add_argument("--data", required=True, help="папка набора (результат export_cycles.py)")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 — слушать всю внутреннюю сеть")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--token", help="токен доступа (по умолчанию случайный); пустая строка не допускается")
    p.add_argument("--no-auth", action="store_true", help="без токена (только для localhost)")
    args = p.parse_args(argv)
    data = Path(args.data)
    if not (data / "cycles.jsonl").exists():
        sys.exit(f"нет {data / 'cycles.jsonl'}: сначала tools/export_cycles.py")
    if args.no_auth and args.host != "127.0.0.1":
        sys.exit("--no-auth допустим только на 127.0.0.1")
    token = "" if args.no_auth else (args.token or secrets.token_urlsafe(12))
    srv = ThreadingHTTPServer((args.host, args.port), make_handler(Store(data), token))
    shown = "localhost" if args.host in ("0.0.0.0", "127.0.0.1") else args.host
    print(f"Разметчик: http://{shown}:{args.port}/" + (f"?token={token}" if token else ""))
    print("Данные:", data.resolve(), "| Ctrl+C — остановить")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
