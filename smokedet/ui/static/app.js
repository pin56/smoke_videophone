"use strict";

const $ = (id) => document.getElementById(id);
const video = $("video");
const overlay = $("overlay");
const timeline = $("timeline");

const C = {
  track: "#36d399",
  hold: "#ffcc33",
  event: "#ff4d4f",
  off: "#7d8796",
  flash: "#ff9f43",
  vapor: "#bfe3ff",
  roi: "#4cc9f0",
  muted: "#8a96a8",
  line: "#253044",
};

const SKELETON = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
  [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]];
const EVENT_TAIL = 3;
const FRESH = 3;
const ALERT_FOR = 4;
const MAX_GAP = 0.45;
const KP_MIN = 30;
const D_MAX = 2.0;
const LEAD = 4;
const RANK = { off: 0, track: 1, hold: 2, event: 3 };
const CYCLES = ["цикл", "цикла", "циклов"];
const EPISODES = ["эпизод", "эпизода", "эпизодов"];
const STATE = { idle: "не открыт", queued: "в очереди", tracking: "трекинг", analyzing: "анализ…", ready: "готов", error: "ошибка" };

const opt = { skel: true, rejects: true, pause: false, all: false };
let clips = [];
let clipId = null;
let userChose = false;
let loading = false;
let data = null;
let focusId = null;
let mediaT = 0;
let frameHandle = null;
let lastDrawn = null;
let dirty = true;
let feedKey = "";
let prevT = 0;
let dragging = false;

const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const lerp = (p, q, a) => p + (q - p) * a;

function plural(n, forms = CYCLES) {
  const a = Math.abs(n) % 100;
  const b = a % 10;
  return forms[a > 10 && a < 15 ? 2 : b === 1 ? 0 : b >= 2 && b <= 4 ? 1 : 2];
}

function fmt(t, frac = true) {
  t = Math.max(0, t || 0);
  const m = Math.floor(t / 60);
  const s = t - m * 60;
  return frac ? `${m}:${s.toFixed(1).padStart(4, "0")}` : `${m}:${String(Math.floor(s)).padStart(2, "0")}`;
}

function wallClock(t) {
  const base = data && data.clip.start_time ? Date.parse(data.clip.start_time) : NaN;
  if (Number.isNaN(base)) return `t = ${t.toFixed(1)} с`;
  const d = new Date(base + t * 1000);
  const p = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`;
}

function setText(el, text) {
  if (el.textContent !== text) el.textContent = text;
}

function cut(s, n) {
  return s.length > n ? s.slice(0, n - 1) + "…" : s;
}

const known = (rt, t) => opt.all || (rt !== null && rt <= t);

function lowerBound(arr, v) {
  let lo = 0;
  let hi = arr.length;
  while (lo < hi) {
    const m = (lo + hi) >> 1;
    if (arr[m] < v) lo = m + 1;
    else hi = m;
  }
  return lo;
}

function sizeCanvas(cv) {
  const r = cv.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const w = Math.max(1, Math.round(r.width * dpr));
  const h = Math.max(1, Math.round(r.height * dpr));
  if (cv.width !== w || cv.height !== h) {
    cv.width = w;
    cv.height = h;
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, r.width, r.height);
  return { ctx, w: r.width, h: r.height };
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  if (ctx.roundRect) ctx.roundRect(x, y, w, h, r);
  else ctx.rect(x, y, w, h);
}

function tag(ctx, text, x, y, bg, fg, size, above, maxW, maxH = Infinity) {
  ctx.save();
  ctx.font = `700 ${size}px system-ui, "Segoe UI", sans-serif`;
  const w = ctx.measureText(text).width + 10;
  const h = size + 7;
  let ty = above ? y - h : y;
  ty = Math.max(2, Math.min(ty, maxH - h - 2));
  const tx = clamp(x, 2, Math.max(2, maxW - w - 2));
  ctx.fillStyle = bg;
  roundRect(ctx, tx, ty, w, h, 4);
  ctx.fill();
  ctx.fillStyle = fg;
  ctx.textBaseline = "middle";
  ctx.fillText(text, tx + 5, ty + h / 2 + 0.5);
  ctx.restore();
  return h;
}

function outlined(ctx, text, x, y, color, size = 11) {
  ctx.save();
  ctx.font = `600 ${size}px system-ui, "Segoe UI", sans-serif`;
  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(0, 0, 0, 0.85)";
  ctx.strokeText(text, x, y);
  ctx.fillStyle = color;
  ctx.fillText(text, x, y);
  ctx.restore();
}

function dot(ctx, x, y, r, fill, stroke) {
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
  if (stroke) {
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = stroke;
    ctx.stroke();
  }
}

function prepare(d) {
  d.byTrack = new Map();
  for (const tr of d.tracks) {
    tr.cycles = [];
    tr.events = [];
    tr.flashes = [];
    tr.t0 = tr.t.length ? tr.t[0] : 0;
    tr.t1 = tr.t.length ? tr.t[tr.t.length - 1] : 0;
    d.byTrack.set(tr.id, tr);
  }
  for (const c of d.cycles) d.byTrack.get(c.track_id)?.cycles.push(c);
  for (const e of d.events) d.byTrack.get(e.track_id)?.events.push(e);
  for (const f of d.flashes) d.byTrack.get(f.track_id)?.flashes.push(f);
  d.passed = d.events.filter((e) => e.passed).sort((a, b) => a.start - b.start);
  return d;
}

function sampleAt(tr, t) {
  const ts = tr.t;
  const n = ts.length;
  if (!n || t < ts[0] - MAX_GAP || t > ts[n - 1] + MAX_GAP) return null;
  const lo = Math.max(0, lowerBound(ts, t + 1e-9) - 1);
  const j = Math.min(lo + 1, n - 1);
  if (ts[lo] <= t && j > lo && ts[j] - ts[lo] <= 2 * MAX_GAP) return { i: lo, j, a: (t - ts[lo]) / (ts[j] - ts[lo]) };
  const k = Math.abs(ts[j] - t) < Math.abs(t - ts[lo]) ? j : lo;
  return Math.abs(ts[k] - t) <= MAX_GAP ? { i: k, j: k, a: 0 } : null;
}

function boxAt(tr, s) {
  const p = tr.box[s.i];
  const q = tr.box[s.j];
  return [0, 1, 2, 3].map((k) => lerp(p[k], q[k], s.a));
}

function pointAt(arr, s) {
  const p = arr[s.i];
  const q = arr[s.j];
  if (p && q) return [lerp(p[0], q[0], s.a), lerp(p[1], q[1], s.a)];
  return s.a < 0.5 ? p || q : q || p;
}

function kpAt(tr, s, k) {
  const ci = tr.kc[s.i][k];
  const cj = tr.kc[s.j][k];
  if (Math.max(ci, cj) < KP_MIN) return null;
  const pi = tr.kp[s.i];
  const pj = tr.kp[s.j];
  if (ci < KP_MIN || cj < KP_MIN) {
    const p = ci >= cj ? pi : pj;
    return [p[2 * k], p[2 * k + 1]];
  }
  return [lerp(pi[2 * k], pj[2 * k], s.a), lerp(pi[2 * k + 1], pj[2 * k + 1], s.a)];
}

function dAt(tr, s) {
  const p = tr.d[s.i];
  const q = tr.d[s.j];
  return p !== null && q !== null ? lerp(p, q, s.a) : p ?? q;
}

function stateAt(tr, s, t) {
  const ev = tr.events.find((e) => e.passed && (opt.all ? e.start : e.reveal_t) <= t && t <= e.end + EVENT_TAIL);
  if (ev) return { kind: "event", ev };
  const c = tr.cycles.find((c) => c.hold_start <= t && t <= c.hold_end);
  if (c && tr.valid[s.i]) return { kind: "hold", cycle: c };
  return { kind: tr.valid[s.i] ? "track" : "off" };
}

function phaseAt(tr, t) {
  for (const c of tr.cycles) {
    if (c.start <= t && t < c.hold_start) return { name: "рука идёт ко рту", cycle: c };
    if (c.hold_start <= t && t <= c.hold_end) return { name: `пауза у рта ${(t - c.hold_start).toFixed(1)} с`, cycle: c };
    if (c.hold_end < t && t <= c.end) return { name: "отвод руки", cycle: c };
  }
  return null;
}

const f2 = (v, n = 2) => (v === null || v === undefined ? "—" : v.toFixed(n));

function lastCycle(tr, t) {
  let best = null;
  for (const c of tr.cycles) if (known(c.reveal_t, t) && (!best || c.end > best.end)) best = c;
  return best;
}

function featureLines(c) {
  const th = data.thresholds;
  const out = [];
  out.push([`пауза ${f2(c.hold_end - c.hold_start, 1)} с · мин. d ${f2(c.min_dist)} · кач. ${f2(c.quality)}`, "#cbd5e1"]);
  if (c.appearance !== null) out.push([`CLIP ${f2(c.appearance, 1)} (порог ${th.min_appearance})${c.look ? " · сигарета" : ""}`, c.appearance >= th.min_appearance ? C.track : C.off]);
  const objs = Object.entries(c.objects).sort((a, b) => b[1] - a[1]).slice(0, 2);
  if (objs.length) out.push([`предметы: ${objs.map(([k, v]) => `${k} ${f2(v)}`).join(", ")}`, "#cbd5e1"]);
  if (c.vapor_net !== null) out.push([`дымка ${f2(c.vapor_net)}${c.vapor ? " · выдох" : ""}`, c.vapor ? C.vapor : "#cbd5e1"]);
  if (!c.valid && c.reason_text) out.push([cut(`не цикл: ${c.reason_text}`, 46), C.off]);
  return out;
}

function visibleAt(t) {
  const out = [];
  for (const tr of data.tracks) {
    if (t < tr.t0 - MAX_GAP || t > tr.t1 + MAX_GAP) continue;
    const s = sampleAt(tr, t);
    if (s) out.push({ tr, s, st: stateAt(tr, s, t) });
  }
  return out.sort((a, b) => RANK[a.st.kind] - RANK[b.st.kind]);
}

function ruleState(tr, t) {
  const done = tr.cycles
    .filter((c) => c.valid && c.reveal_t <= t && c.end >= t - data.thresholds.window)
    .sort((a, b) => a.start - b.start);
  let moves = 0;
  done.forEach((c, i) => {
    if (i === 0 || !c.joined_prev) moves += 1;
  });
  const direct = done.find((c) => c.direct) || null;
  const ev = tr.events.find((e) => e.reveal_t <= t && t <= e.end + EVENT_TAIL) || null;
  return { moves, direct, ev };
}

function frameRect(w, h) {
  const vw = data.clip.width;
  const vh = data.clip.height;
  const k = Math.min(w / vw, h / vh);
  return { k, x: (w - vw * k) / 2, y: (h - vh * k) / 2 };
}

function drawOverlay(t, vis) {
  const { ctx, w, h } = sizeCanvas(overlay);
  if (!data) return;
  const f = frameRect(w, h);
  const X = (x) => f.x + x * f.k;
  const Y = (y) => f.y + y * f.k;
  if (data.roi) {
    const polys = () => {
      for (const poly of data.roi) {
        poly.forEach(([x, y], i) => (i ? ctx.lineTo(X(x), Y(y)) : ctx.moveTo(X(x), Y(y))));
        ctx.closePath();
      }
    };
    ctx.save();
    ctx.beginPath();
    ctx.rect(f.x, f.y, data.clip.width * f.k, data.clip.height * f.k);
    polys();
    ctx.fillStyle = "rgba(6, 9, 14, 0.5)";
    ctx.fill("evenodd");
    ctx.beginPath();
    polys();
    ctx.setLineDash([6, 4]);
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = C.roi;
    ctx.stroke();
    ctx.restore();
  }
  for (const v of vis) drawPerson(ctx, X, Y, v, t, w, h);
}

function drawPerson(ctx, X, Y, v, t, maxW, maxH) {
  const { tr, s, st } = v;
  const col = C[st.kind];
  const [x1, y1, x2, y2] = boxAt(tr, s);
  const bx = X(x1);
  const by = Y(y1);
  const bw = X(x2) - bx;
  const bh = Y(y2) - by;
  ctx.save();
  if (st.kind === "off") {
    ctx.globalAlpha = 0.55;
    ctx.setLineDash([4, 4]);
  }
  ctx.lineWidth = { off: 1, track: 2, hold: 2.5, event: 3.5 }[st.kind];
  if (tr.id === focusId || st.kind === "event") {
    ctx.shadowColor = tr.id === focusId && st.kind !== "event" ? C.roi : col;
    ctx.shadowBlur = 14;
  }
  ctx.strokeStyle = col;
  ctx.strokeRect(bx, by, bw, bh);
  ctx.restore();
  if (st.kind !== "off") {
    if (opt.skel && tr.kp) drawSkeleton(ctx, X, Y, tr, s, col);
    drawHandMouth(ctx, X, Y, tr, s, tr.id === focusId);
  }
  let label = `id${tr.id}`;
  if (st.kind === "event") label = `КУРЕНИЕ? ${st.ev.confidence.toFixed(2)} · id${tr.id}`;
  else if (st.kind === "hold") label = `id${tr.id} · рука у рта ${(t - st.cycle.hold_start).toFixed(1)} с`;
  const big = st.kind === "event";
  let ty = by - 3;
  const ph = st.kind !== "off" ? phaseAt(tr, t) : null;
  if (ph) {
    const dn = dAt(tr, s);
    ty -= tag(ctx, `${ph.name}${dn !== null ? ` · d ${dn.toFixed(2)}` : ""}`, bx, ty, "rgba(15, 23, 42, 0.88)", C.hold, 11, true, maxW, maxH) + 2;
  }
  tag(ctx, label, bx, ty, col, big ? "#fff" : "#0a0d12", big ? 14 : 12, true, maxW, maxH);
  if (st.kind !== "off") badges(ctx, tr, t, bx, by + bh + 4, maxW, maxH, tr.id === focusId || st.kind !== "track" ? lastCycle(tr, t) : null);
}

function drawSkeleton(ctx, X, Y, tr, s, col) {
  const pts = [];
  for (let k = 0; k < 17; k++) pts.push(kpAt(tr, s, k));
  ctx.save();
  ctx.globalAlpha = 0.75;
  ctx.strokeStyle = col;
  ctx.lineWidth = 2;
  ctx.lineCap = "round";
  ctx.beginPath();
  for (const [a, b] of SKELETON) {
    if (!pts[a] || !pts[b]) continue;
    ctx.moveTo(X(pts[a][0]), Y(pts[a][1]));
    ctx.lineTo(X(pts[b][0]), Y(pts[b][1]));
  }
  ctx.stroke();
  for (const p of pts) if (p) dot(ctx, X(p[0]), Y(p[1]), 2.2, col);
  ctx.restore();
}

function drawHandMouth(ctx, X, Y, tr, s, focus) {
  const m = pointAt(tr.mouth, s);
  if (!m) return;
  const mx = X(m[0]);
  const my = Y(m[1]);
  const hnd = pointAt(tr.hand, s);
  if (hnd) {
    const hx = X(hnd[0]);
    const hy = Y(hnd[1]);
    const d = dAt(tr, s);
    const near = d !== null && d < data.thresholds.near_enter;
    ctx.save();
    ctx.strokeStyle = near ? C.hold : "rgba(255, 255, 255, 0.75)";
    ctx.lineWidth = near ? 2.5 : 1.5;
    if (!near) ctx.setLineDash([4, 3]);
    ctx.beginPath();
    ctx.moveTo(hx, hy);
    ctx.lineTo(mx, my);
    ctx.stroke();
    ctx.restore();
    dot(ctx, hx, hy, 4, near ? C.hold : "#fff", "#000");
    if (d !== null && (focus || d < data.thresholds.near_exit)) outlined(ctx, `d ${d.toFixed(2)}`, (hx + mx) / 2 + 6, (hy + my) / 2, near ? C.hold : "#fff");
  }
  dot(ctx, mx, my, 3.5, "#fff", "#000");
}

function badges(ctx, tr, t, x, y, maxW, maxH, last) {
  const items = [];
  if (last && t <= last.reveal_t + FRESH * 2 + 2) for (const [text, fg] of featureLines(last)) items.push([text, "rgba(15, 23, 42, 0.88)", fg]);
  for (const c of tr.cycles) {
    if (!(c.reveal_t <= t && t <= c.reveal_t + FRESH)) continue;
    if (c.valid) {
      items.push([c.joined_prev ? "то же движение" : "цикл ✓", C.track]);
      for (const e of c.evidence) items.push([e, C.flash]);
    } else if (opt.rejects) {
      items.push([cut(`не цикл: ${c.reason_text}`, 44), C.off]);
    }
  }
  for (const f of tr.flashes) if (f.start <= t && t <= f.end + 1) items.push(["вспышка", C.flash]);
  for (const c of tr.cycles) {
    if (c.vapor && c.vapor_t !== null && c.vapor_t <= t && t <= c.vapor_t + 1.5) items.push(["выдох", C.vapor]);
  }
  const shown = items.slice(0, 8);
  y = Math.min(y, maxH - shown.length * 21);
  for (const [text, bg, fg] of shown) y += tag(ctx, text, x, y, bg, fg || "#0a0d12", 11, false, maxW, maxH) + 3;
}

const TL_PAD = 10;

function drawTimeline(t) {
  const { ctx, w, h } = sizeCanvas(timeline);
  if (!data) return;
  const dur = Math.max(data.clip.duration || video.duration || 1, 0.1);
  const X = (s) => TL_PAD + (clamp(s, 0, dur) / dur) * (w - 2 * TL_PAD);
  const top = h - 18;
  const until = (s) => (opt.all ? s : Math.min(s, t));

  ctx.font = "10px system-ui, sans-serif";
  ctx.textBaseline = "top";
  const step = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 1800].find((v) => dur / v <= 10) || 3600;
  for (let s = 0; s <= dur + 1e-6; s += step) {
    ctx.fillStyle = C.line;
    ctx.fillRect(X(s), top, 1, 5);
    ctx.fillStyle = C.muted;
    ctx.fillText(fmt(s, false), X(s) + 3, top + 4);
  }
  ctx.fillStyle = C.muted;
  ctx.fillText("циклы", TL_PAD, 2);
  ctx.fillText("эпизоды", TL_PAD, 32);

  for (const c of data.cycles) {
    if (!known(c.reveal_t, t) || (!c.valid && !opt.rejects)) continue;
    const x = X(c.hold_start);
    ctx.fillStyle = c.valid ? C.hold : "rgba(125, 135, 150, 0.6)";
    ctx.fillRect(x, c.valid ? 14 : 18, Math.max(2, X(c.hold_end) - x), c.valid ? 12 : 6);
  }
  for (const f of data.flashes) {
    if (known(f.end, t)) dot(ctx, X(f.start), 20, 3, C.flash);
  }
  for (const e of data.events) {
    if (!known(e.reveal_t, t)) continue;
    const x = X(e.start);
    const ww = Math.max(3, X(until(e.end)) - x);
    if (e.passed) {
      ctx.fillStyle = C.event;
      roundRect(ctx, x, 44, ww, 10, 3);
      ctx.fill();
    } else {
      ctx.strokeStyle = C.flash;
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 0.5, 44.5, ww, 9);
    }
  }
  if (!opt.all) {

    ctx.fillStyle = "rgba(10, 13, 18, 0.6)";
    ctx.fillRect(X(t), 0, w - X(t), top);
  }
  const px = X(t);
  ctx.fillStyle = "#fff";
  ctx.fillRect(px - 1, 0, 2, top);
  ctx.beginPath();
  ctx.moveTo(px - 5, 0);
  ctx.lineTo(px + 5, 0);
  ctx.lineTo(px, 6);
  ctx.fill();
}

function seekFromPointer(e) {
  if (!data) return;
  const r = timeline.getBoundingClientRect();
  const dur = data.clip.duration || video.duration || 0;
  seek(clamp((e.clientX - r.left - TL_PAD) / (r.width - 2 * TL_PAD), 0, 1) * dur);
}

function drawChart(cv, tr, t, col) {
  const { ctx, w, h } = sizeCanvas(cv);
  const W = data.thresholds.window;
  const t0 = t - W;
  const X = (s) => ((s - t0) / W) * w;
  const Y = (d) => 3 + (1 - clamp(d, 0, D_MAX) / D_MAX) * (h - 6);

  for (const c of tr.cycles) {
    if (c.hold_end < t0 || c.hold_start > t) continue;
    const done = c.reveal_t <= t;
    if (done && !c.valid && !opt.rejects) continue;
    ctx.fillStyle = !done ? "rgba(255, 204, 51, 0.14)" : c.valid ? "rgba(255, 204, 51, 0.34)" : "rgba(125, 135, 150, 0.25)";
    ctx.fillRect(X(c.hold_start), 0, X(Math.min(c.hold_end, t)) - X(c.hold_start), h);
  }
  for (const e of tr.events) {
    if (!e.passed || e.reveal_t > t || e.end < t0) continue;
    ctx.fillStyle = C.event;
    ctx.fillRect(X(Math.max(e.start, t0)), 0, X(Math.min(e.end, t)) - X(Math.max(e.start, t0)), 3);
  }
  ctx.save();
  ctx.setLineDash([3, 3]);
  ctx.lineWidth = 1;
  for (const [v, c, name] of [[data.thresholds.near_enter, C.hold, "у рта"], [data.thresholds.near_exit, C.muted, "отвод"]]) {
    ctx.strokeStyle = c;
    ctx.beginPath();
    ctx.moveTo(0, Y(v));
    ctx.lineTo(w, Y(v));
    ctx.stroke();
    ctx.fillStyle = c;
    ctx.font = "10px system-ui, sans-serif";
    ctx.textBaseline = "bottom";
    ctx.fillText(name, w - 34, Y(v) - 1);
  }
  ctx.restore();

  ctx.save();
  ctx.strokeStyle = col;
  ctx.lineWidth = 1.8;
  ctx.lineJoin = "round";
  ctx.beginPath();
  let pen = false;
  for (let i = lowerBound(tr.t, t0); i < tr.t.length && tr.t[i] <= t; i++) {
    const d = tr.d[i];
    if (d === null || !tr.valid[i]) {
      pen = false;
      continue;
    }
    if (i > 0 && tr.t[i] - tr.t[i - 1] > 2 * MAX_GAP) pen = false;
    if (pen) ctx.lineTo(X(tr.t[i]), Y(d));
    else ctx.moveTo(X(tr.t[i]), Y(d));
    pen = true;
  }
  ctx.stroke();
  ctx.restore();
}

function updatePeople(t, vis) {
  const box = $("people");
  const recent = (tr) => tr.cycles.filter((c) => c.valid && c.reveal_t <= t && c.end >= t - data.thresholds.window).length;
  const height = (v) => v.tr.box[v.s.i][3] - v.tr.box[v.s.i][1];
  const cand = vis
    .filter((v) => v.st.kind !== "off")
    .map((v) => ({ v, score: (v.tr.id === focusId ? 1000 : 0) + RANK[v.st.kind] * 10 + recent(v.tr) }))
    .sort((a, b) => b.score - a.score || height(b.v) - height(a.v))
    .slice(0, 3)
    .map((x) => x.v);
  const want = new Set(cand.map((v) => String(v.tr.id)));
  for (const el of [...box.children]) if (!want.has(el.dataset.id)) el.remove();
  if (!cand.length) {
    box.innerHTML = '<div class="empty" data-id="">нет людей в зоне анализа</div>';
    return;
  }
  cand.forEach((v, i) => {
    const id = v.tr.id;
    let el = box.querySelector(`.person[data-id="${id}"]`);
    if (!el) {
      el = document.createElement("div");
      el.className = "person";
      el.dataset.id = String(id);
      el.innerHTML = '<div class="head"><span class="chip"></span><span class="info"></span></div><canvas></canvas>';
      el.addEventListener("click", () => setFocus(id));
    }
    if (box.children[i] !== el) box.insertBefore(el, box.children[i] || null);
    el.classList.toggle("focus", id === focusId);
    const chip = el.querySelector(".chip");
    chip.className = `chip ${v.st.kind}`;
    setText(chip, `id${id}`);
    const r = ruleState(v.tr, t);
    const what = { event: "эпизод", hold: "рука у рта", track: "наблюдение" }[v.st.kind];
    const ev = r.direct ? ` · ${r.direct.evidence.join(", ")}` : "";
    setText(el.querySelector(".info"), `${what} · движений: ${r.moves}/${data.thresholds.min_cycles}${ev}`);
    drawChart(el.querySelector("canvas"), v.tr, t, C[v.st.kind]);
  });
}

function setCond(id, ok, val) {
  const el = $(id);
  el.classList.toggle("ok", ok);
  setText(el.querySelector(".val"), val);
}

function updateRule(t, vis) {
  let tr = focusId !== null ? data.byTrack.get(focusId) || null : null;
  if (!tr) {
    const best = vis
      .filter((v) => v.st.kind !== "off")
      .map((v) => ({ v, r: ruleState(v.tr, t) }))
      .sort((a, b) => !!b.r.ev - !!a.r.ev || b.r.moves - a.r.moves || RANK[b.v.st.kind] - RANK[a.v.st.kind])[0];
    tr = best ? best.v.tr : null;
  }
  const th = data.thresholds;
  setText($("ruleWho"), tr ? `· id${tr.id}` : "");
  const r = tr ? ruleState(tr, t) : { moves: 0, direct: null, ev: null };
  setCond("condCycles", r.moves >= th.min_cycles, `${Math.min(r.moves, th.min_cycles)}/${th.min_cycles}`);
  setCond("condDirect", !!r.direct, r.direct ? cut(r.direct.evidence.join(", ") || "есть", 22) : "нет");
  let cls = "";
  let text = tr ? "наблюдение: циклов пока нет" : "нет людей в зоне анализа";
  if (r.ev && r.ev.passed) {
    cls = "event";
    text = `ЭПИЗОД · уверенность ${r.ev.confidence.toFixed(2)}`;
  } else if (r.ev) {
    cls = "reject";
    text = `кандидат отклонён: ${r.ev.reason_text || `уверенность ${r.ev.confidence.toFixed(2)} < ${th.confidence_threshold}`}`;
  } else if (r.moves >= th.min_cycles || r.direct) {
    cls = "watch";
    text = "правило выполнено — оценка уверенности…";
  } else if (r.moves > 0) {
    cls = "watch";
    text = `${r.moves} ${plural(r.moves)} — нужен ещё цикл или прямой признак`;
  }
  const v = $("verdict");
  setText(v, text);
  v.className = `verdict ${cls}`;
}

function updateFeed(t) {
  const rows = data.feed.filter((r) => known(r.t, t) && (opt.rejects || r.kind !== "reject"));
  const key = `${clipId}|${rows.length}|${opt.rejects}|${opt.all}`;
  if (key === feedKey) return;
  feedKey = key;
  const ol = $("feed");
  ol.innerHTML = "";
  if (!rows.length) {
    ol.innerHTML = '<li class="empty">анализ ещё ничего не нашёл</li>';
    return;
  }
  const frag = document.createDocumentFragment();
  for (const r of rows.slice(-300).reverse()) {
    const li = document.createElement("li");
    li.className = r.kind + (!opt.all && t - r.t < 1.5 ? " new" : "");
    li.innerHTML = `<time>${fmt(r.t)}</time><span class="who">id${r.track_id}</span><span class="txt"></span>`;
    li.querySelector(".txt").textContent = r.text;
    li.addEventListener("click", () => {
      seek(r.t - LEAD);
      setFocus(r.track_id, true);
    });
    frag.appendChild(li);
  }
  ol.appendChild(frag);
}

function updateAlert(t) {
  const el = $("alert");
  const e = data.passed
    .filter((e) => (opt.all ? e.start <= t && t <= e.end : e.reveal_t <= t && t <= e.reveal_t + ALERT_FOR))
    .pop();
  if (!e) {
    el.classList.add("hidden");
    el.dataset.id = "";
    return;
  }
  if (el.dataset.id !== String(e.id)) {
    el.dataset.id = String(e.id);
    el.textContent = `⚠ Эпизод: id${e.track_id} · ${opt.all ? e.basis : e.reveal_basis}`;
    el.classList.remove("hidden");
    el.style.animation = "none";
    void el.offsetWidth;
    el.style.animation = "";
  }
}

function setHTML(el, key, html) {
  if (el.dataset.key === key) return;
  el.dataset.key = key;
  el.innerHTML = html;
}

function updateCalc(t, vis) {
  let tr = focusId !== null ? data.byTrack.get(focusId) || null : null;
  if (!tr) {
    const v = vis.filter((x) => x.st.kind !== "off").sort((a, b) => RANK[b.st.kind] - RANK[a.st.kind])[0];
    tr = v ? v.tr : null;
  }
  const el = $("calc");
  if (!tr) return setHTML(el, "none", '<div class="empty">нет людей в зоне анализа</div>');
  const s = sampleAt(tr, t);
  const th = data.thresholds;
  const rows = [];
  const row = (k, v, cls = "") => rows.push(`<tr class="${cls}"><td>${k}</td><td>${v}</td></tr>`);
  const dn = s ? dAt(tr, s) : null;
  const ph = phaseAt(tr, t);
  row("человек", `id${tr.id}`);
  row("действие", ph ? ph.name : "наблюдение");
  row("d кисть→рот", dn === null ? "—" : `${dn.toFixed(2)} <small>вход ${th.near_enter} · выход ${th.near_exit}</small>`, dn !== null && dn < th.near_enter ? "ok" : "");
  const c = lastCycle(tr, t);
  if (c) {
    row("последний цикл", `${fmt(c.start)} → ${fmt(c.end)} ${c.valid ? "✓" : "✗"}`, c.valid ? "ok" : "bad");
    row("пауза у рта", `${f2(c.hold_end - c.hold_start, 1)} с <small>${th.min_hold}–${th.max_hold}</small>`);
    row("мин. d / качество", `${f2(c.min_dist)} / ${f2(c.quality)}`);
    row("лицо видно", f2(c.face_visible));
    row("наклон головы", f2(c.head_tilt));
    row("CLIP", c.appearance === null ? "—" : `${f2(c.appearance, 1)} <small>порог ${th.min_appearance} · сигарета ${th.direct_appearance}</small>`, c.appearance !== null && c.appearance >= th.min_appearance ? "ok" : "bad");
    const objs = Object.entries(c.objects).sort((a, b) => b[1] - a[1]);
    row("предметы у рта", objs.length ? objs.map(([k, v]) => `${k} ${f2(v)}`).join(", ") : "нет");
    row("«чужие» кадры", f2(c.neg_fraction));
    row("вспышка / дымка", `${c.flash ? "есть" : "нет"} / ${c.vapor_net === null ? "—" : f2(c.vapor_net)}${c.vapor ? " (выдох)" : ""}`);
    if (c.rescued_from) row("спасён CLIP", c.rescued_from);
    if (!c.valid) row("отказ", c.reason_text || c.reason, "bad");
  } else row("циклов пока", "нет");
  const r = ruleState(tr, t);
  row("движений в окне", `${r.moves}/${th.min_cycles}`, r.moves >= th.min_cycles ? "ok" : "");
  setHTML(el, `${tr.id}|${dn === null ? "" : dn.toFixed(2)}|${ph ? ph.name : ""}|${c ? c.id : ""}|${r.moves}`, `<table>${rows.join("")}</table>`);
}

function updateClipStats() {
  const el = $("clipStats");
  const cs = data.cycles;
  const ok = cs.filter((c) => c.valid).length;
  const reasons = {};
  for (const c of cs) if (!c.valid) reasons[c.reason_text || c.reason] = (reasons[c.reason_text || c.reason] || 0) + 1;
  const top = Object.entries(reasons).sort((a, b) => b[1] - a[1]).slice(0, 4);
  const ev = data.events;
  const rows = [
    ["длительность", fmt(data.clip.duration, false)],
    ["людей отслежено", data.stats.tracks],
    ["касаний рта", cs.length],
    ["засчитано циклов", ok],
    ["отклонено", cs.length - ok],
    ["эпизодов принято", ev.filter((e) => e.passed).length],
    ["кандидатов отклонено", ev.filter((e) => !e.passed).length],
    ["вспышек зажигалки", data.flashes.length],
  ];
  let html = "<table>" + rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("") + "</table>";
  if (top.length) html += '<div class="why">' + top.map(([k, n]) => `<div>${n}× ${k}</div>`).join("") + "</div>";
  const best = data.passed.map((e) => `<div>id${e.track_id} · ${fmt(e.start)}–${fmt(e.end)} · ${e.confidence.toFixed(2)}</div>`).join("");
  if (best) html += `<div class="why ev">${best}</div>`;
  el.innerHTML = html;
}

function render(t) {
  if (!data) {
    sizeCanvas(overlay);
    sizeCanvas(timeline);
    return;
  }
  const vis = visibleAt(t);
  drawOverlay(t, vis);
  drawTimeline(t);
  updatePeople(t, vis);
  updateRule(t, vis);
  updateCalc(t, vis);
  updateFeed(t);
  updateAlert(t);
  setText($("stPeople"), String(vis.filter((v) => v.st.kind !== "off").length));
  setText($("stCycles"), String(data.cycles.filter((c) => c.valid && known(c.reveal_t, t)).length));
  setText($("stEvents"), String(data.passed.filter((e) => known(e.reveal_t, t)).length));
  setText($("clock"), wallClock(t));
  setText($("time"), `${fmt(t)} / ${fmt(data.clip.duration || video.duration, false)}`);
  if (opt.pause && !video.paused && t > prevT && t - prevT < 1) {
    if (data.passed.some((e) => prevT < e.reveal_t && e.reveal_t <= t)) video.pause();
  }
  prevT = t;
}

function onVideoFrame(now, meta) {
  mediaT = meta.mediaTime;
  frameHandle = video.requestVideoFrameCallback(onVideoFrame);
}

function watchFrames() {
  if (!("requestVideoFrameCallback" in video)) return;
  if (frameHandle !== null) video.cancelVideoFrameCallback(frameHandle);
  frameHandle = video.requestVideoFrameCallback(onVideoFrame);
}

function loop() {
  const playing = !video.paused && !video.seeking && frameHandle !== null;
  const t = playing ? mediaT : video.currentTime;
  if (dirty || t !== lastDrawn) {
    render(t);
    lastDrawn = t;
    dirty = false;
  }
  requestAnimationFrame(loop);
}

function seek(t) {
  if (!data) return;
  const end = Math.min(video.duration || Infinity, data.clip.duration || Infinity);
  video.currentTime = clamp(t, 0, Number.isFinite(end) ? Math.max(0, end - 0.05) : 0);
  mediaT = video.currentTime;
  feedKey = "";
  dirty = true;
}

function setFocus(id, keep = false) {
  focusId = focusId === id && !keep ? null : id;
  dirty = true;
}

function togglePlay() {
  if (!data) return;
  if (video.paused) video.play();
  else video.pause();
}

function jumpEvent(dir) {
  if (!data || !data.passed.length) return;
  const t = video.currentTime;
  const starts = data.passed.map((e) => Math.max(0, e.start - LEAD));
  const target = dir > 0 ? starts.find((s) => s > t + 0.5) : [...starts].reverse().find((s) => s < t - 1);
  if (target !== undefined) seek(target);
}

function showWait(c) {
  const w = $("wait");
  if (c && c.state === "ready" && data) {
    w.classList.add("hidden");
    return;
  }
  w.classList.remove("hidden");
  let msg = "Нет клипов";
  let pct = 0;
  if (c) {
    if (c.state === "error") msg = `${c.id}: ошибка — ${c.error}`;
    else if (c.state === "tracking") {
      pct = Math.round(c.progress * 100);
      msg = `${c.id}: трекинг людей и поз… ${pct}%`;
    } else if (c.state === "analyzing") {
      pct = 100;
      msg = `${c.id}: анализ циклов, проверка предметов, события…`;
    } else if (c.state === "ready") {
      pct = 100;
      msg = `${c.id}: загрузка…`;
    } else msg = `${c.id}: ${c.state === "idle" ? "запускаю анализ…" : "в очереди на обработку"}`;
  }
  setText(w.querySelector(".msg"), msg);
  w.querySelector(".bar i").style.width = `${pct}%`;
}

function clipLabel(c) {
  if (c.state === "ready") return `${c.id} · ${c.events ? `${c.events} ${plural(c.events, EPISODES)}` : "без эпизодов"}`;
  if (c.state === "tracking") return `${c.id} · трекинг ${Math.round(c.progress * 100)}%`;
  return `${c.id} · ${STATE[c.state] || c.state}`;
}

async function loadClip(id) {
  clipId = id;
  data = null;
  focusId = null;
  feedKey = "";
  const keepT = (location.hash.match(/[#&]clip=([^&]+)/) || [])[1] === encodeURIComponent(id);
  if (!keepT) history.replaceState(null, "", `#clip=${encodeURIComponent(id)}`);
  $("clip").value = id;
  $("feed").innerHTML = "";
  $("people").innerHTML = "";
  $("alert").classList.add("hidden");
  video.pause();
  video.removeAttribute("src");
  video.load();
  dirty = true;
  const c = clips.find((x) => x.id === id);
  showWait(c);
  if (c && c.state === "idle") fetch(`/api/clips/${encodeURIComponent(id)}`, { cache: "no-store" }).catch(() => {});
  if (!c || c.state !== "ready") return;
  loading = true;
  try {
    const r = await fetch(`/api/clips/${encodeURIComponent(id)}`, { cache: "no-store" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const d = prepare(await r.json());
    if (clipId !== id) return;
    data = d;
    const cl = d.clip;
    $("clipMeta").textContent = `${cl.camera_id} · ${cl.video_name} · ${cl.width}×${cl.height} · ${fmt(cl.duration, false)}` +
      ` · людей ${d.stats.tracks} · циклов ${d.stats.cycles}`;
    $("winSec").textContent = String(d.thresholds.window);
    updateClipStats();
    $("condCycles").querySelector("b").textContent = `≥ ${d.thresholds.min_cycles} ${plural(d.thresholds.min_cycles)} за ${d.thresholds.window} с`;
    const direct = ["сигарета, вейп, дым"];
    if (d.thresholds.flash_as_direct) direct.push("вспышка зажигалки");
    if (d.thresholds.vapor_as_direct) direct.push("выдох");
    $("condDirect").querySelector("small").textContent = direct.join(", ");
    video.src = cl.video_url;
    video.playbackRate = Number($("speed").value);
    const start = Number((location.hash.match(/[#&]t=([\d.]+)/) || [])[1]);
    if (start > 0) video.addEventListener("loadedmetadata", () => seek(start), { once: true });
    prevT = 0;
    showWait(c);
  } catch (e) {
    setText($("wait").querySelector(".msg"), `${id}: не удалось загрузить данные (${e.message})`);
  } finally {
    loading = false;
    dirty = true;
  }
}

async function pollClips() {
  try {
    const r = await fetch("/api/clips", { cache: "no-store" });
    clips = await r.json();
    const sel = $("clip");
    const sig = clips.map(clipLabel).join("|");
    if (sel.dataset.sig !== sig) {
      sel.dataset.sig = sig;
      sel.innerHTML = "";
      for (const c of clips) sel.add(new Option(clipLabel(c), c.id));
      if (clipId !== null) sel.value = clipId;
    }
    const cur = clips.find((c) => c.id === clipId);
    if (clipId === null || (!userChose && !data && !loading && cur && cur.state !== "ready")) {

      const want = decodeURIComponent((location.hash.match(/[#&]clip=([^&]+)/) || [])[1] || "");
      const pick = clips.find((c) => c.id === want) || clips.find((c) => c.state === "ready") || clips[0];
      if (want && pick && pick.id === want) userChose = true;
      if (pick && (pick.id !== clipId || pick.state === "ready")) loadClip(pick.id);
      else showWait(cur);
    } else if (cur && cur.state === "ready" && !data && !loading) {
      loadClip(clipId);
    } else if (!data) {
      showWait(cur);
    }
  } catch (e) {

  }
  const busy = !clips.length || clips.some((c) => c.state !== "ready" && c.state !== "error");
  setTimeout(pollClips, busy ? 1000 : 5000);
}

$("clip").addEventListener("change", (e) => {
  userChose = true;
  loadClip(e.target.value);
});
$("play").addEventListener("click", togglePlay);
$("prevEv").addEventListener("click", () => jumpEvent(-1));
$("nextEv").addEventListener("click", () => jumpEvent(1));
$("speed").addEventListener("change", (e) => {
  video.playbackRate = Number(e.target.value);
});
for (const [id, key] of [["optSkel", "skel"], ["optRejects", "rejects"], ["optPause", "pause"], ["optAll", "all"]]) {
  $(id).checked = opt[key];
  $(id).addEventListener("change", (e) => {
    opt[key] = e.target.checked;
    feedKey = "";
    dirty = true;
  });
}

video.addEventListener("play", () => {
  $("play").textContent = "❚❚";
  watchFrames();
});
video.addEventListener("pause", () => {
  $("play").textContent = "▶";
  dirty = true;
});
video.addEventListener("ended", () => {
  $("play").textContent = "▶";
});
video.addEventListener("loadeddata", () => {
  watchFrames();
  dirty = true;
});
video.addEventListener("seeked", () => {
  dirty = true;
});
video.addEventListener("error", () => {
  if (data) setText($("clipMeta"), "браузер не смог проиграть видео: нужен H.264 в mp4 (сервер перекодирует его, если найдёт ffmpeg)");
});

timeline.addEventListener("pointerdown", (e) => {
  dragging = true;
  timeline.setPointerCapture(e.pointerId);
  seekFromPointer(e);
});
timeline.addEventListener("pointermove", (e) => {
  if (dragging) seekFromPointer(e);
});
timeline.addEventListener("pointerup", () => {
  dragging = false;
});

overlay.addEventListener("click", (e) => {
  if (!data) return;
  const r = overlay.getBoundingClientRect();
  const f = frameRect(r.width, r.height);
  const x = (e.clientX - r.left - f.x) / f.k;
  const y = (e.clientY - r.top - f.y) / f.k;
  const area = (b) => (b[2] - b[0]) * (b[3] - b[1]);
  const hit = visibleAt(video.currentTime)
    .map((v) => ({ v, b: boxAt(v.tr, v.s) }))
    .filter(({ b }) => x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3])
    .sort((a, b) => area(a.b) - area(b.b))[0];
  if (hit) setFocus(hit.v.tr.id);
  else togglePlay();
});

document.addEventListener("keydown", (e) => {
  if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
  if (e.code === "Space") {
    e.preventDefault();
    togglePlay();
  } else if (e.code === "ArrowLeft") seek(video.currentTime - (e.shiftKey ? 0.1 : 5));
  else if (e.code === "ArrowRight") seek(video.currentTime + (e.shiftKey ? 0.1 : 5));
  else if (e.code === "KeyN") jumpEvent(1);
  else if (e.code === "KeyP") jumpEvent(-1);
  else if (e.code === "KeyA") $("optAll").click();
});

window.addEventListener("resize", () => {
  dirty = true;
});

pollClips();
requestAnimationFrame(loop);
