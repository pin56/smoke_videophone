

"use strict";

const $ = (id) => document.getElementById(id);
const pad = (n) => String(n).padStart(2, "0");
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const hms = (sec) => { const d = new Date(sec * 1000); return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`; };
const hm = (sec) => { const d = new Date(sec * 1000); return `${pad(d.getHours())}:${pad(d.getMinutes())}`; };
const day = (sec) => new Date(sec * 1000).toLocaleDateString("ru-RU", { day: "numeric", month: "short" }).replace(".", "");
const dayTime = (sec) => `${day(sec)} ${hm(sec)}`;
const stamp = (sec) => { const d = new Date(sec * 1000); return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${hms(sec)}`; };
const mmss = (sec) => { sec = Math.max(0, Math.round(sec)); const h = Math.floor(sec / 3600); const m = Math.floor((sec % 3600) / 60); return (h ? `${h}:${pad(m)}` : pad(m)) + `:${pad(sec % 60)}`; };
const pct = (v) => `${Math.round(v * 100)}%`;
const now = () => Date.now() / 1000;
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
function plural(n, one, few, many) {
  const m10 = n % 10, m100 = n % 100;
  if (m10 === 1 && m100 !== 11) return one;
  if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few;
  return many;
}
const cyclesText = (n) => `${n} ${plural(n, "цикл", "цикла", "циклов")}`;
const mb = (bytes) => (bytes >= 1 << 30 ? `${(bytes / 2 ** 30).toFixed(1)} ГБ` : `${(bytes / 2 ** 20).toFixed(1)} МБ`);
async function api(url, opts) {
  const r = await fetch(url, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error || `${r.status}`);
  return body;
}

const HEALTH_TEXT = { online: "В сети", no_signal: "Нет сигнала", delay: "Задержка", analyzing: "Анализируется", error: "Ошибка" };
const DET_BADGE = { smoking: "Авто: похоже на курение", suspect: "Авто: подозрение" };
const REVIEW_TEXT = { pending: "Не проверено", confirmed: "Подтверждено оператором", rejected: "Отклонено оператором" };
const JOB_TEXT = { queued: "В очереди", analyzing: "Анализируется", done: "Готово", error: "Ошибка" };

let CONFIG = { retention_days: 7, pre_roll: 10, post_roll: 10, fps: 10, analyzer: "", analyzer_problem: "", ffmpeg: false };
let page = "monitor";
let mode = "live";
let filter = "all";
const cams = [];
const byId = new Map();
let camEvents = [];
let journalVersion = -1;
let serverT = now();

function showPage(hash) {
  const [name, arg] = hash.split("/");
  page = ["monitor", "check", "journal"].includes(name) ? name : "monitor";
  if (page === "journal" && arg) Player.open(+arg);
  if (page === "check" && arg) Jobs.selected = decodeURIComponent(arg);
  if (page === "monitor" && arg === "archive" && mode !== "archive") {
    for (const x of $("mode").children) x.classList.toggle("on", x.dataset.m === "archive");
    Archive.enter();
  }
  for (const p of ["monitor", "check", "journal"]) $(`page-${p}`).classList.toggle("hidden", p !== page);
  for (const a of document.querySelectorAll(".tabs a")) a.classList.toggle("on", a.dataset.page === page);
  if (page === "journal") loadJournal();
  if (page === "check") loadJobs();
  if (page === "monitor") requestAnimationFrame(() => cams.forEach((c) => c.view.reset()));
}
window.addEventListener("hashchange", () => showPage(location.hash.slice(1)));

function segment(el, attr, onPick) {
  el.addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    for (const x of el.querySelectorAll("button")) x.classList.toggle("on", x === b);
    onPick(b.dataset[attr], b);
  });
}

function buildTiles(list) {
  const grid = $("grid");
  for (const info of list) {
    const el = document.createElement("article");
    el.className = "tile";
    el.tabIndex = 0;
    el.innerHTML = `
      <div class="screen">
        <canvas></canvas>
        <span class="cam-id">${esc(info.id)}</span>
        <span class="badge hidden"></span>
        <span class="screen-note hidden"></span>
        <div class="hud"><span class="ts"></span><span class="hud-r"></span></div>
      </div>
      <div class="meta">
        <div class="name"><b>${esc(info.name)}</b><small>${esc(info.zone)}</small></div>
        <div class="kpi"><b class="kv">0</b><small> / 20 с</small></div>
      </div>
      <div class="health"><span class="h-status"></span><span class="h-detail"></span></div>`;
    el.setAttribute("aria-label", `${info.id}, ${info.name}`);
    const cam = {
      info, el, state: null, frame: null, view: new Scene.View(info),
      canvas: el.querySelector("canvas"), badge: el.querySelector(".badge"), note: el.querySelector(".screen-note"),
      ts: el.querySelector(".ts"), hudR: el.querySelector(".hud-r"), kv: el.querySelector(".kv"),
      hStatus: el.querySelector(".h-status"), hDetail: el.querySelector(".h-detail"),
      det: "unknown", health: "online",
    };
    el.addEventListener("click", () => openCam(cam));
    el.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openCam(cam); } });
    grid.appendChild(el);
    cams.push(cam);
    byId.set(info.id, cam);
  }
}

function tileView(cam) {
  if (mode === "archive") {
    const fr = Archive.frameAt(cam.info.id, Archive.cursor);
    if (!fr) return { det: "unknown", health: "gap", frame: null };
    const states = fr[1].map((r) => Scene.STATES[r[7]]);
    const det = states.includes("smoking") ? "smoking" : states.includes("suspect") ? "suspect" : "normal";
    return { det, health: "online", frame: fr };
  }
  const st = cam.state || {};
  const health = st.health || "online";
  let det = st.detection || "unknown";
  if (det !== "unknown" && cam.frame) {
    const states = cam.frame[1].map((r) => Scene.STATES[r[7]]);
    det = states.includes("smoking") ? "smoking" : states.includes("suspect") ? "suspect" : "normal";
  }
  return { det, health, frame: cam.frame };
}

function drawTile(cam) {
  const v = tileView(cam);
  cam.det = v.det; cam.health = v.health;
  const st = cam.state || {};
  if (v.health === "gap") {
    cam.view.blank(cam.canvas, "Нет записи", "камера не передавала кадры в этот момент");
  } else if (mode === "live" && (v.health === "no_signal" || v.health === "error")) {
    cam.view.blank(cam.canvas, v.health === "error" ? "Ошибка" : "Нет сигнала",
      [st.last_frame ? `последний кадр ${dayTime(st.last_frame)}:${pad(new Date(st.last_frame * 1000).getSeconds())}` : "кадров не было", st.reason || ""]);
  } else if (v.frame) {
    cam.view.draw(cam.canvas, v.frame[0], v.frame[1]);
  }
  const blank = v.health === "gap" || (mode === "live" && (v.health === "no_signal" || v.health === "error"));
  cam.ts.textContent = v.frame && !blank ? stamp(v.frame[0]) : "";
  cam.hudR.classList.toggle("hidden", blank);
  if (cam.det !== cam.el.dataset.det) { updateTileMeta(cam); applyLayout(); }
}

function updateTileMeta(cam) {
  const st = cam.state || {};
  const live = mode === "live";
  cam.el.dataset.det = cam.det;
  cam.el.dataset.health = cam.health;
  cam.badge.classList.toggle("hidden", !DET_BADGE[cam.det]);
  cam.badge.dataset.s = cam.det;
  cam.badge.textContent = DET_BADGE[cam.det] || "";
  const observing = cam.det !== "unknown";
  cam.kv.textContent = live ? (observing ? st.recent ?? 0 : "—") : "—";
  cam.kv.parentElement.classList.toggle("hidden", !live);

  let note = "";
  if (live && cam.health === "analyzing") note = "Анализируется: выводов по кадрам пока нет";
  cam.note.textContent = note;
  cam.note.classList.toggle("hidden", !note);

  if (!live) {
    cam.hudR.textContent = "архив · 5 к/с";
    cam.hStatus.textContent = cam.health === "gap" ? "Нет записи" : "Запись";
    cam.hStatus.className = `h-status h-${cam.health === "gap" ? "no_signal" : "online"}`;
    cam.hDetail.textContent = "";
    return;
  }
  cam.hudR.textContent = cam.health === "delay" ? `задержка ${Math.round(st.lag)} с` : "10 к/с";
  cam.hudR.classList.toggle("lag", cam.health === "delay");
  cam.hStatus.textContent = HEALTH_TEXT[cam.health] + (cam.health === "delay" ? ` ${Math.round(st.lag)} с` : "");
  cam.hStatus.className = `h-status h-${cam.health}`;
  const parts = [];
  if (cam.health !== "online") {
    if (st.last_frame) parts.push(`последний кадр ${hms(st.last_frame)}`);
    if (st.reason) parts.push(st.reason);
  }
  const ev = activeEvent(cam);
  if (ev) parts.push(`эпизод №${ev.id}: ${ev.review === "pending" ? "ждёт проверки" : REVIEW_TEXT[ev.review].toLowerCase()}`);
  cam.hDetail.textContent = parts.join(" · ");
}

function activeEvent(cam) {
  const ids = (cam.state?.persons || []).map((p) => p.event_id).filter(Boolean);
  return camEvents.find((e) => ids.includes(e.id)) || null;
}

const ORDER = { smoking: 0, suspect: 1, unknown: 2, normal: 3 };
function applyLayout() {
  const first = $("alarmsFirst").checked;
  const counts = { smoking: 0, suspect: 0, normal: 0, problem: 0 };
  for (const cam of cams) {
    const problem = mode === "live" ? cam.health !== "online" : cam.health === "gap";
    if (cam.det === "smoking") counts.smoking++;
    else if (cam.det === "suspect") counts.suspect++;
    else if (cam.det === "normal") counts.normal++;
    if (problem) counts.problem++;
    const show = filter === "all" || (filter === "problem" ? problem : cam.det === filter);
    cam.el.classList.toggle("hidden", !show);
    cam.el.style.order = first ? ORDER[cam.det] * 100 + (problem ? 50 : 0) + cam.info.scene : cam.info.scene;
  }
  $("nSmoking").textContent = counts.smoking;
  $("nSuspect").textContent = counts.suspect;
  $("nNormal").textContent = counts.normal;
  $("nProblem").textContent = counts.problem;
}

segment($("filter"), "f", (f) => { filter = f; applyLayout(); });
$("alarmsFirst").addEventListener("change", applyLayout);
segment($("cols"), "c", (c) => document.documentElement.style.setProperty("--cols", c));
segment($("mode"), "m", (m) => (m === "archive" ? Archive.enter() : Archive.leave()));

let source = null;
function connect() {
  source = new EventSource("/api/live");
  source.addEventListener("frame", (e) => {
    const data = JSON.parse(e.data);
    serverT = data.t;
    for (const [id, t, rows] of data.cams) {
      const cam = byId.get(id);
      if (cam) cam.frame = t === null ? null : [t, rows];
    }
    if (mode !== "live" || page !== "monitor") { if (CamModal.cam) CamModal.draw(); return; }
    for (const cam of cams) if (!cam.el.classList.contains("hidden")) drawTile(cam);
    if (CamModal.cam) CamModal.draw();
  });
  source.addEventListener("state", (e) => {
    const data = JSON.parse(e.data);
    for (const st of data.cameras) { const cam = byId.get(st.id); if (cam) cam.state = st; }
    if (mode === "live") for (const cam of cams) { const v = tileView(cam); cam.det = v.det; cam.health = v.health; updateTileMeta(cam); }
    applyLayout();
    if (data.journal !== journalVersion) { journalVersion = data.journal; onJournalChanged(); }
    if (CamModal.cam) CamModal.render();
    $("conn").textContent = "";
  });
  source.onerror = () => { $("conn").textContent = "нет связи с сервером"; };
}

const Archive = {
  CHUNK: 60, STEP: 2,
  chunks: new Map(), loading: new Set(),
  from: 0, to: 0, viewFrom: 0, viewTo: 0, cursor: 0, playing: true, speed: 1, marks: [], drawnAt: -1,

  async enter() {
    mode = "archive";
    $("archiveBar").classList.remove("hidden");
    document.body.classList.add("archive-mode");
    await this.refreshRange();
    this.setPeriod(+$("arPeriod").value, true);
    this.playing = true;
    $("arPlay").textContent = "Пауза";
    cams.forEach((c) => c.view.reset());
  },

  leave() {
    mode = "live";
    $("archiveBar").classList.add("hidden");
    document.body.classList.remove("archive-mode");
    for (const x of $("mode").children) x.classList.toggle("on", x.dataset.m === "live");
    cams.forEach((c) => { c.view.reset(); const v = tileView(c); c.det = v.det; c.health = v.health; updateTileMeta(c); });
    applyLayout();
  },

  async openAt(t) {
    for (const x of $("mode").children) x.classList.toggle("on", x.dataset.m === "archive");
    if (location.hash !== "#monitor") location.hash = "#monitor";
    await this.enter();
    if (t < this.viewFrom) { $("arPeriod").value = "1800"; this.setPeriod(1800, false); }
    this.seek(t);
  },

  async refreshRange() {
    const r = await api("/api/archive/range");
    this.from = r.from; this.to = r.to;
  },

  setPeriod(sec, resetCursor) {
    this.viewTo = this.to;
    this.viewFrom = Math.max(this.from, this.to - sec);
    if (resetCursor || this.cursor < this.viewFrom || this.cursor > this.viewTo) this.cursor = Math.max(this.viewFrom, this.viewTo - 60);
    this.renderScale();
    this.loadMarks();
  },

  seek(t) {
    this.cursor = clamp(t, this.viewFrom, this.viewTo);
    cams.forEach((c) => c.view.reset());
    this.drawnAt = -1;
  },

  async loadMarks() {
    const list = await api(`/api/events?source_type=camera&since=${this.viewFrom}&until=${this.viewTo}&limit=500`).catch(() => []);
    this.marks = list;
    const span = this.viewTo - this.viewFrom || 1;
    $("arMarks").innerHTML = list.map((e) => {
      const a = clamp((e.started_at - this.viewFrom) / span, 0, 1), b = clamp((e.started_at + Math.max(e.duration, 2) - this.viewFrom) / span, 0, 1);
      const title = `${e.source_id} · ${hms(e.started_at)} · ${cyclesText(e.cycles)} · авто, ${REVIEW_TEXT[e.review].toLowerCase()}`;
      return `<button class="ar-mark r-${e.review}" data-id="${e.id}" style="left:${(a * 100).toFixed(3)}%;width:${Math.max(0.4, (b - a) * 100).toFixed(3)}%" title="${esc(title)}" aria-label="${esc(title)}"></button>`;
    }).join("");
  },

  renderScale() {
    const span = this.viewTo - this.viewFrom;
    const step = span > 1200 ? 300 : span > 400 ? 120 : 60;
    const first = Math.ceil(this.viewFrom / step) * step;
    let html = "";
    for (let t = first; t <= this.viewTo; t += step) html += `<span style="left:${((t - this.viewFrom) / span) * 100}%">${hm(t)}</span>`;
    $("arScale").innerHTML = html;
  },

  key(t) { return Math.floor(t / this.CHUNK); },

  ensure(t) {
    const k = this.key(t);
    const c = this.chunks.get(k);
    const stale = c && c.partial && now() - c.fetched > 4;
    if ((c && !stale) || this.loading.has(k)) return;
    this.loading.add(k);
    const a = k * this.CHUNK, b = a + this.CHUNK;
    fetch(`/api/archive?from=${a}&to=${b}&step=${this.STEP}`).then((r) => r.json()).then((data) => {
      this.chunks.set(k, { cams: data.cams, partial: b > this.to - 1, fetched: now() });
      if (this.chunks.size > 40) {
        const far = [...this.chunks.keys()].sort((x, y) => Math.abs(x - this.key(this.cursor)) - Math.abs(y - this.key(this.cursor))).slice(40);
        far.forEach((x) => this.chunks.delete(x));
      }
      this.drawnAt = -1;
    }).catch(() => {}).finally(() => this.loading.delete(k));
  },

  frameAt(camId, t) {
    const c = this.chunks.get(this.key(t));
    const frames = c?.cams[camId];
    if (!frames || !frames.length) return null;
    let lo = 0, hi = frames.length - 1, best = -1;
    while (lo <= hi) { const mid = (lo + hi) >> 1; if (frames[mid][0] <= t) { best = mid; lo = mid + 1; } else hi = mid - 1; }
    if (best < 0 || t - frames[best][0] > 1) return null;
    return frames[best];
  },

  tick(dt) {
    if (mode !== "archive") return;
    if (this.playing) {
      this.cursor += dt * this.speed;
      if (this.cursor >= this.viewTo) {
        this.cursor = this.viewTo;
        this.playing = false;
        $("arPlay").textContent = "Смотреть";
        $("arNote").textContent = "конец выбранного периода";
      }
    }
    this.ensure(this.cursor);
    if (this.cursor % this.CHUNK > this.CHUNK - 15) this.ensure(this.cursor + this.CHUNK);
    const frameT = Math.floor(this.cursor * 5) / 5;
    const loaded = this.chunks.has(this.key(this.cursor));
    if (frameT !== this.drawnAt && loaded && page === "monitor") {
      this.drawnAt = frameT;
      for (const cam of cams) {
        drawTile(cam);
        updateTileMeta(cam);
      }
      applyLayout();
    }
    const span = this.viewTo - this.viewFrom || 1;
    $("arCursor").style.left = `${((this.cursor - this.viewFrom) / span) * 100}%`;
    $("arTime").textContent = `${day(this.cursor)}, ${hms(this.cursor)}`;
    if (this.playing) $("arNote").textContent = loaded ? "" : "загрузка записи…";
  },
};

$("arPlay").addEventListener("click", () => {
  if (Archive.cursor >= Archive.viewTo - 0.5) Archive.seek(Archive.viewFrom);
  Archive.playing = !Archive.playing;
  $("arPlay").textContent = Archive.playing ? "Пауза" : "Смотреть";
  $("arNote").textContent = "";
});
segment($("arSpeed"), "s", (s) => { Archive.speed = +s; });
$("arPeriod").addEventListener("change", async () => { await Archive.refreshRange(); Archive.setPeriod(+$("arPeriod").value, false); });
$("arLive").addEventListener("click", () => Archive.leave());
(() => {
  const tl = $("arTimeline");
  const seekTo = (e) => {
    const r = tl.getBoundingClientRect();
    Archive.seek(Archive.viewFrom + clamp((e.clientX - r.left) / r.width, 0, 1) * (Archive.viewTo - Archive.viewFrom));
  };
  let dragging = false;
  tl.addEventListener("pointerdown", (e) => {
    const mark = e.target.closest(".ar-mark");
    if (mark) { Player.open(+mark.dataset.id); return; }
    dragging = true; tl.setPointerCapture(e.pointerId); seekTo(e);
  });
  tl.addEventListener("pointermove", (e) => { if (dragging) seekTo(e); });
  tl.addEventListener("pointerup", () => { dragging = false; });
  tl.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") { Archive.seek(Archive.cursor - 10); e.preventDefault(); }
    if (e.key === "ArrowRight") { Archive.seek(Archive.cursor + 10); e.preventDefault(); }
    if (e.key === " ") { $("arPlay").click(); e.preventDefault(); }
  });
})();
setInterval(async () => {
  if (mode !== "archive") return;
  const atEnd = Archive.viewTo - Archive.cursor < 2;
  await Archive.refreshRange().catch(() => {});
  if (atEnd || Archive.to - Archive.viewTo > 30) { Archive.setPeriod(+$("arPeriod").value, false); }
}, 5000);

function onJournalChanged() {
  loadCamEvents();
  if (page === "journal") loadJournal();
  if (page === "check" || Jobs.active()) loadJobs();
  if (mode === "archive") Archive.loadMarks();
  if (Player.ev) Player.refresh();
}

async function loadCamEvents() {
  camEvents = await api("/api/events?source_type=camera&limit=40").catch(() => camEvents);
  renderCamEvents();
  const pending = await api("/api/events?review=pending&limit=1000").catch(() => null);
  if (pending) {
    const n = pending.filter((e) => !e.active).length;
    $("tabPending").textContent = n ? String(n) : "";
    $("newCount").textContent = n ? `ждут проверки: ${n}` : "";
  }
  for (const cam of cams) updateTileMeta(cam);
}

function reviewButtons(ev, compact) {
  if (ev.review === "pending") {
    return `<button class="btn btn--alarm" data-review="confirmed" data-id="${ev.id}">Подтвердить</button>
      <button class="btn" data-review="rejected" data-id="${ev.id}">Отклонить</button>`;
  }
  return compact ? "" : `<button class="btn btn--ghost" data-review="pending" data-id="${ev.id}">Снять решение</button>`;
}

function clipText(ev) {
  if (ev.clip_status === "saved") return `хранится до ${dayTime(ev.clip_expires_at)}`;
  if (ev.clip_status === "deleted") return `удалён ${day(ev.clip_deleted_at)}`;
  if (ev.clip_status === "recording") return "сохранится после эпизода";
  return "нет";
}

function renderCamEvents() {
  const list = $("events");
  if (!camEvents.length) { list.innerHTML = `<li class="empty">Эпизодов пока нет</li>`; return; }
  list.innerHTML = camEvents.map((e) => {
    const active = !!e.active;
    const dur = active ? now() - e.started_at : e.duration;
    const review = e.review === "pending" ? "Авто · не проверено" : `${REVIEW_TEXT[e.review]} ${e.reviewed_at ? hm(e.reviewed_at) : ""}`;
    const open = active
      ? `<button class="btn btn--ghost" data-archive="${e.started_at}">В архиве</button>`
      : `<button class="btn btn--ghost" data-player="${e.id}">Фрагмент</button>`;
    return `<li class="ev ev--smoking ${e.review !== "pending" ? "done" : ""} r-${e.review}">
      <div class="ev-bar"></div>
      <div>
        <div class="ev-top"><span class="ev-title">Похоже на курение · ${esc(e.source_id)}</span><span class="ev-time">${hms(e.started_at)}</span></div>
        <div class="ev-sub">${esc(e.source_name)} · трек ${e.track ?? "—"}</div>
        <div class="ev-meta">
          ${active ? `<span class="live" data-start="${e.started_at}">идёт ${mmss(dur)}</span>` : `<span>длит. ${mmss(dur)}</span>`}
          <span>${cyclesText(e.cycles)}</span><span>уверенность ${pct(e.confidence)}</span>
        </div>
        <div class="ev-review">${esc(review)}</div>
        <div class="ev-actions">${reviewButtons(e, true)}${open}</div>
      </div></li>`;
  }).join("");
}

async function setReview(id, decision) {
  try {
    await api(`/api/events/${id}/review`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ decision }) });
  } catch (err) { alert(`Не удалось сохранить решение: ${err.message}`); }

  onJournalChanged();
}

document.addEventListener("click", (e) => {
  const r = e.target.closest("[data-review]");
  if (r) { e.stopPropagation(); setReview(+r.dataset.id, r.dataset.review); return; }
  const p = e.target.closest("[data-player]");
  if (p) { e.stopPropagation(); Player.open(+p.dataset.player); return; }
  const a = e.target.closest("[data-archive]");
  if (a) { e.stopPropagation(); Player.close(); CamModal.close(); Archive.openAt(+a.dataset.archive - CONFIG.pre_roll); }
});

let jFilter = { source: "", review: "" };
segment($("jSource"), "v", (v) => { jFilter.source = v; loadJournal(); });
segment($("jReview"), "v", (v) => { jFilter.review = v; loadJournal(); });

async function loadJournal() {
  const q = new URLSearchParams({ limit: "500" });
  if (jFilter.source) q.set("source_type", jFilter.source);
  if (jFilter.review) q.set("review", jFilter.review);
  const list = await api(`/api/events?${q}`).catch(() => null);
  if (!list) return;
  const rows = $("jRows");
  if (!list.length) { rows.innerHTML = `<tr><td colspan="9" class="empty">Записей нет</td></tr>`; return; }
  rows.innerHTML = list.map((e) => {
    const isFile = e.source_type === "file";
    const time = isFile
      ? `<b>${mmss(e.offset_sec)}</b> от начала записи<small>загружен ${dayTime(e.started_at)}</small>`
      : `<b>${hms(e.started_at)}</b><small>${day(e.started_at)}</small>`;
    const src = isFile
      ? `<b>${esc(e.source_name)}</b><small>загруженный файл</small>`
      : `<b>${esc(e.source_id)}</b><small>${esc(e.source_name)}</small>`;
    const dur = e.active ? `<span class="live" data-start="${e.started_at}">идёт ${mmss(now() - e.started_at)}</span>` : mmss(e.duration);
    const det = `Авто${e.demo ? ' <span class="tag">демо</span>' : ""}`;
    const review = e.review === "pending"
      ? `<span class="rv rv-pending">Не проверено</span>`
      : `<span class="rv rv-${e.review}">${REVIEW_TEXT[e.review]}</span><small>${dayTime(e.reviewed_at)}</small>`;
    const actions = e.clip_status === "saved"
      ? `<button class="btn" data-player="${e.id}">Смотреть</button>` + (e.clip_kind === "sim"
        ? `<button class="btn btn--ghost" data-export="${e.id}">Скачать</button>`
        : `<a class="btn btn--ghost" href="${e.download_url}" download>Скачать</a>`)
      : e.active ? `<button class="btn" data-archive="${e.started_at}">В архиве</button>` : `<button class="btn" data-player="${e.id}">Открыть</button>`;
    return `<tr data-row="${e.id}" class="${e.review === "pending" && !e.active ? "row-pending" : ""}">
      <td>${time}</td><td>${src}</td><td class="num">${dur}</td><td class="num">${e.cycles}</td><td class="num">${pct(e.confidence)}</td>
      <td>${det}</td><td>${review}</td><td class="clip-${e.clip_status}">${clipText(e)}</td><td class="actions">${actions}</td></tr>`;
  }).join("");
}
$("jRows").addEventListener("click", (e) => {
  const ex = e.target.closest("[data-export]");
  if (ex) { e.stopPropagation(); Player.open(+ex.dataset.export, { export: true }); return; }
  if (e.target.closest("a, button")) return;
  const row = e.target.closest("[data-row]");
  if (row) Player.open(+row.dataset.row);
});

const CamModal = {
  cam: null, view: null,
  open(cam) {
    if (mode === "archive") {
      Archive.leave();
    }
    this.cam = cam;
    this.view = new Scene.View(cam.info);
    $("mId").textContent = cam.info.id;
    $("mName").textContent = cam.info.name;
    $("mZone").textContent = cam.info.zone;
    this.render();
    if (!$("camModal").open) $("camModal").showModal();
    this.draw();
  },
  close() { if ($("camModal").open) $("camModal").close(); },
  draw() {
    const cam = this.cam; if (!cam) return;
    const st = cam.state || {};
    const canvas = $("bigCanvas");
    if (st.health === "no_signal" || st.health === "error") {
      this.view.blank(canvas, HEALTH_TEXT[st.health], [st.last_frame ? `последний кадр ${dayTime(st.last_frame)}` : "кадров не было", st.reason || ""]);
    } else if (cam.frame) {
      this.view.draw(canvas, cam.frame[0], cam.frame[1], { big: true });
    }
    $("bigHud").innerHTML = `<span>${cam.frame ? stamp(cam.frame[0]) : ""}</span><span>${esc(cam.info.id)} · ${st.health === "delay" ? `задержка ${Math.round(st.lag)} с` : "10 к/с"}</span>`;
  },
  render() {
    const cam = this.cam; if (!cam) return;
    const st = cam.state || {};
    const observing = st.health === "online" || st.health === "delay";
    const det = !observing ? (st.health === "analyzing" ? "выводов пока нет" : "анализ недоступен")
      : st.detection === "smoking" ? "похоже на курение (авто)" : st.detection === "suspect" ? "подозрение: 1 цикл (авто)" : "срабатываний нет";
    const ev = activeEvent(cam);
    const facts = [
      ["Камера", `<span class="h-status h-${st.health}">${HEALTH_TEXT[st.health] || "—"}</span>`],
      st.health !== "online" && st.last_frame ? ["Последний кадр", stamp(st.last_frame)] : null,
      st.reason ? ["Причина", esc(st.reason)] : null,
      ["Анализ", det],
      ev ? ["Эпизод", `№${ev.id} · ${REVIEW_TEXT[ev.review]}`] : null,
    ].filter(Boolean);
    $("mFacts").innerHTML = facts.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
    const t = serverT;
    $("mTracks").innerHTML = (st.persons || []).map((p) => {
      const recent = p.cycles.filter((c) => t - c <= 20).length;
      const text = p.state === "unknown" ? "нет анализа" : p.state === "smoking" ? `похоже на курение · ${cyclesText(recent)} / 20 с`
        : p.state === "suspect" ? "1 цикл / 20 с" : p.role === "walker" ? "проходит" : "без циклов";
      const marks = p.cycles.map((c) => `<i style="left:${(100 - ((t - c) / 60) * 100).toFixed(2)}%"></i>`).join("");
      return `<li class="track" data-s="${p.state}"><div class="track-top"><b>Трек ${p.track}</b><span>${text}</span></div>
        <div class="timeline"><div class="win"></div>${marks}</div></li>`;
    }).join("");
    $("mActions").innerHTML = (ev && ev.review === "pending" ? reviewButtons(ev, true) : "")
      + `<button class="btn btn--ghost" data-archive="${t - 60 + CONFIG.pre_roll}">Архив: последняя минута</button>`;
  },
};
function openCam(cam) { CamModal.open(cam); }
$("camModal").addEventListener("close", () => { CamModal.cam = null; });

const Player = {
  ev: null, clip: null, view: null, pos: 0, playing: true, a: 0, b: 0, evA: 0, evB: 0, kind: "",

  async open(id, opts = {}) {
    let ev;
    try { ev = await api(`/api/events/${id}`); } catch (err) { alert(`Эпизод не найден: ${err.message}`); return; }
    CamModal.close();
    this.stop();
    this.ev = ev;
    this.clip = null;
    const video = $("pVideo"), canvas = $("pCanvas");
    video.classList.add("hidden"); canvas.classList.remove("hidden");
    this.view = new Scene.View({ scene: 0, kind: "street" });
    this.renderSide();
    if (!$("player").open) $("player").showModal();

    const isFile = ev.source_type === "file";
    if (ev.clip_status !== "saved") {
      const title = ev.clip_status === "deleted" ? "Фрагмент удалён" : ev.clip_status === "recording" ? "Эпизод ещё идёт" : "Фрагмент недоступен";
      const sub = ev.clip_status === "deleted" ? `срок хранения истёк ${dayTime(ev.clip_deleted_at)}`
        : ev.clip_status === "recording" ? `фрагмент сохранится через ${CONFIG.post_roll} с после последнего цикла` : "";
      this.kind = "none";
      this.view.blank(canvas, title, sub);
      $("pTime").textContent = "—";
      this.layoutTimeline(0, 1, 0, 0);
      return;
    }
    this.a = ev.clip_start; this.b = ev.clip_end;
    this.evA = isFile ? ev.offset_sec : ev.started_at;
    this.evB = this.evA + ev.duration;
    if (ev.clip_kind === "sim") {
      this.kind = "sim";
      this.view.blank(canvas, "Загрузка фрагмента…");
      const clip = await api(ev.clip_url).catch(() => null);
      if (!clip || this.ev?.id !== ev.id) return;
      this.clip = clip;
      this.view = new Scene.View(clip.camera);
      this.a = clip.frames[0][0]; this.b = clip.frames[clip.frames.length - 1][0];
      this.pos = this.a;
      this.playing = !opts.export;
      $("pPlay").textContent = this.playing ? "Пауза" : "Смотреть";
      this.layoutTimeline(this.a, this.b, this.evA, this.evB);
      this.drawSim();
      if (opts.export) this.exportSim();
    } else {
      this.kind = ev.clip_kind;
      canvas.classList.add("hidden"); video.classList.remove("hidden");
      const base = this.kind === "video" ? this.a : 0;
      video.src = ev.clip_url;
      video.onloadedmetadata = () => { video.currentTime = this.a - base; video.play().catch(() => {}); };
      video.onerror = () => {
        video.classList.add("hidden"); canvas.classList.remove("hidden");
        this.view.blank(canvas, "Браузер не может воспроизвести этот файл", ["скачайте фрагмент и откройте его в видеоплеере", "или установите ffmpeg: сервер перекодирует фрагменты в H.264"]);
      };
      this.layoutTimeline(this.a, this.b, this.evA, this.evB);
      this.playing = true;
      $("pPlay").textContent = "Пауза";
    }
  },

  async refresh() {
    if (!this.ev) return;
    const ev = await api(`/api/events/${this.ev.id}`).catch(() => null);
    if (ev && this.ev && ev.id === this.ev.id) { this.ev = ev; this.renderSide(); }
  },

  renderSide() {
    const ev = this.ev;
    const isFile = ev.source_type === "file";
    $("pKind").textContent = isFile ? "Загруженный файл" : ev.source_id;
    $("pTitle").textContent = `Эпизод №${ev.id}`;
    $("pSub").textContent = ev.source_name;
    const clip = ev.clip_status === "saved"
      ? `хранится до ${dayTime(ev.clip_expires_at)} (осталось ${Math.max(0, Math.ceil((ev.clip_expires_at - now()) / 86400))} дн.)`
      : clipText(ev);
    const facts = [
      ["Время", isFile ? `${mmss(ev.offset_sec)} от начала записи` : `${day(ev.started_at)}, ${hms(ev.started_at)}`],
      ["Длительность", ev.active ? "эпизод идёт" : mmss(ev.duration)],
      ["Циклов «рука ко рту»", ev.cycles],
      ["Уверенность", pct(ev.confidence)],
      ["Обнаружение", `автоматически${ev.demo ? ' <span class="tag">демо-анализ</span>' : ""}`],
      ["Проверка", ev.review === "pending" ? "не проверено оператором" : `${REVIEW_TEXT[ev.review].toLowerCase()}, ${dayTime(ev.reviewed_at)}`],
      ["Фрагмент", clip],
    ];
    if (ev.demo) facts.push(["", '<span class="muted">Эпизод сгенерирован демо-анализом для показа интерфейса, это не результат модели.</span>']);
    $("pFacts").innerHTML = facts.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
    $("pReview").innerHTML = `<div class="review-row">${reviewButtons(ev, false)}</div>`;
    let actions = "";
    if (ev.clip_status === "saved") {
      if (ev.clip_kind === "sim") actions += `<button class="btn" id="pExport">Скачать видео (WebM)</button>`;
      else if (ev.clip_kind === "video") actions += `<a class="btn" href="${ev.download_url}" download>Скачать фрагмент (MP4)</a>`;
      else actions += `<a class="btn" href="${ev.download_url}" download>Скачать исходный файл</a><p class="hint">ffmpeg не найден, поэтому фрагмент не вырезан: скачивается весь файл, эпизод — с ${mmss(ev.clip_start)} по ${mmss(ev.clip_end)}.</p>`;
    }
    if (!isFile) actions += `<button class="btn btn--ghost" data-archive="${ev.started_at}">Открыть в архиве</button>`;
    if (isFile) actions += `<a class="btn btn--ghost" href="#check/${encodeURIComponent(ev.source_id)}" data-close>К заданию</a>`;
    $("pActions").innerHTML = actions + `<p class="hint export-progress hidden" id="pExportNote"></p>`;
    $("pExport")?.addEventListener("click", () => this.exportSim());
  },

  layoutTimeline(a, b, ea, eb) {
    const span = b - a || 1;
    const l = clamp((ea - a) / span, 0, 1), r = clamp((eb - a) / span, 0, 1);
    $("pRange").style.left = `${l * 100}%`;
    $("pRange").style.width = `${Math.max(0.5, (r - l) * 100)}%`;
    $("pRange").classList.toggle("hidden", eb <= ea && this.kind === "none");
  },

  current() {
    if (this.kind === "sim") return this.pos;
    if (this.kind === "video") return this.a + $("pVideo").currentTime;
    if (this.kind === "range") return $("pVideo").currentTime;
    return 0;
  },

  seek(t) {
    t = clamp(t, this.a, this.b);
    if (this.kind === "sim") { this.pos = t; this.view.reset(); this.drawSim(); }
    else if (this.kind === "video") $("pVideo").currentTime = t - this.a;
    else if (this.kind === "range") $("pVideo").currentTime = t;
  },

  frameIndex(t) {
    const f = this.clip.frames;
    let lo = 0, hi = f.length - 1, best = 0;
    while (lo <= hi) { const m = (lo + hi) >> 1; if (f[m][0] <= t) { best = m; lo = m + 1; } else hi = m - 1; }
    return best;
  },

  drawSim() {
    if (!this.clip) return;
    const fr = this.clip.frames[this.frameIndex(this.pos)];
    this.view.draw($("pCanvas"), fr[0], fr[1], { big: true });
    $("pHud").innerHTML = `<span>${stamp(fr[0])}</span><span>${esc(this.clip.camera.id)} · запись</span>`;
  },

  tick(dt) {
    if (!this.ev || this.kind === "none") return;
    if (this.kind === "sim" && this.clip) {
      if (this.playing) {
        const before = this.frameIndex(this.pos);
        this.pos += dt;
        if (this.pos >= this.b) { this.pos = this.b; this.playing = false; $("pPlay").textContent = "Смотреть"; }
        if (this.frameIndex(this.pos) !== before) this.drawSim();
      }
    } else if (this.kind === "range") {
      const v = $("pVideo");
      if (!v.paused && v.currentTime >= this.b) { v.pause(); $("pPlay").textContent = "Смотреть"; }
      if (this.kind !== "sim") $("pHud").innerHTML = "";
    }
    const t = this.current();
    const span = this.b - this.a || 1;
    $("pCursor").style.left = `${clamp((t - this.a) / span, 0, 1) * 100}%`;
    $("pTime").textContent = this.ev.source_type === "file"
      ? `${mmss(t)} · эпизод ${mmss(this.evA)}–${mmss(this.evB)}`
      : `${hms(t)} · эпизод ${hms(this.evA)}–${hms(this.evB)}`;
  },

  toggle() {
    if (this.kind === "sim") {
      if (!this.playing && this.pos >= this.b) this.seek(this.a);
      this.playing = !this.playing;
      $("pPlay").textContent = this.playing ? "Пауза" : "Смотреть";
    } else if (this.kind === "video" || this.kind === "range") {
      const v = $("pVideo");
      if (v.paused) { if (this.current() >= this.b - 0.2) this.seek(this.a); v.play().catch(() => {}); } else v.pause();
      $("pPlay").textContent = v.paused ? "Смотреть" : "Пауза";
    }
  },

  async exportSim() {
    if (!this.clip) return;
    const btn = $("pExport"), note = $("pExportNote");
    if (btn) btn.disabled = true;
    note.classList.remove("hidden");
    note.textContent = "Готовлю видео…";
    const frames = this.clip.frames;
    const view = new Scene.View(this.clip.camera);
    try {
      const blob = await WebmExport.encode({
        n: frames.length, fps: this.clip.fps, width: 960, height: 540,
        render: (i, canvas) => {
          view.draw(canvas, frames[i][0], frames[i][1], { big: true, size: [960, 540], noNoise: true });
          const g = canvas.getContext("2d");
          g.fillStyle = "rgba(255,255,255,0.85)"; g.font = "500 15px Inter, system-ui, sans-serif";
          g.fillText(`${this.clip.camera.id} · ${stamp(frames[i][0])} · эпизод №${this.ev.id} (авто)`, 14, 528);
        },
        onProgress: (k) => { note.textContent = `Готовлю видео… ${Math.round(k * 100)}%`; },
      });
      WebmExport.save(blob, `episode_${this.ev.id}_${this.clip.camera.id}.webm`);
      note.textContent = `Готово: ${mb(blob.size)}`;
    } catch (err) {
      note.textContent = `Не удалось сохранить видео: ${err.message || err}`;
    } finally {
      if (btn) btn.disabled = false;
    }
  },

  stop() {
    const v = $("pVideo");
    v.pause(); v.removeAttribute("src"); v.load();
    this.clip = null; this.kind = "none";
  },
  close() { if ($("player").open) $("player").close(); },
};
$("pPlay").addEventListener("click", () => Player.toggle());
$("pVideo").addEventListener("play", () => { $("pPlay").textContent = "Пауза"; });
$("pVideo").addEventListener("pause", () => { $("pPlay").textContent = "Смотреть"; });
$("player").addEventListener("close", () => {
  Player.stop(); Player.ev = null;
  if (/^#journal\/\d+$/.test(location.hash)) history.replaceState(null, "", "#journal");
});
(() => {
  const tl = $("pTimeline");
  let drag = false;
  const seek = (e) => { const r = tl.getBoundingClientRect(); Player.seek(Player.a + clamp((e.clientX - r.left) / r.width, 0, 1) * (Player.b - Player.a)); };
  tl.addEventListener("pointerdown", (e) => { drag = true; tl.setPointerCapture(e.pointerId); seek(e); });
  tl.addEventListener("pointermove", (e) => { if (drag) seek(e); });
  tl.addEventListener("pointerup", () => { drag = false; });
})();
for (const d of document.querySelectorAll("dialog")) {
  d.addEventListener("click", (e) => { if (e.target === d || e.target.closest("[data-close]")) d.close(); });
}

const Jobs = {
  list: [], uploads: [], selected: null, detail: null, timer: null,
  active() { return this.uploads.length > 0 || this.list.some((j) => j.state === "queued" || j.state === "analyzing"); },
};

async function loadJobs() {
  Jobs.list = await api("/api/jobs").catch(() => Jobs.list);
  renderJobs();
  if (Jobs.selected) loadJobDetail(Jobs.selected);
  clearTimeout(Jobs.timer);
  if (Jobs.active() && (page === "check")) Jobs.timer = setTimeout(loadJobs, 1000);
  else if (Jobs.active()) Jobs.timer = setTimeout(loadJobs, 3000);
}

function analyzerTag(a) {
  if (a === "demo") return ' · <span class="tag">демо-анализ</span>';
  return a === "realtime" ? ' · <span class="tag">потоковый режим</span>' : "";
}

function jobStatus(j) {
  if (j.state === "analyzing") return `${JOB_TEXT.analyzing} · ${Math.round(j.progress * 100)}%`;
  if (j.state === "done") return `${JOB_TEXT.done} · ${j.episodes} ${plural(j.episodes, "эпизод", "эпизода", "эпизодов")}`;
  return JOB_TEXT[j.state];
}

function renderJobs() {
  const ul = $("jobs");
  const ups = Jobs.uploads.map((u) => `<li class="job job--upload">
      <div class="job-top"><b>${esc(u.name)}</b><span class="state s-upload">${u.error ? "Ошибка загрузки" : `Загрузка · ${Math.round(u.progress * 100)}%`}</span></div>
      <div class="job-sub">${mb(u.size)}${u.error ? ` · ${esc(u.error)}` : ""}</div>
      <div class="bar"><i style="width:${u.progress * 100}%"></i></div></li>`).join("");
  const jobs = Jobs.list.map((j) => `<li class="job ${Jobs.selected === j.id ? "on" : ""}" data-job-id="${esc(j.id)}" tabindex="0">
      <div class="job-top"><b>${esc(j.filename)}</b><span class="state s-${j.state}">${jobStatus(j)}</span></div>
      <div class="job-sub">${mb(j.size)}${j.duration ? ` · ${mmss(j.duration)}` : ""} · загружен ${dayTime(j.created_at)}${analyzerTag(j.analyzer)}</div>
      ${j.state === "analyzing" ? `<div class="bar"><i style="width:${j.progress * 100}%"></i></div>` : ""}
      ${j.state === "error" ? `<div class="job-error">${esc(j.error)}</div>` : ""}
      ${j.deleted_at ? `<div class="job-sub">файл удалён ${day(j.deleted_at)} по сроку хранения</div>` : j.expires_at ? `<div class="job-sub">файл и фрагменты хранятся до ${dayTime(j.expires_at)}</div>` : ""}
    </li>`).join("");
  ul.innerHTML = ups + jobs || `<li class="empty">Заданий нет. Загрузите видеофайл.</li>`;
}

$("jobs").addEventListener("click", (e) => {
  const li = e.target.closest("[data-job-id]"); if (!li) return;
  Jobs.selected = li.dataset.jobId;
  renderJobs();
  loadJobDetail(Jobs.selected, true);
});
$("jobs").addEventListener("keydown", (e) => { if (e.key === "Enter") e.target.closest("[data-job-id]")?.click(); });

async function loadJobDetail(id, force) {
  const job = await api(`/api/jobs/${encodeURIComponent(id)}`).catch(() => null);
  if (!job || Jobs.selected !== id) return;
  const prev = Jobs.detail;
  const same = prev && prev.id === job.id && prev.state === job.state && JSON.stringify(prev.events) === JSON.stringify(job.events);
  Jobs.detail = job;
  if (same && !force) {
    const st = document.querySelector("#jobDetail .state");
    if (st) { st.className = `state s-${job.state}`; st.textContent = jobStatus(job); }
    return;
  }
  renderJobDetail(job, prev && prev.id === job.id);
}

function renderJobDetail(job, keepVideo) {
  const box = $("jobDetail");
  const video = keepVideo ? box.querySelector("video") : null;
  const dur = job.duration || 0;
  const eps = job.events.slice().sort((a, b) => a.offset_sec - b.offset_sec);
  const marks = dur ? eps.map((e) => {
    const l = (e.offset_sec / dur) * 100, w = Math.max(0.6, (e.duration / dur) * 100);
    return `<button class="ar-mark r-${e.review}" data-seek="${e.offset_sec}" style="left:${l}%;width:${w}%" title="${mmss(e.offset_sec)} · ${cyclesText(e.cycles)}"></button>`;
  }).join("") : "";
  let body = "";
  if (job.state === "done") {
    body = eps.length ? `<ul class="episodes">${eps.map((e) => `
      <li class="episode r-${e.review}">
        <div class="ep-main">
          <b>${mmss(e.offset_sec)}–${mmss(e.offset_sec + e.duration)}</b>
          <span>${cyclesText(e.cycles)} · уверенность ${pct(e.confidence)}</span>
          <span class="rv rv-${e.review}">${e.review === "pending" ? "Авто · не проверено" : REVIEW_TEXT[e.review]}</span>
        </div>
        <div class="ep-actions">
          <button class="btn" data-seek="${e.offset_sec}">Перейти</button>
          <button class="btn btn--ghost" data-player="${e.id}">Фрагмент</button>
          ${reviewButtons(e, true)}
        </div>
      </li>`).join("")}</ul>`
      : `<p class="empty">Эпизодов, похожих на курение, не найдено. Это не гарантирует их отсутствия в записи.</p>`;
  } else if (job.state === "error") {
    body = `<p class="banner banner--error">Анализ завершился ошибкой: ${esc(job.error)}</p>`;
  } else {
    body = `<p class="empty">${job.state === "queued" ? "Задание ждёт своей очереди." : "Идёт анализ. Эпизоды появятся после завершения."}</p>`;
  }
  const html = `
    <div class="detail-head">
      <div><h2 class="h2">${esc(job.filename)}</h2><p class="muted">${mb(job.size)}${dur ? ` · ${mmss(dur)}` : ""} · загружен ${dayTime(job.created_at)}${analyzerTag(job.analyzer)}</p></div>
      <span class="state s-${job.state}">${jobStatus(job)}</span>
    </div>
    ${job.analyzer === "demo" ? '<p class="banner">Демо-анализ: эпизоды сгенерированы для показа интерфейса и не являются результатом модели.</p>' : ""}
    <div class="detail-video">${job.file_url ? "" : '<p class="empty">Файл удалён по сроку хранения.</p>'}</div>
    <div class="ar-timeline ar-timeline--file" id="fileMarks">${marks}<div class="ar-cursor" id="fileCursor"></div></div>
    <p class="disclaimer">Эпизоды отмечены автоматически. Факт курения и личность человека система не устанавливает.</p>
    ${body}`;
  box.innerHTML = html;
  const holder = box.querySelector(".detail-video");
  if (job.file_url) {
    const v = video || Object.assign(document.createElement("video"), { controls: true, preload: "metadata", src: job.file_url });
    v.playsInline = true;
    v.onerror = () => { holder.innerHTML = '<p class="banner">Браузер не может воспроизвести этот файл (нужен H.264 или VP8/VP9). Эпизоды и фрагменты доступны в журнале.</p>'; };
    holder.appendChild(v);
    v.ontimeupdate = () => { const c = $("fileCursor"); if (c && dur) c.style.left = `${clamp(v.currentTime / dur, 0, 1) * 100}%`; };
  }
}

$("jobDetail").addEventListener("click", (e) => {
  const s = e.target.closest("[data-seek]");
  if (!s) {
    const tl = e.target.closest("#fileMarks");
    const v = $("jobDetail").querySelector("video");
    if (tl && v && Jobs.detail?.duration) { const r = tl.getBoundingClientRect(); v.currentTime = clamp((e.clientX - r.left) / r.width, 0, 1) * Jobs.detail.duration; }
    return;
  }
  const v = $("jobDetail").querySelector("video");
  if (v) { v.currentTime = Math.max(0, +s.dataset.seek - CONFIG.pre_roll); v.play().catch(() => {}); v.scrollIntoView({ block: "nearest", behavior: "smooth" }); }
});

function probeDuration(file) {
  return new Promise((resolve) => {
    const v = document.createElement("video");
    const url = URL.createObjectURL(file);
    let done = false;
    const finish = (d) => { if (done) return; done = true; URL.revokeObjectURL(url); resolve(d); };
    v.preload = "metadata";
    v.onloadedmetadata = () => finish(Number.isFinite(v.duration) ? v.duration : null);
    v.onerror = () => finish(null);
    setTimeout(() => finish(null), 6000);
    v.src = url;
  });
}

function upload(file) {
  return new Promise(async (resolve) => {
    const u = { name: file.name, size: file.size, progress: 0, error: "" };
    Jobs.uploads.push(u);
    renderJobs();
    const duration = await probeDuration(file);
    const xhr = new XMLHttpRequest();
    xhr.open("POST", "/api/jobs");
    xhr.setRequestHeader("X-Filename", encodeURIComponent(file.name));
    if (duration) xhr.setRequestHeader("X-Duration", String(duration));
    xhr.upload.onprogress = (e) => { if (e.lengthComputable) { u.progress = e.loaded / e.total; renderJobs(); } };
    xhr.onload = () => {
      let body = {};
      try { body = JSON.parse(xhr.responseText); } catch (_) {  }
      if (xhr.status === 201) {
        Jobs.uploads = Jobs.uploads.filter((x) => x !== u);
        Jobs.selected = body.id;
      } else {
        u.error = body.error || `ошибка ${xhr.status}`;
        setTimeout(() => { Jobs.uploads = Jobs.uploads.filter((x) => x !== u); renderJobs(); }, 8000);
      }
      loadJobs();
      resolve();
    };
    xhr.onerror = () => { u.error = "сервер недоступен"; renderJobs(); resolve(); };
    xhr.send(file);
  });
}

async function uploadAll(files) {
  for (const f of files) await upload(f);
}
$("fileInput").addEventListener("change", (e) => { uploadAll([...e.target.files]); e.target.value = ""; });
const drop = $("drop");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); uploadAll([...e.dataTransfer.files]); });

let lastFrame = performance.now();
function loop(ts) {
  const dt = Math.min(0.25, (ts - lastFrame) / 1000);
  lastFrame = ts;
  Archive.tick(dt);
  Player.tick(dt);
  requestAnimationFrame(loop);
}

function tickClock() {
  const t = now();
  $("clock").textContent = hms(t);
  $("date").textContent = new Date().toLocaleDateString("ru-RU", { day: "numeric", month: "long", year: "numeric" });
  for (const el of document.querySelectorAll(".live[data-start]")) el.textContent = `идёт ${mmss(t - +el.dataset.start)}`;
}
tickClock();
setInterval(tickClock, 500);

async function init() {
  try {
    CONFIG = await api("/api/config");
  } catch (err) {
    document.body.insertAdjacentHTML("afterbegin", `<p class="banner banner--error">Сервер мониторинга недоступен: запустите monitor.py (${esc(err.message)})</p>`);
    return;
  }
  for (const el of document.querySelectorAll(".retention-days")) el.textContent = `${CONFIG.retention_days} ${plural(Math.round(CONFIG.retention_days), "день", "дня", "дней")}`;
  const note = $("analyzerNote");
  if (CONFIG.analyzer_problem) {
    note.textContent = `Анализ видео недоступен: ${CONFIG.analyzer_problem}. Задания завершатся ошибкой.`;
    note.classList.add("banner--error");
    note.classList.remove("hidden");
  } else if (CONFIG.analyzer === "demo") {
    note.textContent = "Сервер запущен с демо-анализом: эпизоды генерируются для показа интерфейса, модель не используется.";
    note.classList.remove("hidden");
  }
  const st = await api("/api/state");
  buildTiles(st.cameras);
  for (const s of st.cameras) byId.get(s.id).state = s;
  journalVersion = st.journal;
  for (const cam of cams) { const v = tileView(cam); cam.det = v.det; cam.health = v.health; }
  await loadCamEvents();
  applyLayout();
  connect();
  showPage(location.hash.slice(1) || "monitor");
  loadJobs();
  requestAnimationFrame(loop);
}
init();
