

"use strict";

const Scene = (() => {
  const ROLES = ["walker", "idle", "smoker", "sipper"];
  const STATES = ["normal", "suspect", "smoking"];
  const COLORS = { normal: "rgba(220, 224, 230, 0.6)", suspect: "#e0a526", smoking: "#e5484d" };
  const BONES = [
    ["shL", "shR"], ["shL", "hipL"], ["shR", "hipR"], ["hipL", "hipR"],
    ["hipL", "kneeL"], ["kneeL", "ankL"], ["hipR", "kneeR"], ["kneeR", "ankR"],
    ["shL", "elL"], ["elL", "wrL"], ["shR", "elR"], ["elR", "wrR"],
    ["earL", "eyeL"], ["eyeL", "nose"], ["nose", "eyeR"], ["eyeR", "earR"],
  ];
  const LIMBS = [["hipL", "kneeL"], ["kneeL", "ankL"], ["hipR", "kneeR"], ["kneeR", "ankR"],
    ["shL", "elL"], ["elL", "wrL"], ["shR", "elR"], ["elR", "wrR"]];

  const lerp = (a, b, k) => a + (b - a) * k;
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  function rng(seed) {
    let s = seed >>> 0;
    return () => {
      s = (s + 0x6d2b79f5) >>> 0;
      let x = s;
      x = Math.imul(x ^ (x >>> 15), x | 1);
      x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
      return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
    };
  }

  function decode(a) {
    return {
      track: a[0], x: a[1], y: a[2], dir: a[3], phase: a[4], a: a[5],
      role: ROLES[a[6]], state: STATES[a[7]], lit: a[8] === 1, tone: a[9], sway: a[10],
    };
  }

  const heightOf = (p) => 0.2 + (p.y - 0.6) * 0.62;
  const faceOf = (p) => (p.role === "walker" ? p.dir : (p.sway > 3 ? 1 : -1));

  function skeleton(p, aspect, t) {
    const H = heightOf(p);
    const sx = H / aspect;
    const f = faceOf(p);
    const P = (dx, dy) => ({ x: p.x + dx * sx * f, y: p.y + dy * H });
    const swing = Math.sin(p.phase);
    const k = p.a;
    const breath = p.role === "walker" ? 0 : Math.sin(t * 1.3 + p.sway) * 0.004;
    return {
      nose: P(0.035, -0.905 + breath), eyeL: P(0.02, -0.925 + breath), eyeR: P(0.045, -0.925 + breath),
      earL: P(-0.03, -0.915 + breath), earR: P(0.05, -0.915 + breath),
      shL: P(-0.1, -0.8 + breath), shR: P(0.1, -0.8 + breath),
      hipL: P(-0.07, -0.48), hipR: P(0.07, -0.48),
      kneeL: P(-0.06 + swing * 0.07, -0.25), kneeR: P(0.06 - swing * 0.07, -0.25),
      ankL: P(-0.06 + swing * 0.14, 0), ankR: P(0.06 - swing * 0.14, 0),
      elL: P(-0.13 - swing * 0.04, -0.63), wrL: P(-0.13 - swing * 0.09, -0.47),
      elR: P(lerp(0.13 + swing * 0.04, 0.17, k), lerp(-0.63, -0.72, k)),
      wrR: P(lerp(0.12 + swing * 0.09, 0.05, k), lerp(-0.47, -0.875, k)),
    };
  }

  let NOISE = null;
  function noise() {
    if (NOISE) return NOISE;
    const c = document.createElement("canvas");
    c.width = c.height = 160;
    const g = c.getContext("2d");
    const img = g.createImageData(160, 160);
    for (let i = 0; i < img.data.length; i += 4) {
      const v = Math.random() * 255;
      img.data[i] = img.data[i + 1] = img.data[i + 2] = v;
      img.data[i + 3] = 255;
    }
    g.putImageData(img, 0, 0);
    return (NOISE = c);
  }

  function background(cam, w, h) {
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    const g = c.getContext("2d");
    const r = rng(77 + cam.scene * 131);
    const horizon = h * (0.42 + r() * 0.08);

    if (cam.kind === "indoor") {
      const wallG = g.createLinearGradient(0, 0, 0, horizon);
      wallG.addColorStop(0, "#1c2129"); wallG.addColorStop(1, "#2a3039");
      g.fillStyle = wallG; g.fillRect(0, 0, w, horizon);
      for (let i = 0; i < 4; i++) {
        const x = w * (0.08 + i * 0.24 + r() * 0.04);
        g.fillStyle = i % 2 ? "#151a21" : "#323a46";
        g.fillRect(x, horizon - h * 0.26, w * 0.1, h * 0.26);
      }
      g.fillStyle = "rgba(180, 210, 255, 0.18)";
      g.fillRect(w * 0.36, h * 0.08, w * 0.28, h * 0.06);
    } else {
      const sky = g.createLinearGradient(0, 0, 0, horizon);
      sky.addColorStop(0, "#1d2532"); sky.addColorStop(1, "#38404c");
      g.fillStyle = sky; g.fillRect(0, 0, w, horizon);
      let x = -r() * w * 0.1;
      while (x < w) {
        const bw = w * (0.12 + r() * 0.16), bh = h * (0.14 + r() * 0.24);
        g.fillStyle = `hsl(215, 9%, ${11 + r() * 8}%)`;
        g.fillRect(x, horizon - bh, bw - 3, bh);
        for (let wy = horizon - bh + 6; wy < horizon - 8; wy += h * 0.045) {
          for (let wx = x + 5; wx < x + bw - 10; wx += w * 0.025) {
            if (r() < 0.35) { g.fillStyle = `rgba(255, 214, 150, ${0.12 + r() * 0.25})`; g.fillRect(wx, wy, w * 0.01, h * 0.018); }
          }
        }
        x += bw;
      }
    }

    const ground = g.createLinearGradient(0, horizon, 0, h);
    ground.addColorStop(0, cam.kind === "indoor" ? "#3a3f46" : "#2f343b");
    ground.addColorStop(1, cam.kind === "indoor" ? "#22262c" : "#191c21");
    g.fillStyle = ground; g.fillRect(0, horizon, w, h - horizon);

    const vx = w * (0.3 + r() * 0.4);
    g.strokeStyle = "rgba(255, 255, 255, 0.06)"; g.lineWidth = 1;
    for (let i = -6; i <= 6; i++) { g.beginPath(); g.moveTo(vx, horizon); g.lineTo(vx + i * w * 0.22, h); g.stroke(); }
    if (cam.kind === "platform") {
      g.fillStyle = "rgba(250, 204, 21, 0.35)";
      g.beginPath(); g.moveTo(vx - 2, horizon); g.lineTo(vx + 2, horizon); g.lineTo(w * 0.06, h); g.lineTo(-w * 0.02, h); g.fill();
      g.fillStyle = "#14181e"; g.fillRect(0, 0, w, h * 0.09);
      for (let i = 0; i < 5; i++) { g.fillStyle = "#1a1f27"; g.fillRect(w * (0.12 + i * 0.2), h * 0.09, w * 0.018, horizon - h * 0.09 + h * 0.05); }
    } else if (cam.kind === "street") {
      g.fillStyle = "rgba(255, 255, 255, 0.05)"; g.fillRect(0, horizon + h * 0.04, w, h * 0.012);
      const px = w * (0.55 + r() * 0.3);
      g.fillStyle = "rgba(150, 180, 210, 0.08)"; g.fillRect(px, horizon - h * 0.2, w * 0.28, h * 0.24);
      g.strokeStyle = "rgba(200, 220, 240, 0.18)"; g.strokeRect(px, horizon - h * 0.2, w * 0.28, h * 0.24);
      g.fillStyle = "#14181e"; g.fillRect(px - 4, horizon - h * 0.23, w * 0.28 + 8, h * 0.035);
    }
    return c;
  }

  class View {
    constructor(cam) {
      this.cam = cam;
      this.bg = null;
      this.bgKey = "";
      this.smoke = [];
      this.prevA = new Map();
      this.lastT = null;
    }

    reset() { this.smoke = []; this.prevA.clear(); this.lastT = null; }

    prepare(canvas, size) {
      const dpr = size ? 1 : Math.min(window.devicePixelRatio || 1, 2);
      const w = size ? size[0] : canvas.clientWidth, h = size ? size[1] : canvas.clientHeight;
      if (!w || !h) return null;
      const W = Math.round(w * dpr), H = Math.round(h * dpr);
      if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
      const g = canvas.getContext("2d");
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      return { g, w, h, W, H, dpr };
    }

    draw(canvas, t, rows, opts = {}) {
      const c = this.prepare(canvas, opts.size);
      if (!c) return;
      const { g, w, h, W, H, dpr } = c;
      const key = `${W}x${H}`;
      if (this.bgKey !== key) { this.bg = background(this.cam, W, H); this.bgKey = key; }
      g.drawImage(this.bg, 0, 0, w, h);

      const people = rows.map(decode).sort((a, b) => a.y - b.y);
      this.updateSmoke(people, t, w / h);
      const big = !!opts.big;
      const fs = clamp(w / 34, 10, 14);

      for (const p of people) {
        const s = skeleton(p, w / h, t);
        const X = (pt) => pt.x * w, Y = (pt) => pt.y * h;
        const Hp = heightOf(p) * h;
        const body = `hsl(215, 8%, ${p.tone}%)`;

        g.lineCap = "round"; g.lineJoin = "round";
        g.strokeStyle = body; g.fillStyle = body;
        g.lineWidth = Hp * 0.085;
        for (const [a, b] of LIMBS) { g.beginPath(); g.moveTo(X(s[a]), Y(s[a])); g.lineTo(X(s[b]), Y(s[b])); g.stroke(); }
        g.beginPath();
        g.moveTo(X(s.shL), Y(s.shL) - Hp * 0.01); g.lineTo(X(s.shR), Y(s.shR) - Hp * 0.01);
        g.lineTo(X(s.hipR) + Hp * 0.02, Y(s.hipR)); g.lineTo(X(s.hipL) - Hp * 0.02, Y(s.hipL)); g.closePath(); g.fill();
        g.beginPath(); g.arc(X(s.nose) - (s.nose.x - s.earL.x) * w * 0.35, Y(s.nose) - Hp * 0.01, Hp * 0.07, 0, Math.PI * 2); g.fill();

        if (p.role === "smoker" && p.lit) {
          const f = faceOf(p);
          const wx = X(s.wrR), wy = Y(s.wrR), tx = wx + f * Hp * 0.035, ty = wy - Hp * 0.008;
          g.strokeStyle = "rgba(240, 240, 235, 0.9)"; g.lineWidth = Math.max(1, Hp * 0.012);
          g.beginPath(); g.moveTo(wx, wy); g.lineTo(tx, ty); g.stroke();
          const tip = 0.5 + 0.5 * p.a + 0.15 * Math.sin(t * 9);
          const rr = Hp * 0.03 * (1 + p.a);
          const tg = g.createRadialGradient(tx, ty, 0, tx, ty, rr);
          tg.addColorStop(0, `rgba(255, 150, 60, ${tip})`); tg.addColorStop(1, "rgba(255, 120, 40, 0)");
          g.fillStyle = tg; g.beginPath(); g.arc(tx, ty, rr, 0, Math.PI * 2); g.fill();
        } else if (p.role === "sipper") {
          g.fillStyle = "rgba(210, 225, 240, 0.75)";
          g.fillRect(X(s.wrR) - Hp * 0.015, Y(s.wrR) - Hp * 0.05, Hp * 0.03, Hp * 0.06);
        }

        const col = COLORS[p.state];
        g.strokeStyle = col; g.globalAlpha = p.state === "normal" ? 0.55 : 0.95;
        g.lineWidth = big ? 1.8 : 1.2;
        for (const [a, b] of BONES) { g.beginPath(); g.moveTo(X(s[a]), Y(s[a])); g.lineTo(X(s[b]), Y(s[b])); g.stroke(); }
        g.fillStyle = col;
        for (const k in s) { g.beginPath(); g.arc(X(s[k]), Y(s[k]), big ? 2.4 : 1.6, 0, Math.PI * 2); g.fill(); }
        g.globalAlpha = 1;

        if (p.state !== "normal" && p.a > 0.05) {
          g.setLineDash([3, 3]); g.strokeStyle = col; g.lineWidth = 1;
          g.beginPath(); g.moveTo(X(s.wrR), Y(s.wrR)); g.lineTo(X(s.nose), Y(s.nose) + Hp * 0.03); g.stroke();
          g.setLineDash([]);
        }

        const bx = p.x * w - Hp * 0.2, by = p.y * h - Hp * 1.0, bw = Hp * 0.4, bh = Hp * 1.04;
        if (p.state === "normal") {
          g.strokeStyle = "rgba(255, 255, 255, 0.32)"; g.lineWidth = 1;
          const k = Math.min(bw, bh) * 0.18;
          g.beginPath();
          g.moveTo(bx, by + k); g.lineTo(bx, by); g.lineTo(bx + k, by);
          g.moveTo(bx + bw - k, by); g.lineTo(bx + bw, by); g.lineTo(bx + bw, by + k);
          g.moveTo(bx, by + bh - k); g.lineTo(bx, by + bh); g.lineTo(bx + k, by + bh);
          g.moveTo(bx + bw - k, by + bh); g.lineTo(bx + bw, by + bh); g.lineTo(bx + bw, by + bh - k);
          g.stroke();
        } else {
          const alarm = p.state === "smoking";
          g.strokeStyle = col; g.lineWidth = alarm ? 2 : 1.5;
          g.strokeRect(bx, by, bw, bh);
          const label = `трек ${p.track} · ${alarm ? "похоже на курение" : "рука ко рту"}`;
          g.font = `600 ${fs}px Inter, system-ui, sans-serif`;
          const tw = g.measureText(label).width + 10;
          const lx = clamp(bx, 2, w - tw - 2), ly = Math.max(2, by - fs - 8);
          g.fillStyle = col; g.fillRect(lx, ly, tw, fs + 6);
          g.fillStyle = alarm ? "#fff" : "#1a1405";
          g.fillText(label, lx + 5, ly + fs + 1);
        }
      }

      for (const s of this.smoke) {
        const a = (1 - s.life / s.max) * 0.28;
        const rr = s.r * h;
        const sg = g.createRadialGradient(s.x * w, s.y * h, 0, s.x * w, s.y * h, rr);
        sg.addColorStop(0, `rgba(215, 220, 228, ${a})`); sg.addColorStop(1, "rgba(215, 220, 228, 0)");
        g.fillStyle = sg; g.beginPath(); g.arc(s.x * w, s.y * h, rr, 0, Math.PI * 2); g.fill();
      }

      if (opts.noNoise) return;
      g.globalAlpha = 0.07;
      g.globalCompositeOperation = "overlay";
      const ox = Math.random() * 160, oy = Math.random() * 160;
      g.translate(-ox, -oy);
      g.fillStyle = g.createPattern(noise(), "repeat");
      g.fillRect(ox, oy, w, h);
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.globalCompositeOperation = "source-over";
      g.globalAlpha = 1;
    }

    updateSmoke(people, t, aspect) {
      const dt = this.lastT === null ? 0 : t - this.lastT;
      this.lastT = t;
      if (dt < 0 || dt > 1.5) { this.smoke = []; this.prevA.clear(); return; }
      for (const s of this.smoke) { s.life += dt; s.x += s.vx * dt; s.y += s.vy * dt; s.r += dt * 0.006; }
      this.smoke = this.smoke.filter((s) => s.life < s.max);
      for (const p of people) {
        const prev = this.prevA.get(p.track);
        this.prevA.set(p.track, p.a);
        if (p.role !== "smoker" || !p.lit) continue;
        const s = skeleton(p, aspect, t);
        if (prev !== undefined && prev >= 0.98 && p.a < 0.98) {
          for (let i = 0; i < 9; i++) {
            this.smoke.push({
              x: s.nose.x + (Math.random() - 0.5) * 0.004, y: s.nose.y + heightOf(p) * 0.03,
              vx: (Math.random() - 0.3) * 0.012 * faceOf(p), vy: -0.012 - Math.random() * 0.01,
              r: 0.004 + Math.random() * 0.004, life: 0, max: 2.2 + Math.random() * 1.6,
            });
          }
        }
        if (Math.random() < dt * 2.5) {
          this.smoke.push({ x: s.wrR.x, y: s.wrR.y, vx: (Math.random() - 0.5) * 0.004, vy: -0.01, r: 0.002, life: 0, max: 1.8 });
        }
      }
    }

    blank(canvas, title, sub, opts = {}) {
      const c = this.prepare(canvas, opts.size);
      if (!c) return;
      const { g, w, h } = c;
      g.fillStyle = "#0b0c0d"; g.fillRect(0, 0, w, h);
      g.textAlign = "center";
      g.fillStyle = "#c9ccd1";
      g.font = `600 ${clamp(w / 26, 12, 18)}px Inter, system-ui, sans-serif`;
      g.fillText(title, w / 2, h / 2 - (sub ? 4 : -6));
      if (sub) {
        g.fillStyle = "#7d828a";
        g.font = `400 ${clamp(w / 34, 10, 14)}px Inter, system-ui, sans-serif`;
        const lines = Array.isArray(sub) ? sub : [sub];
        lines.forEach((line, i) => g.fillText(line, w / 2, h / 2 + 16 + i * 17));
      }
      g.textAlign = "left";
      this.reset();
    }
  }

  return { View, decode, STATES };
})();
