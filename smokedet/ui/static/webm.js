// Выгрузка фрагмента камеры в видеофайл WebM прямо в браузере, без сервера и сторонних библиотек.
// Основной путь — WebCodecs (VP8) + минимальный муксер Matroska/WebM: быстрее реального времени.
// Без WebCodecs — MediaRecorder с canvas, в реальном времени.
"use strict";

const WebmExport = (() => {
  const enc = new TextEncoder();

  // ---------- EBML ----------
  function idBytes(id) {
    const out = [];
    while (id > 0) { out.unshift(id & 0xff); id = Math.floor(id / 256); }
    return out;
  }
  function sizeBytes(n) { // всегда 8 байт: 0x01 + 7 байт длины
    const out = [0x01];
    for (let i = 6; i >= 0; i--) out.push(Math.floor(n / 2 ** (8 * i)) & 0xff);
    return out;
  }
  function uintBytes(v) {
    const out = [];
    do { out.unshift(v & 0xff); v = Math.floor(v / 256); } while (v > 0);
    return out;
  }
  function el(id, payload) {
    const body = payload instanceof Uint8Array ? payload : Uint8Array.from(payload);
    const head = [...idBytes(id), ...sizeBytes(body.length)];
    const out = new Uint8Array(head.length + body.length);
    out.set(head, 0); out.set(body, head.length);
    return out;
  }
  const uint = (id, v) => el(id, uintBytes(v));
  const str = (id, s) => el(id, enc.encode(s));
  function float64(id, v) {
    const b = new Uint8Array(8);
    new DataView(b.buffer).setFloat64(0, v);
    return el(id, b);
  }
  function concat(parts) {
    const n = parts.reduce((s, p) => s + p.length, 0);
    const out = new Uint8Array(n);
    let o = 0;
    for (const p of parts) { out.set(p, o); o += p.length; }
    return out;
  }

  // chunks: [{data: Uint8Array, ms: число, key: bool}] по возрастанию времени
  function mux(chunks, width, height, durationMs) {
    const header = el(0x1a45dfa3, concat([
      uint(0x4286, 1), uint(0x42f7, 1), uint(0x42f2, 4), uint(0x42f3, 8),
      str(0x4282, "webm"), uint(0x4287, 2), uint(0x4285, 2),
    ]));
    const info = el(0x1549a966, concat([
      uint(0x2ad7b1, 1000000), str(0x4d80, "smokedet"), str(0x5741, "smokedet"), float64(0x4489, durationMs),
    ]));
    const tracks = el(0x1654ae6b, el(0xae, concat([
      uint(0xd7, 1), uint(0x73c5, 1), uint(0x83, 1), str(0x86, "V_VP8"),
      el(0xe0, concat([uint(0xb0, width), uint(0xba, height)])),
    ])));
    const clusters = [];
    let i = 0;
    while (i < chunks.length) { // кластер — от ключевого кадра, не длиннее 30 с (смещение блока — int16 мс)
      const base = Math.round(chunks[i].ms);
      const blocks = [uint(0xe7, base)];
      do {
        const c = chunks[i];
        const rel = Math.round(c.ms) - base;
        const head = new Uint8Array(4);
        head[0] = 0x81; // дорожка 1
        new DataView(head.buffer).setInt16(1, rel);
        head[3] = c.key ? 0x80 : 0x00;
        blocks.push(el(0xa3, concat([head, c.data])));
        i++;
      } while (i < chunks.length && !(chunks[i].key && Math.round(chunks[i].ms) - base > 2000)
               && Math.round(chunks[i].ms) - base < 30000);
      clusters.push(el(0x1f43b675, concat(blocks)));
    }
    const segment = el(0x18538067, concat([info, tracks, ...clusters]));
    return new Blob([header, segment], { type: "video/webm" });
  }

  // render(i, canvas) рисует кадр i из n; fps — частота кадров. onProgress(0..1).
  async function encode({ n, fps, width, height, render, onProgress }) {
    const canvas = document.createElement("canvas");
    canvas.width = width; canvas.height = height;
    if ("VideoEncoder" in window) {
      const config = { codec: "vp8", width, height, bitrate: 1_500_000, framerate: fps };
      // проверка поддержки может не ответить (например, без GPU-процесса) — тогда запись в реальном времени
      const support = await Promise.race([
        VideoEncoder.isConfigSupported(config).catch(() => ({ supported: false })),
        new Promise((r) => setTimeout(() => r({ supported: false }), 2000)),
      ]);
      if (support.supported) return encodeWebCodecs({ n, fps, width, height, render, onProgress, canvas, config });
    }
    return encodeRealtime({ n, fps, render, onProgress, canvas });
  }

  async function encodeWebCodecs({ n, fps, width, height, render, onProgress, canvas, config }) {
    const chunks = [];
    let failure = null;
    const encoder = new VideoEncoder({
      output: (chunk) => {
        const data = new Uint8Array(chunk.byteLength);
        chunk.copyTo(data);
        chunks.push({ data, ms: chunk.timestamp / 1000, key: chunk.type === "key" });
      },
      error: (e) => { failure = e; },
    });
    encoder.configure(config);
    const step = 1e6 / fps;
    for (let i = 0; i < n; i++) {
      if (failure) throw failure;
      render(i, canvas);
      const frame = new VideoFrame(canvas, { timestamp: Math.round(i * step), duration: Math.round(step) });
      encoder.encode(frame, { keyFrame: i % (fps * 2) === 0 });
      frame.close();
      if (encoder.encodeQueueSize > 8) await new Promise((r) => setTimeout(r, 0));
      if (i % 10 === 0) { onProgress?.(i / n); await new Promise((r) => setTimeout(r, 0)); }
    }
    await encoder.flush();
    encoder.close();
    if (failure) throw failure;
    chunks.sort((a, b) => a.ms - b.ms);
    onProgress?.(1);
    return mux(chunks, width, height, (n * 1000) / fps);
  }

  function encodeRealtime({ n, fps, render, onProgress, canvas }) {
    return new Promise((resolve, reject) => {
      const stream = canvas.captureStream(fps);
      const type = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"].find((t) => MediaRecorder.isTypeSupported(t));
      if (!type) { reject(new Error("браузер не умеет записывать видео")); return; }
      const rec = new MediaRecorder(stream, { mimeType: type });
      const parts = [];
      rec.ondataavailable = (e) => e.data.size && parts.push(e.data);
      rec.onstop = () => resolve(new Blob(parts, { type: "video/webm" }));
      rec.onerror = (e) => reject(e.error || new Error("ошибка записи"));
      let i = 0;
      render(0, canvas);
      rec.start(1000);
      const timer = setInterval(() => {
        i++;
        if (i >= n) { clearInterval(timer); rec.stop(); onProgress?.(1); return; }
        render(i, canvas);
        onProgress?.(i / n);
      }, 1000 / fps);
    });
  }

  function save(blob, name) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 30000);
  }

  return { encode, save, mux };
})();
