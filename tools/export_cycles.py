#!/usr/bin/env python
"""Выгрузка циклов «рука ко рту» в кадры для ручной разметки (tools/label_server.py).

Для каждого найденного цикла сохраняется короткая последовательность кадров вокруг человека
(кроп по объединению рамок, зелёная рамка — размечаемый человек, красная кайма — пауза у рта)
и строка в cycles.jsonl с автоматическими признаками. Видео не покидает машину.

    python tools/export_cycles.py --inputs data/ref.csv --out data/labeling/set1
    python tools/export_cycles.py --inputs data/ref.csv --out data/labeling/set1 --rejected all
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smokedet.config import load_config  # noqa: E402
from smokedet.io_video import FrameReader  # noqa: E402
from smokedet.pipeline import LOG_FORMAT, ROOT, Pipeline, read_inputs_csv  # noqa: E402

log = logging.getLogger("export_cycles")

CROP_PAD = 0.3  # поля вокруг объединения рамок, доля размера
TARGET_SIDE = 360  # мелкие кропы увеличиваются до этой длинной стороны, px
MAX_SIDE = 640  # крупные — уменьшаются до неё
JPEG_QUALITY = 88


def want_cycle(c, mode: str) -> bool:
    """none — только валидные; object — ещё отброшенные по предметам (самые интересные «двойники»); all — все."""
    if c.valid:
        return True
    if mode == "all":
        return True
    return mode == "object" and "object" in c.reject_reason


def crop_rect(boxes: np.ndarray, w: int, h: int):
    x1, y1 = boxes[:, 0].min(), boxes[:, 1].min()
    x2, y2 = boxes[:, 2].max(), boxes[:, 3].max()
    px, py = CROP_PAD * (x2 - x1), CROP_PAD * (y2 - y1)
    x1, y1, x2, y2 = max(0, x1 - px), max(0, y1 - py), min(w, x2 + px), min(h, y2 + py)
    return int(x1), int(y1), int(max(x1 + 8, x2)), int(max(y1 + 8, y2))


def render_cycle(reader: FrameReader, sig, c, pad: float, out_dir: Path, w: int, h: int):
    """Пишет JPEG-кадры цикла в out_dir. Возвращает времена кадров (с от начала клипа) или None."""
    i0, i1 = sig.index_at(c.start - pad), sig.index_at(c.end + pad)
    if i1 <= i0:
        return None
    rx1, ry1, rx2, ry2 = crop_rect(sig.boxes[i0 : i1 + 1], w, h)
    scale = float(np.clip(TARGET_SIDE / max(rx2 - rx1, ry2 - ry1), 1.0, 4.0))
    scale = min(scale, MAX_SIDE / max(rx2 - rx1, ry2 - ry1))
    out_dir.mkdir(parents=True, exist_ok=True)
    times = []
    for n, i in enumerate(range(i0, i1 + 1)):
        frame = reader.get(int(sig.frame_idx[i]), float(sig.t[i]))
        if frame is None:
            continue
        crop = frame[ry1:ry2, rx1:rx2]
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA)
        bx1, by1, bx2, by2 = ((sig.boxes[i] - np.array([rx1, ry1, rx1, ry1])) * scale).astype(int)
        cv2.rectangle(crop, (bx1, by1), (bx2, by2), (0, 220, 0), 1)
        if c.hold_start <= sig.t[i] <= c.hold_end:
            cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, crop.shape[0] - 1), (0, 0, 255), 4)
        cv2.imwrite(str(out_dir / f"{n:03d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        times.append(round(float(sig.t[i]), 3))
    return times or None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Выгрузка циклов для разметки")
    p.add_argument("--inputs", required=True, help="CSV со списком клипов, как у run.py")
    p.add_argument("--out", required=True, help="папка набора для разметки")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    p.add_argument("--device")
    p.add_argument("--rejected", choices=["none", "object", "all"], default="object",
                   help="какие отброшенные циклы тоже выгружать (по умолчанию — отброшенные по предметам)")  # fmt: skip
    p.add_argument("--pad", type=float, default=1.0, help="секунд контекста до и после цикла")
    p.add_argument("--cache", help="папка кэша треков (по умолчанию <out>/cache)")
    p.add_argument("--force", action="store_true", help="перерисовать кадры уже выгруженных циклов")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, datefmt="%H:%M:%S")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config, list(args.set) + ([f"detector.device={args.device}"] if args.device else []))
    pipe = Pipeline(cfg)
    index_path = out / "cycles.jsonl"
    records = {}
    if index_path.exists():
        records = {r["id"]: r for r in map(json.loads, index_path.read_text(encoding="utf-8").splitlines()) if r}

    for clip in read_inputs_csv(args.inputs):
        ts = pipe.track(clip, Path(args.cache) if args.cache else out / "cache")
        an = pipe.analyze(clip, ts)
        w, h = ts.meta.width, ts.meta.height
        todo = []
        for tid, cs in an.cycles.items():
            for k, c in enumerate(cs):
                if want_cycle(c, args.rejected):
                    todo.append((c.start, tid, k, c))
        todo.sort(key=lambda x: x[0])
        log.info("[%s] циклов к выгрузке: %d", clip.clip_id, len(todo))
        with FrameReader(clip.video_path, synthetic_from=ts.meta.synthetic_from) as reader:
            for _, tid, k, c in todo:
                cid = f"{clip.clip_id}__t{tid}_{k}"
                frames_dir = out / "frames" / cid
                if cid in records and frames_dir.is_dir() and not args.force:
                    continue
                times = render_cycle(reader, an.signals[tid], c, args.pad, frames_dir, w, h)
                if times is None:
                    log.warning("[%s] цикл %s: кадры не прочитаны", clip.clip_id, cid)
                    continue
                in_event = any(e.passed and e.track_id == tid and e.start - 0.5 <= c.start <= e.end + 0.5 for e in an.events)
                records[cid] = {
                    "id": cid, "clip_id": clip.clip_id, "camera_id": clip.camera_id, "track_id": tid,
                    "start": round(c.start, 2), "end": round(c.end, 2),
                    "hold_start": round(c.hold_start, 2), "hold_end": round(c.hold_end, 2),
                    "times": times, "valid": c.valid, "reject_reason": c.reject_reason, "in_event": in_event,
                    "auto": {
                        "direct": c.direct, "flash": c.flash, "vapor": c.vapor,
                        "vapor_net": None if np.isnan(c.vapor_net) else round(float(c.vapor_net), 3),
                        "neg_fraction": round(float(c.neg_fraction), 3), "strong_negative": c.strong_negative,
                        "min_dist": round(float(c.min_dist), 3), "quality": round(float(c.quality), 3),
                    },
                }  # fmt: skip
        index_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records.values()), encoding="utf-8")
    log.info("готово: %d циклов в %s", len(records), index_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
