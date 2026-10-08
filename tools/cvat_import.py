#!/usr/bin/env python
"""Импорт разметки из CVAT в эталон для evaluate.py.

    python tools/cvat_import.py --inputs runs/goal/all.csv --cache runs/goal/cache --out gt.csv <XML/ZIP или папки>

Вход — выгрузки задач CVAT в формате «CVAT for video 1.1» (Export task dataset, без изображений): ZIP
с annotations.xml или сам XML. Клип определяется по имени видео задачи (<source>), имени задачи или имени
файла выгрузки — по stem видео или clip_id из --inputs. Секунды считаются по pts из кэша треков, поэтому
нужен тот же кэш, что и при экспорте (tools/cvat_export.py).

Выход: CSV camera_id, clip_id, label (POSITIVE|NEGATIVE|IGNORE), start_sec, end_sec, person_gt_id, peak_sec,
x1, y1, x2, y2, scenario, video, note, check. Трек person не переносится; каждый видимый отрезок трека
smoking / not_smoking / ignore — одно событие.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smokedet.cvat import FrameTime, gt_rows, load_trackset, n_frames_of, read_cvat, write_gt  # noqa: E402
from smokedet.pipeline import read_inputs_csv  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Импорт разметки из CVAT в эталон evaluate.py")
    p.add_argument("files", nargs="+", help="XML/ZIP выгрузки CVAT или папки с ними")
    p.add_argument("--inputs", required=True, help="CSV клипов, как у run.py")
    p.add_argument("--cache", required=True, help="папка кэша треков (<clip_id>_tracks.pkl)")
    p.add_argument("--out", required=True, help="эталон CSV")
    args = p.parse_args(argv)

    clips = {}
    for c in read_inputs_csv(args.inputs):
        clips[Path(c.video_path).stem.lower()] = c
        clips[c.clip_id.lower()] = c
    paths = []
    for f in map(Path, args.files):
        paths += sorted(x for x in f.rglob("*") if x.suffix.lower() in (".xml", ".zip")) if f.is_dir() else [f]

    rows, seen, problems = [], {}, []
    for path in paths:
        for cf in read_cvat(path):
            keys = [Path(cf.source).stem, cf.name, cf.origin]
            clip = next((clips[k.lower()] for k in keys if k and k.lower() in clips), None)
            if clip is None:
                problems.append(f"{cf.path}: клип не опознан (видео «{cf.source}», задача «{cf.name}»)")
                continue
            if clip.clip_id in seen:
                problems.append(f"{cf.path}: клип {clip.clip_id} уже взят из {seen[clip.clip_id]}, пропуск")
                continue
            cache = Path(args.cache) / f"{clip.clip_id}_tracks.pkl"
            if not cache.exists():
                problems.append(f"{cf.path}: нет кэша треков {cache}")
                continue
            seen[clip.clip_id] = cf.path
            ts = load_trackset(cache)
            n = n_frames_of(ts)
            stop = cf.stop_frame if cf.stop_frame is not None else n - 1
            if abs(stop + 1 - n) > 1:
                problems.append(f"{clip.clip_id}: кадров в CVAT {stop + 1}, у трекера {n} — время может сдвинуться")
            rows += gt_rows(cf, clip.camera_id, clip.clip_id, Path(clip.video_path).name, FrameTime.from_trackset(ts), stop)

    write_gt(Path(args.out), rows)
    by_label = collections.Counter(r["label"] for r in rows)
    unchecked = [r for r in rows if r["check"] == "true"]
    no_box = [r for r in rows if not r.get("x1")]
    print(f"Клипов {len(seen)}, событий {len(rows)}: " + ", ".join(f"{k} {v}" for k, v in sorted(by_label.items())))
    if unchecked:
        print(f"С неснятым флажком check: {len(unchecked)} (" + ", ".join(
            f"{r['clip_id']}@{r['start_sec']}" for r in unchecked[:10]) + (", …" if len(unchecked) > 10 else "") + ")")
    if no_box:
        print(f"Без рамки в peak_sec: {len(no_box)}")
    for msg in problems:
        print("ВНИМАНИЕ:", msg)
    print(f"Эталон: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
