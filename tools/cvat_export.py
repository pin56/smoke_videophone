#!/usr/bin/env python
"""Экспорт разметки в CVAT: треки людей из кэша трекера и эпизоды из rasmetka_gt.csv.

    python tools/cvat_export.py --inputs runs/goal/all.csv --cache runs/goal/cache --out runs/cvat
        [--marks rasmetka_gt.csv] [--no-persons] [--config ...] [--set section.key=value ...]

Детекция не перезапускается: нужен кэш треков (<cache>/<clip_id>_tracks.pkl) любого прогона run.py
с тем же списком клипов. В --out: <clip_id>.xml («CVAT for video 1.1») на каждое видео, labels.json
(метки проекта CVAT) и index.csv (сводка по клипам).

В CVAT (только локальный: видео объекта нельзя отправлять в облако): создать проект, в Constructor → Raw
вставить labels.json; на каждое видео — задачу проекта с именем clip_id (frame step 1, без start/stop
frame); Actions → Upload annotations → «CVAT 1.1» → <clip_id>.xml. Сомнительные привязки отмечены
атрибутом check: фильтр `attr["check"] == true`, после проверки флажок снять. Обратно — tools/cvat_import.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smokedet.config import load_config  # noqa: E402
from smokedet.cvat import EVENT_LABELS, PERSON, build_clip, labels_spec, load_marks, load_trackset, n_frames_of, write_xml  # noqa: E402
from smokedet.pipeline import ROOT, read_inputs_csv  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Экспорт разметки в CVAT (CVAT for video 1.1)")
    p.add_argument("--inputs", required=True, help="CSV клипов, как у run.py")
    p.add_argument("--cache", required=True, help="папка кэша треков (<clip_id>_tracks.pkl)")
    p.add_argument("--marks", default=str(ROOT / "rasmetka_gt.csv"))
    p.add_argument("--out", required=True)
    p.add_argument("--no-persons", action="store_true", help="без треков людей, только эпизоды")
    p.add_argument("--config", default=None)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args(argv)

    cfg = load_config(args.config, args.set)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    marks = load_marks(args.marks)
    used, index, missing = set(), [], []
    for clip in read_inputs_csv(args.inputs):
        video = Path(clip.video_path)
        cache = Path(args.cache) / f"{clip.clip_id}_tracks.pkl"
        if not cache.exists():
            missing.append(clip.clip_id)
            continue
        ts = load_trackset(cache)
        used.add(video.stem)
        tracks, episodes = build_clip(ts, marks.get(video.stem, []), cfg, persons=not args.no_persons)
        n = n_frames_of(ts)
        write_xml(out / f"{clip.clip_id}.xml", clip.clip_id, video.name, n, ts.meta.width, ts.meta.height, tracks)
        events = [t for t in tracks if t.label != PERSON]
        index.append({
            "clip_id": clip.clip_id, "camera_id": clip.camera_id, "video": video.name, "frames": n,
            "fps": f"{ts.meta.fps:.3f}", **{k: sum(t.label == k for t in events) for k in EVENT_LABELS},
            "check": sum(t.attributes.get("check") == "true" for t in events),
            "persons": sum(t.label == PERSON for t in tracks),
        })  # fmt: skip
        r = index[-1]
        print(f"{clip.clip_id:24s} эпизодов {len(episodes):3d} (smoking {r['smoking']}, not_smoking "
              f"{r['not_smoking']}, ignore {r['ignore']}; проверить {r['check']}), людей {r['persons']}")
    (out / "labels.json").write_text(json.dumps(labels_spec(), ensure_ascii=False, indent=2), encoding="utf-8")
    if index:
        with open(out / "index.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(index[0]))
            w.writeheader()
            w.writerows(index)
    if missing:
        print(f"Нет кэша треков ({len(missing)}): {', '.join(missing)} — прогоните run.py с --out, чей cache указан")
    unused = sorted(set(marks) - used)
    if unused:
        print(f"Отметки без клипа в --inputs ({len(unused)}): {', '.join(unused)}")
    print(f"Готово: {len(index)} XML, метки проекта — {out / 'labels.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
