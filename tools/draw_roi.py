#!/usr/bin/env python
"""Разметка маски зоны интереса мышкой по кадру видео → roi.json.

    python tools/draw_roi.py --video data/clip.mp4 --out data/roi_cam1.json [--frame 0]

ЛКМ — точка; Enter/n — замкнуть полигон; e — режим include/exclude; u — отменить точку;
s — сохранить и выйти; q/Esc — выйти без сохранения.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--max-width", type=int, default=1400)
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Не удалось прочитать кадр {args.frame} из {args.video}")
    h, w = frame.shape[:2]
    k = min(1.0, args.max_width / w)
    view = cv2.resize(frame, (int(w * k), int(h * k)))

    polys = {"include": [], "exclude": []}
    current: list = []
    mode = ["include"]

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            current.append([round(x / k, 1), round(y / k, 1)])

    win = "ROI: LMB point, Enter close, e mode, u undo, s save, q quit"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        img = view.copy()
        for kind, color in (("include", (0, 0, 255)), ("exclude", (255, 0, 255))):
            for poly in polys[kind]:
                cv2.polylines(img, [(np.array(poly) * k).astype(np.int32)], True, color, 2)
        if current:
            pts = (np.array(current) * k).astype(np.int32)
            cv2.polylines(img, [pts], False, (0, 255, 255), 2)
            for pt in pts:
                cv2.circle(img, tuple(int(v) for v in pt), 3, (0, 255, 255), -1)
        cv2.putText(img, f"mode: {mode[0]}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(win, img)
        key = cv2.waitKey(30) & 0xFF
        if key in (13, ord("n")) and len(current) >= 3:
            polys[mode[0]].append(list(current))
            current.clear()
        elif key == ord("e"):
            mode[0] = "exclude" if mode[0] == "include" else "include"
        elif key == ord("u") and current:
            current.pop()
        elif key == ord("s"):
            if len(current) >= 3:
                polys[mode[0]].append(list(current))
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            data = {**polys, "normalized": False, "frame_size": [w, h], "source": str(args.video)}
            Path(args.out).write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"Сохранено: {args.out} (include: {len(polys['include'])}, exclude: {len(polys['exclude'])})")
            break
        elif key in (ord("q"), 27):
            break
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
