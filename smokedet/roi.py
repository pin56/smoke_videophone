"""Маска зоны интереса: PNG/JPG (ненулевые пиксели — зона) или JSON с полигонами.

Форматы JSON:
    {"include": [[[x, y], ...], ...], "exclude": [[[x, y], ...]], "normalized": false}
    {"polygons": [[[x, y], ...], ...]}
    [[x, y], [x, y], ...]                      # один полигон
Координаты в пикселях исходного кадра или (normalized: true / все <= 1) в долях кадра.
Необязательный "frame_size": [w, h] (пишет tools/draw_roi.py) — размер кадра, на котором размечены пиксели:
на видео другого разрешения полигоны масштабируются, как и PNG-маска.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

Polygon = List[List[float]]


def _as_polygons(value) -> List[Polygon]:
    if not value:
        return []
    # один полигон: [[x, y], ...]; несколько: [[[x, y], ...], ...]
    if isinstance(value[0][0], (int, float)):
        return [value]
    return list(value)


class Roi:
    def __init__(self, mask: Optional[np.ndarray]):
        self.mask = mask  # bool HxW или None (весь кадр)

    @classmethod
    def full(cls) -> "Roi":
        return cls(None)

    @classmethod
    def load(cls, path: Optional[str], width: int, height: int) -> "Roi":
        if not path:
            return cls.full()
        p = Path(path)
        if p.suffix.lower() == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):
                data = {"include": data}
            include = _as_polygons(data.get("include") or data.get("polygons"))
            exclude = _as_polygons(data.get("exclude"))
            normalized, size = data.get("normalized"), data.get("frame_size")
            if size and not normalized:
                kx, ky = width / float(size[0]), height / float(size[1])

                def rescale(polys: List[Polygon]) -> List[Polygon]:
                    return [[[x * kx, y * ky] for x, y in poly] for poly in polys]

                include, exclude, normalized = rescale(include), rescale(exclude), False
            return cls.from_polygons(include, exclude, width, height, normalized)
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise IOError(f"Не удалось прочитать маску ROI: {path}")
        if img.shape != (height, width):
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
        return cls(img > 127)

    @classmethod
    def from_polygons(
        cls,
        include: List[Polygon],
        exclude: List[Polygon],
        width: int,
        height: int,
        normalized: Optional[bool] = None,
    ) -> "Roi":
        if normalized is None:
            values = [abs(v) for poly in include + exclude for pt in poly for v in pt]
            normalized = bool(values) and max(values) <= 1.0
        scale = np.array([width, height], dtype=np.float64) if normalized else np.ones(2)

        def to_px(poly: Polygon) -> np.ndarray:
            return np.round(np.asarray(poly, dtype=np.float64) * scale).astype(np.int32)

        mask = np.zeros((height, width), np.uint8)
        if include:
            cv2.fillPoly(mask, [to_px(p) for p in include], 1)
        else:
            mask[:] = 1
        if exclude:
            cv2.fillPoly(mask, [to_px(p) for p in exclude], 0)
        return cls(mask.astype(bool))

    def contains(self, x: float, y: float) -> bool:
        if self.mask is None:
            return True
        h, w = self.mask.shape
        xi = int(min(max(x, 0), w - 1))
        yi = int(min(max(y, 0), h - 1))
        return bool(self.mask[yi, xi])

    def draw(self, frame: np.ndarray, color=(255, 200, 0), thickness: int = 2) -> None:
        if self.mask is None:
            return
        contours, _ = cv2.findContours(self.mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, color, thickness)


def anchor_point(box: np.ndarray, anchor: str) -> tuple:
    x1, y1, x2, y2 = box[:4]
    cx = (x1 + x2) / 2
    if anchor == "feet":
        return cx, y2
    if anchor == "head":
        return cx, y1 + 0.1 * (y2 - y1)
    return cx, (y1 + y2) / 2
