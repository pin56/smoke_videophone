"""YOLO (детекция и поза), экспортированная ultralytics в OpenVINO, без predict() ultralytics — для CPU.

predict() готовит вход в torch (BGR→RGB flip, contiguous, float, /255) и строит Results: на 2 ядрах это треть
времени вызова позы на мелком батче. Здесь предобработка встроена в граф OpenVINO (вход — uint8 BGR NHWC), а
декодирование и NMS — в numpy, с той же математикой, что ultralytics (utils/nms.py: порог по уверенности, рамки
xywh→xyxy, жадный NMS как torchvision.ops.nms, не больше max_det; рамки и точки обрезаются по входу).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml

MAX_DET = 300  # как max_det в ultralytics


def is_openvino_dir(path: str) -> bool:
    p = Path(path)
    return p.is_dir() and p.name.endswith("_openvino_model") and any(p.glob("*.xml"))


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Жадный NMS (как torchvision.ops.nms): индексы оставленных рамок по убыванию уверенности."""
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes.T
    area = (x2 - x1) * (y2 - y1)
    keep = []
    while order.size:
        i, rest = order[0], order[1:]
        keep.append(i)
        w = np.clip(np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]), 0, None)
        h = np.clip(np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]), 0, None)
        inter = w * h
        order = rest[inter / (area[i] + area[rest] - inter + 1e-9) <= iou]
    return np.asarray(keep, dtype=np.int64)


def decode(pred: np.ndarray, conf: float, iou: float, nc: int, cls: int = 0, max_det: int = MAX_DET):
    """Выход головы одного изображения (4 + nc + доп. каналы, N) -> (рамки xyxy (K, 4), уверенность (K,),
    доп. каналы (K, E)) класса cls после порога и NMS, по убыванию уверенности."""
    p = pred.T
    score = p[:, 4 + cls]
    if nc > 1:  # ultralytics берёт класс с наибольшей оценкой: человек остаётся, только если он — лучший класс
        m = (score > conf) & (np.argmax(p[:, 4 : 4 + nc], axis=1) == cls)
    else:
        m = score > conf
    p, score = p[m], score[m]
    xy, wh = p[:, :2], p[:, 2:4] / 2
    boxes = np.concatenate([xy - wh, xy + wh], axis=1)
    keep = nms(boxes, score, iou)[:max_det]
    return boxes[keep], score[keep], p[keep, 4 + nc :]


class OvYolo:
    """Модель ultralytics в формате OpenVINO (папка *_openvino_model) на CPU: вход — батч uint8 BGR (B, H, W, 3)."""

    def __init__(self, path: str, threads: int = 0):
        import openvino as ov
        from openvino.preprocess import ColorFormat, PrePostProcessor

        d = Path(path)
        meta = yaml.safe_load((d / "metadata.yaml").read_text(encoding="utf-8"))
        self.task = meta["task"]
        self.nc = len(meta["names"])
        self.kpt_shape = tuple(meta.get("kpt_shape") or ())
        core = ov.Core()
        model = core.read_model(next(d.glob("*.xml")))
        ppp = PrePostProcessor(model)
        ppp.input().tensor().set_element_type(ov.Type.u8).set_layout(ov.Layout("NHWC")).set_color_format(ColorFormat.BGR)
        ppp.input().preprocess().convert_element_type(ov.Type.f32).convert_color(ColorFormat.RGB).scale(255.0)
        ppp.input().model().set_layout(ov.Layout("NCHW"))
        config = {"PERFORMANCE_HINT": "LATENCY"}
        if threads > 0:
            config["INFERENCE_NUM_THREADS"] = threads
        self.compiled = core.compile_model(ppp.build(), "CPU", config)
        self.request = self.compiled.create_infer_request()

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        """(B, H, W, 3) uint8 BGR -> выход головы (B, 4 + nc + доп. каналы, N)."""
        self.request.infer({0: np.ascontiguousarray(batch)})
        return self.request.get_output_tensor(0).data.copy()

    def detect(self, batch: np.ndarray, conf: float, iou: float, cls: int = 0) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Рамки, уверенность и доп. каналы (у позы — точки (K, 17, 3)) класса cls в координатах входа."""
        h, w = batch.shape[1:3]
        out = []
        for pred in self(batch):
            boxes, score, extra = decode(pred, conf, iou, self.nc, cls)
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
            if self.kpt_shape:
                extra = extra.reshape(len(extra), *self.kpt_shape)
                extra[..., 0] = extra[..., 0].clip(0, w)
                extra[..., 1] = extra[..., 1].clip(0, h)
            out.append((boxes, score, extra))
        return out
