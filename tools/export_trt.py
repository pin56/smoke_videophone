#!/usr/bin/env python
"""Экспорт pose-модели в TensorRT FP16 — инференс позы примерно вдвое быстрее PyTorch FP16.

    python tools/export_trt.py [--model weights/yolo11m-pose.pt] [--imgsz 1280] [--workspace 3]

Нужны пакеты tensorrt-cu12 (10.x), onnx, onnxslim (см. requirements.txt). Engine привязан к GPU и версии
TensorRT: собирать на той машине, где будет прогон (на RTX 2060 — ~4 мин). Результат —
weights/<имя>-fp16-dyn<imgsz>.engine; включается через --set detector.pose_model=<путь к engine>.

Вход динамический (до imgsz по большей стороне), поэтому кадр, как и в PyTorch, приводится к прямоугольнику
1280×736, а не к квадрату 1280×1280. Батч — 1: уточнение позы на кропах (батч кропов) идёт на исходном .pt
(tracking.refine_model_path). Числа позы немного отличаются от PyTorch FP16, пороги подбирать в том же режиме.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from smokedet.pipeline import project_path  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="weights/yolo11m-pose.pt")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--workspace", type=float, default=3.0, help="ГиБ памяти GPU на сборку")
    p.add_argument("--device", default="0")
    args = p.parse_args(argv)

    from ultralytics import YOLO

    src = Path(project_path(args.model))
    if not src.exists():
        raise SystemExit(f"{src}: нет весов (tools/prepare_weights.py)")
    dst = src.with_name(f"{src.stem}-fp16-dyn{args.imgsz}.engine")
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:  # ONNX и engine ultralytics кладёт рядом с весами
        work = Path(tmp) / src.name
        shutil.copy(src, work)
        out = YOLO(str(work)).export(format="engine", imgsz=args.imgsz, dynamic=True, batch=1, quantize=16,
                                     workspace=args.workspace, device=args.device, simplify=True)  # fmt: skip
        shutil.copy(out, dst)
    print(f"{dst} ({time.time() - t0:.0f} с); включить: --set detector.pose_model={dst.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
