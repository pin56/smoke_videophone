#!/usr/bin/env python
"""Подготовка весов к офлайн-запуску: скачать всё, что требует конфиг, и «запечь» словарь верификатора.

    python tools/prepare_weights.py [--config configs/default.yaml] [--set section.key=value ...]

Запускать один раз с интернетом (до площадки). Скачивает в weights/ pose-модель, open-vocab модель
верификатора и COCO-модель для отката, затем сохраняет open-vocab модель с заданным словарём
(weights/<модель>-vocab-<хэш>.pt). С таким файлом прогон не загружает текстовый энкодер CLIP.
"""
from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smokedet.config import load_config  # noqa: E402
from smokedet.export import file_sha256  # noqa: E402
from smokedet.pipeline import project_path  # noqa: E402
from smokedet.verifier import ObjectVerifier, baked_path, vocabulary  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Скачать веса и запечь словарь верификатора")
    p.add_argument("--config", default=None)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config, args.set)

    from ultralytics import YOLO

    for path in (cfg.detector.pose_model, cfg.verifier.model, cfg.verifier.fallback_model):
        YOLO(project_path(path))  # скачивает недостающий файл по указанному пути
        print(f"{path}: {file_sha256(project_path(path))}")
    vcfg = dataclasses.replace(
        cfg.verifier, model=project_path(cfg.verifier.model), fallback_model=project_path(cfg.verifier.fallback_model)
    )
    v = ObjectVerifier(vcfg, "cpu")
    if v.mode != "open_vocab":
        print("Open-vocab модель недоступна: словарь не запечён", file=sys.stderr)
        return 1
    baked = baked_path(vcfg.model, vocabulary(vcfg))
    if not baked.is_file():
        print(f"Не удалось сохранить {baked}", file=sys.stderr)
        return 1
    print(f"словарь верификатора: {baked} ({'уже был' if not v.clip_used else 'запечён сейчас'}): {file_sha256(str(baked))}")

    import clip

    from smokedet.appearance import weights_path

    wdir = project_path(cfg.appearance.weights_dir)
    clip.load(cfg.appearance.model, device="cpu", download_root=wdir)  # скачивает недостающий файл
    print(f"CLIP внешнего вида {cfg.appearance.model}: {file_sha256(weights_path(cfg.appearance.model, wdir))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
