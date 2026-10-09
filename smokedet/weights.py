"""Веса потокового режима из релиза GitHub: недостающие файлы скачиваются в weights/, sha256 сверяется.

Папки OpenVINO лежат в релизе zip-архивами (папка в корне архива). Файл, которого нет в RELEASE, не трогается:
.pt Ultralytics и CLIP при отсутствии скачивают сами библиотеки. Репозиторий закрытый — нужен токен
(GH_TOKEN / GITHUB_TOKEN или `gh auth login`); без доступа веса готовит tools/prepare_weights.py.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import List, Optional
from urllib.request import Request, urlopen

from .config import Config

log = logging.getLogger(__name__)

REPO = "pin56/smoke_videophone"
TAG = "v1.1"
# путь в weights/ → (файл релиза, sha256 файла релиза, размер)
RELEASE = {
    "yolo11n.pt": ("yolo11n.pt", "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1", 5613764),
    "yolo11m-pose.pt": ("yolo11m-pose.pt", "29b17eaf3a3117cbea906090dbedf9159f7c6a49db58ec8b99ed2dfde1cf6eb2", 42459307),
    "ViT-L-14.pt": ("ViT-L-14.pt", "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836", 932768134),
    "yolo11n_openvino_model": ("yolo11n_openvino_model.zip",
                               "e3923bb51c2a39c3d6e09fd3c0d65390b070812eb7449cfd25522eee2e811de0", 10843238),
    "yolo11m-pose_openvino_model": ("yolo11m-pose_openvino_model.zip",
                                    "cdc35d84b8fb328e09c0483d16ef0181502665cc89f78cef9c6e37988c98cf30", 84050084),
    "ViT-L-14-visual168_openvino": ("ViT-L-14-visual168_openvino.zip",
                                    "63d070dc4b5280ad5af51894b4f0b872bb812cd18a0fa3e6c042132325190013", 1216248939),
}  # fmt: skip


def needed(cfg: Config, device: str) -> List[Path]:
    """Веса, которые потоковый режим с этим конфигом загрузит на device (cpu — ещё IR CLIP при appearance.openvino)."""
    from .appearance import openvino_path, weights_path
    from .pipeline import project_path

    paths = [Path(project_path(cfg.realtime.det_model)), Path(project_path(cfg.realtime.pose_model))]
    if cfg.realtime.objects:
        clip_file = weights_path(cfg.appearance.model, project_path(cfg.appearance.weights_dir))
        paths.append(Path(clip_file))
        if device == "cpu" and cfg.appearance.openvino:
            paths.append(Path(openvino_path(clip_file, cfg.appearance.input_px)).parent)
    return paths


def _token() -> Optional[str]:
    tok = os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if tok:
        return tok
    try:
        out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _request(url: str, token: Optional[str], accept: str) -> Request:
    req = Request(url, headers={"Accept": accept, "User-Agent": "smokedet"})
    if token:  # не уходит на редирект к хранилищу файлов (там своя подпись в ссылке)
        req.add_unredirected_header("Authorization", f"Bearer {token}")
    return req


def download(name: str, dst: Path, sha256: str) -> None:
    """Файл релиза TAG → dst (через .part), с проверкой sha256."""
    token = _token()
    api = f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}"
    try:
        with urlopen(_request(api, token, "application/vnd.github+json"), timeout=30) as r:
            assets = {a["name"]: a for a in json.load(r)["assets"]}
    except Exception as e:  # noqa: BLE001 — 404 у закрытого репозитория без токена, нет сети
        raise RuntimeError(f"релиз {REPO}@{TAG} недоступен ({e}): нужен GH_TOKEN или `gh auth login`, "
                           f"иначе tools/prepare_weights.py") from e  # fmt: skip
    if name not in assets:
        raise RuntimeError(f"в релизе {TAG} нет файла {name}")
    total = assets[name]["size"]
    part = dst.with_name(dst.name + ".part")
    h, done, step = hashlib.sha256(), 0, 0
    with urlopen(_request(assets[name]["url"], token, "application/octet-stream"), timeout=60) as r, open(part, "wb") as f:
        for chunk in iter(lambda: r.read(1 << 20), b""):
            f.write(chunk)
            h.update(chunk)
            done += len(chunk)
            if total and done * 10 // total > step:
                step = done * 10 // total
                log.info("%s: %d %% (%d из %d МБ)", name, step * 10, done >> 20, total >> 20)
    if h.hexdigest() != sha256:
        part.unlink()
        raise RuntimeError(f"{name}: sha256 не совпал, файл удалён")
    part.replace(dst)


def ensure(paths: List[Path], fetch=download) -> List[Path]:
    """Скачивает отсутствующие веса из RELEASE (zip распаковывается рядом); возвращает скачанные."""
    got = []
    for p in paths:
        if p.exists() or p.name not in RELEASE:
            continue
        name, sha, size = RELEASE[p.name]
        p.parent.mkdir(parents=True, exist_ok=True)
        log.info("нет %s: скачиваю %s из релиза %s (%d МБ)", p, name, TAG, size >> 20)
        if name.endswith(".zip"):
            tmp = p.parent / name
            fetch(name, tmp, sha)
            try:
                with zipfile.ZipFile(tmp) as z:
                    z.extractall(p.parent / f".{p.name}.tmp")
                shutil.move(str(p.parent / f".{p.name}.tmp" / p.name), str(p))
            finally:
                tmp.unlink(missing_ok=True)
                shutil.rmtree(p.parent / f".{p.name}.tmp", ignore_errors=True)
        else:
            fetch(name, p, sha)
        got.append(p)
    return got


def ensure_for(cfg: Config, device: str) -> None:
    """Перед загрузкой моделей run_rt.py/serve.py: ошибка скачивания — предупреждение (модель сообщит о себе сама)."""
    try:
        ensure(needed(cfg, device))
    except Exception as e:  # noqa: BLE001
        log.warning("веса не скачаны: %s", e)
