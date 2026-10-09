"""Скачивание весов из релиза: без сети, загрузчик подменён."""
import zipfile

import pytest

from smokedet import weights
from smokedet.config import Config


def fake_fetch(files):
    """Загрузчик, который пишет заранее заданное содержимое файла релиза."""

    def fetch(name, dst, sha):
        assert sha == weights.RELEASE[next(k for k, v in weights.RELEASE.items() if v[0] == name)][1]
        files[name](dst)

    return fetch


def test_ensure_file_and_zip(tmp_path):
    def zip_dir(dst):
        with zipfile.ZipFile(dst, "w") as z:
            z.writestr("yolo11n_openvino_model/yolo11n.xml", "<xml/>")
            z.writestr("yolo11n_openvino_model/metadata.yaml", "task: detect")

    fetch = fake_fetch({"yolo11n.pt": lambda d: d.write_bytes(b"pt"), "yolo11n_openvino_model.zip": zip_dir})
    have = tmp_path / "w" / "yolo11m-pose.pt"
    have.parent.mkdir()
    have.write_bytes(b"old")
    paths = [tmp_path / "w" / "yolo11n.pt", tmp_path / "w" / "yolo11n_openvino_model", have, tmp_path / "w" / "other.pt"]
    got = weights.ensure(paths, fetch)
    assert got == paths[:2]
    assert (tmp_path / "w" / "yolo11n.pt").read_bytes() == b"pt"
    assert (tmp_path / "w" / "yolo11n_openvino_model" / "yolo11n.xml").exists()
    assert sorted(p.name for p in (tmp_path / "w").iterdir()) == ["yolo11m-pose.pt", "yolo11n.pt", "yolo11n_openvino_model"]
    assert have.read_bytes() == b"old"  # есть — не трогается
    assert weights.ensure(paths, fetch) == []


def test_needed_cpu_profile():
    pytest.importorskip("clip")
    cfg = Config()
    cfg.realtime.det_model = "weights/yolo11n_openvino_model"
    cfg.appearance.model, cfg.appearance.input_px, cfg.appearance.openvino = "ViT-L/14", 168, True
    names = [p.name for p in weights.needed(cfg, "cpu")]
    assert names == ["yolo11n_openvino_model", "yolo11m-pose.pt", "ViT-L-14.pt", "ViT-L-14-visual168_openvino"]
    assert all(n in weights.RELEASE for n in names)
    assert "ViT-L-14-visual168_openvino" not in [p.name for p in weights.needed(cfg, "0")]
