"""Верификатор без модели: кэш детекций при смене conf/imgsz/классов, запечённый словарь, поиск питья по треку."""
import dataclasses

import numpy as np
import pytest
from test_logic import make_signal, puffs

import smokedet.verifier as vmod
from smokedet.config import CycleConfig, VaporConfig, VerifierConfig
from smokedet.cycles import detect_cycles
from smokedet.vapor import VaporMeter
from smokedet.verifier import ObjectVerifier, baked_path, vocabulary


class Arr:  # вместо тензора ultralytics
    def __init__(self, a):
        self.a = np.asarray(a)

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class Boxes:
    def __init__(self, rows):
        self.cls = Arr([r[0] for r in rows])
        self.conf = Arr([r[1] for r in rows])
        self.xyxy = Arr(np.array([r[2] for r in rows], float).reshape(-1, 4))

    def __len__(self):
        return len(self.cls.a)


class Result:
    def __init__(self, rows):
        self.boxes = Boxes(rows)


class FakeYOLO:
    """Отдаёт в центре кропа каждый класс словаря с уверенностью из canned (если она ≥ conf)."""

    canned = {}
    calls = []
    vocab_sets = []
    saved = []

    def __init__(self, path):
        self.path = path
        self.names = {0: "person"}

    def set_classes(self, classes):
        self.names = dict(enumerate(classes))
        FakeYOLO.vocab_sets.append(list(classes))

    def save(self, path):
        FakeYOLO.saved.append(path)

    def predict(self, crops, imgsz, conf, device, verbose):
        FakeYOLO.calls.append((len(crops), imgsz, conf))
        out = []
        for crop in crops:
            h, w = crop.shape[:2]
            box = [w * 0.45, h * 0.45, w * 0.55, h * 0.55]
            out.append(Result([(i, self.canned[n], box) for i, n in self.names.items() if self.canned.get(n, 0) >= conf]))
        return out


class FakeReader:
    def __init__(self, path, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def get(self, idx, t=None):
        return np.zeros((480, 640, 3), np.uint8)


@pytest.fixture
def fake(monkeypatch, tmp_path):
    monkeypatch.setattr(vmod, "FrameReader", FakeReader)
    FakeYOLO.canned, FakeYOLO.calls, FakeYOLO.vocab_sets, FakeYOLO.saved = {}, [], [], []
    cfg = VerifierConfig(model=str(tmp_path / "world.pt"), track_negative_classes=[], track_drink_frames=0)
    return ObjectVerifier(cfg, "cpu", factory=FakeYOLO), cfg


def _setup(starts=(3,)):
    sig = make_signal(puffs(list(starts)))
    return {1: detect_cycles(sig, CycleConfig())}, {1: sig}


def test_conf_filter_applied_after_cache(fake):
    v, cfg = fake
    FakeYOLO.canned = {"beverage can": 0.2}
    cycles, signals = _setup()
    v.verify("c", "v.mp4", cycles, signals, cfg)
    assert not cycles[1][0].valid and cycles[1][0].reject_reason == "object:beverage can"
    n_calls = len(FakeYOLO.calls)
    assert FakeYOLO.calls[0][2] == vmod.PREDICT_CONF  # модель видит и слабые детекции
    cycles, signals = _setup()
    v.verify("c", "v.mp4", cycles, signals, dataclasses.replace(cfg, conf=0.3))
    assert cycles[1][0].valid and "beverage can" not in cycles[1][0].objects  # отфильтровано по новому conf
    assert len(FakeYOLO.calls) == n_calls  # из кэша, без повторного предсказания


def test_imgsz_and_geometry_invalidate_cache(fake):
    v, cfg = fake
    cycles, signals = _setup()
    v.verify("c", "v.mp4", cycles, signals, cfg)
    n_calls = len(FakeYOLO.calls)
    cycles, signals = _setup()
    v.verify("c", "v.mp4", cycles, signals, dataclasses.replace(cfg, imgsz=320))
    assert len(FakeYOLO.calls) == n_calls + 1 and FakeYOLO.calls[-1][1] == 320
    cycles, signals = _setup()
    signals[1].scale[:] = 45.0  # другая нормировка — другие кропы
    v.verify("c", "v.mp4", cycles, signals, cfg)
    assert len(FakeYOLO.calls) == n_calls + 2


def test_class_lists_change_vocabulary(fake):
    v, cfg = fake
    FakeYOLO.canned = {"jar": 0.6}
    cycles, signals = _setup()
    v.verify("c", "v.mp4", cycles, signals, dataclasses.replace(cfg, negative_classes=["bottle"]))
    assert FakeYOLO.vocab_sets[-1] == vocabulary(dataclasses.replace(cfg, negative_classes=["bottle"]))
    assert cycles[1][0].valid  # «jar» нет в словаре
    cycles, signals = _setup()
    v.verify("c", "v.mp4", cycles, signals, cfg)  # снова полный словарь: детекции пересчитаны
    assert not cycles[1][0].valid and cycles[1][0].reject_reason == "object:jar"


def test_baked_vocabulary_skips_set_classes(monkeypatch, tmp_path):
    cfg = VerifierConfig(model=str(tmp_path / "world.pt"))
    classes = vocabulary(cfg)
    baked = baked_path(cfg.model, classes)
    assert baked.parent == tmp_path and baked.name.startswith("world-vocab-")
    FakeYOLO.vocab_sets, FakeYOLO.saved = [], []
    v = ObjectVerifier(cfg, "cpu", factory=FakeYOLO)
    assert v.clip_used and FakeYOLO.saved == [str(baked)]

    class BakedYOLO(FakeYOLO):
        def __init__(self, path):
            assert path == str(baked)
            self.names = dict(enumerate(classes))

    baked.write_bytes(b"x")
    FakeYOLO.vocab_sets = []
    v = ObjectVerifier(cfg, "cpu", factory=BakedYOLO)
    assert not v.clip_used and v.weights == str(baked) and FakeYOLO.vocab_sets == []
    assert v.describe()["weights"] == str(baked)


def test_fallback_to_coco_when_open_vocab_fails(tmp_path):
    class Broken(FakeYOLO):
        def set_classes(self, classes):
            raise RuntimeError("нет CLIP")

    cfg = VerifierConfig(model=str(tmp_path / "world.pt"), fallback_model=str(tmp_path / "coco.pt"))
    v = ObjectVerifier(cfg, "cpu", factory=Broken)
    assert v.mode == "coco" and v.weights == cfg.fallback_model and not v.positive


def test_track_drink_rejects_cycles(fake):
    v, cfg = fake
    cfg = dataclasses.replace(cfg, track_drink_frames=3, negative_min_fraction=1.1, strong_negative_frames=99)
    FakeYOLO.canned = {"mug": 0.7}  # кружка у руки весь трек
    cycles, signals = _setup([3, 15])
    v.verify("c", "v.mp4", cycles, signals, cfg)
    assert all(not c.valid and c.reject_reason == "track_object:mug" for c in cycles[1])
    cycles, signals = _setup([3, 15])
    v.verify("c", "v.mp4", cycles, signals, dataclasses.replace(cfg, track_drink_conf=0.8))
    assert all(c.valid for c in cycles[1])


def test_vapor_cache_depends_on_signals(monkeypatch):
    import smokedet.vapor as vap

    calls = []
    monkeypatch.setattr(vap, "FrameReader", FakeReader)
    monkeypatch.setattr(VaporMeter, "_measure_cycle", lambda self, r, sig, c, o, cfg: calls.append(1) or (0.0, 0.0))
    meter, cfg = VaporMeter(), VaporConfig()
    cycles, signals = _setup()
    meter.measure("c", "v.mp4", cycles, signals, cfg)
    meter.measure("c", "v.mp4", cycles, signals, cfg)
    assert len(calls) == 1
    signals[1].mouth[:, 1] += 5  # другая геометрия — замер заново
    meter.measure("c", "v.mp4", cycles, signals, cfg)
    assert len(calls) == 2
