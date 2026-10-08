import numpy as np
import pytest

from smokedet.ov_yolo import decode, nms


def random_boxes(rng, n):
    xy = rng.uniform(0, 300, (n, 2))
    wh = rng.uniform(5, 80, (n, 2))
    return np.concatenate([xy, xy + wh], axis=1).astype(np.float32), rng.uniform(0, 1, n).astype(np.float32)


def test_nms_matches_torchvision():
    torch = pytest.importorskip("torch")
    tv = pytest.importorskip("torchvision")
    rng = np.random.default_rng(0)
    for _ in range(20):
        boxes, scores = random_boxes(rng, 200)
        ref = tv.ops.nms(torch.from_numpy(boxes), torch.from_numpy(scores), 0.5).numpy()
        assert np.array_equal(nms(boxes, scores, 0.5), ref)


def test_nms_suppresses_overlap():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    keep = nms(boxes, np.array([0.8, 0.9, 0.5], dtype=np.float32), 0.5)
    assert keep.tolist() == [1, 2]


def test_decode_pose_threshold_and_layout():
    # 2 якоря: xywh, оценка человека, 17 точек x3
    pred = np.zeros((4 + 1 + 51, 2), dtype=np.float32)
    pred[:4, 0] = [50, 60, 20, 40]
    pred[4] = [0.9, 0.1]
    pred[5:, 0] = np.arange(51)
    boxes, conf, extra = decode(pred, conf=0.25, iou=0.7, nc=1)
    assert boxes.tolist() == [[40, 40, 60, 80]]
    assert conf.tolist() == pytest.approx([0.9])
    assert extra.reshape(-1, 17, 3)[0, 1].tolist() == [3, 4, 5]


def test_decode_detect_keeps_person_only_if_best_class():
    pred = np.zeros((4 + 3, 2), dtype=np.float32)
    pred[:4] = [[50, 150], [50, 150], [10, 10], [10, 10]]
    pred[4:, 0] = [0.6, 0.7, 0.0]  # лучший класс — не человек
    pred[4:, 1] = [0.6, 0.2, 0.0]
    boxes, conf, _ = decode(pred, conf=0.25, iou=0.7, nc=3)
    assert boxes[:, 0].tolist() == [145]
    assert conf.tolist() == pytest.approx([0.6])
