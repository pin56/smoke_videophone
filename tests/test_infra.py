"""Инфраструктура: суффикс -dirty в manifest, ключ кэша треков, время кадров по pts, чтение кадров."""
import subprocess

import cv2
import numpy as np
import pytest

from smokedet.export import git_commit
from smokedet import io_video
from smokedet.io_video import FrameClock, FrameReader
from smokedet.pipeline import track_cache_key


def _git(repo, *args):
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def test_git_commit_dirty_only_for_tracked(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.txt").write_text("1")
    _git(tmp_path, "add", "a.txt")
    _git(tmp_path, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "init")
    sha = git_commit(tmp_path)
    assert sha and not sha.endswith("-dirty")
    (tmp_path / "video.mp4").write_text("x")  # неотслеживаемые файлы (видео, runs/) не в счёт
    assert git_commit(tmp_path) == sha
    (tmp_path / "a.txt").write_text("2")
    assert git_commit(tmp_path) == sha + "-dirty"


def test_track_cache_key_depends_on_tracker_yaml(tmp_path):
    video = tmp_path / "v.mp4"
    video.write_bytes(b"0" * 10)
    tracker = tmp_path / "bt.yaml"
    tracker.write_text("track_buffer: 50\n")
    k1 = track_cache_key(str(video), "cfg", str(tracker))
    assert k1 == track_cache_key(str(video), "cfg", str(tracker))
    assert k1 != track_cache_key(str(video), "cfg2", str(tracker))
    tracker.write_text("track_buffer: 30\n")
    assert k1 != track_cache_key(str(video), "cfg", str(tracker))
    # встроенный трекер ultralytics (файла нет в проекте) — в ключ идёт имя
    assert track_cache_key(str(video), "cfg", "bytetrack.yaml") != track_cache_key(str(video), "cfg", "botsort.yaml")


def test_frame_clock_uses_pts():
    clock = FrameClock(10.0)
    pts = [0.0, 95.0, 210.0, 300.0]  # переменный fps: время по pts, а не idx/fps
    assert [clock.time(i, p) for i, p in enumerate(pts)] == pytest.approx([0.0, 0.095, 0.21, 0.3])


@pytest.mark.parametrize("bad", [0.0, -5.0, 150.0, 9000.0])
def test_frame_clock_falls_back_without_jump(bad):
    clock = FrameClock(10.0)
    ts = [clock.time(0, 0.0), clock.time(1, 120.0), clock.time(2, 220.0), clock.time(3, bad), clock.time(4, 420.0)]
    assert ts[:3] == pytest.approx([0.0, 0.12, 0.22])
    # с первого негодного pts — idx/fps со сдвигом последнего годного, дальше pts не используется
    assert ts[3:] == pytest.approx([0.32, 0.42])
    assert all(b > a for a, b in zip(ts, ts[1:]))


def test_frame_clock_bad_first_frame():
    clock = FrameClock(25.0)
    assert [clock.time(i, -1.0) for i in range(3)] == pytest.approx([0.0, 0.04, 0.08])


class VfrCapture:
    """VideoCapture с переменным fps, seek — как в OpenCV/FFmpeg: номер кадра считается по pts (pts * fps)."""

    FPS = 25.0

    def __init__(self, pts_ms, valid_pts=True):
        self.pts_ms = pts_ms
        self.valid_pts = valid_pts
        self.next = 0  # кадр, который вернёт grab()
        self.cur = -1  # последний захваченный кадр

    def _seek_number(self, number):
        # первый кадр, чей «номер по pts» не меньше запрошенного
        self.next = next((i for i, p in enumerate(self.pts_ms) if round(p / 1000 * self.FPS) >= number), len(self.pts_ms))

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            self._seek_number(value)
        elif prop == cv2.CAP_PROP_POS_MSEC:
            self._seek_number(value * self.FPS / 1000)

    def get(self, prop):
        assert prop == cv2.CAP_PROP_POS_MSEC
        return self.pts_ms[self.cur] if self.valid_pts else 0.0

    def grab(self):
        if self.next >= len(self.pts_ms):
            return False
        self.cur, self.next = self.next, self.next + 1
        return True

    def retrieve(self):
        return True, np.full((2, 2, 3), self.cur, np.int32)

    def read(self):
        return self.retrieve() if self.grab() else (False, None)

    def release(self):
        pass


@pytest.mark.parametrize("real_fps, valid_pts", [(26.0, True), (30.0, True), (26.0, False)])
def test_frame_reader_exact_frame_on_vfr(monkeypatch, real_fps, valid_pts):
    # заявлено 25 кадр/с: при 26 pts уходит от idx/fps на 0.6 с за клип (время по pts), при 30 — на 2.7 с,
    # и часы с 301-го кадра переходят на idx/fps со сдвигом; без pts время синтетическое с первого кадра
    pts_ms = [i * 1000 / real_fps for i in range(400)]
    opened = []
    monkeypatch.setattr(io_video, "_open", lambda path: opened.append(1) or VfrCapture(pts_ms, valid_pts))
    clock = FrameClock(VfrCapture.FPS)
    times = [clock.time(i, p if valid_pts else 0.0) for i, p in enumerate(pts_ms)]
    assert (clock.synthetic_from is None) == (real_fps == 26.0 and valid_pts)
    with FrameReader("v.mp4", synthetic_from=clock.synthetic_from) as reader:
        for idx in (350, 120, 360, 10, 399, 200):  # назад и далеко вперёд — через seek
            assert int(reader.get(idx, times[idx])[0, 0, 0]) == idx
        assert reader.pts_seek
    if clock.synthetic_from is None:
        assert len(opened) == 1  # все переходы — seek по pts, без чтения с начала
