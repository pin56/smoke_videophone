"""Инфраструктура: суффикс -dirty в manifest, ключ кэша треков, время кадров по pts."""
import subprocess

import pytest

from smokedet.export import git_commit
from smokedet.io_video import FrameClock
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
