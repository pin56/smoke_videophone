"""Ролики с разметкой поверх видео — посмотреть и проверить разметку глазами.

Рамки — эпизоды из выгрузки CVAT (tools/cvat_export.py или правленая в CVAT): smoking / not_smoking / ignore,
у эпизода — номер человека и флажок check («привязка к человеку под вопросом»). Подписи сверху и шкала
снизу — отметки rasmetka_gt.csv: подъём руки / затяжка / опускание.

Запуск из корня проекта:
    python tools/render_markup.py --cvat runs/cvat --gt markup/rasmetka_gt.csv --videos videos/true videos/false --out runs/markup
По умолчанию — клипы, где есть эпизод smoking; --clips 1,sm_3 — выбранные, --all — все. --persons рисует
и треки всех людей (метка person).

Кадры CVAT нумеруются подряд, как у трекера; время отметок — pts кадра (io_video.FrameClock), как в render.py.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from smokedet.cvat import PERSON, CvatFile, CvatTrack, Mark, load_marks, read_cvat  # noqa: E402
from smokedet.io_video import FrameClock, probe_video  # noqa: E402

BGR = {"smoking": (53, 57, 229), "not_smoking": (71, 160, 67), "ignore": (110, 110, 110), PERSON: (229, 136, 30)}
TITLE = {"smoking": "КУРИТ", "not_smoking": "НЕ КУРИТ", "ignore": "ИГНОР", PERSON: ""}
WHITE, DARK = (255, 255, 255), (30, 30, 30)
MARK_PAD = 0.5  # с: отметка видна чуть раньше и позже — у «17/17/17» нулевая длина
FONTS = ["C:/Windows/Fonts/arial.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]  # fmt: skip


class TrackBoxes:
    """Рамки трека CVAT по номеру кадра — как CvatTrack.box_at, но без сортировки на каждый вызов."""

    def __init__(self, track: CvatTrack):
        shapes = track.sorted_shapes()
        self.frames = np.array([s.frame for s in shapes], dtype=int)
        self.boxes = np.array([s.box for s in shapes], dtype=float).reshape(-1, 4)
        self.outside = np.array([s.outside for s in shapes], dtype=bool)

    def at(self, frame: int) -> Optional[np.ndarray]:
        k = int(np.searchsorted(self.frames, frame, side="right")) - 1
        if k < 0 or self.outside[k]:
            return None
        if k + 1 >= len(self.frames) or self.outside[k + 1] or self.frames[k] == frame:
            return self.boxes[k]
        w = (frame - self.frames[k]) / (self.frames[k + 1] - self.frames[k])
        return self.boxes[k] + (self.boxes[k + 1] - self.boxes[k]) * w


def phase(m: Mark, t: float) -> Optional[str]:
    """Фаза отметки в момент t; None — отметка не активна."""
    if not (m.start - MARK_PAD <= t <= m.end + MARK_PAD):
        return None
    if m.kind == "continue":
        return "продолжают курить"
    if m.puff is None:
        return ""
    if t < m.puff - MARK_PAD:
        return "рука ко рту"
    if t <= m.puff + MARK_PAD:
        return "затяжка"
    return "отвод руки"


def mark_lines(marks: List[Mark], t: float) -> List[tuple]:
    """Строки подписи для активных в t отметок: (текст, метка). Циклы SMOKE нумеруются по порядку."""
    smoke = [m for m in marks if m.label == "smoking" and m.kind != "continue"]
    out = []
    for m in marks:
        ph = phase(m, t)
        if ph is None:
            continue
        head = TITLE[m.label]
        if m in smoke:
            head += f" — цикл {smoke.index(m) + 1}/{len(smoke)}"
        out.append((f"{head}: {ph} ({m.text})" if ph else f"{head} ({m.text})", m.label))
    return out


def track_text(tr: CvatTrack) -> str:
    who = tr.attributes.get("person", "")
    parts = [TITLE[tr.label]] if TITLE.get(tr.label) else []
    if who:
        parts.append(f"чел. {who}")
    if tr.attributes.get("check", "").lower() == "true":
        parts.append("проверить")
    return " · ".join(parts)


def _font(size: int):
    for path in FONTS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


class Painter:
    """Подписи с кириллицей: PIL рисует плашку один раз, дальше она копируется в кадр."""

    def __init__(self, height: int):
        self.size = max(14, round(height / 38))
        self.font = _font(self.size)
        self.cache: Dict[tuple, np.ndarray] = {}

    def sprite(self, text: str, color: tuple) -> np.ndarray:
        key = (text, color)
        if key not in self.cache:
            if len(self.cache) > 4000:
                self.cache.clear()
            pad = max(2, self.size // 5)
            asc, desc = self.font.getmetrics()
            w = int(self.font.getlength(text)) + 2 * pad
            img = Image.new("RGB", (max(w, 1), asc + desc + 2 * pad), color[::-1])
            ImageDraw.Draw(img).text((pad, pad), text, font=self.font, fill=WHITE)
            self.cache[key] = np.asarray(img)[:, :, ::-1].copy()
        return self.cache[key]

    def label(self, frame: np.ndarray, text: str, x: float, y: float, color: tuple, above: bool = True) -> int:
        """Плашка у точки (x, y): над ней (above) или под ней. Возвращает высоту плашки."""
        spr = self.sprite(text, color)
        h, w = spr.shape[:2]
        H, W = frame.shape[:2]
        x0 = min(max(0, int(x)), max(0, W - w))
        y0 = min(max(0, int(y - h if above else y)), max(0, H - h))
        crop = spr[: H - y0, : W - x0]
        frame[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]] = crop
        return h


def draw_timeline(frame: np.ndarray, marks: List[Mark], duration: float, t: float) -> None:
    """Шкала клипа внизу кадра: отметки цветом метки, затяжки — белые риски, текущий момент — курсор."""
    H, W = frame.shape[:2]
    h = max(10, H // 36)
    y0 = H - h
    frame[y0:] = (frame[y0:] * 0.3).astype(np.uint8)
    dur = max(duration, 1e-6)

    def px(s: float) -> int:
        return int(round(min(max(s / dur, 0.0), 1.0) * (W - 1)))

    for m in marks:
        a, b = px(m.start - MARK_PAD), px(min(m.end, duration) + MARK_PAD)
        cv2.rectangle(frame, (a, y0 + 2), (max(b, a + 2), H - 3), BGR[m.label], -1)
        if m.puff is not None:
            cv2.line(frame, (px(m.puff), y0 + 2), (px(m.puff), H - 3), WHITE, 1)
    x = px(t)
    cv2.line(frame, (x, y0 - h // 2), (x, H), WHITE, max(2, h // 6))


class Writer:
    """H.264 через ffmpeg (смотрится в браузере и в Telegram), без ffmpeg — mp4v средствами OpenCV."""

    def __init__(self, path: Path, w: int, h: int, fps: float):
        ff = shutil.which("ffmpeg")
        self.proc, self.cv = None, None
        if ff:
            cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                   "-r", f"{fps:.3f}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                   "-pix_fmt", "yuv420p", str(path)]  # fmt: skip
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        else:
            self.cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, frame: np.ndarray) -> None:
        if self.proc is not None:
            self.proc.stdin.write(frame.tobytes())
        else:
            self.cv.write(frame)

    def close(self) -> None:
        if self.proc is not None:
            self.proc.stdin.close()
            if self.proc.wait() != 0:
                raise RuntimeError("ffmpeg завершился с ошибкой")
        else:
            self.cv.release()


def _even(v: float) -> int:
    return max(2, int(round(v / 2)) * 2)  # yuv420p требует чётных сторон


def render_clip(video: Path, cf: CvatFile, marks: List[Mark], out: Path, title: str,
                persons: bool = False, max_width: int = 1280) -> int:  # fmt: skip
    """Рисует разметку на видео, возвращает число прочитанных кадров."""
    meta = probe_video(str(video))
    duration = meta.duration
    tracks = [(tr, TrackBoxes(tr)) for tr in cf.tracks if persons or tr.label != PERSON]
    tracks.sort(key=lambda p: p[0].label != PERSON)  # люди — под эпизодами
    cap = cv2.VideoCapture(str(video))
    clock = FrameClock(meta.fps)
    writer = painter = None
    idx, scale, thick = 0, 1.0, 1
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = clock.time(idx, cap.get(cv2.CAP_PROP_POS_MSEC))
            if writer is None:  # размер — по кадру: у повёрнутых .MOV он не совпадает с метаданными
                fh, fw = frame.shape[:2]
                scale = min(1.0, max_width / fw) if max_width else 1.0
                size = (_even(fw * scale), _even(fh * scale))
                writer, painter = Writer(out, *size, meta.fps), Painter(size[1])
                thick = max(1, round(size[1] / 360))
            if frame.shape[1::-1] != size:
                frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            for tr, tb in tracks:
                box = tb.at(idx)
                if box is None:
                    continue
                x1, y1, x2, y2 = (box * scale).astype(int)
                color = BGR.get(tr.label, WHITE)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick if tr.label == PERSON else 2 * thick)
                text = track_text(tr)
                if text:
                    painter.label(frame, text, x1, y1 - thick, color)
            y = painter.label(frame, title, 0, 0, DARK, above=False)
            x = painter.cache[(title, DARK)].shape[1]
            stamp, fs = f"{t:.1f} s", painter.size / 30  # время меняется каждый кадр — не через кэш PIL
            (tw, _), _ = cv2.getTextSize(stamp, cv2.FONT_HERSHEY_SIMPLEX, fs, thick)
            cv2.rectangle(frame, (x, 0), (x + tw + painter.size // 2, y - 1), DARK, -1)
            cv2.putText(frame, stamp, (x + painter.size // 4, y - y // 4), cv2.FONT_HERSHEY_SIMPLEX, fs, WHITE, thick,
                        cv2.LINE_AA)  # fmt: skip
            for text, label in mark_lines(marks, t):
                y += painter.label(frame, text, 0, y, BGR[label], above=False)
            draw_timeline(frame, marks, duration, t)
            writer.write(frame)
            idx += 1
    finally:
        cap.release()
        if writer is not None:
            writer.close()
    return idx


def load_index(cvat_dir: Path) -> Dict[str, dict]:
    path = cvat_dir / "index.csv"
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["clip_id"]: r for r in csv.DictReader(f)}


def find_video(name: str, dirs: List[Path]) -> Optional[Path]:
    for d in dirs:
        if (d / name).exists():
            return d / name
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cvat", default="runs/cvat", help="папка с XML (CVAT for video 1.1) и index.csv экспорта")
    ap.add_argument("--gt", default="markup/rasmetka_gt.csv", help="отметки подъём/затяжка/опускание")
    ap.add_argument("--videos", nargs="+", default=["videos/true", "videos/false"], help="где искать видео")
    ap.add_argument("--out", default="runs/markup")
    ap.add_argument("--clips", default="", help="clip_id через запятую (имена XML без .xml)")
    ap.add_argument("--all", action="store_true", help="все клипы, а не только с эпизодами smoking")
    ap.add_argument("--persons", action="store_true", help="рисовать треки всех людей")
    ap.add_argument("--max-width", type=int, default=1280, help="уменьшать кадр до этой ширины (0 — не уменьшать)")
    args = ap.parse_args(argv)

    cvat_dir, out_dir, dirs = Path(args.cvat), Path(args.out), [Path(d) for d in args.videos]
    index = load_index(cvat_dir)
    marks = load_marks(args.gt) if Path(args.gt).exists() else {}
    wanted = {c.strip() for c in args.clips.split(",") if c.strip()}
    rows = []
    for xml in sorted(cvat_dir.glob("*.xml")):
        clip = xml.stem
        if wanted and clip not in wanted:
            continue
        cf = read_cvat(xml)[0]
        n_smoke = sum(tr.label == "smoking" for tr in cf.tracks)
        if not wanted and not args.all and not n_smoke:
            continue
        name = index.get(clip, {}).get("video") or cf.source
        video = find_video(name, dirs) if name else None
        if video is None:
            print(f"[{clip}] видео {name!r} не найдено в {', '.join(map(str, dirs))} — пропуск")
            continue
        clip_marks = marks.get(video.stem, [])
        out = out_dir / f"{clip}.mp4"
        print(f"[{clip}] {video.name}: эпизодов smoking {n_smoke}, отметок {len(clip_marks)} -> {out}", flush=True)
        n = render_clip(video, cf, clip_marks, out, clip, persons=args.persons, max_width=args.max_width)
        expected = cf.stop_frame + 1 if cf.stop_frame is not None else None
        if expected is not None and expected != n:
            print(f"[{clip}] ВНИМАНИЕ: кадров в видео {n}, в CVAT {expected} — рамки могут сдвинуться")
        rows.append({"clip_id": clip, "video": str(video), "out": str(out), "frames": n, "cvat_frames": expected,
                     "smoking": n_smoke, "marks": len(clip_marks)})  # fmt: skip
    if rows:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "index.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    print(f"готово: {len(rows)} роликов в {out_dir}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
