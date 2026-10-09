"""Внешний вид цикла: похоже ли на курение то, что у рта в паузе (CLIP без дообучения).

Признаки позы (пауза, положение кисти, локтя, головы, свет) курение от питья и касания лица не отличают:
на размеченных циклах классификатор по ним даёт AUC 0.54–0.64, внутри клипа — около 0.5. CLIP по кропу
«рот + кисть» в кадрах паузы сравнивает описания «курит» и «пьёт / трогает лицо / говорит по телефону…»:
AUC 0.88 на размеченных кандидатных циклах и 0.86 внутри клипа (где камера и человек одни и те же).

Оценка цикла — сумма по двум видам кропа (узкий `mouth_scale` роста со словарём MOUTH_PROMPTS, широкий
`wide_scale` со словарём WIDE_PROMPTS) максимума по кадрам паузы log Σexp(логиты «курит») −
log Σexp(логиты «не курит»). Модель не дообучается: на ~30 клипах обученный классификатор запоминает
камеру (у многих камер все циклы одного класса), а не действие. Цикл с оценкой ниже `min_score`
считается не курением, если у него нет прямого признака (предмет, вспышка, пар).
"""
from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import AppearanceConfig
from .cycles import Cycle
from .geometry import TrackSignal
from .io_video import FrameReader
from .vapor import signals_digest

log = logging.getLogger(__name__)

PAD_VALUE = 114  # заливка за краем кадра (серый, как у letterbox ultralytics)

# (курит, не курит). Узкий кроп: рот и кисть крупно
MOUTH_PROMPTS = (
    ["a person smoking a cigarette", "a person holding a cigarette at their mouth", "a person smoking",
     "a person exhaling smoke", "a person vaping"],
    ["a person drinking from a bottle", "a person drinking from a cup", "a person drinking from a can",
     "a person touching their face", "a person scratching their nose", "a person wiping their mouth",
     "a person talking on the phone", "a person eating", "a person eating a lollipop", "a person yawning",
     "a person adjusting their glasses", "a person standing"],
)  # fmt: skip
# Широкий кроп: голова, плечи, руки
WIDE_PROMPTS = (
    ["a surveillance camera photo of a person smoking a cigarette", "a person with a cigarette in their mouth",
     "a person holding a cigarette between two fingers near the lips", "a person taking a drag from a cigarette",
     "a person smoking", "a man smoking a cigarette outdoors", "a woman smoking a cigarette outdoors",
     "a person holding a lit cigarette"],
    ["a person drinking from a bottle", "a person drinking from a cup", "a person drinking from a can",
     "a person drinking beer", "a person touching their face with their hand", "a person scratching their face",
     "a person rubbing their eyes", "a person wiping their nose", "a person covering their mouth while coughing",
     "a person resting their chin on their hand", "a person talking on a mobile phone", "a person eating food",
     "a person biting their nails", "a person adjusting their hat or hood", "a person adjusting their glasses",
     "a person yawning", "a person blowing their nose", "a person with empty hands near their face"],
)  # fmt: skip

# Кодировщик: (кропы BGR, номер вида 0/1) -> оценка «курит» каждого кропа
Encoder = Callable[[List[np.ndarray], int], np.ndarray]


def square_crop(frame: np.ndarray, cx: float, cy: float, side: float) -> np.ndarray:
    """Квадрат со стороной side вокруг (cx, cy); часть за краем кадра заливается PAD_VALUE."""
    h, w = frame.shape[:2]
    side = max(8, int(round(side)))
    x1, y1 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x2, y2 = x1 + side, y1 + side
    out = np.full((side, side) + frame.shape[2:], PAD_VALUE, dtype=frame.dtype)
    sx1, sy1, sx2, sy2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if sx2 > sx1 and sy2 > sy1:
        out[sy1 - y1 : sy2 - y1, sx1 - x1 : sx2 - x1] = frame[sy1:sy2, sx1:sx2]
    return out


def plan_frames(sig: TrackSignal, c: Cycle, max_frames: int) -> np.ndarray:
    """Отсчёты сигнала для оценки: до max_frames равномерно по паузе (иначе — ближайший к её середине)."""
    idx = np.nonzero((sig.t >= c.hold_start) & (sig.t <= c.hold_end))[0]
    if len(idx) == 0:
        return np.array([sig.index_at((c.hold_start + c.hold_end) / 2)])
    pick = np.unique(np.linspace(0, len(idx) - 1, min(max_frames, len(idx))).round().astype(int))
    return idx[pick]


def crop_geometry(sig: TrackSignal, c: Cycle, ks: np.ndarray) -> Tuple[np.ndarray, float]:
    """Центры кропов (середина между ртом и кистью в каждом отсчёте) и рост человека за цикл, px."""
    span = np.nonzero((sig.t >= c.start) & (sig.t <= c.end))[0]
    if len(span) == 0:
        span = ks
    height = float(np.median(sig.boxes[span, 3] - sig.boxes[span, 1]))
    mouth_all = sig.mouth[span]
    ok = np.isfinite(mouth_all).all(axis=1)
    if ok.any():
        mouth_med = np.median(mouth_all[ok], axis=0)
    else:  # рот не найден: верх рамки по центру
        b = np.median(sig.boxes[span], axis=0)
        mouth_med = np.array([(b[0] + b[2]) / 2, b[1] + 0.12 * height])
    centers = []
    for k in ks:
        m = sig.mouth[k] if np.isfinite(sig.mouth[k]).all() else mouth_med
        hd = sig.hand[k] if np.isfinite(sig.hand[k]).all() else m
        centers.append((np.asarray(m, float) + np.asarray(hd, float)) / 2)
    return np.array(centers), height


def weights_path(model: str, weights_dir: str) -> str:
    """Файл весов CLIP, который clip.load скачивает в weights_dir (или путь, если model — сам файл)."""
    import clip

    url = clip.clip._MODELS.get(model)
    return str(Path(weights_dir) / Path(url).name) if url else model


OPEN_CLIP = "oc:"  # appearance.model «oc:<архитектура>/<тег>» — open_clip (MobileCLIP2, SigLIP2), иначе OpenAI CLIP


def load_clip(model: str, device: str, weights_dir: str, px: int = 0):
    """CLIP по appearance.model -> (модель, предобработка PIL, токенизатор, файл весов). Веса open_clip скачиваются
    в weights_dir; ветви MobileOne/FastViT у MobileCLIP сливаются (timm reparameterize_model): быстрее, ответ тот же.
    px > 0 — вход ViT OpenAI CLIP px x px (resize_vit)."""
    if not model.startswith(OPEN_CLIP):
        import clip

        m, pre = clip.load(model, device=device, download_root=weights_dir)
        if px and px != m.visual.input_resolution:
            m, pre = resize_vit(m, px), clip.clip._transform(px)
        return m, pre, clip.tokenize, weights_path(model, weights_dir)
    if px:
        raise ValueError("appearance.input_px — только для OpenAI CLIP")
    import open_clip
    from open_clip.pretrained import download_pretrained, get_pretrained_cfg
    from timm.utils import reparameterize_model

    arch, tag = model[len(OPEN_CLIP):].split("/", 1)
    m, _, pre = open_clip.create_model_and_transforms(arch, pretrained=tag, cache_dir=weights_dir, device=device)
    m = reparameterize_model(m.eval())
    weights = str(download_pretrained(get_pretrained_cfg(arch, tag), cache_dir=weights_dir))
    return m, pre, open_clip.get_tokenizer(arch), weights


def resize_vit(model, px: int):
    """ViT OpenAI CLIP на вход px x px: позиционные векторы сетки патчей интерполируются бикубически. Кропы рта на
    CCTV — десятки пикселей, родной вход их только растягивает, а цена ViT растёт с числом патчей."""
    import torch

    v = model.visual
    patch = v.conv1.kernel_size[0]
    if px % patch:
        raise ValueError(f"appearance.input_px={px} не кратно патчу {patch}")
    pe = v.positional_embedding.data
    g0, g1 = int(round((pe.shape[0] - 1) ** 0.5)), px // patch
    grid = pe[1:].float().reshape(1, g0, g0, -1).permute(0, 3, 1, 2)
    grid = torch.nn.functional.interpolate(grid, size=(g1, g1), mode="bicubic", align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(g1 * g1, -1).to(pe.dtype)
    v.positional_embedding = torch.nn.Parameter(torch.cat([pe[:1], grid]))
    v.input_resolution = px
    return model


def openvino_path(weights: str, px: int = 0) -> str:
    """IR визуальной части CLIP (tools/prepare_weights.py) рядом с файлом весов: weights/<файл>-visual_openvino
    (при appearance.input_px — <файл>-visual<px>_openvino)."""
    p = Path(weights)
    return str(p.with_name(f"{p.stem}-visual{px or ''}_openvino") / "model.xml")


def export_openvino(model, weights: str, res: int, px: int = 0) -> str:
    """Визуальная часть модели load_clip (вход res x res; px — appearance.input_px) -> IR OpenVINO (веса f32, батч
    любой); путь к model.xml."""
    import openvino as ov
    import torch

    out = openvino_path(weights, px)
    if not Path(out).exists():
        visual, r = model.visual.float().eval(), res
        ov.save_model(ov.convert_model(visual, example_input=torch.zeros(1, 3, r, r), input=[(-1, 3, r, r)]), out,
                      compress_to_fp16=False)  # fmt: skip
    return out


def view_plan(cfg: AppearanceConfig) -> List[Tuple[float, int]]:
    """Виды кропа: (сторона, доля роста; словарь 0 — узкий, 1 — широкий)."""
    views = [(cfg.mouth_scale, 0), (cfg.wide_scale, 1)]
    return views + [(cfg.tight_scale, 0)] if cfg.tight_scale > 0 else views


def cycle_score(view_scores: Sequence[np.ndarray], agg: str = "max") -> float:
    """Сумма по видам кропа оценки по кадрам: максимум (max) или среднее двух лучших (top2)."""
    if agg == "top2":
        return float(sum(np.sort(s)[-2:].mean() for s in view_scores))
    if agg != "max":
        raise ValueError(f"appearance.frame_agg: {agg}")
    return float(sum(np.max(s) for s in view_scores))


def rescuable(c: Cycle, cfg: AppearanceConfig) -> bool:
    """Невалидный цикл, который высокая оценка может вернуть: отброшен верификатором за предмет из
    rescue_objects или при поиске циклов только по причинам из rescue_reasons."""
    if c.valid:
        return False
    kind, _, obj = c.reject_reason.partition(":")
    if kind in ("object", "track_object"):
        return obj in cfg.rescue_objects
    return all(r in cfg.rescue_reasons for r in c.reject_reason.split(","))


def apply_score(c: Cycle, score: float, cfg: AppearanceConfig, keep: Optional[Callable[[Cycle], bool]] = None) -> None:
    """Оценка цикла -> appearance, look и валидность (min_score, rescue_score); keep(c) — оставить цикл
    с низкой оценкой (прямой признак)."""
    c.appearance = score
    c.look = bool(c.appearance >= cfg.direct_score)  # nan -> False
    if not c.valid:
        if c.appearance >= cfg.rescue_score:
            c.valid, c.rescued_from, c.reject_reason = True, c.reject_reason, ""
    elif c.appearance < cfg.min_score and not (keep and keep(c)):
        c.valid = False
        c.reject_reason = f"appearance:{c.appearance:.2f}"


class ClipEncoder:
    """CLIP (OpenAI) с векторами описаний обоих видов. На CUDA веса в FP16 (так их грузит clip.load). На CPU при
    cfg.openvino визуальная часть — через OpenVINO (IR из tools/prepare_weights.py; на CPU с bf16 OpenVINO считает
    в bf16): CLIP-L/336 ~100 мс на кроп против ~650 у torch, косинус векторов с torch >= 0.998."""

    def __init__(self, cfg: AppearanceConfig, device: str, weights_dir: str):
        import torch
        from PIL import Image

        self.torch, self.Image = torch, Image
        self.device = f"cuda:{device}" if device.isdigit() else device
        self.px = cfg.input_px
        self.model, self.preprocess, tokenize, self.weights = load_clip(cfg.model, self.device, weights_dir, self.px)
        self.model.eval()
        self.res = int(self.preprocess(Image.new("RGB", (8, 8))).shape[-1])
        self.visual = None
        ir = openvino_path(self.weights, self.px)
        if self.device == "cpu" and cfg.openvino:
            if Path(ir).exists():
                import openvino as ov

                self.visual = ov.Core().compile_model(ir, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
            else:
                log.warning("нет %s (tools/prepare_weights.py): CLIP на CPU через torch, в несколько раз медленнее", ir)
        self.npos, self.text = [], []
        with torch.no_grad():
            for pos, neg in (MOUTH_PROMPTS, WIDE_PROMPTS):
                t = self.model.encode_text(tokenize(pos + neg).to(self.device)).float()
                self.text.append(t / t.norm(dim=-1, keepdim=True))
                self.npos.append(len(pos))
        self.scale = float(self.model.logit_scale.detach().exp())

    def embed(self, crops: List[np.ndarray]):
        """Кропы BGR -> нормированные векторы изображений (torch, float32)."""
        torch = self.torch
        x = torch.stack([self.preprocess(self.Image.fromarray(np.ascontiguousarray(c[:, :, ::-1]))) for c in crops])
        with torch.no_grad():
            if self.visual is not None:
                e = torch.from_numpy(self.visual(x.numpy())[0]).float()
            else:
                e = self.model.encode_image(x.to(self.device)).float()
        return e / e.norm(dim=-1, keepdim=True)

    def __call__(self, crops: List[np.ndarray], view: int) -> np.ndarray:
        torch = self.torch
        e = self.embed(crops)
        with torch.no_grad():
            lg = self.scale * e @ self.text[view].T
            n = self.npos[view]
            s = torch.logsumexp(lg[:, :n], dim=1) - torch.logsumexp(lg[:, n:], dim=1)
        return s.cpu().numpy()


class AppearanceScorer:
    """Оценка внешнего вида валидных циклов клипа. Модель грузится при первой надобности; если не
    загрузилась (нет пакета clip или весов офлайн) — оценка выключается с предупреждением, циклы не
    отбрасываются. Результат кэшируется в памяти по циклу, параметрам и отпечатку сигналов клипа:
    tune.py перебирает min_score без повторного чтения видео."""

    def __init__(self, device: str = "cpu", weights_dir: str = "weights", encoder: Optional[Encoder] = None):
        self.device = device
        self.weights_dir = weights_dir
        self.encoder = encoder
        self.failed = ""
        self.cache: Dict[tuple, float] = {}
        self._sha: Dict[str, Optional[str]] = {}

    def _encoder(self, cfg: AppearanceConfig) -> Optional[Encoder]:
        if self.encoder is None and not self.failed:
            try:
                self.encoder = ClipEncoder(cfg, self.device, self.weights_dir)
                log.info("оценка внешнего вида: CLIP %s на %s", cfg.model, self.device)
            except Exception as e:  # noqa: BLE001 — любая ошибка загрузки выключает только эту стадию
                self.failed = f"{type(e).__name__}: {e}"
                log.warning("CLIP %s не загрузился, оценка внешнего вида выключена: %s", cfg.model, self.failed)
        return self.encoder

    def describe(self, cfg: AppearanceConfig) -> dict:
        from .export import file_sha256

        weights = getattr(self.encoder, "weights", None)
        if weights and weights not in self._sha:  # info() зовут после каждого клипа, файл ~0.9 ГБ
            self._sha[weights] = file_sha256(weights)
        return {"model": cfg.model, "weights": weights, "sha256": self._sha.get(weights),
                "available": self.encoder is not None, "error": self.failed, "min_score": cfg.min_score}  # fmt: skip

    @staticmethod
    def _key(clip_id: str, c: Cycle, cfg: AppearanceConfig, digest: str) -> tuple:
        params = (cfg.model, cfg.input_px, cfg.max_frames, cfg.mouth_scale, cfg.wide_scale, cfg.tight_scale,
                  cfg.frame_agg)  # fmt: skip
        return clip_id, digest, c.track_id, round(c.start, 3), round(c.hold_start, 3), round(c.hold_end, 3), params

    def score(
        self,
        clip_id: str,
        video_path: str,
        cycles: Dict[int, List[Cycle]],
        signals: Dict[int, TrackSignal],
        cfg: AppearanceConfig,
        keep: Optional[Callable[[Cycle], bool]] = None,
        synthetic_from: Optional[int] = None,
    ) -> None:
        """Заполняет appearance и look у валидных циклов; цикл с оценкой ниже min_score помечает невалидным,
        если keep(c) не велит его оставить (прямой признак). Цикл, отброшенный верификатором за предмет из
        rescue_objects или только по причинам из rescue_reasons, тоже оценивается и с оценкой не ниже rescue_score
        снова валиден (rescued_from)."""
        digest = signals_digest(signals)
        scored = [c for cs in cycles.values() for c in cs if c.valid or rescuable(c, cfg)]
        todo = [c for c in scored if self._key(clip_id, c, cfg, digest) not in self.cache]
        if todo:
            encoder = self._encoder(cfg)
            if encoder is None:
                return
            for c, s in zip(todo, self._score_cycles(video_path, todo, signals, cfg, encoder, synthetic_from)):
                self.cache[self._key(clip_id, c, cfg, digest)] = s
        for c in scored:
            apply_score(c, self.cache[self._key(clip_id, c, cfg, digest)], cfg, keep)

    @staticmethod
    def _score_cycles(
        video_path: str,
        todo: List[Cycle],
        signals: Dict[int, TrackSignal],
        cfg: AppearanceConfig,
        encoder: Encoder,
        synthetic_from: Optional[int] = None,
    ) -> List[float]:
        """Кадры всех циклов читаются за один проход по возрастанию номера (как в vapor.py); кропы цикла
        оцениваются, как только прочитан его последний кадр. nan — ни один кадр не прочитан."""
        views = view_plan(cfg)
        plans = []
        need: Dict[int, List[Tuple[int, int]]] = defaultdict(list)  # кадр -> [(номер цикла, номер кропа)]
        times: Dict[int, float] = {}
        for i, c in enumerate(todo):
            sig = signals[c.track_id]
            ks = plan_frames(sig, c, cfg.max_frames)
            centers, height = crop_geometry(sig, c, ks)
            frames = [int(sig.frame_idx[k]) for k in ks]
            plans.append((centers, height, max(frames)))
            for j, (k, f) in enumerate(zip(ks, frames)):
                need[f].append((i, j))
                times[f] = float(sig.t[k])
        crops: List[List[List[np.ndarray]]] = [[[] for _ in views] for _ in todo]
        results = [float("nan")] * len(todo)
        with FrameReader(video_path, synthetic_from=synthetic_from) as reader:
            for f in sorted(need):
                frame = reader.get(f, times[f])
                for i, j in need[f]:
                    centers, height, last = plans[i]
                    if frame is not None:
                        cx, cy = centers[j]
                        for v, (s, _) in enumerate(views):
                            crops[i][v].append(square_crop(frame, cx, cy, s * height))
                    if f == last and crops[i][0]:
                        results[i] = cycle_score([encoder(crops[i][v], tv) for v, (_, tv) in enumerate(views)],
                                                 cfg.frame_agg)  # fmt: skip
                        crops[i] = [[] for _ in views]
        return results
