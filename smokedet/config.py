"""Конфигурация пайплайна: dataclass-схема, загрузка из YAML и переопределения из CLI (--set key=value)."""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class VideoConfig:
    target_fps: float = 10.0  # частота анализа; остальные кадры пропускаются
    fallback_fps: float = 25.0  # если в контейнере нет корректного FPS


@dataclass
class DetectorConfig:
    pose_model: str = "weights/yolo11m-pose.pt"
    imgsz: int = 1280  # CCTV: люди мелкие, поэтому вход больше стандартных 640
    conf: float = 0.1  # низкий порог: ByteTrack использует слабые детекции во втором проходе
    iou: float = 0.6
    tracker: str = "configs/bytetrack_cctv.yaml"
    device: str = "auto"  # auto | cpu | mps | 0
    half: bool = True  # FP16, только на CUDA
    refine_small: bool = True  # повторная поза на увеличенном кропе для мелких людей
    refine_max_height: int = 220
    refine_imgsz: int = 320
    refine_pad: float = 0.15


@dataclass
class TrackingConfig:
    min_person_height: int = 72  # по правилам 80px; запас вниз почти не штрафуется (IGNORE)
    roi_anchor: str = "feet"  # feet | center | head — какая точка человека должна быть в зоне
    stitch_max_gap: float = 2.0  # сек между концом одного трека и началом другого
    stitch_max_dist: float = 0.6  # расстояние центров в высотах bbox


@dataclass
class GeometryConfig:
    min_kpt_conf: float = 0.3
    hand_extrapolation: float = 0.3  # кисть = запястье + k * (запястье - локоть)
    scale_bbox_ratio: float = 0.12  # нижняя граница нормировки: доля высоты bbox
    scale_window: float = 2.0  # сек, скользящая медиана нормировки
    smooth_window: int = 3  # отсчётов, медианное сглаживание d(t)
    # неправдоподобная рука (чужая в толпе) не учитывается в отсчёте; 0 — проверка отключена
    max_upper_arm: float = 3.0  # плечо (плечо→локоть) длиннее k*scale
    max_forearm: float = 3.0  # предплечье (локоть→запястье) длиннее k*scale
    max_wrist_outside: float = 1.0  # запястье за рамкой человека дальше k*scale


@dataclass
class CycleConfig:
    near_enter: float = 0.6  # d ниже — кисть у рта
    near_exit: float = 0.9  # d выше — кисть отведена (гистерезис)
    min_hold: float = 0.4  # по правилам ~0.5 с, с допуском на дискретизацию
    max_hold: float = 6.0  # по правилам ~5 с
    max_missing: float = 0.6  # сек пропуска точек, допустимые внутри паузы
    lookaround: float = 2.0  # сек до/после паузы для поиска фаз подхода и отвода
    phone_ear_ratio: float = 0.75  # d(кисть, ухо) / d(кисть, рот) ниже — похоже на телефон
    min_wrist_above_elbow: float = 0.5  # доля кадров паузы, где запястье выше локтя
    max_wrist_above_mouth: float = 0.1  # запястье выше рта больше чем на k*scale — рука у головы
    max_elbow_raise: float = 0.15  # локоть выше плеча больше чем на k*scale — рука у головы
    min_face_visible: float = 0.5  # доля кадров паузы с уверенно найденным носом
    max_head_tilt: float = 0.0  # нос поднялся над ушами в паузе больше чем на k*scale — питьё; 0 — отключено


@dataclass
class VerifierConfig:
    enabled: bool = True
    model: str = "weights/yolov8s-worldv2.pt"  # open-vocabulary: YOLO-World или YOLOE
    fallback_model: str = "weights/yolo11s.pt"  # COCO-модель, если open-vocab недоступен
    imgsz: int = 640
    conf: float = 0.12
    half: bool = True  # FP16, только на CUDA
    frames_per_cycle: int = 6
    crop_margin: float = 0.9  # поле вокруг рта и кисти в единицах нормировки
    wide_crop_margin: float = 2.0  # второй, широкий кроп для крупных предметов; 0 — отключить
    positive_classes: List[str] = field(
        default_factory=lambda: ["cigarette", "vape", "electronic cigarette", "smoke", "vapor cloud"]
    )
    negative_classes: List[str] = field(
        default_factory=lambda: [
            "bottle", "water bottle", "cup", "mug", "drinking glass", "beverage can", "can", "jar",
            "glass jar", "thermos", "cell phone", "sandwich", "food", "spoon", "fork",
        ]  # fmt: skip
    )
    # детектор часто называет так саму кисть: учитываются в доле негативов, но не как сильный негатив
    weak_negative_classes: List[str] = field(default_factory=lambda: ["cell phone"])
    direct_min_frames: int = 2  # прямой признак: предмет в нескольких соседних кадрах
    negative_min_fraction: float = 0.5  # доля кропов с «чужим» предметом, чтобы отбросить цикл
    strong_negative_conf: float = 0.25  # сильный негатив: предмет с такой уверенностью...
    strong_negative_frames: int = 2  # ...минимум в стольких кадрах цикла отбрасывает цикл
    # «двойники» сигареты: ищутся по всему треку (палочку в паузе детектор принимает за сигарету).
    # «lollipop stick» / «candy on a stick» сюда не годятся: срабатывают на настоящие сигареты
    track_negative_classes: List[str] = field(default_factory=lambda: ["lollipop"])
    track_scan_step: float = 0.5  # с между просматриваемыми кадрами трека; 0 — отключить
    track_negative_conf: float = 0.5  # уверенность находки
    track_negative_radius: float = 1.0  # центр предмета не дальше от кисти или рта, в единицах нормировки
    track_negative_frames: int = 3  # столько кадров с находкой в окне вокруг цикла отбрасывают цикл
    track_negative_window: float = 60.0  # с, окно вокруг пика цикла; 0 — весь трек
    # предметы для питья у кисти или рта по всему треку (тот же проход, что и для «двойников», свои пороги)
    track_drink_classes: List[str] = field(
        default_factory=lambda: ["bottle", "water bottle", "cup", "mug", "beverage can", "drinking glass"]
    )
    track_drink_conf: float = 0.6  # на true/false не разделяет (true_9 даёт больше находок, чем false_8): подбирать на эталоне
    track_drink_radius: float = 1.0
    track_drink_frames: int = 3  # 0 — отключить
    track_drink_window: float = 60.0


@dataclass
class FlashConfig:
    enabled: bool = True
    min_area: float = 0.8  # новый пересвет у рта/кистей, в единицах scale^2
    max_ring: float = 0.6  # пересвет вокруг человека выше — светлеет окружение (дверь, фары)
    max_hand_dist: float = 1.5  # кисть не дальше от рта, в единицах нормировки
    min_samples: int = 2  # отсчётов подряд: одиночный кадр — блик
    max_gap: float = 0.35  # с между отсчётами одной вспышки
    attach_before: float = 2.0  # с: цикл может начаться раньше вспышки на столько...
    attach_after: float = 10.0  # ...или позже её на столько (прикуривание → затяжка)


@dataclass
class VaporConfig:
    enabled: bool = True
    zone_radius: float = 1.6  # зона воздуха вокруг рта, в единицах нормировки
    head_radius: float = 0.6  # голова исключается: круг вокруг центра лица
    hand_radius: float = 0.6  # кисть исключается
    body_offset: float = 0.5  # тело исключается: внутри рамки ниже рта на столько
    min_rise: int = 10  # пиксель дымки: светлее фона на столько уровней (0..255) и не насыщеннее
    min_net: float = 0.22  # чистая дымка (доля зоны) в двух годных кадрах подряд — пар
    bg_before: float = 2.5  # с: фон — кадры от начала цикла минус bg_before...
    bg_gap: float = 0.2  # ...до начала цикла минус bg_gap
    bg_min_frames: int = 3
    window_before: float = 0.5  # с: замер от конца паузы минус столько...
    window_after: float = 3.0  # ...до конца паузы плюс столько
    min_kept: float = 0.3  # кадр годен, если от зоны осталось не меньше этой доли (соседи, край кадра)...
    max_shift: float = 0.25  # ...и рамка сдвинулась от фона не больше чем на долю высоты


@dataclass
class EpisodeConfig:
    window: float = 20.0  # два полных цикла в пределах 20 с
    min_cycles: int = 2
    # с: цикл, начавшийся не позже этого после конца предыдущего, — то же движение; < 0 — отключено
    min_cycle_gap: float = 0.05
    merge_gap: float = 15.0  # циклы одного человека объединяются при паузе <= 15 с
    box_window: float = 0.5  # с: рамка события — медиана рамок трека в ±box_window от пика; 0 — рамка в пике
    confidence_threshold: float = 0.5
    cycle_weight: float = 0.45  # вклад одного идеального цикла в confidence
    evidence_weight: float = 0.6  # вклад прямого признака
    evidence_conf_ref: float = 0.4  # уверенность верификатора, считающаяся «полной»
    negative_penalty: float = 0.5
    event_negative_fraction: float = 0.5  # доля циклов с «чужим» предметом, отклоняющая событие
    flash_as_direct: bool = True  # вспышка зажигалки — прямой признак (1 цикл + вспышка = эпизод)
    flash_evidence: float = 1.0  # вклад вспышки в confidence, доля от полного прямого признака
    vapor_as_direct: bool = True  # выдох пара — прямой признак (1 цикл + пар = эпизод)
    vapor_evidence: float = 1.0  # вклад пара в confidence, доля от полного прямого признака


@dataclass
class ReportConfig:
    team: str = ""
    model_version: str = "smokedet-0.1.0"
    data_sources: List[str] = field(
        default_factory=lambda: [
            "COCO (предобучение YOLO11-pose, Ultralytics)",
            "Objects365/GoldG (предобучение YOLO-World, Ultralytics)",
            "Эталонный набор организатора — только подбор порогов",
        ]
    )
    fine_tuning: str = "нет: используются предобученные веса, на данных объекта подбираются только пороги"
    notes: str = ""


SECTIONS = {
    "video": VideoConfig,
    "detector": DetectorConfig,
    "tracking": TrackingConfig,
    "geometry": GeometryConfig,
    "cycle": CycleConfig,
    "verifier": VerifierConfig,
    "flash": FlashConfig,
    "vapor": VaporConfig,
    "episode": EpisodeConfig,
    "report": ReportConfig,
}


@dataclass
class Config:
    video: VideoConfig = field(default_factory=VideoConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    cycle: CycleConfig = field(default_factory=CycleConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    flash: FlashConfig = field(default_factory=FlashConfig)
    vapor: VaporConfig = field(default_factory=VaporConfig)
    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def hash(self, sections: Optional[List[str]] = None) -> str:
        data = self.to_dict()
        if sections is not None:
            data = {k: data[k] for k in sections}
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16]


def _coerce(default: Any, value: Any) -> Any:
    """Приводит значение из YAML или --set к типу значения по умолчанию: YAML читает «--set detector.device=0»
    как число 0 (select_device счёл бы его «auto»), а «verifier.positive_classes=cigarette» — как строку
    (словарь верификатора разобрал бы её по буквам)."""
    if value is None or isinstance(default, bool):
        return value
    if isinstance(default, str) and isinstance(value, (int, float)):
        return str(value)
    if isinstance(default, float) and isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    if isinstance(default, list) and isinstance(value, str):
        return [value]
    return value


def _section_from_dict(cls, data: Dict[str, Any]):
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise KeyError(f"Неизвестные параметры в секции {cls.__name__}: {sorted(unknown)}")
    values = {}
    for name, value in data.items():
        f = fields[name]
        default = f.default_factory() if f.default is dataclasses.MISSING else f.default
        values[name] = _coerce(default, value)
    return cls(**values)


def config_from_dict(data: Dict[str, Any]) -> Config:
    unknown = set(data) - set(SECTIONS)
    if unknown:
        raise KeyError(f"Неизвестные секции конфигурации: {sorted(unknown)}")
    return Config(**{name: _section_from_dict(cls, data.get(name) or {}) for name, cls in SECTIONS.items()})


def apply_overrides(data: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Применяет переопределения вида ``episode.confidence_threshold=0.6``."""
    data = copy.deepcopy(data)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Ожидается KEY=VALUE, получено: {item}")
        key, raw = item.split("=", 1)
        parts = key.strip().split(".")
        if len(parts) != 2:
            raise ValueError(f"Ключ должен быть вида секция.параметр: {key}")
        section, name = parts
        if not isinstance(data.get(section), dict):  # «cycle:» без параметров в YAML — это None
            data[section] = {}
        data[section][name] = yaml.safe_load(raw)
    return data


def load_config(path: Optional[str] = None, overrides: Optional[List[str]] = None) -> Config:
    data: Dict[str, Any] = {}
    if path:
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    data = apply_overrides(data, overrides or [])
    return config_from_dict(data)


def with_overrides(cfg: Config, overrides: List[str]) -> Config:
    return config_from_dict(apply_overrides(cfg.to_dict(), overrides))


def resolve_path(path: str, base: Path) -> str:
    """Относительные пути из конфигов и CSV считаются от корня проекта / файла списка."""
    p = Path(path)
    return str(p if p.is_absolute() else (base / p))
