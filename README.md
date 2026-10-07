# smokedet — детекция эпизодов, похожих на курение, по видео

Кейс хакатона НТУ «Сириус» × Видеофор (Гатчина, 8–10 октября 2026). Подробный план — [PLAN.md](PLAN.md).

Система **не подтверждает** факт курения и не делает выводов о личности. Она находит наблюдаемый эпизод
поведения, похожего на курение, для тревоги оператору.

## Как работает

```
видео → YOLO11-pose + ByteTrack (~10 кадр/с, уточнение позы мелких людей на кропе) + замер света у рта
      → склейка оборванных треков
      → сигнал d(t) = |кисть − рот| / размер человека   (только в зоне интереса и при высоте ≥ 72px)
      → циклы жеста: подход → пауза 0.5–5 с → отвод   (фильтры: телефон у уха, кисть ниже локтя, нет подхода)
      → верификатор YOLO-World на кропах паузы: сигарета/вейп/дым → прямой признак; бутылка/чашка/телефон → отказ
      → вспышка зажигалки: новый пересвет у рта и кисти (не вокруг человека) → прямой признак для циклов рядом
      → выдох пара: у рта после паузы светлее и бледнее, чем до жеста (человек стоит) → прямой признак цикла
      → эпизод: ≥2 цикла за 20 с ИЛИ 1 цикл + прямой признак; объединение при паузе ≤ 15 с; confidence → порог
      → predictions.csv/json, manifest.json, processing_log.csv, debug/*.json, rendered/*.mp4
```

| Модуль | Файл |
|---|---|
| Конфиг (все пороги) | `configs/default.yaml`, `smokedet/config.py` |
| Чтение видео | `smokedet/io_video.py` |
| Зона интереса | `smokedet/roi.py`, `tools/draw_roi.py` |
| Поза + трекинг + склейка | `smokedet/tracking.py`, `configs/bytetrack_cctv.yaml` |
| Сигнал «кисть → рот» | `smokedet/geometry.py` |
| Циклы жеста | `smokedet/cycles.py` |
| Верификатор предметов | `smokedet/verifier.py` |
| Вспышка зажигалки | `smokedet/flash.py` |
| Выдох пара или дыма | `smokedet/vapor.py` |
| Правило эпизода, confidence | `smokedet/episodes.py` |
| Выходные файлы, manifest | `smokedet/export.py` |
| Ролик с отметками | `smokedet/render.py` |
| Протокол оценки | `smokedet/evaluation.py`, `evaluate.py` |
| Подбор порогов | `tune.py` |

## Установка

```bash
uv venv .venv --python 3.11 && uv pip install --python .venv/bin/python -r requirements.txt
# или: python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
source .venv/bin/activate
```

На Windows интерпретатор — `.venv/Scripts/python.exe`. torch ставится под своё железо (команды — в начале
`requirements.txt`). Точная копия проверенного окружения (Windows, CUDA 12.6):

```bash
uv pip install --python .venv/Scripts/python.exe -r requirements.lock     --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
```

Веса и офлайн-словарь верификатора — **до выезда на объект**, там может не быть интернета:

```bash
python tools/prepare_weights.py   # weights/: pose, YOLO-World, COCO-откат yolo11s.pt + словарь *-vocab-<хэш>.pt
```

С запечённым словарём прогону не нужен CLIP. Если поменяли списки классов верификатора — запустите скрипт снова.

## Запуск

Один клип:

```bash
python run.py --input data/clip.mp4 --camera-id cam_fountain --start-time 2026-10-09T12:00:00 \
    --roi data/roi_fountain.json --out runs/demo --render
```

Набор клипов (скрытый набор — **одна команда**):

```bash
python run.py --inputs data/hidden.csv --out runs/final
```

`hidden.csv` (пути — относительно CSV; `clip_id`, `start_time`, `roi` необязательны):

```csv
video_path,camera_id,clip_id,start_time,roi
cam1_001.mp4,cam_fountain,cam1_001,2026-10-09T12:00:00,roi_fountain.json
```

Параметры без правки YAML: `--set episode.confidence_threshold=0.6 --set detector.imgsz=960`, устройство: `--device cpu|mps|0`.

Маска зоны: PNG (белое — зона) или JSON с полигонами. Нарисовать по кадру:

```bash
python tools/draw_roi.py --video data/clip.mp4 --out data/roi_fountain.json
```

Пиксельные полигоны из `draw_roi.py` хранят размер кадра (`frame_size`): на видео другого разрешения той же камеры
они масштабируются, как и PNG-маска.

## Выход (`--out`)

| Файл | Содержимое |
|---|---|
| `predictions.csv` / `.json` | `camera_id, clip_id, event_id, start_sec, end_sec, confidence, label, person_track_id, peak_sec, x1, y1, x2, y2` — одна строка на событие, рамка человека в `peak_sec` в пикселях исходного кадра |
| `manifest.json` | порог, правила объединения, версия, git-коммит, веса + sha256, лицензии, промпты, оборудование, полный конфиг |
| `processing_log.csv` | длительность клипа, время по стадиям, `realtime_factor` (>1 — быстрее реального времени) |
| `debug/<clip>.json` | все циклы (включая отброшенные с причиной: `short_hold`, `long_hold`, `phone_like`, `hand_low`, `hand_above_mouth`, `face_hidden`, `no_approach`, `object:bottle`…) и события (отклонённые — с `reject_reason`, например `objects_in_100%_cycles`) — для разбора ошибок |
| `rendered/<clip>.mp4` | с `--render`: зона, рамки (зелёная — человек, жёлтая — кисть у рта, красная — событие `SMOKING-LIKE 0.82`) |
| `cache/` | треки; повторный запуск с другими порогами не пересчитывает детекцию |

## Оценка и подбор порогов

```bash
python evaluate.py --pred runs/ref/predictions.csv --gt data/ref_gt.csv \
    --durations runs/ref/processing_log.csv --out runs/ref/report
```

Протокол как в кейсе: эталон ±3 с, пересечение ≥ 1 с, IoU рамок ≥ 0.30, сопоставление один к одному
по наибольшему пересечению, предсказание на ≥ 50% в IGNORE не считается FP. Формат эталона:
`camera_id, clip_id, label (POSITIVE|NEGATIVE|IGNORE), start_sec, end_sec [, person_gt_id, peak_sec, x1, y1, x2, y2, scenario]`.
Колонка `scenario` (P01…N08) даёт разбивку FP/FN по сценариям. Если организатор выдаст свой валидатор и формат —
адаптировать `smokedet/evaluation.py::parse_items`.

Подбор — **только на эталонном наборе**:

```bash
python tune.py --inputs data/ref.csv --gt data/ref_gt.csv --out runs/tune \
    --grid episode.confidence_threshold=0.3,0.4,0.5,0.6 --grid cycle.near_enter=0.5,0.6,0.7
```

## Фиксация результата (9 октября)

1. Подобрать пороги на эталоне (`tune.py`), записать лучшие в `configs/default.yaml`.
2. Закоммитить код и конфиг (в `manifest.json` попадёт коммит без `-dirty`).
3. `python run.py --inputs data/hidden.csv --out runs/final` → сдать `manifest.json`, `predictions.csv`, архив с кодом.
4. После фиксации порог, постобработку и предсказания не менять.

## Скорость (MacBook, Apple Silicon, MPS, кадр 1920×1080, только поза + трекинг)

| Модель | imgsz 1280 | imgsz 960 |
|---|---|---|
| yolo11m-pose (по умолчанию) | 8.3 кадр/с | 13.1 кадр/с |
| yolo11s-pose | 18.9 кадр/с | 27.6 кадр/с |

При анализе 10 кадр/с конфигурация по умолчанию работает примерно в реальном времени (×0.8).
Быстрее: `--set detector.pose_model=weights/yolo11s-pose.pt --set detector.imgsz=960` или `--set video.target_fps=6`.

## Тесты

```bash
python -m pytest -q tests
```

Синтетические сигналы проверяют поиск циклов, правило эпизода, объединение, фильтры; отдельно — протокол оценки, ROI, конфиг.

## Данные, модели, лицензии

- Предобученные веса Ultralytics (YOLO11-pose — COCO; YOLO-World v2 — Objects365/GoldG), лицензия **AGPL-3.0**. Дообучения нет.
- Промпты верификатора — `verifier.positive_classes` / `negative_classes` / `track_negative_classes` / `track_drink_classes` в конфиге (дублируются в `manifest.json`).
- Обработка локальная. Исходные записи объекта не публикуются и не коммитятся (`data/`, `*.mp4` в `.gitignore`).

## Ограничения

- Ключевые точки лица на людях ниже ~100px шумят → пропуски в дальней зоне.
- Питьё и еда дают тот же жест; защита — верификатор предметов и правило двух циклов.
- Чупа-чупс: палочку в паузе детектор принимает за сигарету. Поэтому конфету (`lollipop`) ищем у кисти и рта по всему
  треку, а не только в паузах (`verifier.track_negative_*`). Это работает на крупном плане; при росте человека ~100 px
  конфета занимает пару пикселей и не видна.
- Муляж сигареты на CCTV почти не виден, поэтому основной сигнал — повторяющийся жест.
- Вспышка зажигалки видна, только когда пламя засвечивает лицо и кисти (темно, крупный план). Маленький огонёк
  днём — несколько пикселей, его не находят ни яркость, ни YOLO-World (классы `lighter`/`flame` проверены).
- Выдох пара замеряется, только пока человек стоит и зону у рта не закрывают соседи: фон за идущим человеком
  другой. Порог `vapor.min_net` подобран по нескольким выдохам в своих роликах — уточнять на эталоне. На муляжах
  пара нет, поэтому он только повышает уверенность и не обязателен.
- Касание лица, почёсывание носа и поправление маски по позе не отличить от затяжки: если жест повторился
  дважды за 20 с с паузой у рта, это эпизод по правилу двух циклов. Отсекает их только порог confidence, который
  выбирается на эталоне: у событий с прямым признаком (пар, вспышка, предмет) confidence выше.
- Обработка офлайн: границы события уточняются по фазам подхода/отвода, поэтому задержка ≈ 0.
