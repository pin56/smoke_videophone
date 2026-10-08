# CLAUDE.md

## Контекст

Кейс хакатона НТУ «Сириус» × Видеофор (8–10 октября 2026): найти эпизоды поведения, похожего на курение, в видео
стационарных CCTV-камер (Гатчина). Требования кейса — `TZ.md`, план и обоснование решений — `PLAN.md`, запуск —
`README.md`. Ручная разметка пользователя — `rasmetka.md`, из неё `tools/match_markup.py` строит `rasmetka_gt.csv`
(только время, без рамок). Документация, комментарии и логи в коде — на русском.

Правила кейса, которые влияют на код:
- Главная метрика — Event F1. Предсказание засчитывается, если интервал совпадает с эталоном (эталон расширен на ±3 с, пересечение ≥ 1 с) **и** IoU рамки человека в `peak_sec` ≥ 0.3. Поэтому пайплайн работает по отдельным людям, а не по клипу целиком.
- Положительный эпизод: ≥ 2 полных цикла «рука ко рту → пауза 0.5–5 с → отвод» за 20 с ИЛИ 1 цикл + прямой признак (сигарета/вейп/дым). Циклы одного человека объединяются при паузе ≤ 15 с; события разных людей не объединяются.
- Пороги подбираются только на эталонном наборе. До фиксации `manifest.json` и предсказаний скрытый набор нельзя использовать для подстройки.

## Команды

Окружение: `.venv` (Python 3.11, uv). Pip в venv нет, автоустановка ultralytics не работает: пакеты ставить
`uv pip install --python .venv/bin/python <pkg>`. На Windows интерпретатор — `.venv/Scripts/python.exe`.
Версии — `requirements.txt`, снимок окружения — `requirements.lock` (пересоздавать `uv pip freeze --color never`).

```bash
.venv/bin/python -m pytest -q tests                                   # все тесты (без моделей, ~8 с)
.venv/bin/python -m pytest -q tests/test_logic.py::test_merge_gap     # один тест
.venv/bin/python run.py --input videos/1.mp4 --camera-id cam1 --out runs/x --render   # один клип
.venv/bin/python run.py --inputs data/hidden.csv --out runs/final [--workers 2]   # набор (CSV: video_path,camera_id[,clip_id,start_time,roi])
.venv/bin/python run_rt.py --inputs runs/rt/all.csv --out runs/rt/final [--no-objects]   # потоковый режим
.venv/bin/python tools/tune_rt.py --inputs runs/rt/all.csv --cache runs/rt/final/cache --grid appearance.min_score=0,0.5
.venv/bin/python ui.py --inputs data/demo.csv --out runs/final        # веб-интерфейс повтора на 127.0.0.1:8765
.venv/bin/python ui.py --videos videos --out runs/ui                  # то же по папке, анализ при открытии клипа
.venv/bin/python monitor.py --analyzer demo --out runs/monitor        # мониторинг: стена камер, архив, журнал
.venv/bin/python monitor.py --analyzer realtime --enqueue videos/true videos/false
.venv/bin/python evaluate.py --pred runs/x/predictions.csv --gt gt.csv --durations runs/x/processing_log.csv --out runs/x/report
.venv/bin/python tune.py --inputs ref.csv --gt ref_gt.csv --out runs/tune --grid episode.confidence_threshold=0.3,0.5 [--cv auto|clip|camera|none]
.venv/bin/python tools/eval_markup.py --run runs/x [--run runs/y]   # оценка по rasmetka_gt.csv (только время, без IoU)
.venv/bin/python tools/tune_markup.py --inputs runs/goal/all.csv --cache runs/goal/cache --grid appearance.min_score=4,6 --diag
.venv/bin/python tools/cvat_export.py --inputs ref.csv --cache runs/x/cache --out runs/cvat
.venv/bin/python tools/cvat_import.py runs/cvat_done --inputs ref.csv --cache runs/x/cache --out gt.csv
.venv/bin/python tools/export_cycles.py --inputs ref.csv --out data/labeling/set1   # кадры циклов для разметки
.venv/bin/python tools/label_server.py --data data/labeling/set1   # веб-разметчик циклов (обучение — tools/train_cycle_clf.py)
.venv/bin/python tools/draw_roi.py --video data/clip.mp4 --out data/roi_cam1.json
.venv/bin/python tools/render_markup.py --cvat runs/cvat --out runs/markup
```

Любой параметр переопределяется через `--set section.key=value`. Линтера и сборки нет.

## Архитектура

Пайплайн разделён на две стадии, чтобы перебор порогов не требовал повторной детекции:

1. **`Pipeline.track()`** (`smokedet/tracking.py`) — тяжёлая: YOLO-pose + ByteTrack на каждом `stride`-м кадре, уточнение
   позы мелких людей на кропе, `flash.LightMeter` (пересвет у рта/кисти). Результат — `TrackSet` (bbox + 17 точек COCO +
   `light`), не зависит от ROI и порогов, кэшируется в `<out>/cache/<clip_id>_tracks.pkl`. Ключ (`pipeline.track_cache_key`) —
   видео (размер, mtime) + хэш секций `detector`/`video` + хэш YAML трекера. Время наблюдения — pts кадра (`io_video.FrameClock`:
   в CCTV с переменным fps `idx/fps` уходит до 0.7 с); `render.py` считает время так же.
2. **`Pipeline.analyze(clip, trackset, cfg)`** — лёгкая, зависит только от порогов; `tune.py` вызывает её много раз
   (подбор — `smokedet/tuning.py`). Цепочка: `stitch_tracks` → `geometry.build_signal` (`d(t)` кисть→рот) →
   `cycles.detect_cycles` → `verifier.verify` (open-vocab детектор в паузе и по треку) → `flash` → `vapor` →
   `appearance` (CLIP zero-shot по кропам «рот + кисть») → `episodes.build_events`. Отброшенные циклы остаются с
   `valid=False` и `reject_reason`; `appearance` может вернуть цикл (`rescued_from`). `export.event_records` берёт только
   события с `passed`.

Веб-интерфейс (`ui.py`, `smokedet/ui/`) — поверх готового `Analysis`; `reveal_t` у циклов/событий — момент, к которому
анализ о них «знает», фронтенд в повторе показывает только открытое. Время анализа — pts, как `currentTime` браузера.
Видео отдаётся с HTTP Range (без него не работает перемотка). Причины отказа и классы верификатора переводятся в
`payload.CYCLE_REASONS`/`OBJECT_NAMES` — новую причину или класс добавляйте туда.

Мониторинг (`monitor.py`, `smokedet/monitor/`, `ui/static/wall.html`) — только stdlib, пайплайн импортируется лениво
(`analysis.PipelineAnalyzer`), сервер работает без torch/cv2. В журнале `detection` всегда `auto`, решение оператора —
поле `review`. Формат кадра — `Person.encode` ↔ `Scene.decode` в `scene.js`: меняйте оба. `parse_range` продублирован из
`ui/server.py`, потому что тот импортирует cv2.

Потоковый режим (`run_rt.py`, `smokedet/realtime.py`) — отдельный путь без `Pipeline.track`/`analyze`; общие только
`build_signal`, `detect_cycles`, `build_events` и `appearance`. 1 Гц детектор + трекер + поза на кропах, при подходе руки
ко рту — плотная проверка человека (`PersonJob`). Порог `appearance.min_score` при `objects_per_event` проверяется по
лучшему циклу события, а не по каждому. Правило режима — `realtime.rule_overrides` (YAML → rule_overrides → `--set`).
Кэш `<out>/cache/<clip>_rt.pkl`: ключ включает секцию `realtime` кроме `RULE_FIELDS` — параметр, не влияющий на отсчёты
стадий 1–3, добавляйте в `RULE_FIELDS`, иначе он сбросит кэш. Повтор из кэша (`replay_job`) даёт то же, что поток.
`load` в `realtime_log.csv` — время / длительность (≤ 1 — успевает); два GPU-прогона одновременно его искажают.

## Неочевидные моменты

- `configs/default.yaml` обязан совпадать с dataclass-значениями в `smokedet/config.py` (`test_config_roundtrip`): новый параметр — в оба файла. Неизвестные ключи — ошибка. Значения приводятся к типу умолчания (`config._coerce`): `--set detector.device=0` — строка `"0"`.
- Относительные пути к весам и конфигу трекера — от корня проекта (`pipeline.project_path`), пути в CSV клипов — от CSV.
- Трекер сбрасывается между клипами `model.track(..., persist=False)`. Уточнение позы — **отдельный** экземпляр `YOLO`: `predict()` на основной модели запустил бы колбэки трекера.
- Константы `SAT`, `BG_*`, `ZONE_RADIUS` в `smokedet/flash.py` влияют на кэш треков, но не входят в ключ: при изменении поднимайте `tracking.CACHE_VERSION`. `tracking` импортирует `LightMeter` лениво (цикл импортов flash → geometry → tracking).
- Верификатор и пар перечитывают кадры (`io_video.FrameReader`) за один проход по возрастанию индекса. Seek `CAP_PROP_POS_FRAMES` на переменном fps промахивается, поэтому дальний переход — по pts из трекера: `reader.get(idx, t)`.
- FP16 (`detector.half`, `verifier.half`) действует только на CUDA через `tracking.half_kwargs`. Не передавайте `half` в ultralytics 8.4 — предупреждение на каждом кадре. FP16 сдвигает точки позы и меняет пограничные события: пороги подбирайте в режиме финального прогона.
- `run.py --workers N` — процессы spawn, у каждого свои модели; на 6 ГБ с CLIP-L два процесса медленнее одного.
- Кэш верификатора в памяти; модель запускается с `PREDICT_CONF`, `cfg.conf` применяется при агрегации. Без open-vocab модели верификатор откатывается на COCO (только негативные классы).
- Словарь YOLO-World запекается в `weights/<модель>-vocab-<хэш классов>.pt`. После правки списков классов запустите `tools/prepare_weights.py` с интернетом, иначе офлайн верификатор откатится на COCO. Он же качает CLIP для `appearance` (`weights/ViT-L-14-336px.pt`) и все веса конфига — запускать до выезда.
- Классификаторы, обученные на своих клипах, проверять «оставить камеру» и AUC внутри клипа: «оставить клип» показывает AUC 0.8 у модели, выучившей камеру.
- `manifest.json` пишет git-коммит с `-dirty` при незакоммиченных изменениях: перед финальным прогоном закоммитить код.
- CVAT (`smokedet/cvat.py`): кадр — последовательный индекс, секунды — по pts из кэша треков, поэтому импорт требует тот же кэш, что экспорт.
- Формат эталона в `smokedet/evaluation.py::parse_items` — предположение; при выдаче формата организатором адаптировать её.
- Pose-модель — `yolo11m-pose`. `yolo26m-pose`, `yolo11x-pose` и TensorRT-engine (`tools/export_trt.py`) на разметке пользователя хуже по F1; на эталоне организатора сравнение стоит повторить.
