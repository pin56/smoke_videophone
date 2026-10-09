#!/usr/bin/env python
"""Лаборатория VLM-арбитра потокового режима: локальная мультимодальная модель (Qwen-VL через transformers)
смотрит кропы лучших циклов события и выбирает, что делает человек; событие с малой вероятностью «курит / вейп»
снимается (вето), отвергнутое CLIP — может вернуться. Вызов — один на событие, не на цикл.

    python tools/vlm_lab.py extract --inputs runs/rt_noobj/all.csv --cache runs/cpu/l168/cache --config l168.yaml \\
        --emb runs/clip_lab/L168.pkl --lab views=0+1+2 --lab agg=top2 --out runs/vlm/l168
    python tools/vlm_lab.py vlm --data runs/vlm/l168 --model Qwen/Qwen3-VL-2B-Instruct --out runs/vlm/l168/q2b.json
    python tools/vlm_lab.py score --inputs ... --cache ... --config ... --emb ... --lab ... --data runs/vlm/l168 \\
        --vlm runs/vlm/l168/q2b.json --grid vlm.thr=0,0.1,0.3 --grid vlm.hi=inf,9

События — повтор из кэша run_rt.py с оценками CLIP из банка tools/clip_lab.py (как clip_lab score), поэтому
extract и score обязаны получать одинаковые --config/--set/--lab. Ключи сетки vlm.*: thr — вето прошедшего события
при оценке VLM ниже thr; hi — вето только при лучшей оценке CLIP события ниже hi (inf — всегда); rescue — событие,
отвергнутое CLIP (appearance / many_cycles), возвращается при оценке VLM не ниже rescue (none — нет).
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

log = logging.getLogger("vlm_lab")
SCALES = (0.7, 1.2)  # стороны кропа, доли роста: голова и кисти / человек с окружением
CROP_PX = 448
CLASSES = (
    ("smoke", "smoking a cigarette"),
    ("vape", "vaping (an electronic cigarette)"),
    ("drink", "drinking from a bottle, can or cup"),
    ("eat", "eating food"),
    ("phone", "using a mobile phone"),
    ("other", "touching the face (scratching, wiping the mouth, adjusting glasses or mask) or something else"),
)  # fmt: skip
POSITIVE = ("smoke", "vape")
SCENE = ("The {n} images are crops from a low-resolution outdoor CCTV camera. They show the same person at {n} "
         "different moments when the person's hand is raised to the face. ")  # fmt: skip
OBJECTS = (
    ("smoke", "a cigarette"),
    ("vape", "an electronic cigarette or vape"),
    ("drink", "a bottle, can or cup"),
    ("eat", "food"),
    ("phone", "a mobile phone"),
    ("other", "nothing (an empty hand) or something else"),
)  # fmt: skip
# вопрос -> (текст, варианты); act — что делает, obj — что в руке (имитация курения — пустая рука), yn — да/нет
QUESTIONS = {
    "act": (SCENE + "What is the person most likely doing?\n{options}\nAnswer with a single letter.", CLASSES),
    "obj": (SCENE + "What does the person hold in the hand near the mouth?\n{options}\nAnswer with a single letter.",
            OBJECTS),
    "yn": (SCENE + "Is the person holding a cigarette or an electronic cigarette (vape) in the hand near the mouth? "
           "Answer Yes or No.", (("smoke", "Yes"), ("other", "No"))),
}  # fmt: skip
REJECTS = ("appearance", "many_cycles")  # отказы rt_events по CLIP — кандидаты на возврат


def event_key(clip_id: str, e) -> str:
    return f"{clip_id}|{e.track_id}|{e.start:.2f}"


def clip_bank(a):
    """Повтор клипов и оценки кадров CLIP из банка — как clip_lab score."""
    from clip_lab import TextBank, replay_clips
    from smokedet.realtime import project_path

    cfg0, clips = replay_clips(a)
    if not a.emb:  # без банка: события без стадии 4 — все геометрические кандидаты
        return clips, None, None, None
    with open(a.emb, "rb") as fh:
        bank = pickle.load(fh)
    text = TextBank(bank["model"], project_path(cfg0.appearance.weights_dir))
    names = ["wide" if v >= 0.5 else "mouth" for v in bank["views"]]
    lab = {"frames": "u8", "views": "0+1", "agg": "max", "hand": 0.0, "diff": 0.0, "stop": None, "prompts": "base"}
    for item in a.lab:
        k, v = item.split("=", 1)
        lab[k] = v
    lab["views"] = [int(v) for v in str(lab["views"]).split("+")]
    lab["hand"], lab["diff"] = float(lab["hand"]), float(lab["diff"])
    fs = {k: (text.frame_scores(r["emb"], names, lab["prompts"]),
              text.frame_scores(r["rest"], ["hand", "mouth"], lab["prompts"]) if "rest" in r else None)
          for k, r in bank["recs"].items()}  # fmt: skip
    return clips, bank, fs, lab


def clip_events(clips, cfg, bank, fs, lab):
    """(clip, run, signals, events) по клипам: циклы получают оценку CLIP из банка, события — rt_events."""
    from clip_lab import cycle_appearance, cycle_key
    from smokedet import appearance as ap
    from smokedet.realtime import keep_cycle, rt_events

    keep = keep_cycle(cfg)
    for clip, run, cycles0, signals in clips:
        cycles = copy.deepcopy(cycles0)
        for cs in cycles.values() if bank is not None else ():
            for c in cs:
                key = cycle_key(clip.clip_id, c)
                if (c.valid or ap.rescuable(c, cfg.appearance)) and key in bank["recs"]:
                    ap.apply_score(c, cycle_appearance(bank["recs"][key], fs[key], lab)[0], cfg.appearance, keep)
        yield clip, run, signals, rt_events(cycles, signals, cfg)


def candidate(e) -> bool:
    return e.passed or e.reject_reason.split(":")[0] in REJECTS


def evaluate(gt, clips_events, decide=None):
    """Оценка по разметке: decide(clip_id, e) -> прошло ли событие (None — как есть)."""
    from eval_markup import RunClip, evaluate_run
    from match_markup import canonical_stem
    from smokedet.export import event_records

    rc = {}
    for clip, run, _, events in clips_events:
        if decide is not None:
            events = [copy.copy(e) for e in events]
            for e in events:
                e.passed = decide(clip.clip_id, e)
        vp = Path(clip.video_path)
        c = RunClip(canonical_stem(vp.stem), clip.clip_id, vp.parent.name, run.meta.duration)
        c.preds = event_records(events, clip.camera_id, clip.clip_id)
        rc[c.stem] = c
    return evaluate_run(gt, rc), rc


def extract(a) -> int:
    import cv2

    from clip_lab import hold_samples
    from eval_markup import MERGE_GAP, load_gt
    from smokedet import appearance as ap
    from smokedet.io_video import FrameReader
    from smokedet.realtime import load_rt_config

    cfg = load_rt_config(a.config, a.set)
    clips, bank, fs, lab = clip_bank(a)
    ce = list(clip_events(clips, cfg, bank, fs, lab))
    res, _ = evaluate(load_gt(a.gt, MERGE_GAP), ce)
    verdict = {m["pred_event_id"]: ("dup" if m.get("dup") else m["result"]) for m in res["overall"]["matches"]
               if m["result"] in ("TP", "FP")}  # fmt: skip
    out = Path(a.out)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    items, t0 = [], time.perf_counter()
    for clip, run, signals, events in ce:
        need = defaultdict(list)  # кадр -> (событие, номер цикла, центр, рост)
        n_passed = 0
        for e in events:
            if e.passed:
                n_passed += 1
            if not candidate(e):
                continue
            scored = [c for c in e.cycles if np.isfinite(c.appearance)]
            pool = sorted(scored, key=lambda c: -c.appearance) if scored else sorted(e.cycles, key=lambda c: c.hold_start - c.hold_end)
            item = {"key": event_key(clip.clip_id, e), "clip_id": clip.clip_id, "folder": Path(clip.video_path).parent.name,
                    "track": int(e.track_id), "start": round(e.start, 2), "end": round(e.end, 2), "passed": bool(e.passed),
                    "reject": e.reject_reason, "label": verdict.get(f"{clip.clip_id}_{n_passed:03d}", "") if e.passed else "",
                    "clip_best": max((c.appearance for c in scored), default=float("nan")), "n_cycles": len(e.cycles),
                    "images": {str(s): [None] * min(a.top, len(pool)) for s in SCALES}, "times": []}  # fmt: skip
            for j, c in enumerate(pool[: a.top]):
                sig = signals[c.track_id]
                ks = hold_samples(sig, c)
                d = np.where(np.isfinite(sig.d[ks]), sig.d[ks], np.inf)
                k = int(ks[int(np.argmin(d))])
                centers, height = ap.crop_geometry(sig, c, np.array([k]))
                need[int(sig.frame_idx[k])].append((item, j, centers[0], height, float(sig.t[k])))
                item["times"].append(round(float(sig.t[k]), 2))
            items.append(item)
        with FrameReader(clip.video_path, synthetic_from=run.meta.synthetic_from) as reader:
            for f in sorted(need):
                frame = reader.get(f, need[f][0][4])
                if frame is None:
                    continue
                for item, j, (cx, cy), height, _ in need[f]:
                    for s in SCALES:
                        crop = cv2.resize(ap.square_crop(frame, cx, cy, s * height), (CROP_PX, CROP_PX), interpolation=cv2.INTER_CUBIC)
                        name = f"{item['key'].replace('|', '_')}_{j}_{int(s * 100):03d}.jpg"
                        # imwrite на Windows молча не пишет путь с кириллицей
                        (out / "crops" / name).write_bytes(cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes())
                        item["images"][str(s)][j] = f"crops/{name}"  # по убыванию оценки CLIP цикла
        for item in items:
            item["images"] = {s: [p for p in ps if p] for s, ps in item["images"].items()}
        log.info("%s: событий-кандидатов %d", clip.clip_id, sum(1 for e in events if candidate(e)))
    with open(out / "events.json", "w", encoding="utf-8") as fh:
        json.dump(items, fh, ensure_ascii=False, indent=1)
    n_pass = sum(i["passed"] for i in items)
    print(f"{out / 'events.json'}: событий {len(items)} (прошли {n_pass}, отвергнуты CLIP {len(items) - n_pass}), "
          f"{time.perf_counter() - t0:.0f} с")  # fmt: skip
    return 0


def vlm(a) -> int:
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    data = Path(a.data)
    items = json.loads((data / "events.json").read_text(encoding="utf-8"))
    if a.limit:
        items = items[: a.limit]
    if a.threads:
        torch.set_num_threads(a.threads)
    proc = AutoProcessor.from_pretrained(a.model)
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}.get(a.dtype)
    kw = {"dtype": dtype or torch.float16, "device_map": a.device}
    if a.quant == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.float16,
                                                       llm_int8_enable_fp32_cpu_offload=a.lean)  # fmt: skip
    elif a.device == "cpu" and dtype is None:
        kw["dtype"] = torch.bfloat16
    if a.lean:  # таблица эмбеддингов и lm_head (по 1.2 ГБ у 8B) — в RAM, иначе 8B nf4 не влезает в 6 ГБ
        kw["device_map"] = {"": a.device, "model.language_model.embed_tokens": "cpu", "lm_head": "cpu"}
    model = AutoModelForImageTextToText.from_pretrained(a.model, **kw).eval()
    prompt, classes = QUESTIONS[a.question]
    yn = a.question == "yn"
    letters = [c[1] for c in classes] if yn else "ABCDEF"[: len(classes)]  # у yn ответ — сами слова
    ids = [proc.tokenizer.encode(x, add_special_tokens=False)[0] for x in letters]
    if a.lean:  # веса на CPU accelerate держит meta-тензорами — берём их из safetensors
        from safetensors import safe_open
        from transformers.utils import cached_file

        wmap = json.loads(Path(cached_file(a.model, "model.safetensors.index.json")).read_text())["weight_map"]

        def tensor(name):
            with safe_open(cached_file(a.model, wmap[name]), "pt") as fh:
                return fh.get_tensor(name).to(torch.float16)

        class CpuEmbed(torch.nn.Module):  # поиск строк в RAM, результат — на GPU
            def __init__(self, w):
                super().__init__()
                self.emb = torch.nn.Embedding.from_pretrained(w, freeze=True)

            def forward(self, x):
                return self.emb(x.cpu()).to(a.device)

        model.model.language_model.embed_tokens = CpuEmbed(tensor("model.language_model.embed_tokens.weight"))
        head = torch.nn.Linear(model.lm_head.in_features, len(ids), bias=False, dtype=torch.float16)
        head.weight.data.copy_(tensor("lm_head.weight")[ids])  # нужны только логиты букв ответа
        model.lm_head = head.to(a.device)
        ids = list(range(len(ids)))
    orders = [list(range(len(classes)))]
    if a.orders > 1 and not yn:
        orders.append(orders[0][::-1])  # обратный порядок вариантов: снимает предпочтение буквы
    scores, t_all = {}, 0.0
    for n, item in enumerate(items):
        paths = item["images"][a.scale][: a.k]
        if not paths:
            continue
        imgs = [Image.open(data / p).convert("RGB") for p in paths]
        if a.px != CROP_PX:
            imgs = [im.resize((a.px, a.px), Image.BICUBIC) for im in imgs]
        probs = np.zeros(len(classes))
        mass = []
        t0 = time.perf_counter()
        for order in orders:
            options = "\n".join(f"{letters[i]}. {classes[c][1]}" for i, c in enumerate(order))
            msg = [{"role": "user", "content": [*({"type": "image", "image": im} for im in imgs),
                                                {"type": "text", "text": prompt.format(n=len(imgs), options=options)}]}]  # fmt: skip
            inputs = proc.apply_chat_template(msg, tokenize=True, add_generation_prompt=True, return_dict=True,
                                              return_tensors="pt").to(a.device)  # fmt: skip
            with torch.inference_mode():
                logits = model(**inputs).logits[0, -1].float()
            mass.append(float(torch.softmax(logits, 0)[ids].sum()))  # при --lean всегда 1
            p = torch.softmax(logits[ids], 0).cpu().numpy()
            for i, c in enumerate(order):
                probs[c] += p[i] / len(orders)
        dt = time.perf_counter() - t0
        t_all += dt
        smoke = float(sum(probs[i] for i, (c, _) in enumerate(classes) if c in POSITIVE))
        scores[item["key"]] = {"smoke": smoke, "p": {c: round(float(probs[i]), 4) for i, (c, _) in enumerate(classes)},
                               "mass": round(min(mass), 3), "sec": round(dt, 3)}  # fmt: skip
        if n % 20 == 0:
            log.info("%d/%d %s %s smoke=%.2f mass=%.2f %.2f с", n, len(items), item["key"], item["label"] or item["reject"],
                     smoke, min(mass), dt)  # fmt: skip
    meta = {"question": a.question, "model": a.model, "quant": a.quant, "lean": a.lean, "scale": a.scale, "k": a.k, "px": a.px, "orders": a.orders,
            "device": a.device, "dtype": str(kw["dtype"]).split(".")[-1], "threads": torch.get_num_threads(),
            "sec_per_event": t_all / max(1, len(scores)),
            "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0}  # fmt: skip
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "scores": scores}, fh, ensure_ascii=False, indent=1)
    print(f"{a.out}: событий {len(scores)}, {meta['sec_per_event']:.2f} с/событие, память {meta['max_mem_gb']:.1f} ГБ")
    return 0


def sheets(a) -> int:
    """Листы для ручной разметки событий: строка — событие (номер, клип, интервал), до k кропов вида scale с
    временем кадра; index.csv — номер -> ключ события."""
    import csv
    import re

    import cv2

    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = json.loads((data / "events.json").read_text(encoding="utf-8"))
    items.sort(key=lambda i: ([int(x) if x.isdigit() else x for x in re.split(r"(\d+)", i["clip_id"])], i["start"]))
    tile, head, font = 240, 30, cv2.FONT_HERSHEY_SIMPLEX
    rows = []
    for n, item in enumerate(items, 1):
        row = np.full((head + tile, tile * a.k, 3), 255, np.uint8)
        for j, p in enumerate(item["images"][a.scale][: a.k]):
            im = cv2.imdecode(np.fromfile(str(data / p), np.uint8), cv2.IMREAD_COLOR)
            im = cv2.resize(im, (tile, tile), interpolation=cv2.INTER_AREA)
            cv2.putText(im, f"{item['times'][j]:.1f}s", (5, tile - 8), font, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(im, f"{item['times'][j]:.1f}s", (5, tile - 8), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            row[head:, j * tile : (j + 1) * tile] = im
        video = f" ({item['video']})" if item.get("video") else ""
        text = f"#{n}  {item['clip_id']}{video}  {item['start']:.1f}-{item['end']:.1f} s"
        cv2.putText(row, text, (6, 22), font, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.line(row, (0, 0), (row.shape[1], 0), (0, 0, 255), 3)
        rows.append(row)
        item["n"] = n
    paths = []
    for s in range(0, len(rows), a.per_sheet):
        path = out / f"sheet_{s // a.per_sheet + 1:02d}.jpg"
        path.write_bytes(cv2.imencode(".jpg", np.vstack(rows[s : s + a.per_sheet]), [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes())
        paths.append(path)
    with open(out / "index.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["n", "key", "clip_id", "track", "start", "end", "times", "label"])
        for i in items:
            w.writerow([i["n"], i["key"], i["clip_id"], i["track"], i["start"], i["end"], " ".join(map(str, i["times"])), i["label"]])
    print(f"{out}: событий {len(items)}, листов {len(paths)}")
    return 0


def auc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def score(a) -> int:
    from clip_lab import parse_grid
    from eval_markup import MERGE_GAP, load_gt
    from smokedet.realtime import load_rt_config

    cfg = load_rt_config(a.config, a.set)
    clips, bank, fs, lab = clip_bank(a)
    ce = list(clip_events(clips, cfg, bank, fs, lab))
    gt = load_gt(a.gt, MERGE_GAP)
    gt_true = {s for s, g in gt.items() if g.episodes}
    items = {i["key"]: i for i in json.loads((Path(a.data) / "events.json").read_text(encoding="utf-8"))}
    for path in a.vlm:
        run = json.loads(Path(path).read_text(encoding="utf-8"))
        vs, m = run["scores"], run["meta"]
        tp = [vs[k]["smoke"] for k, i in items.items() if i["label"] == "TP" and k in vs]
        fp = [vs[k]["smoke"] for k, i in items.items() if i["label"] in ("FP", "dup") and k in vs]
        fpf = [vs[k]["smoke"] for k, i in items.items() if i["label"] in ("FP", "dup") and i["folder"] == "false" and k in vs]
        rej = [vs[k]["smoke"] for k, i in items.items() if not i["passed"] and k in vs]
        ctp = [items[k]["clip_best"] for k, i in items.items() if i["label"] == "TP"]
        cfp = [items[k]["clip_best"] for k, i in items.items() if i["label"] in ("FP", "dup")]
        print(f"\n## {Path(path).name}: {m['model']} {m['quant']} вид {m['scale']} k={m['k']} px={m['px']} "
              f"порядков {m['orders']} — {m['sec_per_event']:.2f} с/событие, {m['max_mem_gb']:.1f} ГБ")
        print(f"AUC TP против FP: VLM {auc(tp, fp):.3f}, CLIP {auc(ctp, cfp):.3f}; медиана VLM: TP {np.median(tp):.2f}, "
              f"FP {np.median(fp):.2f} (в false {np.median(fpf) if fpf else float('nan'):.2f}), "
              f"отвергнутые CLIP {np.median(rej) if rej else float('nan'):.2f}")  # fmt: skip
        print("| точка | вызовов VLM | TP | FP | FN | F1 | true с TP | затяжек | время | FP в false | дублей | пропущены |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|")  # fmt: skip
        for point in parse_grid(a.grid):
            thr = float(point.get("vlm.thr", "0"))
            hi = float(point.get("vlm.hi", "inf"))
            rs = point.get("vlm.rescue", "none")
            rescue = None if rs == "none" else float(rs)
            calls = 0

            def decide(clip_id, e):
                nonlocal calls
                v = vs.get(event_key(clip_id, e))
                if e.passed:
                    best = max((c.appearance for c in e.cycles if np.isfinite(c.appearance)), default=-math.inf)
                    if v is None or best >= hi:
                        return True
                    calls += 1
                    return v["smoke"] >= thr
                if rescue is not None and v is not None and e.reject_reason.split(":")[0] in REJECTS:
                    calls += 1
                    return v["smoke"] >= rescue
                return False

            res, rc = evaluate(gt, ce, decide)
            t = res["total"]
            hit = {mm["clip_id"] for mm in res["overall"]["matches"] if mm["result"] == "TP"}
            trues = sorted(s for s in gt_true if s in rc and rc[s].folder == "true")
            missed = [s for s in trues if s not in hit]
            neg = res["per_folder"].get("false", {})
            cover = "—" if t["time_cover"] is None else f"{t['time_cover']:.0%}"
            name = " ".join(f"{k.split('.')[-1]}={v}" for k, v in point.items()) or "база"
            print(f"| {name} | {calls} | {t['tp']} | {t['fp']} | {t['fn']} | {t['f1']:.3f} | {len(trues) - len(missed)}/{len(trues)} "
                  f"| {t['marks_hit']}/{t['marks']} | {cover} | {neg.get('fp', 0)} | {t['dup']} | {', '.join(missed)} |", flush=True)  # fmt: skip
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=("extract", "vlm", "score", "sheets"))
    p.add_argument("--inputs")
    p.add_argument("--cache", help="папка кэша прогона run_rt.py (<out>/cache)")
    p.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    p.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    p.add_argument("--emb", help="банк эмбеддингов clip_lab.py extract")
    p.add_argument("--lab", action="append", default=[], metavar="KEY=VALUE", help="параметры clip_lab (без префикса lab.)")
    p.add_argument("--gt", default=str(ROOT / "rasmetka_gt.csv"))
    p.add_argument("--out", help="extract: папка кропов и events.json; vlm: файл оценок (.json)")
    p.add_argument("--top", type=int, default=3, help="extract: циклов события (лучшие по CLIP)")
    p.add_argument("--data", help="vlm, score: папка extract")
    p.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    p.add_argument("--quant", default="none", choices=("none", "nf4"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--scale", default=str(SCALES[0]), choices=[str(s) for s in SCALES])
    p.add_argument("--k", type=int, default=3, help="vlm: кропов на событие")
    p.add_argument("--px", type=int, default=CROP_PX, help="vlm: сторона кропа на входе")
    p.add_argument("--lean", action="store_true", help="vlm: эмбеддинги в RAM, lm_head — только строки букв")
    p.add_argument("--question", default="act", choices=tuple(QUESTIONS), help="vlm: вопрос к модели")
    p.add_argument("--dtype", default="auto", choices=("auto", "fp32", "bf16", "fp16"),
                   help="vlm: auto — fp16 на GPU, bf16 на CPU")
    p.add_argument("--threads", type=int, default=0, help="vlm: потоков torch (0 — по умолчанию)")
    p.add_argument("--limit", type=int, default=0, help="vlm: только первые N событий (замер скорости)")
    p.add_argument("--orders", type=int, default=2, help="vlm: 1 — прямой порядок вариантов, 2 — ещё обратный")
    p.add_argument("--vlm", action="append", default=[], help="score: файлы оценок vlm")
    p.add_argument("--grid", action="append", default=[], metavar="vlm.KEY=V1,V2")
    p.add_argument("--per-sheet", type=int, default=8, help="sheets: событий на лист")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    logging.getLogger("smokedet").setLevel(logging.WARNING)
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    return {"extract": extract, "vlm": vlm, "score": score, "sheets": sheets}[a.mode](a)


if __name__ == "__main__":
    sys.exit(main())
