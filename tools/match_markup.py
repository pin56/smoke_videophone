"""Сопоставление разметки rasmetka.md с видео и предсказаниями прогонов.

Запуск из корня проекта: читает markup/*.md, пишет markup/rasmetka_gt.csv и сверяет его с runs/true_now и runs/false.
Импорт (tools/eval_markup.py) берёт отсюда имена файлов (canonical_stem), папку видео (FILES) и разбор времени (sec).

Студенты (smoke1..17, видео — жёсткие ссылки videos/students -> videos/true|false) — в rasmetka_students.md
того же формата: курят только события, подтверждённые пользователем 2026-10-09 (smoke6 т1 и т24, smoke7 т1,
smoke15 т6), отметки — циклы этих треков из data/labeling/students; остальные 14 клипов — false (имитация).

rasmetka_sheets.md — курение, которого нет в rasmetka.md: события потокового прогона l168 на старых true-клипах,
подтверждённые пользователем по листам runs/vlm/true_sheets (2026-10-09), отметки — их циклы.
"""
import csv, re, collections
import unicodedata

ALIAS = {"25_11_6_8_35_37": "2025_11_06_08_35_37", "распитие": "распитие", "распитие (2)": "распитие (2)"}
# имена из разметки, которые на деле части 600-с файлов (время в разметке — внутри файла)
PART = {"2025_10_22_12_38_58": "2025_10_22_12_38_57", "2025_10_22_12_38_59": "2025_10_22_12_38_57"}
for i in range(27, 36):
    PART[f"2025_12_08_13_02_{i}"] = "2025_12_08_13_02_26"
_FILES = {  # stem -> (папка, clip_id в runs/true_now|false)
    **{s: ("true", c) for s, c in {"1": "true_0", "2": "true_1", "2025_10_22_12_38_57": "true_2",
       "2025_12_08_13_02_26": "true_3", "3": "true_4", "4": "true_5", "5": "true_6", "_88": "true_7",
       "sm_1": "true_11", "sm_2": "true_12", "sm_3": "true_13", "sm_4": "true_14", "sm_5": "true_15",
       "sm_6": "true_16", "sm_7": "true_17", "sm_9": "true_18", "sm_y_2": "true_19",
       "smoking_Jar (1)": "true_20", "smoking_Jar (2)": "true_21"}.items()},
    **{s: ("false", c) for s, c in {"1_drink": "false_1", "2025_11_06_08_13_58": "false_2",
       "2025_11_06_08_15_57": "false_3", "2025_11_06_08_19_11": "false_4", "2025_11_06_08_23_06": "false_5",
       "2025_11_06_08_33_31": "false_6", "2025_11_06_08_35_37": "false_7", "drinking_Jar": "false_8",
       "smoking_noyabrsk_2": "false_9", "smoking_noyabrsk_3": "false_10", "распитие (2)": "false_11",
       "распитие": "false_12"}.items()},
    **{f"smoke{i}": ("true" if i in (6, 7, 15) else "false", f"smoke{i}") for i in range(1, 18)},
}
MARKUP = ("markup/rasmetka.md", "markup/rasmetka_students.md", "markup/rasmetka_sheets.md")

FILES = _FILES
LOW = {k.lower(): k for k in _FILES}


def canonical_stem(name):
    """Имя видео из разметки или имя файла без расширения -> stem файла (ALIAS, PART, регистр как у файла)."""
    name = unicodedata.normalize("NFC", name.strip())
    stem = ALIAS.get(name, name)
    stem = PART.get(stem, stem)
    return LOW.get(stem.lower(), stem)


def sec(s):
    s = s.strip().replace("–", "-")
    if not s or s == "-":
        return None
    s = s.split("-")[0] if re.fullmatch(r"\d+-\d+", s) else s  # диапазон секунд → начало
    if re.fullmatch(r"\d+:\d+-\d+:\d+", s):
        s = s.split("-")[0]
    m = re.fullmatch(r"(\d+):(\d+)", s)
    if m:
        return int(m[1]) * 60 + int(m[2])
    return float(s) if re.fullmatch(r"\d+(\.\d+)?", s) else None

def main():
    rows, prev = [], None
    lines = [ln for path in MARKUP for ln in open(path, encoding="utf-8").read().splitlines()[1:]]
    for ln in lines:
        p = [x.strip().strip('"') for x in ln.split("\t")]
        p += [""] * (5 - len(p))
        if p[0] == "" and p[1] == "":  # пустое имя и метка: смотрим сдвинутые колонки
            pass
        name, lab = p[0], p[1]
        if not name and prev:
            name = prev  # строки 212–214 без имени: продолжение 1_drink
            lab = lab or "не курит"
        if not name:
            continue
        # строки 132/140 с кавычками: поля сдвинуты на пустую колонку
        if lab == "" and p[2] and "курит" in p[2]:
            pass
        prev = name
        rows.append((name, lab.lower(), p[2:5], ln))

    out = []
    for name, lab, t, ln in rows:
        stem = canonical_stem(name)
        if "курит" in lab and "не" not in lab:
            l = "SMOKE"
        elif "не курит" in lab:
            l = "NOSMOKE"
        elif "игнор" in lab:
            l = "IGNORE"
        else:
            l = "SMOKE" if "курит" in name else "NOSMOKE"
        out.append(dict(video=stem, label=l, up=sec(t[0]), puff=sec(t[1]), down=sec(t[2]), raw=ln))

    bad = [o for o in out if o["video"] not in FILES]
    with open("markup/rasmetka_gt.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video", "folder", "clip_id", "label", "up_sec", "puff_sec", "down_sec"])
        for o in out:
            fo, cid = FILES.get(o["video"], ("?", "?"))
            w.writerow([o["video"], fo, cid, o["label"], o["up"], o["puff"], o["down"]])
    print("строк", len(out), "не сопоставлено", len(bad))
    for o in bad: print("  ?", repr(o["raw"][:70]))

    def preds(path):
        d = collections.defaultdict(list)
        for r in csv.DictReader(open(path, encoding="utf-8")):
            d[r["clip_id"]].append((float(r["start_sec"]), float(r["end_sec"])))
        return d
    P = {**preds("runs/true_now/predictions.csv"), **preds("runs/false/predictions.csv")}
    dur = {}
    for r in ("true_now", "false"):
        for x in csv.DictReader(open(f"runs/{r}/processing_log.csv", encoding="utf-8")):
            dur[x["clip_id"]] = float(x["duration_sec"])

    print("\nклип | SMOKE-отметок | NOSMOKE | IGNORE | событий | отметки SMOKE внутри событий | NOSMOKE внутри событий | время > длины")
    by = collections.defaultdict(list)
    for o in out: by[o["video"]].append(o)
    for v, os_ in by.items():
        fo, cid = FILES.get(v, ("?", "?"))
        ev = P.get(cid, [])
        def inside(o):
            t = o["puff"] if o["puff"] is not None else o["up"]
            return t is not None and any(s - 3 <= t <= e + 3 for s, e in ev)
        sm = [o for o in os_ if o["label"] == "SMOKE" and o["up"] is not None]; ns = [o for o in os_ if o["label"] == "NOSMOKE"]
        ig = [o for o in os_ if o["label"] == "IGNORE"]
        over = [o for o in os_ if any(x is not None and x > dur.get(cid, 1e9) for x in (o["up"], o["puff"], o["down"]))]
        print(f"{fo}/{v} [{cid}] | {len(sm)} | {len(ns)} | {len(ig)} | {len(ev)} | {sum(map(inside, sm))}/{len(sm)} | {sum(map(inside, ns))}/{len(ns)} | {len(over)}")


if __name__ == "__main__":
    main()
