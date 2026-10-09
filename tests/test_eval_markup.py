"""tools/eval_markup.py: эпизоды из отметок, «продолжают курить», связь прогона с разметкой по имени видео."""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import eval_markup as em  # noqa: E402

GT_COLS = ["video", "folder", "clip_id", "label", "up_sec", "puff_sec", "down_sec"]


def write_csv(path: Path, cols, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)


def test_merge_marks_gap():
    eps = em.merge_marks([(30, 31), (5, 6), (10, 12), (47, 48)], gap=15)
    assert eps == [(5, 12, 2), (30, 31, 1), (47, 48, 1)]  # зазор 16 с — отдельные эпизоды
    assert em.merge_marks([(0, 1), (16, 17)], gap=15) == [(0, 17, 2)]  # зазор ровно 15 с объединяет


def test_mark_interval_partial():
    assert em.mark_interval(12.0, 13.0, None) == (12.0, 13.0)
    assert em.mark_interval(None, 16.0, None) == (16.0, 16.0)
    assert em.mark_interval(None, None, None) is None


def test_load_gt_continue_and_labels(tmp_path):
    gt = tmp_path / "gt.csv"
    write_csv(gt, GT_COLS, [
        ["3", "true", "x", "SMOKE", 85, 85, 85],
        ["3", "true", "x", "SMOKE", "", "", ""],  # «продолжают курить»
        ["3", "true", "x", "SMOKE", 97, 97, 98],
        ["3", "true", "x", "SMOKE", 131, 131, 131],
        ["3", "true", "x", "SMOKE", "", "", ""],  # в конце: до конца клипа
        ["d", "false", "y", "NOSMOKE", 13.0, "", 16.0],
        ["d", "false", "y", "IGNORE", 20, "", 22],
    ])  # fmt: skip
    clips = em.load_gt(str(gt))
    assert clips["3"].episodes == [(85, 98, 2), (131, 131, 1)]
    assert (85, 97, "continue") in clips["3"].ignores
    assert (131, float("inf"), "continue") in clips["3"].ignores
    assert clips["d"].nosmoke == [(13.0, 16.0)]
    assert clips["d"].ignores == [(20, 22, "IGNORE")]


def make_run(tmp_path: Path, clips, preds, debug=None) -> Path:
    run = tmp_path / "run"
    run.mkdir()
    write_csv(run / "processing_log.csv", ["camera_id", "clip_id", "video_path", "duration_sec"],
              [["c", cid, path, dur] for cid, path, dur in clips])  # fmt: skip
    write_csv(run / "predictions.csv", ["camera_id", "clip_id", "event_id", "start_sec", "end_sec", "confidence"],
              [["c", cid, f"{cid}_{i}", s, e, 0.9] for i, (cid, s, e) in enumerate(preds)])  # fmt: skip
    for cid, data in (debug or {}).items():
        (run / "debug").mkdir(exist_ok=True)
        (run / "debug" / f"{cid}.json").write_text(json.dumps(data), encoding="utf-8")
    return run


def test_evaluate_run(tmp_path):
    gt = tmp_path / "gt.csv"
    write_csv(gt, GT_COLS, [
        ["sm_1", "true", "true_11", "SMOKE", 90, 91, 93],
        ["sm_1", "true", "true_11", "SMOKE", 106, 107, 108],
        ["sm_1", "true", "true_11", "SMOKE", "", "", ""],
        ["sm_1", "true", "true_11", "SMOKE", 341, 341, 341],
        ["1_drink", "false", "false_1", "NOSMOKE", 15, "", 16],
        ["распитие", "false", "false_12", "IGNORE", 10, 12, 14],
    ])  # fmt: skip
    dbg = {"cycles": [{"start": 340.0, "end": 342.0, "valid": False, "reject_reason": "object:cup", "appearance": 3.5}],
           "events": []}  # fmt: skip
    run = make_run(tmp_path, [
        ("a", "Z:/v/true/sm_1.mp4", 600.0),
        ("b", "Z:/v/false/1_drink.mp4", 55.0),
        ("c", "Z:/v/false/IMG_2162.MOV", 126.0),  # false без разметки: всё — FP
        ("d", "Z:/v/true/new.mp4", 30.0),  # true без разметки: исключается
        ("e", "Z:/v/false/распитие.wmv", 66.0),
    ], [
        ("a", 104.0, 112.0),  # TP: эпизод 90–108
        ("a", 200.0, 210.0),  # в «продолжают курить» 108–341 — не FP
        ("b", 14.0, 18.0),  # FP на питье
        ("c", 50.0, 55.0),  # FP
        ("d", 1.0, 5.0),  # исключён
        ("e", 10.5, 13.5),  # IGNORE
    ], debug={"a": dbg})  # fmt: skip
    res = em.evaluate_run(em.load_gt(str(gt)), em.load_run(str(run)), str(run))
    o = res["overall"]
    assert (o["tp"], o["fp"], o["fn"], o["ignored_preds"]) == (1, 2, 1, 2)
    assert res["excluded"] == ["new"]
    assert res["per_folder"]["false"]["fp"] == 2
    assert abs(res["per_folder"]["false"]["fp_per_hour"] - 2 / ((55 + 126 + 66) / 3600)) < 1e-9
    fp = {m["clip"]: m for m in o["matches"] if m["result"] == "FP"}
    assert fp["b"]["note"].startswith("питьё: 15–16")
    assert fp["c"]["note"] == "без NOSMOKE-отметки рядом"
    fn = [m for m in o["matches"] if m["result"] == "FN"][0]
    assert fn["gt_start"] == 341 and fn["gt_marks"] == 1
    assert "object×1" in fn["note"] and "CLIP max 3.5" in fn["note"]
    assert res["single_mark_episodes"] == [("sm_1", 341, 341)]
    em.write_outputs(tmp_path / "out", "run", res)
    assert "FN" in (tmp_path / "out" / "summary.md").read_text(encoding="utf-8")


def test_duplicate_prediction(tmp_path):
    gt = tmp_path / "gt.csv"
    write_csv(gt, GT_COLS, [["sm_3", "true", "x", "SMOKE", 48, 49, 51], ["sm_3", "true", "x", "SMOKE", 59, 60, 62]])
    run = make_run(tmp_path, [("a", "Z:/v/true/sm_3.mp4", 117.0)], [("a", 47.0, 52.0), ("a", 58.0, 63.0)])
    res = em.evaluate_run(em.load_gt(str(gt)), em.load_run(str(run)))
    assert (res["total"]["tp"], res["total"]["fp"], res["total"]["dup"]) == (1, 1, 1)
    assert res["total"]["f1_nodup"] == 1.0


def test_coverage_counts_point_marks():
    """Отметка-точка (подъём = затяжка = опускание) внутри предсказания — попадание."""
    g = em.ClipGT("x", "true", episodes=[(5.0, 20.0, 3)], marks=[(5.0, 5.0), (12.0, 12.0), (30.0, 30.0)])
    hit, inside, total = em.coverage(g, [{"start_sec": "4", "end_sec": "13"}])
    assert hit == 2
    assert (inside, total) == (8.0, 15.0)
