#!/usr/bin/env python3
"""
Build and evaluate a LOCAL, ground-truth-labeled plate test set from real
images already captured at this gate (gate_photos/), because the Roboflow
dataset's camera angle, lighting, and gate geometry may not match this
installation.

This tool has two modes:

  label     Interactively review vehicle images and record the correct
            plate transcription (or mark the image unusable / a
            screen-replay). Writes to
            datasets/local-gate-evaluation/ground_truth.csv.

  evaluate  Run the current ANPR pipeline against every labeled, usable,
            non-screen-replay image and report detection + OCR metrics —
            broken out by lighting/distance/tilt/blur/glare/plate-position/
            screen-replay, not just a single blended number.

Ground-truth images are NOT used for training until reviewed, annotated,
and explicitly moved into a separate training set — this tool never writes
into datasets/nigerian-license-plate/, and evaluate mode never trains
anything.

Usage:
    python tools/build_local_plate_test_set.py label
    python tools/build_local_plate_test_set.py evaluate
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

GROUND_TRUTH_CSV = REPO_ROOT / "datasets" / "local-gate-evaluation" / "ground_truth.csv"
REPORTS_DIR = REPO_ROOT / "reports"

CSV_FIELDS = [
    "image_path", "plate_ground_truth", "usable", "is_screen_replay",
    "lighting", "distance", "tilt", "blur", "glare", "plate_position",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "notes", "reviewed_at",
]

CATEGORY_FIELDS = ["lighting", "distance", "tilt", "blur", "glare", "plate_position", "is_screen_replay"]

CATEGORY_CHOICES = {
    "lighting": ["daylight", "low_light"],
    "distance": ["near", "distant"],
    "tilt": ["upright", "tilted"],
    "blur": ["sharp", "blurry"],
    "glare": ["none", "glare"],
    "plate_position": ["front", "rear", "unknown"],
}


def load_ground_truth() -> list:
    if not GROUND_TRUTH_CSV.exists():
        return []
    with open(GROUND_TRUTH_CSV, newline="") as f:
        return list(csv.DictReader(f))


def save_ground_truth(rows: list):
    GROUND_TRUTH_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(GROUND_TRUTH_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def _prompt_choice(label: str, choices: list, default: str) -> str:
    choice_str = "/".join(f"[{c[0]}]{c[1:]}" if i == 0 else c for i, c in enumerate(choices))
    while True:
        raw = input(f"  {label} ({choice_str}, default={default}): ").strip().lower()
        if not raw:
            return default
        for c in choices:
            if raw == c or raw == c[0]:
                return c
        print(f"  Not understood — choose one of {choices}")


def label_images(images_dir: Path, limit: int = None):
    from anpr.postprocessing import normalize_plate_for_storage

    existing = load_ground_truth()
    already_labeled = {row["image_path"] for row in existing}

    candidates = sorted(
        p for p in glob.glob(str(images_dir / "vehicle_*.jpg"))
        if p not in already_labeled
    )
    if limit:
        candidates = candidates[:limit]

    if not candidates:
        print("No new (unlabeled) vehicle images found.")
        return

    print(f"{len(candidates)} unlabeled image(s) to review. "
          f"Open each file in an image viewer alongside this terminal — this "
          f"tool does not itself display images over SSH/headless sessions.\n")

    for path in candidates:
        print("=" * 70)
        print(f"Image: {path}")
        usable_raw = input("  Usable for evaluation? [Y/n]: ").strip().lower()
        usable = usable_raw not in ("n", "no")

        row = {"image_path": path, "usable": str(usable), "reviewed_at": time.strftime("%Y-%m-%d %H:%M:%S")}

        if not usable:
            reason = input("  Why unusable (free text, e.g. 'no plate visible'): ").strip()
            row.update({"notes": reason, "is_screen_replay": "False",
                        "plate_ground_truth": ""})
            existing.append(row)
            save_ground_truth(existing)
            continue

        is_screen = input("  Is this a computer-screen/photo-of-a-photo (replay) image? [y/N]: ").strip().lower()
        row["is_screen_replay"] = str(is_screen in ("y", "yes"))

        plate_raw = input("  Correct plate transcription (blank if no plate visible): ").strip()
        row["plate_ground_truth"] = normalize_plate_for_storage(plate_raw) if plate_raw else ""

        for field in ("lighting", "distance", "tilt", "blur", "glare", "plate_position"):
            row[field] = _prompt_choice(field, CATEGORY_CHOICES[field], CATEGORY_CHOICES[field][0])

        bbox_raw = input("  Plate bounding box as 'x1,y1,x2,y2' pixels (optional, blank to skip): ").strip()
        if bbox_raw:
            try:
                x1, y1, x2, y2 = (int(v.strip()) for v in bbox_raw.split(","))
                row.update({"bbox_x1": x1, "bbox_y1": y1, "bbox_x2": x2, "bbox_y2": y2})
            except ValueError:
                print("  Could not parse bbox — skipping bbox for this image.")

        row["notes"] = input("  Notes (optional): ").strip()

        existing.append(row)
        save_ground_truth(existing)   # save after every image — never lose work to a crash

    print(f"\nSaved {len(candidates)} label(s) to {GROUND_TRUTH_CSV}")


# ── EVALUATION ────────────────────────────────────────────────────────────

def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def _character_accuracy(pred: str, truth: str) -> float:
    if not truth:
        return 1.0 if not pred else 0.0
    dist = _levenshtein(pred, truth)
    return max(0.0, 1.0 - dist / len(truth))


def evaluate(images_dir: Path):
    import cv2
    import config as cfg
    from anpr import build_pipeline

    rows = load_ground_truth()
    usable_rows = [r for r in rows if r.get("usable", "True") == "True"
                   and r.get("is_screen_replay", "False") != "True"]
    replay_rows = [r for r in rows if r.get("is_screen_replay", "False") == "True"]

    if not usable_rows:
        print("No usable, non-screen-replay labeled rows to evaluate. Run "
              "'python tools/build_local_plate_test_set.py label' first.", file=sys.stderr)
        sys.exit(1)

    plate_config = dict(cfg.load())
    plate_config["plate_consensus_frames"] = 1   # single static image per row — see docstring
    pipeline = build_pipeline(plate_config)
    if not pipeline.detector.is_ready():
        print("WARNING: plate model is not available — detection metrics will show 0 recall.",
              file=sys.stderr)

    per_row = []
    for row in usable_rows:
        img = cv2.imread(row["image_path"])
        if img is None:
            continue
        t0 = time.perf_counter()
        result = pipeline.process_frames([img])
        elapsed = time.perf_counter() - t0

        truth = row.get("plate_ground_truth", "")
        pred = result.plate_number or ""
        per_row.append({
            "image_path": row["image_path"],
            "truth": truth,
            "pred": pred,
            "status": result.status,
            "detected_box": result.bounding_box is not None,
            "exact_match": bool(truth) and pred == truth,
            "character_accuracy": _character_accuracy(pred, truth) if truth else None,
            "no_read": bool(truth) and not pred,
            "false_positive": (not truth) and result.bounding_box is not None,
            "recognition_time_s": elapsed,
            "categories": {f: row.get(f, "") for f in CATEGORY_FIELDS},
        })

    overall = _summarize(per_row)
    by_category = {}
    for field in CATEGORY_FIELDS:
        by_category[field] = {}
        values = sorted({r["categories"].get(field, "") for r in per_row if r["categories"].get(field, "")})
        for v in values:
            subset = [r for r in per_row if r["categories"].get(field) == v]
            by_category[field][v] = _summarize(subset)

    report = {
        "num_labeled_rows": len(rows),
        "num_usable_evaluated": len(per_row),
        "num_screen_replay_rows_excluded_from_main_metrics": len(replay_rows),
        "overall": overall,
        "by_category": by_category,
        "note": "Detection and OCR accuracy here reflect THIS gate's own camera/lighting/angle "
                "on a locally labeled set — they are separate from, and not a substitute for, "
                "reports/plate_detector_metrics.json (Roboflow validation-set detector mAP).",
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "local_evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    _write_summary_md(report)

    print(json.dumps(overall, indent=2))
    print(f"\nFull report: {report_path}")
    print(f"Summary: {REPORTS_DIR / 'local_evaluation_summary.md'}")

    if replay_rows:
        replay_metrics = _evaluate_rows_only(pipeline, replay_rows)
        print("\nScreen/replay images (evaluated separately, not blended into main metrics):")
        print(json.dumps(replay_metrics, indent=2))


def _evaluate_rows_only(pipeline, rows):
    import cv2
    results = []
    for row in rows:
        img = cv2.imread(row["image_path"])
        if img is None:
            continue
        result = pipeline.process_frames([img])
        truth = row.get("plate_ground_truth", "")
        pred = result.plate_number or ""
        results.append({
            "truth": truth, "pred": pred,
            "detected_box": result.bounding_box is not None,
            "exact_match": bool(truth) and pred == truth,
            "no_read": bool(truth) and not pred,
            "false_positive": (not truth) and result.bounding_box is not None,
            "character_accuracy": _character_accuracy(pred, truth) if truth else None,
            "recognition_time_s": 0.0,
        })
    return _summarize(results)


def _summarize(per_row: list) -> dict:
    if not per_row:
        return {"num_images": 0}

    with_truth = [r for r in per_row if r["truth"]]
    without_truth = [r for r in per_row if not r["truth"]]

    recall = (
        sum(1 for r in with_truth if r["detected_box"]) / len(with_truth)
        if with_truth else None
    )
    precision = (
        sum(1 for r in per_row if r["detected_box"] and not r.get("false_positive")) /
        sum(1 for r in per_row if r["detected_box"])
        if any(r["detected_box"] for r in per_row) else None
    )
    char_accs = [r["character_accuracy"] for r in with_truth if r["character_accuracy"] is not None]
    times = [r["recognition_time_s"] for r in per_row if r.get("recognition_time_s") is not None]

    return {
        "num_images": len(per_row),
        "num_with_ground_truth_plate": len(with_truth),
        "num_without_ground_truth_plate": len(without_truth),
        "detection_recall": round(recall, 4) if recall is not None else None,
        "detection_precision": round(precision, 4) if precision is not None else None,
        "character_accuracy_avg": round(statistics.mean(char_accs), 4) if char_accs else None,
        "exact_match_accuracy": (
            round(sum(1 for r in with_truth if r["exact_match"]) / len(with_truth), 4)
            if with_truth else None
        ),
        "no_read_rate": (
            round(sum(1 for r in with_truth if r["no_read"]) / len(with_truth), 4)
            if with_truth else None
        ),
        "false_positive_rate": (
            round(sum(1 for r in without_truth if r["false_positive"]) / len(without_truth), 4)
            if without_truth else None
        ),
        "avg_recognition_time_s": round(statistics.mean(times), 4) if times else None,
    }


def _write_summary_md(report: dict):
    lines = ["# Local Gate Evaluation Summary", "",
              f"Labeled rows: {report['num_labeled_rows']} — "
              f"evaluated (usable, non-replay): {report['num_usable_evaluated']}", "",
              "## Overall", "", "```json",
              json.dumps(report["overall"], indent=2), "```", "",
              "## By category", ""]
    for field, buckets in report["by_category"].items():
        if not buckets:
            continue
        lines.append(f"### {field}")
        lines.append("")
        for value, metrics in buckets.items():
            lines.append(f"**{value}** (n={metrics['num_images']})")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(metrics, indent=2))
            lines.append("```")
            lines.append("")
    (REPORTS_DIR / "local_evaluation_summary.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["label", "evaluate"])
    parser.add_argument("--images-dir", default=str(REPO_ROOT / "gate_photos"))
    parser.add_argument("--limit", type=int, default=None, help="Max images to label in this session")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    if args.mode == "label":
        label_images(images_dir, args.limit)
    else:
        evaluate(images_dir)


if __name__ == "__main__":
    main()
