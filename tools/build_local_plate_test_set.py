#!/usr/bin/env python3
"""
Build and evaluate a LOCAL, ground-truth-labeled plate test set from real
images already captured at this gate (gate_photos/), because the Roboflow
dataset's camera angle, lighting, and gate geometry may not match this
installation.

This tool has two modes:

  label     Interactively review vehicle images and record the correct
            plate transcription AND the vehicle's colour, body type and
            brand (or mark the image unusable / a screen-replay). Writes to
            datasets/local-gate-evaluation/ground_truth.csv.

            The operator is already opening every gate photo to transcribe
            the plate, so the three extra prompts cost almost nothing per
            image and build the plate test set and all three attribute
            datasets in a single pass. Attribute answers are optional: a
            blank is recorded as "not labelled" and skipped by every
            metric, which is strictly better than a guess.

  evaluate  Run the current ANPR pipeline against every labeled, usable,
            non-screen-replay image and report detection + OCR metrics —
            broken out by lighting/distance/tilt/blur/glare/plate-position/
            screen-replay, not just a single blended number.

  evaluate-attributes
            The same, for colour/type/brand: per-class precision and
            recall, macro-F1, the unknown/other rate, and brand top-1 vs
            top-3 — each broken out by the same lighting and distance
            buckets.

Ground-truth images are NOT used for training until reviewed, annotated,
and explicitly moved into a separate training set — this tool never writes
into datasets/nigerian-license-plate/, and evaluate mode never trains
anything.

Usage:
    python tools/build_local_plate_test_set.py label
    python tools/build_local_plate_test_set.py evaluate
    python tools/build_local_plate_test_set.py evaluate-attributes
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

# Attribute columns are APPENDED, never inserted: a ground_truth.csv
# written before this change still loads, its attribute cells simply come
# back empty and are treated as "not labelled".
CSV_FIELDS = [
    "image_path", "plate_ground_truth", "usable", "is_screen_replay",
    "lighting", "distance", "tilt", "blur", "glare", "plate_position",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "notes", "reviewed_at",
    "colour_ground_truth", "type_ground_truth", "brand_ground_truth",
    "vbox_x1", "vbox_y1", "vbox_x2", "vbox_y2",
]

ATTRIBUTE_FIELDS = {
    "colour": "colour_ground_truth",
    "type": "type_ground_truth",
    "brand": "brand_ground_truth",
}

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


def _attribute_label_spaces() -> dict:
    """
    The closed label spaces from the attributes package, so the labelling
    tool and the model can never drift apart. Imported lazily: labelling
    must work on a machine with no torch installed.
    """
    from attributes.models import BRANDS, COLOURS, TYPES
    return {"colour": COLOURS, "type": TYPES, "brand": BRANDS}


def _prompt_label_space(attribute: str, choices: list) -> str:
    """
    Prompt for one attribute from its closed list.

    Accepts a number, a full label, or a unique prefix. Blank means "not
    labelled" and is recorded as such — a guessed label is worse than a
    missing one, because it silently corrupts every metric computed from
    this file.
    """
    numbered = "  ".join(f"{i + 1}:{c}" for i, c in enumerate(choices))
    print(f"  {attribute} options: {numbered}")
    while True:
        raw = input(f"  {attribute} (number/name, blank = not labelled): ").strip().lower()
        if not raw:
            return ""
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return choices[int(raw) - 1]
        if raw in choices:
            return raw
        matches = [c for c in choices if c.startswith(raw)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            print(f"  Ambiguous — matches {matches}")
        else:
            print(f"  Not understood — choose a number 1-{len(choices)}, a label, "
                  "or press Enter to leave it unlabelled")


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

        # ── Vehicle attributes ────────────────────────────────────────────
        # Three prompts on an image the operator already has open. Leaving
        # one blank is a valid, useful answer.
        spaces = _attribute_label_spaces()
        print("  ── vehicle attributes (Enter to skip any) ──")
        for attribute, field_name in ATTRIBUTE_FIELDS.items():
            row[field_name] = _prompt_label_space(attribute, spaces[attribute])

        vbox_raw = input("  Vehicle bounding box as 'x1,y1,x2,y2' pixels "
                         "(optional, blank to skip): ").strip()
        if vbox_raw:
            try:
                x1, y1, x2, y2 = (int(v.strip()) for v in vbox_raw.split(","))
                row.update({"vbox_x1": x1, "vbox_y1": y1, "vbox_x2": x2, "vbox_y2": y2})
            except ValueError:
                print("  Could not parse vehicle bbox — skipping it for this image.")

        row["notes"] = input("  Notes (optional): ").strip()

        existing.append(row)
        save_ground_truth(existing)   # save after every image — never lose work to a crash

    print(f"\nSaved {len(candidates)} label(s) to {GROUND_TRUTH_CSV}")
    _print_label_coverage(existing)


def _print_label_coverage(rows: list):
    """How much of each dataset now exists — the honest answer to 'can I train yet?'."""
    usable = [r for r in rows if r.get("usable", "True") == "True"]
    print("\nLabel coverage across the whole ground-truth file "
          f"({len(usable)} usable row(s)):")
    print(f"  plate   {sum(1 for r in usable if r.get('plate_ground_truth')):4d}")
    for attribute, field_name in ATTRIBUTE_FIELDS.items():
        labelled = [r[field_name] for r in usable if r.get(field_name)]
        distinct = sorted(set(labelled))
        print(f"  {attribute:7s} {len(labelled):4d}   {len(distinct)} distinct class(es)"
              + (f": {', '.join(distinct[:8])}" if distinct else ""))


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


# ── ATTRIBUTE EVALUATION ──────────────────────────────────────────────────

def _per_class_metrics(pairs: list, label_space: list) -> dict:
    """
    Per-class precision/recall/F1 and a confusion matrix from
    [(truth, prediction), ...]. Computed directly so no scikit-learn is
    needed on the Pi.

    A prediction of None (the model declined, or nothing was produced) is
    NEVER counted as correct. It is tracked as an abstention: declining is
    better than being wrong, but it is not being right.
    """
    if not pairs:
        return {"n": 0}

    answered = [(t, p) for t, p in pairs if p is not None]
    correct = sum(1 for t, p in answered if t == p)
    classes = sorted({t for t, _ in pairs} | {p for _, p in answered})

    per_class = {}
    precisions, recalls, f1s = [], [], []
    for label in classes:
        tp = sum(1 for t, p in answered if t == label and p == label)
        fp = sum(1 for t, p in answered if t != label and p == label)
        fn = sum(1 for t, p in pairs if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        support = sum(1 for t, _ in pairs if t == label)
        per_class[label] = {"precision": round(precision, 4), "recall": round(recall, 4),
                            "f1": round(f1, 4), "support": support}
        if support:                       # macro-average over true classes only
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

    matrix = {}
    for truth, prediction in pairs:
        matrix.setdefault(truth, {})
        key = prediction if prediction is not None else "(abstained)"
        matrix[truth][key] = matrix[truth].get(key, 0) + 1

    def mean(values):
        return round(sum(values) / len(values), 4) if values else 0.0

    return {
        "n": len(pairs),
        "answered": len(answered),
        "abstained": len(pairs) - len(answered),
        "abstention_rate": round((len(pairs) - len(answered)) / len(pairs), 4),
        "accuracy_on_answered": round(correct / len(answered), 4) if answered else None,
        "accuracy_over_all": round(correct / len(pairs), 4),
        "macro_precision": mean(precisions),
        "macro_recall": mean(recalls),
        "macro_f1": mean(f1s),
        "per_class": per_class,
        "confusion_matrix": matrix,
    }


def _escape_rate(pairs: list, escape_label: str) -> dict:
    """
    How often the model predicted its escape class (`unknown` / `other`).

    A model that NEVER says `unknown` at night is broken regardless of its
    headline accuracy, so this is reported separately rather than being
    folded into accuracy.
    """
    predicted = sum(1 for _t, p in pairs if p == escape_label)
    truth = sum(1 for t, _p in pairs if t == escape_label)
    correct = sum(1 for t, p in pairs if t == escape_label and p == escape_label)
    return {
        "escape_label": escape_label,
        "predicted_count": predicted,
        "predicted_rate": round(predicted / len(pairs), 4) if pairs else None,
        "true_count": truth,
        "recall_on_escape_class": round(correct / truth, 4) if truth else None,
    }


def evaluate_attributes():
    """
    Run the attribute pipeline over every labelled, usable, non-replay row
    and report per-attribute metrics, broken out by the same lighting and
    distance buckets the plate evaluation uses.

    Reports nothing it did not measure: an attribute with no ground-truth
    labels is reported as such, not as 0% or 100%.
    """
    import cv2

    import attributes as attr
    import config as cfg
    from attributes.models import ESCAPE_LABEL

    rows = load_ground_truth()
    usable = [r for r in rows if r.get("usable", "True") == "True"
              and r.get("is_screen_replay", "False") != "True"]
    if not usable:
        print("No usable, non-screen-replay labelled rows. Run "
              "'python tools/build_local_plate_test_set.py label' first.",
              file=sys.stderr)
        sys.exit(1)

    labelled_counts = {a: sum(1 for r in usable if r.get(f)) 
                       for a, f in ATTRIBUTE_FIELDS.items()}
    if not any(labelled_counts.values()):
        print("No attribute labels in the ground-truth file — every "
              "colour/type/brand cell is blank.\n"
              "Attribute accuracy for this gate has NOT been established. Run "
              "'label' and answer the three attribute prompts.", file=sys.stderr)
        sys.exit(1)

    settings = attr.settings.load()
    pipeline = attr.build_pipeline(settings)
    status = pipeline.status()
    if not status["classifier"]["trained"]:
        print("WARNING: the attribute checkpoint is untrained (stub weights). "
              "Colour and brand will abstain on every image and the numbers "
              "below will reflect that, not model quality.\n", file=sys.stderr)

    from anpr import build_pipeline as build_plate_pipeline
    plate_config = dict(cfg.load())
    plate_config["plate_consensus_frames"] = 1     # one static image per row
    plate_pipeline = build_plate_pipeline(plate_config)

    per_row = []
    for row in usable:
        image = cv2.imread(row["image_path"])
        if image is None:
            continue

        # Prefer the operator's hand-drawn plate box: it isolates attribute
        # accuracy from plate-detector recall, so a localisation failure is
        # not silently charged to the colour head.
        plate_box = None
        try:
            plate_box = tuple(int(row[k]) for k in
                              ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"))
        except (KeyError, TypeError, ValueError):
            plate_result = plate_pipeline.process_frames([image])
            plate_box = plate_result.bounding_box

        t0 = time.perf_counter()
        result = pipeline.process_frames([image], plate_box=plate_box)
        elapsed = time.perf_counter() - t0

        entry = {
            "image_path": row["image_path"],
            "localised": result.status != "VEHICLE_NOT_LOCALISED",
            "latency_s": elapsed,
            "categories": {f: row.get(f, "") for f in CATEGORY_FIELDS},
        }
        for attribute, field_name in ATTRIBUTE_FIELDS.items():
            truth = (row.get(field_name) or "").strip().lower() or None
            value = getattr(result, attribute)
            entry[attribute] = {
                "truth": truth,
                "pred": value.value if value.is_usable else None,
                "status": value.status,
                "confidence": value.confidence,
                "top_k": [s.label for s in value.top_k],
            }
        per_row.append(entry)

    def summarise(subset: list) -> dict:
        out = {"num_images": len(subset),
               "localisation_rate": (round(sum(1 for r in subset if r["localised"]) /
                                           len(subset), 4) if subset else None)}
        for attribute in ATTRIBUTE_FIELDS:
            pairs = [(r[attribute]["truth"], r[attribute]["pred"])
                     for r in subset if r[attribute]["truth"]]
            if not pairs:
                out[attribute] = {"n": 0,
                                  "note": "no ground-truth labels for this attribute"}
                continue
            metrics = _per_class_metrics(pairs, [])
            metrics["escape_class"] = _escape_rate(pairs, ESCAPE_LABEL[attribute])
            if attribute == "brand":
                # Brand is reported top-1 AND top-3, never top-1 alone.
                considered = [r for r in subset if r["brand"]["truth"]]
                hits = sum(1 for r in considered
                           if r["brand"]["truth"] in (r["brand"]["top_k"] or []))
                metrics["top3_accuracy"] = (round(hits / len(considered), 4)
                                            if considered else None)
                metrics["top1_accuracy"] = metrics["accuracy_over_all"]
            out[attribute] = metrics
        latencies = [r["latency_s"] for r in subset]
        out["latency_ms"] = {
            "mean": round(statistics.mean(latencies) * 1000, 2) if latencies else None,
            "median": round(statistics.median(latencies) * 1000, 2) if latencies else None,
        }
        return out

    by_category = {}
    for field in CATEGORY_FIELDS:
        by_category[field] = {}
        values = sorted({r["categories"].get(field, "") for r in per_row
                         if r["categories"].get(field, "")})
        for value in values:
            subset = [r for r in per_row if r["categories"].get(field) == value]
            by_category[field][value] = summarise(subset)

    report = {
        "num_labelled_rows": len(rows),
        "num_evaluated": len(per_row),
        "labelled_per_attribute": labelled_counts,
        "model_status": status,
        "overall": summarise(per_row),
        "by_category": by_category,
        "note": ("Measured on THIS gate's own locally labelled images. Not a "
                 "public-benchmark number and not transferable to another "
                 "camera, mounting angle or lighting regime. Screen-replay "
                 "rows are excluded."),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = REPORTS_DIR / "local_attribute_evaluation.json"
    path.write_text(json.dumps(report, indent=2))
    _write_attribute_summary_md(report)

    for attribute in ATTRIBUTE_FIELDS:
        metrics = report["overall"][attribute]
        if not metrics.get("n"):
            print(f"{attribute:8s} no ground-truth labels")
            continue
        line = (f"{attribute:8s} n={metrics['n']:4d} "
                f"acc(answered)={metrics['accuracy_on_answered']} "
                f"macro-F1={metrics['macro_f1']} "
                f"abstained={metrics['abstention_rate']:.0%}")
        if attribute == "brand":
            line += f" top3={metrics.get('top3_accuracy')}"
        print(line)
    print(f"\nFull report: {path}")
    print(f"Summary: {REPORTS_DIR / 'local_attribute_evaluation_summary.md'}")


def _write_attribute_summary_md(report: dict):
    lines = ["# Local Vehicle Attribute Evaluation", "",
             f"Labelled rows: {report['num_labelled_rows']} — "
             f"evaluated: {report['num_evaluated']}", "",
             f"Labels available per attribute: {report['labelled_per_attribute']}", "",
             "## Overall", "",
             "| Attribute | n | Answered | Abstained | Acc (answered) | Acc (all) | "
             "Macro-F1 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for attribute in ("colour", "type", "brand"):
        m = report["overall"].get(attribute, {})
        if not m.get("n"):
            lines.append(f"| {attribute} | 0 | - | - | - | - | - |")
            continue
        lines.append(f"| {attribute} | {m['n']} | {m['answered']} | "
                     f"{m['abstained']} ({m['abstention_rate']:.0%}) | "
                     f"{m['accuracy_on_answered']} | {m['accuracy_over_all']} | "
                     f"{m['macro_f1']} |")

    brand = report["overall"].get("brand", {})
    if brand.get("n"):
        lines += ["", f"Brand top-1 {brand.get('top1_accuracy')} · "
                      f"top-3 {brand.get('top3_accuracy')}"]

    for attribute in ("colour", "type", "brand"):
        m = report["overall"].get(attribute, {})
        if not m.get("n"):
            continue
        lines += ["", f"### {attribute} — per class", "",
                  "| Class | Precision | Recall | F1 | Support |",
                  "| --- | --- | --- | --- | --- |"]
        for label, stats in sorted(m["per_class"].items()):
            lines.append(f"| {label} | {stats['precision']} | {stats['recall']} | "
                         f"{stats['f1']} | {stats['support']} |")
        escape = m.get("escape_class", {})
        lines += ["", f"Escape class `{escape.get('escape_label')}`: predicted "
                      f"{escape.get('predicted_count')} time(s) "
                      f"({escape.get('predicted_rate')}), recall on true "
                      f"{escape.get('escape_label')} = "
                      f"{escape.get('recall_on_escape_class')}"]

    lines += ["", "## By category", ""]
    for field, buckets in report["by_category"].items():
        if not buckets:
            continue
        lines += [f"### {field}", "",
                  "| Bucket | n | Localised | colour acc | type acc | brand top-1 | "
                  "brand top-3 |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for value, metrics in buckets.items():
            def acc(name):
                m = metrics.get(name, {})
                return m.get("accuracy_on_answered") if m.get("n") else "-"
            brand_metrics = metrics.get("brand", {})
            lines.append(
                f"| {value} | {metrics['num_images']} | {metrics['localisation_rate']} | "
                f"{acc('colour')} | {acc('type')} | "
                f"{brand_metrics.get('top1_accuracy', '-') if brand_metrics.get('n') else '-'} | "
                f"{brand_metrics.get('top3_accuracy', '-') if brand_metrics.get('n') else '-'} |")
        lines.append("")

    lines += ["", "## Scope", "", report["note"], ""]
    (REPORTS_DIR / "local_attribute_evaluation_summary.md").write_text("\n".join(lines) + "\n")


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
    parser.add_argument("mode", choices=["label", "evaluate", "evaluate-attributes",
                                        "coverage"])
    parser.add_argument("--images-dir", default=str(REPO_ROOT / "gate_photos"))
    parser.add_argument("--limit", type=int, default=None, help="Max images to label in this session")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    if args.mode == "label":
        label_images(images_dir, args.limit)
    elif args.mode == "coverage":
        _print_label_coverage(load_ground_truth())
    elif args.mode == "evaluate-attributes":
        evaluate_attributes()
    else:
        evaluate(images_dir)


if __name__ == "__main__":
    main()
