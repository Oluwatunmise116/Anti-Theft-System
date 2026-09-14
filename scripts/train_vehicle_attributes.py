#!/usr/bin/env python3
"""
Train the multi-head vehicle attribute model (colour / type / brand).

Follows scripts/train_plate_detector.py's semantics: runs are written under
runs/vehicle_attributes/<name>/, `--compare` summarises finished runs
without training, and `--promote` copies a run's weights to the production
path only after the refusal checks below pass.

    python scripts/train_vehicle_attributes.py --check-data
    python scripts/train_vehicle_attributes.py --run-name mnv3_v1 --epochs 30
    python scripts/train_vehicle_attributes.py --compare mnv3_v1 mnv3_v2
    python scripts/train_vehicle_attributes.py --run-name mnv3_v1 --promote

ARCHITECTURE
------------
One MobileNetV3-Large backbone, three linear heads (see
attributes/classifiers.py — the network is built by the SAME function
inference uses, so a checkpoint can never be trained against one
architecture and loaded into another).

PER-HEAD LOSS MASKING (primary strategy)
----------------------------------------
Each batch is drawn from ONE source dataset, and only the heads that
dataset labels contribute gradient. Batches alternate across sources. The
backbone sees every image; each head sees only images that label it. This
is what lets a large colour corpus and a small local type/brand corpus
train one shared representation.

    --strategy frozen   is the documented fallback: train the backbone on
                        colour alone (the largest, densest source), freeze
                        it, then train type and brand on frozen features.
                        Slightly worse, far easier to debug, and heads can
                        be promoted independently. Use it only if masked
                        training proves unstable, and say so in the report.

DATA SOURCES
------------
  colour  UFPR-VCR (10,039 images; frontal and rear views, occlusions,
          varied lighting, and — the reason it is preferred over VCoR
          here — real nighttime scenes). Not redistributable: request it
          from the authors and point --colour-data at the extracted
          directory.
  type    locally labelled gate captures.
  brand   locally labelled gate captures, cropped to the detected badge.

Stanford Cars is deliberately not used: it stops around 2012 and is
US-market, so it does not describe the Nigerian vehicle mix at this gate.

REFUSALS
--------
This script will not promote a model that is at chance, and will not train
a head that has too few examples or too few distinct classes to mean
anything. Both refusals print what is missing.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from attributes.models import BRANDS, COLOURS, TYPES              # noqa: E402

RUNS_DIR = REPO_ROOT / "runs" / "vehicle_attributes"
PRODUCTION_MODEL_PATH = REPO_ROOT / "models" / "vehicle_attributes_mnv3.pt"
LOCAL_GROUND_TRUTH = REPO_ROOT / "datasets" / "local-gate-evaluation" / "ground_truth.csv"
REPORTS_DIR = REPO_ROOT / "reports"

HEADS = ("colour", "type", "brand")
LABEL_SPACES = {"colour": COLOURS, "type": TYPES, "brand": BRANDS}

#: A head with fewer than this many examples, or fewer than this many
#: distinct classes, cannot produce a number anyone should act on.
MIN_EXAMPLES_PER_HEAD = 200
MIN_CLASSES_PER_HEAD = 3
MIN_EXAMPLES_PER_CLASS = 20

#: Promotion gate. "At chance" for a head with K classes is 1/K; a model
#: must clear it by this margin on the validation split to be promoted.
CHANCE_MARGIN = 0.10


# ── data ──────────────────────────────────────────────────────────────────

def load_local_rows() -> list:
    """Locally labelled gate captures from the Phase 2 review loop."""
    if not LOCAL_GROUND_TRUTH.exists():
        return []
    with open(LOCAL_GROUND_TRUTH, newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for row in rows:
        if row.get("usable", "True") != "True":
            continue
        # Screen-replay images are excluded from TRAINING as well as
        # evaluation: a model taught on photographs of a monitor learns
        # moire and screen glare, not vehicles.
        if row.get("is_screen_replay", "False") == "True":
            continue
        path = row.get("image_path", "")
        if not path or not os.path.exists(path):
            continue
        labels = {}
        for head in HEADS:
            value = (row.get(f"{head}_ground_truth") or "").strip().lower()
            if value and value in LABEL_SPACES[head]:
                labels[head] = value
        if labels:
            out.append({"path": path, "labels": labels,
                        "lighting": row.get("lighting", ""),
                        "distance": row.get("distance", "")})
    return out


def load_colour_rows(colour_data: str) -> list:
    """
    UFPR-VCR, as either an ImageFolder-style tree (one directory per colour)
    or a CSV with image_path,colour columns.
    """
    if not colour_data:
        return []
    root = Path(colour_data)
    if not root.exists():
        return []

    if root.is_file() and root.suffix == ".csv":
        with open(root, newline="") as fh:
            rows = list(csv.DictReader(fh))
        out = []
        for row in rows:
            path = row.get("image_path", "")
            colour = (row.get("colour") or row.get("color") or "").strip().lower()
            if path and os.path.exists(path) and colour in COLOURS:
                out.append({"path": path, "labels": {"colour": colour}})
        return out

    out = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        colour = child.name.strip().lower()
        if colour not in COLOURS:
            # A directory whose name is outside the closed label space is
            # skipped loudly rather than silently widening the space.
            print(f"  skipping '{child.name}/': not in COLOURS", file=sys.stderr)
            continue
        for image in child.rglob("*"):
            if image.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                out.append({"path": str(image), "labels": {"colour": colour}})
    return out


def survey(sources: dict) -> dict:
    """Per-head example and class counts, and whether each head is trainable."""
    report = {}
    for head in HEADS:
        counter = Counter()
        for rows in sources.values():
            for row in rows:
                if head in row["labels"]:
                    counter[row["labels"][head]] += 1
        total = sum(counter.values())
        usable_classes = {k: v for k, v in counter.items() if v >= MIN_EXAMPLES_PER_CLASS}
        trainable = (total >= MIN_EXAMPLES_PER_HEAD
                     and len(usable_classes) >= MIN_CLASSES_PER_HEAD)
        blockers = []
        if total < MIN_EXAMPLES_PER_HEAD:
            blockers.append(f"{total} examples < {MIN_EXAMPLES_PER_HEAD} required")
        if len(usable_classes) < MIN_CLASSES_PER_HEAD:
            blockers.append(f"{len(usable_classes)} class(es) with "
                            f">={MIN_EXAMPLES_PER_CLASS} examples < "
                            f"{MIN_CLASSES_PER_HEAD} required")
        report[head] = {
            "examples": total,
            "distinct_classes": len(counter),
            "classes_above_minimum": len(usable_classes),
            "distribution": dict(counter.most_common()),
            "trainable": trainable,
            "blockers": blockers,
            "chance_accuracy": round(1.0 / len(LABEL_SPACES[head]), 4),
        }
    return report


def print_survey(report: dict, sources: dict):
    print("\nData sources")
    for name, rows in sources.items():
        print(f"  {name:8s} {len(rows):6d} image(s)")
    print("\nPer-head readiness")
    for head, info in report.items():
        state = "TRAINABLE" if info["trainable"] else "NOT TRAINABLE"
        print(f"  {head:7s} {info['examples']:6d} example(s), "
              f"{info['distinct_classes']} class(es)  -> {state}")
        if info["distribution"]:
            top = ", ".join(f"{k}={v}" for k, v in
                            list(info["distribution"].items())[:8])
            print(f"          {top}")
        for blocker in info["blockers"]:
            print(f"          blocked: {blocker}")


# ── training ──────────────────────────────────────────────────────────────

class AttributeDataset:
    """Rows -> (tensor, {head: class_index}). Torch Dataset, built lazily."""

    def __init__(self, rows, heads, input_size=224, train=False):
        self.rows = rows
        self.heads = heads
        self.input_size = input_size
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import cv2
        import numpy as np
        import torch

        from attributes.classifiers import preprocess

        row = self.rows[index]
        image = cv2.imread(row["path"])
        if image is None:
            image = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        if self.train:
            # Deliberately mild: a horizontal flip is label-preserving for
            # all three heads. No colour jitter — it would destroy the
            # colour head's only signal.
            if random.random() < 0.5:
                image = image[:, ::-1].copy()
        tensor = torch.from_numpy(preprocess(image, self.input_size))[0]
        targets = {}
        for head in self.heads:
            label = row["labels"].get(head)
            # -1 is the masking sentinel: CrossEntropyLoss(ignore_index=-1)
            # drops it, so a head sees gradient only from rows that label it.
            targets[head] = (LABEL_SPACES[head].index(label)
                             if label in LABEL_SPACES[head] else -1)
        return tensor, targets


def split_rows(rows, val_fraction=0.2, seed=42):
    """
    Random split. NOTE: this splits by ROW, not by vehicle. When several
    frames of the same vehicle exist, group them before splitting or the
    validation score measures memorisation. Grouping needs a vehicle id the
    labelling tool does not yet capture — recorded here as a known
    limitation rather than silently ignored.
    """
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    cut = max(1, int(len(shuffled) * val_fraction))
    return shuffled[cut:], shuffled[:cut]


def train(args, sources: dict, survey_report: dict) -> Path:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    from attributes.classifiers import build_network, checkpoint_payload

    trainable_heads = [h for h in HEADS if survey_report[h]["trainable"]
                       and h in args.heads]
    if not trainable_heads:
        print("\nREFUSING TO TRAIN: no head has enough labelled data.\n"
              "Nothing here can produce a number worth acting on. Label more "
              "images with:\n  python tools/build_local_plate_test_set.py label",
              file=sys.stderr)
        sys.exit(1)

    print(f"\nTraining heads: {', '.join(trainable_heads)}  "
          f"(strategy: {args.strategy})")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # One loader per source. Masked training alternates batches between
    # them, so each head only ever sees rows that label it.
    loaders, val_loaders = {}, {}
    for name, rows in sources.items():
        usable = [r for r in rows if any(h in r["labels"] for h in trainable_heads)]
        # A source too small to yield both a train and a validation split
        # contributes nothing and would otherwise build an empty DataLoader.
        if len(usable) < 2:
            if usable:
                print(f"  {name:8s} skipped — {len(usable)} row(s) cannot be split "
                      "into train and validation")
            continue
        train_rows, val_rows = split_rows(usable, args.val_fraction, args.seed)
        if not train_rows or not val_rows:
            print(f"  {name:8s} skipped — split produced an empty side "
                  f"({len(train_rows)} train / {len(val_rows)} val)")
            continue
        loaders[name] = DataLoader(
            AttributeDataset(train_rows, trainable_heads, args.input_size, train=True),
            batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=False)
        val_loaders[name] = DataLoader(
            AttributeDataset(val_rows, trainable_heads, args.input_size),
            batch_size=args.batch, shuffle=False, num_workers=args.workers)
        print(f"  {name:8s} train={len(train_rows)} val={len(val_rows)}")

    if not loaders:
        print("\nREFUSING TO TRAIN: no source has enough rows to form a "
              "train/validation split.", file=sys.stderr)
        sys.exit(1)

    model = build_network(len(COLOURS), len(TYPES), len(BRANDS),
                          pretrained=not args.no_pretrained)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    if args.strategy == "frozen":
        # Fallback strategy: colour trains the backbone, then it is frozen.
        print("  stage 1: backbone + colour head")
        _run_epochs(model, {k: v for k, v in loaders.items() if k == "colour"},
                    val_loaders, ["colour"], criterion, args, tag="stage1")
        for parameter in model.features.parameters():
            parameter.requires_grad = False
        for parameter in model.shared.parameters():
            parameter.requires_grad = False
        remaining = [h for h in trainable_heads if h != "colour"]
        if remaining:
            print(f"  stage 2: {', '.join(remaining)} heads on frozen features")
            _run_epochs(model, loaders, val_loaders, remaining, criterion, args,
                        tag="stage2")
    else:
        _run_epochs(model, loaders, val_loaders, trainable_heads, criterion, args,
                    tag="masked")

    metrics = evaluate_heads(model, val_loaders, trainable_heads)

    run_dir = RUNS_DIR / (args.run_name or time.strftime("run_%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, LABEL_SPACES, metrics),
               run_dir / "weights.pt")
    summary = {
        "run_name": run_dir.name,
        "strategy": args.strategy,
        "heads_trained": trainable_heads,
        "epochs": args.epochs,
        "input_size": args.input_size,
        "seed": args.seed,
        "data_survey": survey_report,
        "validation_metrics": metrics,
        "note": ("Validation accuracy on a row-level split of locally labelled "
                 "data. Not a public-benchmark number."),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nRun written to {run_dir}")
    for head in trainable_heads:
        m = metrics[head]
        print(f"  {head:7s} val accuracy {m['accuracy']:.4f} "
              f"(chance {m['chance']:.4f}, n={m['n']})")
    return run_dir


def _run_epochs(model, loaders, val_loaders, heads, criterion, args, tag=""):
    import torch

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    for epoch in range(args.epochs):
        model.train()
        iterators = {name: iter(loader) for name, loader in loaders.items()}
        steps = max((len(loader) for loader in loaders.values()), default=0)
        running, counted = 0.0, 0

        for _step in range(steps):
            # Alternate sources so the backbone sees every corpus and no
            # single large source dominates an epoch.
            for name in list(iterators):
                try:
                    tensors, targets = next(iterators[name])
                except StopIteration:
                    continue
                outputs = model(tensors, heads=tuple(heads))
                loss = None
                for head in heads:
                    target = targets[head]
                    if (target >= 0).sum() == 0:
                        continue      # this batch labels nothing for this head
                    head_loss = criterion(outputs[head], target)
                    loss = head_loss if loss is None else loss + head_loss
                if loss is None:
                    continue
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                running += float(loss)
                counted += 1

        scheduler.step()
        mean_loss = running / counted if counted else float("nan")
        print(f"    [{tag}] epoch {epoch + 1}/{args.epochs} loss {mean_loss:.4f}")


def evaluate_heads(model, val_loaders, heads) -> dict:
    """Validation accuracy and per-class counts per head. Measured, not assumed."""
    import torch

    model.eval()
    correct = {h: 0 for h in heads}
    total = {h: 0 for h in heads}
    confusion = {h: Counter() for h in heads}

    with torch.inference_mode():
        for loader in val_loaders.values():
            for tensors, targets in loader:
                outputs = model(tensors, heads=tuple(heads))
                for head in heads:
                    target = targets[head]
                    mask = target >= 0
                    if mask.sum() == 0:
                        continue
                    predicted = outputs[head][mask].argmax(dim=-1)
                    truth = target[mask]
                    correct[head] += int((predicted == truth).sum())
                    total[head] += int(mask.sum())
                    for p, t in zip(predicted.tolist(), truth.tolist()):
                        confusion[head][(LABEL_SPACES[head][t],
                                         LABEL_SPACES[head][p])] += 1

    metrics = {}
    for head in heads:
        n = total[head]
        chance = 1.0 / len(LABEL_SPACES[head])
        metrics[head] = {
            "n": n,
            "accuracy": round(correct[head] / n, 4) if n else 0.0,
            "chance": round(chance, 4),
            "beats_chance_by": round((correct[head] / n) - chance, 4) if n else 0.0,
            "confusion": {f"{t}->{p}": c for (t, p), c in confusion[head].most_common(40)},
        }
    return metrics


# ── compare / promote ─────────────────────────────────────────────────────

def compare_runs(run_names: list):
    rows = []
    for name in run_names:
        summary_path = RUNS_DIR / name / "summary.json"
        if not summary_path.exists():
            print(f"  {name}: no summary.json — run not completed", file=sys.stderr)
            continue
        rows.append(json.loads(summary_path.read_text()))
    if not rows:
        print("No completed runs to compare.", file=sys.stderr)
        sys.exit(1)

    print(f"{'run':22s} {'strategy':8s} " + " ".join(f"{h:>18s}" for h in HEADS))
    for summary in rows:
        cells = []
        for head in HEADS:
            m = summary["validation_metrics"].get(head)
            cells.append(f"{m['accuracy']:.4f} (ch {m['chance']:.2f})" if m
                         else f"{'not trained':>18s}")
        print(f"{summary['run_name']:22s} {summary['strategy']:8s} "
              + " ".join(f"{c:>18s}" for c in cells))
    print("\nAccuracies are on each run's own validation split of locally "
          "labelled data.\nThey are comparable to each other, and to nothing else.")


def promote(run_dir: Path):
    summary_path = run_dir / "summary.json"
    weights = run_dir / "weights.pt"
    if not summary_path.exists() or not weights.exists():
        print(f"Refusing to promote: {run_dir} has no completed run.", file=sys.stderr)
        sys.exit(1)

    summary = json.loads(summary_path.read_text())
    metrics = summary["validation_metrics"]

    failures = []
    for head, m in metrics.items():
        if m["n"] == 0:
            failures.append(f"{head}: no validation examples")
        elif m["accuracy"] <= m["chance"] + CHANCE_MARGIN:
            failures.append(
                f"{head}: validation accuracy {m['accuracy']:.4f} does not beat "
                f"chance ({m['chance']:.4f}) by the required {CHANCE_MARGIN:.2f}")

    if failures:
        print("REFUSING TO PROMOTE — this model is at or near chance:",
              file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        print("\nPromoting it would put a coin-flip in front of an operator and "
              "label it a reading.", file=sys.stderr)
        sys.exit(1)

    PRODUCTION_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights, PRODUCTION_MODEL_PATH)
    print(f"Promoted {run_dir.name} -> {PRODUCTION_MODEL_PATH}")
    for head, m in metrics.items():
        print(f"  {head:7s} val accuracy {m['accuracy']:.4f} (chance {m['chance']:.4f})")
    print("\nRestart the application to load it. Re-run "
          "'python tools/build_local_plate_test_set.py evaluate-attributes' to "
          "measure it on the local test set.")


# ── CLI ───────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--colour-data", default=None,
                   help="UFPR-VCR directory (one folder per colour) or a CSV")
    p.add_argument("--heads", nargs="+", default=list(HEADS), choices=list(HEADS))
    p.add_argument("--strategy", choices=["masked", "frozen"], default="masked",
                   help="masked = per-head loss masking (primary); "
                        "frozen = colour-trained backbone then frozen features")
    p.add_argument("--run-name", default=None)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--input-size", type=int, default=224)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pretrained", action="store_true",
                   help="start from random weights instead of ImageNet")
    p.add_argument("--check-data", action="store_true",
                   help="survey the datasets and stop")
    p.add_argument("--compare", nargs="+", default=None, metavar="RUN_NAME")
    p.add_argument("--promote", action="store_true",
                   help=f"copy the run's weights to {PRODUCTION_MODEL_PATH.name} "
                        "if it beats chance")
    return p


def main() -> int:
    args = build_arg_parser().parse_args()

    if args.compare:
        compare_runs(args.compare)
        return 0

    sources = {
        "colour": load_colour_rows(args.colour_data),
        "local": load_local_rows(),
    }
    survey_report = survey(sources)
    print_survey(survey_report, sources)

    if not sources["colour"]:
        print("\nNo colour corpus supplied. UFPR-VCR is not redistributable: "
              "request it\nfrom the authors, extract it, and pass "
              "--colour-data <dir>. Without it the\ncolour head has only "
              "locally labelled examples to learn from.")

    if args.check_data:
        return 0 if any(i["trainable"] for i in survey_report.values()) else 1

    if args.promote and args.run_name and (RUNS_DIR / args.run_name / "weights.pt").exists():
        # Promote an existing run without retraining it.
        promote(RUNS_DIR / args.run_name)
        return 0

    run_dir = train(args, sources, survey_report)
    if args.promote:
        promote(run_dir)
    else:
        print(f"\nNot promoted. Pass --promote to copy it to "
              f"{PRODUCTION_MODEL_PATH.name} after reviewing the numbers above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
