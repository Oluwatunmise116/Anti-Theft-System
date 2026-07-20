#!/usr/bin/env python3
"""
Reproducible download + validation for the Nigerian license-plate detection
dataset (https://universe.roboflow.com/nigerianlpd/nigerian-license-plate).

This is a DETECTION dataset — bounding boxes only, no plate-text
transcriptions. It is used to train scripts/train_plate_detector.py. It is
NOT sufficient to train an OCR text recognizer; see README_ANPR.md.

Usage:
    export ROBOFLOW_API_KEY=...      # never commit this — see .env.example
    export ROBOFLOW_WORKSPACE=nigerianlpd
    export ROBOFLOW_PROJECT=nigerian-license-plate   # default
    export ROBOFLOW_VERSION=         # optional — leave empty to auto-select latest
    python scripts/download_nigerian_plate_dataset.py

What this script does:
  1. Reads credentials from environment variables only (optionally via a
     local, uncommitted .env file) — never from source code.
  2. Lists all available dataset versions and picks the latest one unless
     ROBOFLOW_VERSION pins a specific version.
  3. Downloads that version in Ultralytics YOLO format to
     datasets/nigerian-license-plate/.
  4. Validates the export: data.yaml, split directories, image/label
     pairing, bounding-box validity, class-id validity; reports corrupt
     images and empty-annotation images SEPARATELY (an empty annotation is
     not the same failure as a corrupt file).
  5. Normalizes the single class's semantic name to "license_plate" while
     keeping its index at 0.
  6. Writes reports/dataset_audit.json with the full audit trail and the
     exact dataset version used.
  7. Preserves any LICENSE/README shipped in the Roboflow export.

If you don't have API access to this workspace, Roboflow requires you to
fork the project into your own workspace first — see MANUAL STEP below.
This script will not fabricate or guess an API key.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = REPO_ROOT / "datasets" / "nigerian-license-plate"
REPORTS_DIR = REPO_ROOT / "reports"
TARGET_CLASS_NAME = "license_plate"

MANUAL_STEP = """
MANUAL STEP REQUIRED
=====================
Roboflow Universe projects are sometimes only downloadable after you fork
them into your own workspace (this is a Roboflow access-control mechanism,
not something this script can bypass):

  1. Open https://universe.roboflow.com/nigerianlpd/nigerian-license-plate
  2. Click "Fork Dataset" (top right) and fork it into your own Roboflow
     workspace.
  3. Open your fork, click "Versions" and generate/select a version
     (or use the version the maintainers already generated).
  4. Get your API key from https://app.roboflow.com/settings/api and set:

       export ROBOFLOW_API_KEY=<your key>
       export ROBOFLOW_WORKSPACE=<your workspace slug>
       export ROBOFLOW_PROJECT=nigerian-license-plate
       # ROBOFLOW_VERSION is optional — leave unset to auto-select the latest

  5. Re-run this script.

Never hardcode the API key in source control. Use a local, uncommitted
.env file (see .env.example) or your shell environment.
""".strip()


def _load_dotenv_if_present():
    """Best-effort .env loader — no hard dependency on python-dotenv."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
        return
    except ImportError:
        pass
    # Minimal manual parser fallback
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def get_credentials() -> dict:
    _load_dotenv_if_present()
    api_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    workspace = os.environ.get("ROBOFLOW_WORKSPACE", "").strip()
    project = os.environ.get("ROBOFLOW_PROJECT", "nigerian-license-plate").strip()
    version = os.environ.get("ROBOFLOW_VERSION", "").strip()
    return {"api_key": api_key, "workspace": workspace, "project": project, "version": version}


def select_version(project_handle, pinned_version: str):
    """
    List all versions of the project and pick the one to use. Prefers the
    highest version number ("latest") unless pinned_version is given.
    Prints every available version so the choice is auditable.
    """
    versions = project_handle.versions()
    if not versions:
        raise RuntimeError("Project has no published versions to download.")

    parsed = []
    for v in versions:
        # Roboflow's Version.id looks like "workspace/project/<n>"
        try:
            num = int(str(v.version).split("/")[-1])
        except (ValueError, AttributeError):
            num = -1
        parsed.append((num, v))
    parsed.sort(key=lambda t: t[0], reverse=True)

    print("Available dataset versions (newest first):")
    for num, v in parsed:
        created = getattr(v, "created", "unknown date")
        images = getattr(v, "images", "?")
        print(f"  - version {num}: created={created} images={images}")

    if pinned_version:
        for num, v in parsed:
            if str(num) == str(pinned_version):
                print(f"\nUsing pinned ROBOFLOW_VERSION={pinned_version}")
                return num, v
        raise RuntimeError(f"ROBOFLOW_VERSION={pinned_version} not found among published versions.")

    latest_num, latest_v = parsed[0]
    print(f"\nNo ROBOFLOW_VERSION pinned — auto-selecting the latest: version {latest_num}")
    return latest_num, latest_v


def download_dataset(creds: dict, dest: Path) -> int:
    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: the 'roboflow' package is not installed.\n"
              "Install it with: pip install roboflow", file=sys.stderr)
        sys.exit(2)

    if not creds["api_key"] or not creds["workspace"]:
        print("ERROR: ROBOFLOW_API_KEY and ROBOFLOW_WORKSPACE must be set "
              "(never hardcoded — use environment variables or a local .env).",
              file=sys.stderr)
        print(MANUAL_STEP)
        sys.exit(2)

    rf = Roboflow(api_key=creds["api_key"])
    workspace = rf.workspace(creds["workspace"])
    project = workspace.project(creds["project"])

    version_num, version_handle = select_version(project, creds["version"])

    if dest.exists():
        print(f"Removing existing dataset directory: {dest}")
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading version {version_num} in Ultralytics YOLO format to {dest} ...")
    version_handle.download("yolov8", location=str(dest))
    print("Download complete.")
    return version_num


# ── VALIDATION ────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _is_valid_image(path: Path) -> bool:
    try:
        import cv2
        img = cv2.imread(str(path))
        return img is not None and img.size > 0
    except Exception:
        return False


def _validate_split(split_dir: Path, class_count: int, dataset_root: Path) -> dict:
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    result = {
        "split": split_dir.name,
        "images_dir_exists": images_dir.is_dir(),
        "labels_dir_exists": labels_dir.is_dir(),
        "num_images": 0,
        "num_label_files": 0,
        "images_missing_labels": [],
        "labels_missing_images": [],
        "corrupt_images": [],
        "empty_annotation_images": [],
        "invalid_bounding_boxes": [],
        "invalid_class_ids": [],
    }
    if not images_dir.is_dir() or not labels_dir.is_dir():
        return result

    image_files = {p.stem: p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS}
    label_files = {p.stem: p for p in labels_dir.glob("*.txt")}
    result["num_images"] = len(image_files)
    result["num_label_files"] = len(label_files)

    result["images_missing_labels"] = sorted(set(image_files) - set(label_files))
    result["labels_missing_images"] = sorted(set(label_files) - set(image_files))

    for stem, img_path in image_files.items():
        if not _is_valid_image(img_path):
            result["corrupt_images"].append(str(img_path.relative_to(dataset_root)))

        label_path = label_files.get(stem)
        if label_path is None:
            continue
        lines = [l.strip() for l in label_path.read_text().splitlines() if l.strip()]
        if not lines:
            result["empty_annotation_images"].append(str(img_path.relative_to(dataset_root)))
            continue
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                result["invalid_bounding_boxes"].append(f"{label_path.name}: '{line}'")
                continue
            cls_id_str, cx, cy, w, h = parts
            try:
                cls_id = int(cls_id_str)
                cx, cy, w, h = float(cx), float(cy), float(w), float(h)
            except ValueError:
                result["invalid_bounding_boxes"].append(f"{label_path.name}: '{line}'")
                continue
            if not (0 <= cls_id < class_count):
                result["invalid_class_ids"].append(f"{label_path.name}: class_id={cls_id}")
            if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0
                    and 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                result["invalid_bounding_boxes"].append(f"{label_path.name}: '{line}' (out of [0,1] range)")

    return result


def normalize_class_name(data_yaml_path: Path) -> dict:
    """Rewrite the single class's name to 'license_plate', keeping index 0."""
    import yaml
    with open(data_yaml_path) as f:
        data = yaml.safe_load(f)

    names = data.get("names")
    before = dict(names) if isinstance(names, dict) else list(names) if names else names

    if isinstance(names, dict):
        # {0: "0"} or {0: "plate"} etc.
        data["names"] = {0: TARGET_CLASS_NAME}
    elif isinstance(names, list):
        data["names"] = [TARGET_CLASS_NAME] + list(names[1:])
    else:
        data["names"] = [TARGET_CLASS_NAME]
    data["nc"] = 1

    with open(data_yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    return {"before": before, "after": data["names"]}


def validate_dataset(dataset_dir: Path) -> dict:
    data_yaml = dataset_dir / "data.yaml"
    audit = {
        "dataset_dir": str(dataset_dir),
        "data_yaml_exists": data_yaml.exists(),
        "class_name_normalization": None,
        "splits": [],
    }
    if not data_yaml.exists():
        audit["error"] = "data.yaml not found — download may have failed or used a different format."
        return audit

    import yaml
    with open(data_yaml) as f:
        data = yaml.safe_load(f)

    class_count = int(data.get("nc", 1) or 1)
    audit["class_name_normalization"] = normalize_class_name(data_yaml)

    for split in ("train", "valid", "test"):
        split_dir = dataset_dir / split
        if split_dir.is_dir():
            audit["splits"].append(_validate_split(split_dir, class_count, dataset_dir))
        else:
            audit["splits"].append({"split": split, "images_dir_exists": False,
                                     "labels_dir_exists": False, "note": "split directory not present"})

    return audit


def preserve_license(dataset_dir: Path):
    for name in ("README.dataset.txt", "README.roboflow.txt", "LICENSE", "LICENSE.txt"):
        src = dataset_dir / name
        if src.exists():
            print(f"Preserved dataset licence/attribution file: {src.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", default=str(DATASET_DIR), help="Destination directory for the dataset")
    parser.add_argument("--skip-download", action="store_true",
                         help="Only re-validate an already-downloaded dataset at --dest")
    args = parser.parse_args()

    dest = Path(args.dest)
    version_used = None

    if not args.skip_download:
        creds = get_credentials()
        version_used = download_dataset(creds, dest)
    else:
        print(f"--skip-download set: validating existing dataset at {dest}")

    if not dest.exists():
        print(f"ERROR: dataset directory {dest} does not exist.", file=sys.stderr)
        sys.exit(1)

    preserve_license(dest)

    print("\nValidating dataset export ...")
    audit = validate_dataset(dest)
    audit["dataset_version"] = version_used
    audit["target_class_name"] = TARGET_CLASS_NAME

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "dataset_audit.json"
    with open(report_path, "w") as f:
        json.dump(audit, f, indent=2)

    print(f"\nDataset audit written to {report_path}")
    print(f"Exact dataset version used: {version_used!r}")

    total_corrupt = sum(len(s.get("corrupt_images", [])) for s in audit["splits"])
    total_empty = sum(len(s.get("empty_annotation_images", [])) for s in audit["splits"])
    total_bad_boxes = sum(len(s.get("invalid_bounding_boxes", [])) for s in audit["splits"])
    total_bad_classes = sum(len(s.get("invalid_class_ids", [])) for s in audit["splits"])

    print(f"Corrupt images: {total_corrupt}")
    print(f"Empty-annotation images (background/no-plate images — reported "
          f"separately from corruption): {total_empty}")
    print(f"Invalid bounding boxes: {total_bad_boxes}")
    print(f"Invalid class ids: {total_bad_classes}")

    if total_corrupt or total_bad_boxes or total_bad_classes:
        print("\nWARNING: dataset has integrity issues — review reports/dataset_audit.json "
              "before training. Corrupt images and invalid boxes will be skipped by "
              "scripts/train_plate_detector.py, not silently included.")


if __name__ == "__main__":
    main()
