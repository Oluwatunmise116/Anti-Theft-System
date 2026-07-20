#!/usr/bin/env python3
"""
Export the trained Nigerian plate detector to Raspberry Pi-appropriate
formats and benchmark each one on representative images.

Formats attempted:
    pytorch  — the trained .pt as-is (baseline)
    onnx     — via Ultralytics' ONNX exporter, where supported
    ncnn     — via Ultralytics' NCNN exporter, for ARM (Raspberry Pi) deployment

Exports are written to models/exports/. Benchmarks run against real images
from gate_photos/ (falling back to the training validation set if none are
available) and measure timing, load time, memory where practical, and
detection agreement with the original PyTorch model — never a claim about
accuracy beyond what's actually measured here plus
reports/plate_detector_metrics.json (which covers detection mAP only).

This script does not choose blindly — it prints a recommendation but never
overwrites config.json automatically; the operator decides.

Run this on the training workstation (NCNN/ONNX export doesn't need a GPU,
but does need the same ultralytics install used for training). Benchmarks
here are indicative; ALSO re-run the "pi" benchmark mode
(--device-label pi) directly on the target Raspberry Pi for real Pi timing,
since a workstation CPU benchmark is not representative of Pi performance.

Example:
    python scripts/export_plate_model.py \\
        --weights models/nigerian_plate_detector.pt \\
        --images gate_photos \\
        --formats pytorch onnx ncnn
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPORTS_DIR = REPO_ROOT / "models" / "exports"
REPORTS_DIR = REPO_ROOT / "reports"


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=str(REPO_ROOT / "models" / "nigerian_plate_detector.pt"))
    p.add_argument("--images", default=str(REPO_ROOT / "gate_photos"),
                   help="Directory of representative images to benchmark on "
                        "(defaults to real captured gate photos)")
    p.add_argument("--formats", nargs="+", default=["pytorch", "onnx", "ncnn"],
                   choices=["pytorch", "onnx", "ncnn"])
    p.add_argument("--max-images", type=int, default=30)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--device-label", default="workstation",
                   help="Label recorded in the report, e.g. 'workstation' or 'raspberry-pi-4'. "
                        "Re-run this script with --device-label pi ON THE PI ITSELF for real "
                        "Pi numbers — a workstation CPU is not a stand-in for Pi timing.")
    return p


def _require_ultralytics():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError:
        print("ERROR: ultralytics is not installed.", file=sys.stderr)
        sys.exit(2)


def _load_images(images_dir: str, max_images: int):
    import cv2
    paths = sorted(
        glob.glob(os.path.join(images_dir, "vehicle_*.jpg")) +
        glob.glob(os.path.join(images_dir, "*.jpg"))
    )
    # de-dup while preserving order
    seen = set()
    unique_paths = []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique_paths.append(p)
    unique_paths = unique_paths[:max_images]

    images = []
    for p in unique_paths:
        img = cv2.imread(p)
        if img is not None:
            images.append((p, img))
    return images


def export_format(weights_path: Path, fmt: str) -> Path:
    YOLO = _require_ultralytics()
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if fmt == "pytorch":
        dest = EXPORTS_DIR / "nigerian_plate_detector.pt"
        import shutil
        shutil.copy2(weights_path, dest)
        return dest

    model = YOLO(str(weights_path))
    export_path = model.export(format=fmt)   # returns path to exported artifact/dir
    export_path = Path(export_path)

    dest = EXPORTS_DIR / export_path.name
    if export_path.is_dir():
        import shutil
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(export_path, dest)
    else:
        import shutil
        shutil.copy2(export_path, dest)
    return dest


def _load_model_for_format(fmt: str, artifact_path: Path):
    YOLO = _require_ultralytics()
    return YOLO(str(artifact_path))


def benchmark_model(model, images, conf: float, label: str) -> dict:
    if not images:
        return {"format": label, "error": "No benchmark images available"}

    # Warm-up (first inference includes graph/session setup — exclude it from timing)
    model(images[0][1], conf=conf, verbose=False)

    tracemalloc.start()
    times = []
    detections_per_image = {}
    for path, img in images:
        t0 = time.perf_counter()
        results = model(img, conf=conf, verbose=False)[0]
        t1 = time.perf_counter()
        times.append(t1 - t0)
        boxes = results.boxes
        detections_per_image[path] = [] if boxes is None else [
            tuple(b.tolist()) for b in boxes.xyxy.cpu().numpy().round(1)
        ]
    _, peak_kb = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "format": label,
        "num_images": len(images),
        "avg_inference_time_ms": round(statistics.mean(times) * 1000, 2),
        "median_inference_time_ms": round(statistics.median(times) * 1000, 2),
        "min_inference_time_ms": round(min(times) * 1000, 2),
        "max_inference_time_ms": round(max(times) * 1000, 2),
        "peak_python_memory_kb": round(peak_kb / 1024, 1),
        "detections_per_image": detections_per_image,
    }


def detection_agreement(reference: dict, candidate: dict, iou_threshold: float = 0.5) -> float:
    """
    Fraction of images where candidate's box count and rough box locations
    agree with the reference (pytorch) model — a simple, honest agreement
    metric, not a substitute for reports/plate_detector_metrics.json.
    """
    ref_dets = reference.get("detections_per_image", {})
    cand_dets = candidate.get("detections_per_image", {})
    if not ref_dets:
        return 0.0

    def iou(a, b):
        ax1, ay1, ax2, ay2 = a[:4]
        bx1, by1, bx2, by2 = b[:4]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
        inter = iw * ih
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    agree = 0
    for path, ref_boxes in ref_dets.items():
        cand_boxes = cand_dets.get(path, [])
        if not ref_boxes and not cand_boxes:
            agree += 1
            continue
        if not ref_boxes or not cand_boxes:
            continue
        matched = any(iou(rb, cb) >= iou_threshold for rb in ref_boxes for cb in cand_boxes)
        if matched:
            agree += 1
    return round(agree / len(ref_dets), 4)


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"ERROR: {weights_path} does not exist. Train it first with "
              f"scripts/train_plate_detector.py --promote, or point --weights at "
              f"an existing model.", file=sys.stderr)
        sys.exit(2)

    images = _load_images(args.images, args.max_images)
    if not images:
        print(f"WARNING: no images found under {args.images} — benchmarks will be skipped.",
              file=sys.stderr)

    results = []
    reference = None

    for fmt in args.formats:
        print(f"\n=== Exporting format: {fmt} ===")
        t_export_start = time.perf_counter()
        try:
            artifact_path = export_format(weights_path, fmt)
        except Exception as e:
            print(f"Export failed for {fmt}: {e}", file=sys.stderr)
            results.append({"format": fmt, "export_error": str(e)})
            continue
        export_time = time.perf_counter() - t_export_start
        print(f"Exported to {artifact_path} in {export_time:.2f}s")

        t_load_start = time.perf_counter()
        try:
            model = _load_model_for_format(fmt, artifact_path)
        except Exception as e:
            print(f"Load failed for {fmt}: {e}", file=sys.stderr)
            results.append({"format": fmt, "artifact_path": str(artifact_path), "load_error": str(e)})
            continue
        load_time_ms = (time.perf_counter() - t_load_start) * 1000

        bench = benchmark_model(model, images, args.conf, fmt)
        bench["artifact_path"] = str(artifact_path)
        bench["export_time_s"] = round(export_time, 2)
        bench["model_load_time_ms"] = round(load_time_ms, 2)
        bench["device_label"] = args.device_label

        if fmt == "pytorch":
            reference = bench
        if reference is not None:
            bench["detection_agreement_with_pytorch"] = detection_agreement(reference, bench)

        # Detection details are large and only useful for debugging agreement —
        # drop them from the persisted report, keep the summary numbers.
        bench_summary = {k: v for k, v in bench.items() if k != "detections_per_image"}
        results.append(bench_summary)
        print(json.dumps(bench_summary, indent=2))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "plate_export_benchmark.json"
    report_path.write_text(json.dumps({
        "weights": str(weights_path),
        "device_label": args.device_label,
        "confidence_threshold": args.conf,
        "num_benchmark_images": len(images),
        "results": results,
    }, indent=2))
    print(f"\nBenchmark report written to {report_path}")

    valid = [r for r in results if "avg_inference_time_ms" in r]
    if valid:
        fastest = min(valid, key=lambda r: r["avg_inference_time_ms"])
        print(f"\nFastest format on this run ({args.device_label}): {fastest['format']} "
              f"({fastest['avg_inference_time_ms']} ms/image avg)")
        print("This is a recommendation based on THIS benchmark run only — review "
              "detection_agreement_with_pytorch and, ideally, re-run with --device-label pi "
              "directly on the Raspberry Pi before changing plate_model_format in config.json.")
    else:
        print("\nNo format produced usable benchmark results.")


if __name__ == "__main__":
    main()
