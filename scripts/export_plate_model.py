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

This script also exports the multi-head VEHICLE ATTRIBUTE model
(--attributes). That path is a plain image classifier rather than an
Ultralytics detector, so it uses torch.onnx + onnxruntime directly, adds
INT8 dynamic quantisation, and checks TOP-1 LABEL AGREEMENT against the
PyTorch baseline instead of IoU-matched box agreement — a classifier has no
boxes to match. The agreement check is the same idea in both cases: an
export that silently disagrees with its baseline is worse than no export.

Example:
    python scripts/export_plate_model.py \\
        --weights models/nigerian_plate_detector.pt \\
        --images gate_photos \\
        --formats pytorch onnx ncnn

    python scripts/export_plate_model.py --attributes \\
        --images gate_photos --device-label pi
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
sys.path.insert(0, str(REPO_ROOT))          # import attributes/ from the repo root
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
    p.add_argument("--attributes", action="store_true",
                   help="export and benchmark the multi-head vehicle attribute "
                        "model instead of the plate detector")
    p.add_argument("--attr-weights", default=None,
                   help="attribute checkpoint (default: the configured "
                        "attr_model_path)")
    p.add_argument("--attr-formats", nargs="+",
                   default=["pytorch", "onnx", "onnx-int8"],
                   help="attribute export formats to benchmark")
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


# ── VEHICLE ATTRIBUTE MODEL EXPORT ────────────────────────────────────────
# A classifier, not a detector: no boxes, so agreement is measured on top-1
# labels. INT8 here is ONNX Runtime's dynamic quantisation, which needs no
# calibration set — appropriate for a MobileNet whose activations are
# already well conditioned, and honest about being dynamic rather than
# claiming a calibrated static quantisation that was never performed.

ATTR_EXPORTS_DIR = REPO_ROOT / "models" / "exports"


def _attr_relative(path: Path) -> str:
    """Repo-relative where possible; absolute paths are not put in reports."""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return Path(path).name


def _attr_peak_rss_mb():
    """Process RSS in MB, or None where it cannot be measured."""
    try:
        import psutil
        return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)
    except Exception:
        return None


def _attr_load_crops(images_dir: str, max_images: int) -> list:
    """
    Vehicle crops to benchmark on. Uses the COCO detector to crop real
    vehicles, falling back to whole images so the script still runs on a
    machine with no detector available.
    """
    import cv2

    paths = sorted(
        p for p in glob.glob(os.path.join(images_dir, "vehicle_*.jpg"))
    ) or sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    if not paths:
        return []
    if len(paths) > max_images:
        step = len(paths) / float(max_images)
        paths = [paths[min(len(paths) - 1, int(i * step))] for i in range(max_images)]

    try:
        from attributes.vehicle_detector import VehicleDetector
        detector = VehicleDetector()
        detector.is_ready()
    except Exception:
        detector = None

    crops = []
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            continue
        crop = image
        if detector is not None:
            boxes = detector.detect_vehicles(image)
            if boxes:
                largest = max(boxes, key=lambda b: (b["box"][2] - b["box"][0]) *
                                                   (b["box"][3] - b["box"][1]))
                x1, y1, x2, y2 = largest["box"]
                crop = image[y1:y2, x1:x2]
        crops.append((os.path.basename(path), crop))
    return crops


def attr_export(checkpoint: Path, fmt: str, input_size: int) -> Path:
    """Export the attribute checkpoint to `fmt`. Returns the artifact path."""
    import torch

    from attributes.classifiers import build_network
    from attributes.models import BRANDS, COLOURS, TYPES

    ATTR_EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if fmt == "pytorch":
        dest = ATTR_EXPORTS_DIR / "vehicle_attributes_mnv3.pt"
        import shutil
        shutil.copy2(checkpoint, dest)
        return dest

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    spaces = payload.get("label_spaces") or {"colour": COLOURS, "type": TYPES,
                                             "brand": BRANDS}
    model = build_network(len(spaces["colour"]), len(spaces["type"]),
                          len(spaces["brand"]))
    model.load_state_dict(payload["state_dict"], strict=False)
    model.eval()

    class MultiHeadLogits(torch.nn.Module):
        """ONNX needs plain tensors, and named outputs the loader can map."""

        def __init__(self, wrapped):
            super().__init__()
            self.wrapped = wrapped

        def forward(self, pixel_values):
            out = self.wrapped(pixel_values)
            return out["colour"], out["type"], out["brand"]

    # eval() BEFORE export: torch.onnx.export restores each module's original
    # training flag afterwards, and a freshly built wrapper defaults to
    # training=True, which would silently re-enable dropout for anything run
    # after the export — including this script's own agreement check.
    exportable = MultiHeadLogits(model).eval()

    fp32_path = ATTR_EXPORTS_DIR / "vehicle_attributes_mnv3.onnx"
    generator = torch.Generator().manual_seed(0)
    dummy = torch.randn(1, 3, input_size, input_size, generator=generator)
    with torch.inference_mode():
        torch.onnx.export(
            exportable, (dummy,), str(fp32_path),
            input_names=["pixel_values"], output_names=["colour", "type", "brand"],
            dynamic_axes={"pixel_values": {0: "batch"}},
            opset_version=17, do_constant_folding=True, dynamo=False)
    _attr_write_onnx_metadata(fp32_path, spaces, input_size,
                              trained=bool(payload.get("metrics")))

    if fmt == "onnx":
        return fp32_path

    if fmt == "onnx-int8":
        from onnxruntime.quantization import QuantType, quantize_dynamic
        int8_path = ATTR_EXPORTS_DIR / "vehicle_attributes_mnv3_int8.onnx"
        quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QUInt8)
        _attr_write_onnx_metadata(int8_path, spaces, input_size,
                                  trained=bool(payload.get("metrics")))
        return int8_path

    if fmt == "ncnn":
        # NCNN conversion for a plain torch classifier goes through pnnx,
        # which Ultralytics' exporter only wires up for YOLO models. Rather
        # than pretend, this reports what is needed.
        raise RuntimeError(
            "NCNN export of the attribute classifier is not implemented here. "
            "Ultralytics' NCNN exporter covers YOLO models only; a plain "
            "classifier needs pnnx (pip install pnnx) run against the ONNX "
            "graph produced above. The ONNX and ONNX-INT8 paths are exported "
            "and benchmarked.")

    raise RuntimeError(f"unknown attribute export format: {fmt}")


def _attr_write_onnx_metadata(path: Path, label_spaces: dict, input_size: int,
                              trained: bool):
    """
    Bake the label spaces into the graph. An ONNX file carries no
    vocabulary, and guessing an ordering would mislabel every prediction.
    """
    import json as _json

    import onnx

    model = onnx.load(str(path))
    for key, value in (("label_spaces", _json.dumps(label_spaces)),
                       ("input_size", str(input_size)),
                       ("trained", "true" if trained else "false")):
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(model, str(path))


def attr_benchmark(fmt: str, artifact: Path, crops: list, input_size: int,
                   threads: int) -> dict:
    """Latency, load time and peak RSS for one attribute export format."""
    import numpy as np

    from attributes.classifiers import AttributeClassifier

    model_format = "pytorch" if fmt == "pytorch" else "onnx"
    rss_before = _attr_peak_rss_mb()
    t0 = time.perf_counter()
    classifier = AttributeClassifier(str(artifact), input_size=input_size,
                                     model_format=model_format, threads=threads,
                                     allow_stub=False)
    if not classifier.is_ready():
        return {"format": fmt, "load_error": classifier.status()["load_error"]}
    load_ms = (time.perf_counter() - t0) * 1000
    rss_after = _attr_peak_rss_mb()

    if not crops:
        return {"format": fmt, "error": "no benchmark crops available"}

    # Warm-up: the first pass includes graph/session setup.
    for _ in range(3):
        classifier.predict(crops[0][1], heads=("colour", "type", "brand"))

    times, predictions = [], {}
    for name, crop in crops:
        t0 = time.perf_counter()
        out = classifier.predict(crop, heads=("colour", "type", "brand"))
        times.append((time.perf_counter() - t0) * 1000)
        predictions[name] = {head: scored[0].label for head, scored in out.items()}

    times.sort()
    def pct(p):
        return round(times[min(len(times) - 1, int(round(p / 100 * (len(times) - 1))))], 2)

    return {
        "format": fmt,
        "artifact_path": _attr_relative(artifact),
        "artifact_size_mb": round(os.path.getsize(artifact) / (1024 * 1024), 2),
        "model_load_time_ms": round(load_ms, 2),
        "num_images": len(times),
        "avg_inference_time_ms": round(statistics.mean(times), 2),
        "median_inference_time_ms": round(statistics.median(times), 2),
        "p95_inference_time_ms": pct(95),
        "min_inference_time_ms": round(min(times), 2),
        "max_inference_time_ms": round(max(times), 2),
        "rss_before_load_mb": rss_before,
        "rss_after_load_mb": rss_after,
        "rss_increase_mb": (round(rss_after - rss_before, 1)
                            if rss_before is not None and rss_after is not None else None),
        "torch_threads": threads,
        "predictions": predictions,
    }


def label_agreement(reference: dict, candidate: dict) -> dict:
    """
    Top-1 label agreement per head against the PyTorch baseline.

    The classifier analogue of detection_agreement()'s IoU matching: same
    purpose, appropriate measure. This says nothing about whether either
    model is CORRECT — only whether the export preserved the baseline's
    behaviour.
    """
    ref = reference.get("predictions") or {}
    cand = candidate.get("predictions") or {}
    shared = sorted(set(ref) & set(cand))
    if not shared:
        return {"images_compared": 0}

    out = {"images_compared": len(shared)}
    for head in ("colour", "type", "brand"):
        agree = sum(1 for name in shared if ref[name].get(head) == cand[name].get(head))
        out[head] = round(agree / len(shared), 4)
    out["all_heads"] = round(
        sum(1 for name in shared if ref[name] == cand[name]) / len(shared), 4)
    return out


def run_attribute_export(args) -> int:
    from attributes import settings as attr_settings

    settings = attr_settings.load()
    checkpoint = Path(args.attr_weights or
                      attr_settings.resolve_path(settings["attr_model_path"]))
    if not checkpoint.exists():
        print(f"ERROR: attribute checkpoint {checkpoint} does not exist.\n"
              "Train and promote one first:\n"
              "  python scripts/train_vehicle_attributes.py --promote",
              file=sys.stderr)
        return 2

    input_size = int(settings.get("attr_input_size", 224))
    threads = int(settings.get("attr_torch_threads", 1))
    crops = _attr_load_crops(args.images, args.max_images)
    print(f"Benchmarking on {len(crops)} vehicle crop(s) from {args.images}")
    print(f"Checkpoint: {checkpoint}  input {input_size}px  torch threads {threads}\n")

    results, reference = [], None
    for fmt in args.attr_formats:
        print(f"=== Attribute format: {fmt} ===")
        t0 = time.perf_counter()
        try:
            artifact = attr_export(checkpoint, fmt, input_size)
        except Exception as exc:
            print(f"Export failed for {fmt}: {exc}", file=sys.stderr)
            results.append({"format": fmt, "export_error": str(exc)})
            continue
        export_s = time.perf_counter() - t0

        bench = attr_benchmark(fmt, artifact, crops, input_size, threads)
        bench["export_time_s"] = round(export_s, 2)
        bench["device_label"] = args.device_label
        if fmt == "pytorch":
            reference = bench
        if reference is not None and "predictions" in bench:
            bench["label_agreement_with_pytorch"] = label_agreement(reference, bench)

        summary = {k: v for k, v in bench.items() if k != "predictions"}
        results.append(summary)
        print(json.dumps(summary, indent=2))
        print()

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / f"attribute_export_benchmark_{args.device_label}.json"
    report_path.write_text(json.dumps({
        "checkpoint": _attr_relative(checkpoint),
        "device_label": args.device_label,
        "input_size": input_size,
        "torch_threads": threads,
        "num_benchmark_crops": len(crops),
        "latency_budget_ms": settings.get("attr_latency_budget_ms"),
        "results": results,
        "note": ("Latency, load time and memory measured on this device. Label "
                 "agreement compares each export against the PyTorch baseline "
                 "and is NOT an accuracy claim about either."),
    }, indent=2))
    print(f"Benchmark report written to {report_path}")

    valid = [r for r in results if "avg_inference_time_ms" in r]
    if valid:
        fastest = min(valid, key=lambda r: r["avg_inference_time_ms"])
        budget = settings.get("attr_latency_budget_ms", 150.0)
        print(f"\nFastest on {args.device_label}: {fastest['format']} "
              f"({fastest['avg_inference_time_ms']} ms/crop avg, "
              f"budget {budget} ms)")
        for r in valid:
            agreement = r.get("label_agreement_with_pytorch", {})
            if r["format"] != "pytorch" and agreement.get("all_heads") is not None:
                print(f"  {r['format']}: top-1 agreement with PyTorch "
                      f"{agreement['all_heads']:.3f} across all heads")
        print("Change attr_model_format in config.json only after reviewing "
              "agreement, and only from a run made ON THE PI.")
    else:
        print("\nNo attribute format produced usable benchmark results.")
    return 0


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.attributes:
        sys.exit(run_attribute_export(args))

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
