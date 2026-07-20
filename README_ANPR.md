# Nigerian ANPR Pipeline

This document covers the automatic number-plate recognition (ANPR)
subsystem: architecture, deployment, configuration, testing, and known
limitations.

**Current implementation (NLPDRS).** Detection and recognition follow the
[NLPDRS Nigerian License Plate Detection and Recognition System](https://github.com/esssyjr/NLPDRS-Nierian-License-Plate-Detection-and-Recognition-System-):
a YOLOv8 segmentation model trained on Nigerian vehicles (annotated to
segment just the plate-number portion of the plate), whose weights are
committed at `models/nlpdrs_plate_segment.pt`, followed by EasyOCR on the
cropped region — the recognized text is concatenated and spaces stripped.
On top of NLPDRS's single-image technique, this app adds a thin
multi-frame majority vote over the capture burst and the alphanumeric
storage normalization used for DB lookup. The training/export tooling
described further below (sections 3-5) belongs to the previous
locally-trained detector and is kept only for optional retraining — the
running app does not depend on it.

## 1. What was wrong with the old implementation

The previous `gate_manager.py` ANPR code had several compounding accuracy
problems, all fixed in this rewrite:

- **Wrong model.** `models/license_plate_detector.pt` was a *generic*
  plate detector trained on non-Nigerian plates, and if missing, the app
  downloaded one of three near-identical generic models from unrelated
  third-party GitHub mirrors at runtime. None of these were trained on
  Nigerian plate geometry, fonts, or backgrounds.
- **Single-frame decisions.** One frame was captured and analyzed once —
  no temporal voting, so a single motion-blurred or glare-heavy frame
  could produce (or lose) a reading with nothing to check it against.
- **Aggressive, unconditional character substitution.** OCR text was
  forced into whatever Nigerian-plate-shaped string required the fewest
  edits, with no cap on how many characters could be changed and no
  confidence penalty for doing so — a low-confidence OCR read could be
  "corrected" into a *different*, confidently-wrong plate.
- **One broad regex.** A single `^[A-Z]{1,3}[0-9]{2,4}[A-Z]{1,3}$` pattern
  covered too much ground to usefully validate structure, while also
  implicitly telling the corrector "any shape in this range is fine."
- **Confidence and regex-match were conflated.** Matching the regex added
  a flat score bonus regardless of how weak the underlying OCR confidence
  was — a plausible-looking wrong answer could outscore a correct-but-shy
  one.
- **640×480 capture.** Plates are a small fraction of a 640×480 frame at
  typical gate distances; a lot of the pixels needed to resolve individual
  characters were simply never captured.
- **No crop quality gating.** Blurry, tiny, or glare-blown crops were sent
  to OCR anyway, with no sharpness/exposure check and no rejection path.

## 2. Architecture

```
anpr/
    models.py          PlateDetection, PlateOCRCandidate, PlateRecognitionResult
    detector.py         PlateDetector — loads ONE configured Nigerian YOLO model, no downloads
    recognizer.py        OCRBackend abstraction — EasyOCR (default) / PaddleOCR (optional)
    preprocessing.py     crop quality gating, rectification, preprocessing variants
    postprocessing.py    normalization, display formatting, safe character correction
    consensus.py         multi-frame temporal voting
    __init__.py           PlateRecognitionPipeline — orchestrates all of the above
```

Pipeline flow for one gate capture (NLPDRS technique):

```
capture 8-12 raw frames (burst)
        |
select best frames (sharpness, exposure) --------- anpr.preprocessing.select_best_frames
        |
for each selected frame:
    PlateDetector.detect() on the FULL frame -> first (highest-conf) box
        |                                        --- models/nlpdrs_plate_segment.pt, loaded once
    crop the detected plate-number region
        |
    OCRBackend.recognize(crop)  -> concatenate text, strip spaces --- anpr.recognizer (EasyOCR)
        |
    normalize_plate_for_storage() + plausibility guard (5-10 chars, letters+digits, no overlay words)
        |
majority vote across frames -> PlateRecognitionResult (CONFIRMED / LOW_CONFIDENCE / NOT_DETECTED)
```

Detection accuracy (did we find a box?) and OCR accuracy (did we read the
text correctly?) are measured and reported **separately** — see
`reports/plate_detector_metrics.json` (detector-only, from a real training
run) vs. `reports/local_evaluation_report.json` (OCR/end-to-end, from real
gate photos). Nothing in this codebase presents detector mAP as
plate-reading accuracy.

### Typed results, not bare strings

`PlateRecognitionPipeline.process_frames()` returns a
`PlateRecognitionResult` (see `anpr/models.py`) carrying status,
per-stage confidences, consensus count, bounding box, and a rejection
reason — not a bare string. `anpr.detect_plate_string(result)` is a thin
compatibility wrapper for the one remaining call site that only wants a
string (`gate_manager.detect_plate`, used by the exit page's quick
"anpr-now" check).

## 3. Dataset

Source: <https://universe.roboflow.com/nigerianlpd/nigerian-license-plate>
— an **object-detection** dataset (bounding boxes only, no plate-text
transcription). It trains the detector; it cannot and does not train an
OCR recognizer. OCR uses a general-purpose, pretrained local OCR engine
(EasyOCR) restricted to the A-Z0-9 plate alphabet, evaluated separately
against real gate photos (`tools/build_local_plate_test_set.py`).

```
cp .env.example .env
# edit .env: ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE (fork the dataset into
# your own workspace first if you don't have direct access — the script
# prints the exact manual steps if this is required)
python scripts/download_nigerian_plate_dataset.py
```

This:
1. Lists every published dataset version and auto-selects the latest
   unless `ROBOFLOW_VERSION` pins one — the exact version used is printed
   and written to the audit report.
2. Downloads it in Ultralytics YOLO format to
   `datasets/nigerian-license-plate/`.
3. Validates: `data.yaml` presence, `train/valid/test` split directories,
   image↔label pairing, bounding-box range validity, class-id validity;
   reports **corrupt images** and **empty-annotation images** as separate
   categories (an image with no plate is not the same problem as a
   corrupt file).
4. Normalizes the single class's name to `license_plate` (index stays 0).
5. Preserves any `LICENSE`/`README.roboflow.txt` shipped in the export.
6. Writes `reports/dataset_audit.json`.

Re-validate an already-downloaded copy without re-downloading://
```
python scripts/download_nigerian_plate_dataset.py --skip-download \
    --dest datasets/nigerian-license-plate
```

## 4. Training the detector

Run on a GPU workstation or cloud notebook — **not** on the Raspberry Pi.

```
python scripts/train_plate_detector.py \
    --data datasets/nigerian-license-plate/data.yaml \
    --model yolov8n.pt \
    --imgsz 640 \
    --epochs 120 \
    --patience 25 \
    --batch -1 \
    --seed 42 \
    --run-name ng_plate_640
```

Compare at least two configurations (this repo's default is imgsz
640 vs. 960 — evaluate both before picking one) before promoting a model:

```
python scripts/train_plate_detector.py --imgsz 960 --run-name ng_plate_960
python scripts/train_plate_detector.py --compare ng_plate_640 ng_plate_960
```

Then promote the better one (only after reviewing its validation plots):

```
python scripts/train_plate_detector.py --imgsz 640 --run-name ng_plate_640 --promote
```

`--promote` copies `runs/plate_detector/<run>/weights/best.pt` to
`models/nigerian_plate_detector.pt`, backing up any existing production
model first, and refuses to promote a model with `mAP50 == 0`.

**Augmentations used:** small rotation (±5°), mild shear/perspective,
moderate brightness/contrast (HSV jitter), moderate scale, reduced mosaic
(0.5 instead of Ultralytics' default 1.0, which tiles small plates into
unreadable fragments). **Horizontal/vertical flips are disabled** — a
mirrored plate is not a plate a camera will ever see, and would teach the
model on physically impossible text.

Reports produced (real numbers from the validation split only — never
training-set performance, never fabricated):
- `reports/plate_detector_metrics.json` — mAP50, mAP50-95, precision,
  recall, exact dataset version, exact model architecture, image size,
  selected confidence/IoU thresholds, per run.
- `reports/plate_detector_summary.md` — human-readable comparison table
  across runs.
- Ultralytics also writes validation PR curves, confusion matrix, and
  prediction mosaics under `runs/plate_detector/<run>/` — inspect
  `val_batch*_pred.jpg` for real false-positive/negative examples before
  promoting.

## 5. Export and Raspberry Pi benchmarking

```
python scripts/export_plate_model.py \
    --weights models/nigerian_plate_detector.pt \
    --images gate_photos \
    --formats pytorch onnx ncnn
```

Exports land in `models/exports/`. The script benchmarks every format it
successfully exports against real captured gate images (`gate_photos/`),
measuring average/median inference time, model load time, peak Python
memory, and detection agreement against the PyTorch baseline (IoU-matched
box comparison) — written to `reports/plate_export_benchmark.json`.

A workstation CPU is not representative of Raspberry Pi timing — re-run
with `--device-label pi` **on the Pi itself** before deciding
`plate_model_format` in `config.json`:

```
python scripts/export_plate_model.py --device-label raspberry-pi-5 \
    --formats pytorch onnx ncnn
```

Then set, in `config.json`:
```
"plate_model_path": "models/exports/<chosen-artifact>",
"plate_model_format": "pytorch" | "onnx" | "ncnn"
```

## 6. Application configuration (`config.json`)

```json
{
  "camera_index": 0,
  "camera_width": 1280,
  "camera_height": 720,
  "plate_model_path": "models/nlpdrs_plate_segment.pt",
  "plate_model_format": "pytorch",
  "plate_detection_confidence": 0.35,
  "plate_detection_iou": 0.45,
  "plate_min_width_pixels": 100,
  "plate_consensus_frames": 3,
  "plate_capture_frame_count": 10,
  "plate_max_corrections": 2,
  "plate_min_confirm_confidence": 0.55,
  "plate_ocr_backend": "easyocr",
  "plate_debug_mode": false,
  "plate_crop_padding": 6,
  "plate_debug_max_age_hours": 24,
  "plate_debug_max_storage_mb": 200
}
```

The camera worker (`face_manager.py`) requests `camera_width`/
`camera_height` (1280×720 by default — plates lose too much detail at
640×480) but **verifies the resolution the driver actually returns** and
logs a warning if it differs; the live MJPEG preview may be downscaled for
bandwidth, but the raw frame used for plate detection and face encoding is
never touched by that downscale.

Changing `plate_*` settings requires restarting the app (the pipeline is
built once and reused — `gate_manager.reset_pipeline()` forces a rebuild
on next use if you need it from a script/REPL).

## 7. Safe character correction

`anpr/postprocessing.py` replaces the old unconditional O↔0/I↔1/S↔5/B↔8/
G↔6/Z↔2 substitution with:

- The raw OCR text is always preserved.
- A character is only corrected when doing so makes the string match one
  of eleven explicit Nigerian plate structures (`PLATE_STRUCTURES`) — not
  "whichever correction happens to look plate-shaped."
- Every correction is recorded (`{"pos", "from", "to"}`).
- `plate_max_corrections` (default 2) caps how many characters may be
  changed; candidates needing more are **rejected**, not force-fit.
- Each correction reduces `combined_confidence`
  (`CORRECTION_CONFIDENCE_PENALTY = 0.08` per character); matching a
  structure adds only a small bonus (`STRUCTURE_MATCH_BONUS = 0.06`) — a
  regex match is evidence, never proof, and can't turn a weak OCR read
  into a confident one by itself (see
  `test_score_candidate_structure_bonus_is_small_not_decisive`).

`normalize_plate_for_storage("ABC-123-DE")` → `"ABC123DE"` (DB lookup key).
`format_plate_for_display("ABC123DE")` → `"ABC-123-DE"` (falls back to the
raw string if no known structure matches, rather than hiding a valid-but-
uncommon plate).

## 8. Multi-frame consensus

`anpr/consensus.py` groups OCR candidates by normalized text and scores
each group by **frame agreement first** (`count * 10.0`), then average
confidence, crop size, and correction count. A candidate is only
`CONFIRMED` when it meets both `plate_consensus_frames` (default 3
distinct agreeing frames) and `plate_min_confirm_confidence` (default
0.55). Similar-but-different readings are kept as separate groups — the
runner-up is recorded under `debug_information.ambiguous_alternative`
rather than silently merged.

## 9. Gate security

- Database lookup (`db.get_open_trip`) is **always** an exact match on the
  normalized plate — no fuzzy matching decides entry/exit.
- `/gate/entry/confirm` requires an explicit `confirm_low_confidence: true`
  before logging a trip whose plate reading was left unedited but was not
  `CONFIRMED` — the entry template shows a confirmation dialog rather than
  silently accepting a low-confidence read (see `gate_entry.html`
  `confirmEntry()`).
- Every trip records `plate_source` (`auto` / `manual_correction` /
  `manual_entry`) and `plate_confidence` — an operator edit is always
  distinguishable from an accepted automatic reading.
- Face/fingerprint verification at exit is unchanged: it still compares
  against *this trip's own* stored biometric (`gate_manager.run_exit_verify`
  / `run_fp_exit`), never a general holders-database search — the ANPR
  changes do not touch this.
- Manual plate entry remains available at every stage — a missing/broken
  model or a `NOT_DETECTED` result never blocks the gate workflow.

## 10. Local evaluation (this gate's own camera/lighting)

The Roboflow dataset's camera angle/lighting/gate geometry may not match
your installation. Build a locally labeled test set from real captures:

```
python tools/build_local_plate_test_set.py label
```

Review each image (open it in a viewer alongside the terminal), enter the
correct transcription (or mark it unusable / a screen-replay), and tag
lighting/distance/tilt/blur/glare/plate-position. Labels are written
incrementally to `datasets/local-gate-evaluation/ground_truth.csv` — never
used for training until reviewed and explicitly moved elsewhere.

```
python tools/build_local_plate_test_set.py evaluate
```

Reports detection recall/precision, character accuracy, exact-match
accuracy, no-read rate, false-positive rate, and average recognition time
— both overall and broken out **per category** (daylight vs. low-light,
near vs. distant, upright vs. tilted, sharp vs. blurry, glare vs. none,
front vs. rear, and screen/replay images evaluated separately) — written
to `reports/local_evaluation_report.json` and
`reports/local_evaluation_summary.md`.

## 11. Debugging

Set `"plate_debug_mode": true` in `config.json` to save, per capture
session, under a timestamped `gate_photos/debug/<ts>_<capture_id>/`
folder: the raw detection crop per frame, the rectified crop, and (in the
result's `debug_information`) every OCR candidate with its preprocessing
method and confidence breakdown.

Debug images can contain personal information (faces, plates) — they are
**off by default**, served only via `/gate/debug/<file>` which 404s
immediately when `plate_debug_mode` is false, and pruned automatically by
`gate_manager.cleanup_debug_artifacts()` (called at app startup) based on
`plate_debug_max_age_hours` / `plate_debug_max_storage_mb`.

## 12. Installation

Development/training workstation:
```
pip install -r requirements.txt
```

Raspberry Pi (inference only — no `roboflow`, no `pytest`):
```
pip install -r requirements-pi.txt
```

Both were pinned against a verified working environment: Python 3.13.5,
Debian 13 (trixie), 64-bit (aarch64) — this repository's own dev
environment is, in fact, a Raspberry Pi 5. If you're on 32-bit Raspberry
Pi OS, the pinned `torch`/`opencv-python` 64-bit wheels will not install —
you'll need 32-bit-compatible builds or `piwheels` equivalents; check
`python3 -c "import platform; print(platform.architecture())"` first.

## 13. Running

```
python app.py
```
(unchanged — `gm.warm_up_ocr()` preloads the OCR backend and
`gm.cleanup_debug_artifacts()` prunes stale debug sessions at startup.)

Systemd unit (adjust `WorkingDirectory`/`User`):
```ini
[Unit]
Description=License Gate ANPR Server
After=network.target

[Service]
WorkingDirectory=/home/pi/database
ExecStart=/home/pi/database/env/bin/python app.py
Restart=on-failure
User=pi

[Install]
WantedBy=multi-user.target
```
Model acquisition (training + export) is always a separate, explicit,
offline step — the Flask process never downloads a model while serving a
request.

## 14. Tests

```
pytest tests/ -q
```

- `tests/test_plate_postprocessing.py` — normalization, display
  formatting, correction budget/rejection, overlay-word rejection,
  confidence scoring.
- `tests/test_plate_consensus.py` — temporal voting, frame-vs-variant
  counting, ambiguous-alternative recording.
- `tests/test_plate_pipeline.py` — end-to-end pipeline with fake
  detector/OCR components (missing model, no detections, tiny crops,
  empty OCR, overlay words, multiple detections, ROI fallback, camera
  resolution negotiation).
- `tests/test_plate_routes.py` — Flask routes: manual entry vs. auto vs.
  correction provenance, the low-confidence confirmation gate, debug-photo
  gating/traversal, model-download-is-disabled-at-runtime.

Only hardware interfaces (camera, YOLO weights, sensor) are mocked/faked;
the database tests use a real temporary SQLite file through
`database.init_db()`.

## 15. Known limitations

- The committed weights (`models/nlpdrs_plate_segment.pt`) are the NLPDRS
  project's model, trained on 300+ Nigerian vehicle images collected in
  Kano State (reported 0.87 precision / 0.91 recall / 0.93 mAP@0.5 on
  their test data). If the file is removed, `gate_manager.plate_model_status()`
  reports `model_present: false` and the app falls back to manual plate
  entry rather than substituting an unrelated model.
- EasyOCR remains the default OCR backend; PaddleOCR is wired up as an
  alternative (`anpr/recognizer.py`) but not benchmarked here — do that
  comparison locally with `tools/build_local_plate_test_set.py evaluate`
  once both are installed.
- The safe-correction structure list (`PLATE_STRUCTURES`) covers the
  common Nigerian formats documented in public references; a plate format
  outside this list will not be auto-corrected (it can still match exactly
  if the OCR read it perfectly) — extend the list if a real, verified
  format is missing.
- NCNN export requires the `ncnn`/`pnnx` toolchain to be available at
  export time; if unavailable, `scripts/export_plate_model.py` reports the
  export error for that format and continues with the others rather than
  failing the whole run.

## 16. Collecting more real gate images

- Prioritize the conditions `tools/build_local_plate_test_set.py evaluate`
  buckets separately: low-light/night, distant vehicles, tilted approach
  angles, and glare (headlight reflection on the plate at night is a
  common Nigerian-gate failure mode not well represented in a
  general-purpose Roboflow dataset).
- Capture both front and rear plates if this gate photographs both.
- Avoid capturing the same vehicle/plate repeatedly in near-identical
  conditions — a large set of near-duplicate frames inflates apparent
  accuracy without adding real signal; `tools/build_local_plate_test_set.py`
  labels by individual file so dedupe before training on any of this data.
- Once ~200-500 locally labeled images exist, consider a second Roboflow
  project (or extending the existing one, if your fork allows it) seeded
  with this gate's own images, so the detector eventually sees this
  gate's specific mounting height/angle/lens distortion during training,
  not just Roboflow's contributor images.
