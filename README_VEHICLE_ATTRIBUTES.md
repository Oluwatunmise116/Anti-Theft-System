# Vehicle attributes: colour, body type, brand

Extends the ANPR gate with three advisory vehicle attributes. Read
`README_ANPR.md` first — this subsystem follows its conventions and depends
on its plate detector.

> **Attributes never decide anything.** The existing invariant is that entry
> and exit are decided by an exact match on the normalised plate, with no
> fuzzy matching. This subsystem extends that invariant rather than
> breaching it: a disagreement between the registered vehicle and what the
> camera sees routes to the operator-confirmation dialog, never to a relay.

> **Accuracy is not established.** No attribute model has been trained. The
> numbers in this document are latency, memory and agreement measurements —
> all made on the target Pi. There is no accuracy figure anywhere here, and
> there will not be one until `evaluate-attributes` runs against labelled
> local data.

---

## Architecture

```
attributes/
    models.py            typed results, closed label spaces, statuses
    settings.py          attr_* config schema, validation, safe defaults
    vehicle_detector.py  COCO boxes + the contains-plate-box linkage
    classifiers.py       MobileNetV3-Large backbone, three heads
    logo.py              single-class badge detector (brand, stage 1)
    brand_classifier.py  whole-vehicle BEiT make+model classifier (brand)
    consensus.py         per-attribute temporal voting
    __init__.py          orchestration
```

### Flow

```
best frames (anpr.preprocessing.select_best_frames)
  + the plate box from the existing PlateDetector
    -> COCO vehicle boxes, linked to the plate by CONTAINMENT
    -> vehicle crop -> MobileNetV3-Large -> colour head, type head
    -> brand (attr_brand_backend):
         vehicle: vehicle crop -> BEiT make+model -> summed per make
         logo:    vehicle crop -> logo detector -> badge crop -> brand head
    -> per-attribute voting
    -> VehicleAttributeResult
```

Attributes are computed **after** the plate result has been pushed, so the
barrier never waits on a brand classifier.

### Vehicle localisation is by containment, not by size

The vehicle whose attributes are reported must be the vehicle carrying the
plate that was just read. With two vehicles in frame — routine when a car
queues behind another — choosing the largest or most confident box attaches
the wrong body to the right plate. The result is a confidently wrong colour
and brand on a trip record that looks complete, and the error never appears
in aggregate accuracy because the plate was correct.

So the box is chosen **only** by containment of the plate box, controlled by
`attr_plate_containment_fraction` (default 0.90, tolerating a few pixels of
detector jitter). If no box contains the plate, the result is
`VEHICLE_NOT_LOCALISED`. There is deliberately no fallback to the largest,
nearest or most confident box. When several boxes contain the plate, the
tightest one wins — a containment tie-break, not a size heuristic.

`bus`, `truck` and `motorcycle` from COCO are **authoritative** for those
coarse types and skip the learned head entirely. The head only resolves fine
body style within `car`.

### Brand: whole-vehicle model (default) or two stages

`attr_brand_backend` picks the method: `auto` (default) uses the
whole-vehicle model below when `models/car_brands_beit/` exists, otherwise
the two-stage logo pipeline; `vehicle` and `logo` force one. The rest of
this section describes the logo pipeline.


A badge occupies a few dozen pixels at gate distance, and an end-to-end
classifier latches onto body shape instead. So:

1. a YOLOv8n trained on a single `vehicle_logo` class, run on the **vehicle
   crop** (not the full frame), detects badge **presence** only;
2. the badge crop, upscaled to 224, goes into the brand head.

**No logo, no brand.** The result is `UNKNOWN` with zero confidence. There is
no fallback to classifying the whole body — that is the failure mode this
design exists to avoid. Brand is reported **top-1 and top-3**, never top-1
alone, and it votes only across frames where a badge was actually found, so
its vote count has an honest denominator.

### Label spaces are closed, with real escape classes

```python
COLOURS = ["white","black","silver","grey","red","blue","green","gold",
           "brown","orange","yellow","unknown"]
TYPES   = ["sedan","suv","hatchback","minivan","pickup","bus","truck",
           "motorcycle","tricycle","other"]
BRANDS  = ["toyota","honda","mercedes-benz","lexus","nissan","hyundai","kia",
           "ford","volkswagen","mitsubishi","peugeot","innoson","mazda","bmw",
           "audi","chevrolet","suzuki","other"]
```

`unknown` and `other` are trainable classes, not post-hoc thresholds. A night
frame whose colour genuinely cannot be determined is labelled `unknown`
during labelling and is expected to be predicted as `unknown`. When a head
wins on its escape class the status is `UNKNOWN`, never `CONFIRMED`: the head
declining to commit is an honest non-answer, not a finding.

**Adjust these lists to the vehicle mix at the installation before
training.** A label with no training examples will never be predicted; a
label removed after training invalidates the checkpoint.

### Voting

`attributes/consensus.py` mirrors `anpr/consensus.py` with the same weights
(frame agreement dominates confidence) and the same rule: never `CONFIRMED`
unless both the agreement bar and the confidence bar are met, and the
runner-up is preserved as an alternative. It could not call the plate module
directly — that is typed to `PlateOCRCandidate` / `PlateRecognitionResult`
and scores plate-specific evidence — so the decision rule is reproduced
rather than imported.

### Statuses

`CONFIRMED` · `LOW_CONFIDENCE` · `UNKNOWN` · `VEHICLE_NOT_LOCALISED` ·
`MODEL_UNAVAILABLE` · `FAILED`

---

## Model selection: what was measured

Three candidate checkpoints were tested against this gate's own 113 vehicle
crops before being adopted or rejected. Two failed.

### dima806/vehicle_10_types_image_detection — REJECTED

The card reports ~93%. On this hardware the published checkpoint is
effectively collapsed, and that is not a domain-gap argument:

| Measurement | Result |
| --- | --- |
| Mean top-1 confidence over 113 gate crops | 0.167 (chance is 0.100) |
| Maximum top-1 seen on any crop | 0.200 |
| Output entropy | 98.1% of uniform |
| Crops clearing a 0.50 confidence bar | 0 of 113 |
| Predicted class distribution | `family sedan` on 100 of 113 |
| Logit range on any input | ±0.66 |
| Prediction on **random noise** | `family sedan` |

All weights load with no missing keys and the classifier head has normal
weight statistics, so this is not a loading fault. A model that returns the
same class for random noise as for a real car has no usable signal. My
preprocessing was separately verified to match the repo's own
`AutoImageProcessor` to three decimals, so it is not a preprocessing fault
either.

The author's 93% is on the author's split. It does not transfer here, and
the checkpoint should not be quoted at all.

**Re-checked, including the repo's intermediate checkpoints** (revision
`7619f8d`, via the official `transformers.pipeline`):

* The published `model.safetensors` is byte-identical to `checkpoint-190`
  (epoch 10), not the final epoch-30 `checkpoint-570`.
* The training log explains the flat softmax: final eval loss 1.69 against
  2.30 for a uniform guess, at 90.75% eval accuracy — the logits barely
  grew during training. So low confidence alone is not proof of a broken
  model; the ranking had to be tested separately.
* The ranking fails too. All three weight sets give the same result on 116
  gate crops: `family sedan` 103, `SUV` 7, `minibus` 6. On 28 crops typed
  by eye, **0 of 6 SUVs** (Lexus RX, Ford Escape x3, Qashqai, RAV4) were
  called SUV; 9 of 28 were right overall, all of them sedans.
* Its label space has no hatchback, minivan or pickup, which are 12 of
  those 28 crops (Toyota Matrix, Sienna).

**Consequence:** fine body style (sedan vs SUV vs hatchback) has no working
model. COCO's `bus` / `truck` / `motorcycle` — the highest-value
distinctions at a Nigerian gate — are already authoritative and unaffected.
`car` stays unresolved until a type head is trained locally.

### piotreksl/vehicle-color-recognition — REJECTED

Loads cleanly (0 missing keys) but fails unambiguous input:

| Input | Top-3 |
| --- | --- |
| Pure blue patch | brown 0.14, pink 0.09, black 0.09 |
| Pure white patch | black 0.11, silver 0.11, white 0.09 |
| Pure red patch | red 0.11, pink 0.11, black 0.10 |

Chance over 14 classes is 0.071. A solid patch is out of distribution for a
model trained on photographs, so this alone is suggestive rather than
conclusive — but combined with a mean top-1 of 0.44 on real crops and a
confident `pink` for a red Corolla, it is not shippable. The checkpoint also
carries no class-name list, so the label ordering was an assumption.

### The HSV baseline — ADOPTED, and now the bar

Per your suggestion, `attributes/colour_baseline.py`. It needs no
checkpoint, runs in **2.0 ms median** on a vehicle crop, and its confidence
is the fraction of body pixels supporting the winner — a real quantity, not
a softmax.

| Input | Result |
| --- | --- |
| Ten solid colour patches incl. boundary silver/grey | 10 of 10 exact, confidence 1.00 |
| Uniform random noise, six seeds | never reaches a confirmable confidence |
| Blown-out capture (53% sensor-clipped) | `unknown`, "colour cannot be judged" |

Two design points worth stating, because both came out of measurement:

* **Sensor-clipped pixels do not vote.** On the red Corolla, 56% of body
  pixels had a channel pinned at ≥250 and the paint reads as pale pink in
  the raw data. Counting those votes `white` for a red car. Clipped pixels
  are excluded, and above 35% clipped the answer is `unknown` rather than
  the colour of the glare. This is the same mechanism that will matter for
  direct sun and headlight bloom at a real gate.
* **The palette is pinned to the closed label space.** An early version
  emitted `purple` and `pink`, which are not in `COLOURS`, so correct
  answers silently became `unknown` — while `brown`, which *is* in the
  space, was unreachable. A test now asserts the palette equals the label
  space exactly.

On 113 gate crops it returns a colour for 72% and declines on 28%, the
declines being predominantly blown-out photographs of a monitor.

**This is the number a trained head has to beat.** `attr_colour_backend`
defaults to `auto`: the head when a checkpoint exists, the baseline
otherwise. Compare them per lighting bucket with `evaluate-attributes`
before promoting a head, and expect the head to earn its ~87 ms.

### lamnt2008/car_brands_classification — ADOPTED for brand, advisory

A BEiT-base fine-tuned on 107 make+model classes from the Vietnamese market
(revision `f28052a`, Apache-2.0). It was trained on whole cars, so it takes
the **vehicle crop**. `attributes/brand_classifier.py` sums the 107
probabilities per make onto `BRANDS`; Ferrari and VinFast map to `other`.
Fetch it with `python scripts/prepare_brand_model.py` (inference files
only, pinned revision, ~350 MB); the app never downloads it.

Measured on this gate's crops (the largest COCO car box in each of 115
`gate_photos/vehicle_*` images; brands checked by eye on 28 of them, many
of which are repeat shots of the same few cars, so this is a sanity check
and **not** an accuracy figure):

| Measurement | Result |
| --- | --- |
| Correct make, 28 eyeballed crops | 17 |
| Wrong make at >= 0.55 (would show as CONFIRMED) | 4 — Toyota Matrix -> Mazda 0.94 and 0.55, Matrix -> Kia 0.56, Skoda Octavia -> Kia 0.94 |
| The labelled red Corolla (ConvNeXt said Hyundai 0.71) | Toyota 0.92 |
| Toyota predictions checked | 12 of 12 correct |
| Random noise | Kia ~0.72 on three seeds |
| Latency, fp32, 1-4 threads | 680-880 ms; 1.6 s while the Pi was thermally throttled at 83 C |
| Dynamic int8 | slower (1.27 s) and 11% top-1 disagreement — not used |

What to expect:

* **No Volkswagen, Peugeot, Innoson or Skoda classes.** Those vehicles are
  forced onto another make, sometimes confidently. No confidence threshold
  separates these errors from correct answers.
* The softmax is not calibrated (noise reads as Kia). In the gate the
  input is always a car box that contains the plate, so noise never
  reaches it, but treat CONFIRMED as "likely", not "verified".
* It runs after the plate result is pushed and costs ~0.7 s per frame, so
  it overruns `attr_latency_budget_ms` on its own. With the default one
  attribute frame that only delays the attribute message, never the
  barrier. Raising `attr_capture_frame_count` will not buy more brand
  votes: the budget stops the loop first.
* Brand remains advisory: a mismatch at exit routes to the operator
  dialog, and the UI shows top-3.

### Brand — the logo pipeline

Agreed, and the architecture already reflects the surveillance-logo
literature: a single-class detector for **presence only**, then a separate
classifier on the badge crop, precisely because detector classification
accuracy on 30x30-pixel badges is too low. No badge means `unknown`; the
body crop is never used as a fallback.

Candidates surveyed and why none is wired in:

| Candidate | Why not |
| --- | --- |
| `dima806/car_brands_image_detection` | ~69% headline, but Ford recall 0.08, Kia 0.10, Chevrolet 0.16, BMW 0.18 — the common brands fail |
| `therealcyberlord/stanford-car-vit-patch16` | Stanford Cars: US-market, stops around 2012. Wrong market and a decade stale |
| `Jordo23/vehicle-classifier` | Worth a trial — EfficientNet-B4 on VMMRdb, ships ONNX, and its card is honest that top-5 is the meaningful measure. Make/model granularity is finer than this application's 13-brand space, so it needs a mapping layer |
| `haydarkadioglu/brand-eye` | General brand logos, not car marques |

With `attr_brand_backend: logo`, brand reports `UNKNOWN` until a logo
detector is trained.

## Measured performance

All measured on the target Raspberry Pi 5 (8 GB, aarch64, Debian 13,
Python 3.13.5, torch 2.12.0). Nothing below is estimated.

### Per stage, one frame at 1280x720

| Stage | Median | Note |
| --- | --- | --- |
| COCO vehicle detection, imgsz 640 | 366 ms | default Ultralytics input |
| COCO vehicle detection, imgsz 320 | 109 ms | identical box to within 25 px on the sample |
| Backbone @224, 1 torch thread | 87 ms | |
| Backbone @224, 2 threads | 105 ms | |
| Backbone @224, 4 threads | 165 ms | more threads is **slower** |
| **Full path, imgsz 320 + PyTorch** | **204 ms** | localisation 113 + classify 91 |

Thread count is the counter-intuitive one: small convolutions do not amortise
thread-synchronisation cost, and extra threads also steal cores from the
camera and ANPR workers. `attr_torch_threads` defaults to **1** because that
is what measured fastest, not out of caution.

### Export formats (12 real vehicle crops, `--device-label pi`)

| Format | Size | Load | Mean | Median | P95 | RSS + | Top-1 agreement |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pytorch | 13.48 MB | 351 ms | 75.7 ms | 76.0 ms | 77.8 ms | 8.5 MB | baseline |
| onnx | 13.23 MB | 88 ms | 51.4 ms | 53.1 ms | 54.7 ms | 24.3 MB | 1.000 |
| onnx-int8 | 3.50 MB | 155 ms | 58.5 ms | 59.9 ms | 63.8 ms | 0.0 MB | 1.000 |

Both exports reproduce the PyTorch baseline's top-1 on every head, on every
crop. That is an agreement check, not an accuracy claim about any of them.

NCNN is **not** implemented for this model. Ultralytics' NCNN exporter covers
YOLO models only; a plain classifier needs `pnnx` against the ONNX graph. The
script says so and exports the two ONNX paths rather than pretending.

### Against the 150 ms budget

The **attribute models fit**: 51 ms (ONNX) to 87 ms (PyTorch) per crop.

The **full path does not**, at 204 ms, because the COCO localisation the
design mandates costs ~113 ms of that even at reduced input size. Two
measured routes close the gap:

* switch `attr_model_format` to `onnx` (saves ~40 ms, agreement verified);
* lower `attr_vehicle_detection_imgsz` to 256 (72 ms measured, saves a
  further ~40 ms).

Together those land at roughly 123 ms. Neither is enabled by default:
**recall on small and distant vehicles at reduced detector input size has not
been measured**, and tuning to hit a latency number without measuring the
accuracy cost is exactly the trade this project does not make. Measure the
`distant` bucket with `evaluate-attributes` first.

Because the budget admits one frame at present, `attr_capture_frame_count`
and `attr_consensus_frames` both default to 1. Raise them together once the
per-frame cost drops. The budget mechanism is safe either way: exceeding it
skips remaining frames, which lowers the vote count and yields
`LOW_CONFIDENCE` — never a false `CONFIRMED`.

---

## Camera or uploaded image

Both the entry and exit pages offer **Auto-Detect Vehicle** (live camera) and
**Upload Image** (a still JPEG or PNG). The upload feeds exactly the same
pipeline: plate detection, then the vehicle-attribute pass over the same
image.

What differs is provenance, and it matters. A camera capture is evidence the
vehicle was physically at the gate; an uploaded file is not. So:

* the trip records `vehicle_image_source` as `camera` or `upload`;
* the capture page shows an amber banner while analysing an upload;
* the trip log shows an `uploaded` badge beside the plate.

Uploading authorises nothing. Entry still requires a plate, a low-confidence
plate reading still needs explicit operator confirmation, and exit still
requires a verified face, fingerprint or passcode.

Uploaded bytes are validated by **decoding** them, never by trusting the
filename or content type, and are re-encoded to JPEG before storage. That
normalises the format, strips EXIF (which can carry location and device
metadata the gate has no reason to keep), and means a file that merely looks
like an image cannot be stored under a `.jpg` name. The stored filename is
constructed server-side from the capture id, so a crafted upload name cannot
escape the gate photo directory.

A single still image cannot meet the multi-frame plate consensus bar, so an
uploaded plate comes back `LOW_CONFIDENCE` and requires confirmation. That is
the existing rule, not a new one.

| Key | Default | Meaning |
| --- | --- | --- |
| `gate_upload_enabled` | `true` | Offer the upload path at all |
| `gate_upload_max_mb` | `12` | Per-image size limit |
| `gate_upload_max_pixels` | `40000000` | Resolution ceiling, guarding against a decompression bomb sized to pass the byte limit |

## Configuration

`attr_*` keys in `config.json`, mirroring the `plate_*` convention. Validated
and clamped on read by `attributes/settings.py`; a bad edit degrades one
setting and is reported, never taking the gate down.

| Key | Default | Meaning |
| --- | --- | --- |
| `attr_enabled` | `true` | Master switch |
| `attr_colour_enabled` / `attr_type_enabled` / `attr_brand_enabled` | `true` | Per-head switches |
| `attr_model_path` | `models/vehicle_attributes_mnv3.pt` | Multi-head checkpoint |
| `attr_model_format` | `pytorch` | `pytorch` / `onnx` / `ncnn` |
| `attr_logo_model_path` | `models/vehicle_logo_detector.pt` | Single-class badge detector |
| `attr_input_size` | `224` | Head input |
| `attr_vehicle_detection_imgsz` | `320` | COCO detector input |
| `attr_plate_containment_fraction` | `0.90` | Plate-in-vehicle threshold |
| `attr_logo_detection_confidence` | `0.30` | Badge presence threshold |
| `attr_capture_frame_count` | `1` | Frames given attribute inference |
| `attr_consensus_frames` | `1` | Agreeing frames needed to confirm |
| `attr_colour_min_confidence` / `type` / `brand` | `0.55` | Per-head confidence bars |
| `attr_latency_budget_ms` | `150.0` | Hard ceiling on the whole path |
| `attr_torch_threads` | `1` | Measured optimum |
| `attr_debug_mode` | `false` | Keeps per-frame votes in the result |

---

## Workflow

### 1. Label

```bash
python tools/build_local_plate_test_set.py label
python tools/build_local_plate_test_set.py coverage
```

The existing plate review loop now also asks for colour, type and brand from
the closed lists, plus an optional vehicle box. The operator already has the
image open, so three extra prompts build all three datasets in one pass.
Blank is a valid answer and is recorded as "not labelled" — a guessed label
is worse than a missing one, because it corrupts every metric computed from
the file. Per-category tagging (lighting, distance, tilt, blur, glare,
plate position, screen replay) and the incremental after-every-image write
are unchanged.

### 2. Train

```bash
python scripts/train_vehicle_attributes.py --check-data
python scripts/train_vehicle_attributes.py --colour-data <UFPR-VCR dir> \
    --run-name mnv3_v1 --epochs 30
python scripts/train_vehicle_attributes.py --compare mnv3_v1 mnv3_v2
python scripts/train_vehicle_attributes.py --run-name mnv3_v1 --promote
```

Per-head loss masking is the primary strategy: each batch comes from one
source, batches alternate across sources, and `ignore_index=-1` means a head
sees gradient only from rows that label it. `--strategy frozen` is the
documented fallback (colour trains the backbone, then it is frozen and the
other heads train on frozen features).

Colour fine-tunes on **UFPR-VCR** (10,039 images; frontal and rear,
occlusions, varied lighting, and nighttime scenes — the nighttime coverage is
why it is preferred over VCoR here). It is not redistributable: request it
from the authors and pass `--colour-data`. Type and brand fine-tune on
locally labelled gate captures, because no public dataset matches the
Nigerian vehicle mix — Stanford Cars stops around 2012 and is US-market.

**Refusals.** Training is refused for a head with fewer than 200 examples or
fewer than 3 classes holding 20+ examples. Promotion is refused for any head
whose validation accuracy fails to beat chance by 0.10. Both print exactly
what is missing.

### 3. Export and benchmark, on the Pi

```bash
python scripts/export_plate_model.py --attributes \
    --images gate_photos --device-label pi
```

Extends the existing export script rather than duplicating it. Adds INT8
dynamic quantisation and replaces IoU box agreement with **top-1 label
agreement** against the PyTorch baseline, since a classifier has no boxes.
Reports latency, load time and peak RSS per format, measured on whatever
machine it runs on — run it on the Pi.

### 4. Evaluate

```bash
python tools/build_local_plate_test_set.py evaluate-attributes
```

Reports per attribute, and broken out by the existing lighting and distance
buckets: accuracy over answered images and over all images, macro precision /
recall / F1, the full per-class table, confusion matrices, the escape-class
(`unknown` / `other`) rate, and brand top-1 **and** top-3.

An abstention is never scored as correct. The escape rate is reported
separately because a model that never says `unknown` at night is broken
regardless of its headline accuracy.

---

## Integration rules

* Attributes are advisory flags for a human operator, never inputs to
  allow/deny logic.
* A mismatch between the registered vehicle and the observed attributes sets
  `requires_operator_confirmation` on the exit lookup and routes to the
  existing low-confidence confirmation dialog. It never auto-denies. Given
  realistic brand accuracy, auto-denying on a brand mismatch would strand
  legitimate residents weekly.
* Attribute mismatch is the actual anti-theft value: it is the signal that
  catches a cloned plate on a different vehicle. That is precisely why it
  must reach a human rather than a relay.
* Every stored attribute carries its own confidence, vote count and a
  `*_source` column using the same vocabulary as `plate_source`
  (`auto` / `manual_correction` / `manual_entry`). An operator edit is always
  distinguishable from an automatic reading, in the database and in the UI.
* A missing or broken attribute model never blocks the gate, exactly as a
  missing plate model does not.
* Corrections are constrained to the closed label spaces: an operator cannot
  widen a label space through a route.

---

## Database

Additive, idempotent migration in `database.init_db()`. The existing
colour/type/brand value and confidence columns are reused; these are added:

```
vehicle_image_source    (camera | upload)
vehicle_color_source    vehicle_color_votes
vehicle_type_source     vehicle_type_votes
vehicle_brand_source    vehicle_brand_votes
attr_status             attr_coco_class        attr_vehicle_box_json
attr_model_version      attr_processing_ms     attr_mismatch_flags_json
```

No trip record is deleted or rewritten, and a database from any earlier
version migrates in place. `close_trip()` still clears biometrics and photos,
but keeps the attribute columns: they are non-biometric observations forming
the anti-theft audit trail.

---

## Troubleshooting

**Everything reports `MODEL_UNAVAILABLE`.** No checkpoint has been trained.
Expected until `train_vehicle_attributes.py --promote` succeeds. The coarse
COCO type (bus / truck / motorcycle) still works, because that comes from a
trained detector rather than the stubbed heads.

**Brand is always `UNKNOWN`.** Either no logo detector is present, or no
badge was found. By design, brand is never guessed from the body crop.

**`VEHICLE_NOT_LOCALISED`.** No vehicle box contained the plate box. Check
`attr_plate_containment_fraction`, and check the `distant` bucket in
`evaluate-attributes` if `attr_vehicle_detection_imgsz` has been lowered.

**Nothing is ever `CONFIRMED`.** `attr_consensus_frames` is higher than the
number of frames the latency budget actually admits. See the budget section.

**Attributes are slow.** Check `attr_torch_threads` is 1, not 4.

---

## Superseded implementation

An earlier Hugging Face based implementation (three separate models) lives in
`superseded/` and is imported by nothing. It was moved rather than deleted
because it was never committed to git. See `superseded/README.md`.
