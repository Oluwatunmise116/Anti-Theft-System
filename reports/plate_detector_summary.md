# Nigerian Plate Detector — Training Summary

Detector (bounding-box) metrics only. OCR/plate-reading accuracy is a separate, later-stage measurement — see tools/build_local_plate_test_set.py.

| Run | imgsz | epochs | mAP50 | mAP50-95 | precision | recall |
|---|---|---|---|---|---|---|
| ng_plate_test | 640 | 1 | 0.9950 | 0.7968 | 1.0000 | 0.9948 |

## How to read this
- mAP50 / mAP50-95 / precision / recall are computed by Ultralytics on the **validation split only** (never the training split).
- These numbers describe how reliably the model draws a box around a plate — they say nothing about whether the OCR stage reads the text correctly.
- False positives/negatives should be spot-checked visually in each run's `val_batch*_pred.jpg` files before promoting a model to production.
