"""
Whole-vehicle brand classifier: lamnt2008/car_brands_classification.

WHAT THE MODEL IS
-----------------
A BEiT-base (microsoft/beit-base-patch16-224-pt22k-ft22k) fine-tuned on
107 make+model classes from the Vietnamese market — `ToyotaCorollaAltis`,
`FordRanger`, `LexusRX` and so on. It was trained on whole-car images, so it
is fed the VEHICLE crop, not a badge crop. That is the opposite of the
two-stage logo design in logo.py, and is selected by `attr_brand_backend`.

This module collapses the 107 make+model probabilities onto the closed
BRANDS label space by summing per make. The model-level label is discarded:
on this gate's crops it names the right make far more often than the right
model (a Corolla is routinely called a Yaris).

KNOWN LIMITS (measured on this gate's crops, see README_VEHICLE_ATTRIBUTES.md)
-----------------------------------------------------------------------------
* It has NO Volkswagen, Peugeot, Innoson or Skoda classes. Such a vehicle is
  forced onto some other make, sometimes at high confidence (a Skoda
  Octavia read as Kia at 0.94). No threshold separates these errors.
* Random noise reads as Kia at ~0.7. In the gate this cannot happen — the
  input is always a COCO car box that contains the plate — but it means the
  softmax is not a calibrated confidence.
* ~750 ms per crop on the Pi 5, fp32, one thread. More threads did not help
  and dynamic int8 was slower. It runs after the plate result is pushed, so
  it never delays the barrier.

Nothing here downloads a model: fetch the weights with
scripts/prepare_brand_model.py. Loading uses local_files_only.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import List, Optional

from .models import BRANDS, ScoredLabel

log = logging.getLogger("attributes.brand_classifier")

MODEL_ID = "lamnt2008/car_brands_classification"
#: The exact commit prepare_brand_model.py fetches and that was evaluated.
MODEL_REVISION = "f28052ad70f4988eabd619b455f6fec3211afcfc"

#: Model label prefix -> BRANDS label. Longest prefixes first so that
#: "MercedesBenz" is not shadowed by anything shorter. A make missing from
#: BRANDS maps to the escape class `other`.
MAKE_PREFIXES = (
    ("MercedesBenz", "mercedes-benz"),
    ("Mitsubishi", "mitsubishi"),
    ("Chevrolet", "chevrolet"),
    ("Hyundai", "hyundai"),
    ("Ferrari", "other"),
    ("Vinfast", "other"),
    ("Toyota", "toyota"),
    ("Suzuki", "suzuki"),
    ("Nissan", "nissan"),
    ("Honda", "honda"),
    ("Lexus", "lexus"),
    ("Mazda", "mazda"),
    ("Audi", "audi"),
    ("Ford", "ford"),
    ("BMW", "bmw"),
    ("Kia", "kia"),
)


class BrandModelError(RuntimeError):
    """Raised when the configured brand model is missing or unusable."""


def brand_for_model_label(label: str) -> str:
    """`ToyotaCorollaAltis` -> `toyota`. Unrecognised makes -> `other`."""
    for prefix, brand in MAKE_PREFIXES:
        if label.startswith(prefix):
            return brand if brand in BRANDS else "other"
    return "other"


def aggregate_to_brands(probabilities, id2label: dict, k: int = 3) -> List[ScoredLabel]:
    """Sum make+model probabilities per brand; top `k` brands, best first."""
    totals = {}
    for index, p in enumerate(probabilities):
        label = id2label.get(index, id2label.get(str(index), ""))
        brand = brand_for_model_label(label)
        totals[brand] = totals.get(brand, 0.0) + float(p)
    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)
    return [ScoredLabel(brand, confidence) for brand, confidence in ranked[:max(1, k)]]


class VehicleBrandClassifier:
    """
    Thread-safe wrapper around the BEiT make+model classifier.

    Mirrors AttributeClassifier: loaded lazily once behind a lock, reused
    for the life of the process, never downloaded at runtime.
    """

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.input_size = 224
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)
        self.id2label: dict = {}
        self.version = "unloaded"
        self._model = None
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._load_error: Optional[str] = None

    # ── model access ──────────────────────────────────────────────────────
    def model_present(self) -> bool:
        return bool(self.model_dir) and os.path.exists(
            os.path.join(self.model_dir, "config.json"))

    def _load(self) -> bool:
        if self._model is not None:
            return True
        if self._load_error is not None:
            return False
        with self._lock:
            if self._model is not None:
                return True
            try:
                if not self.model_present():
                    raise BrandModelError(
                        f"Brand model not found at '{self.model_dir}'. Fetch it with "
                        "scripts/prepare_brand_model.py. Nothing is downloaded at "
                        "runtime.")
                from transformers import BeitForImageClassification

                model = BeitForImageClassification.from_pretrained(
                    self.model_dir, local_files_only=True)
                model.eval()
                self.id2label = {int(k): v for k, v in model.config.id2label.items()}
                self._read_preprocessor_config()
                self._model = model
                self.version = f"{MODEL_ID}@{MODEL_REVISION[:7]}"
            except Exception as exc:
                self._load_error = f"{type(exc).__name__}: {exc}"
                log.warning("Brand model load failed: %s", self._load_error)
                return False
        return True

    def _read_preprocessor_config(self):
        path = os.path.join(self.model_dir, "preprocessor_config.json")
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as fh:
            cfg = json.load(fh)
        size = cfg.get("size") or {}
        self.input_size = int(size.get("height", self.input_size))
        if isinstance(cfg.get("image_mean"), list):
            self.mean = tuple(cfg["image_mean"])
        if isinstance(cfg.get("image_std"), list):
            self.std = tuple(cfg["image_std"])

    def is_ready(self) -> bool:
        return self._load()

    def status(self) -> dict:
        return {
            "model_id": MODEL_ID,
            "model_dir": self.model_dir,
            "model_present": self.model_present(),
            "model_loaded": self._model is not None,
            "version": self.version,
            "load_error": self._load_error,
        }

    # ── inference ─────────────────────────────────────────────────────────
    def predict(self, vehicle_crop_bgr, k: int = 3) -> List[ScoredLabel]:
        """
        Top-`k` brands for a whole-vehicle crop.

        Raises ValueError on an unusable crop and BrandModelError when no
        model could be loaded, so the caller records a structured failure.
        """
        if not self._load():
            raise BrandModelError(self._load_error or "brand model unavailable")

        import cv2
        import numpy as np
        import torch

        from .classifiers import is_valid_crop

        if not is_valid_crop(vehicle_crop_bgr):
            raise ValueError("empty or malformed crop")

        # Matches the repo's BeitImageProcessor: plain resize (bilinear),
        # no centre crop, rescale to [0,1], normalise with mean/std 0.5.
        rgb = cv2.cvtColor(vehicle_crop_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.input_size, self.input_size),
                         interpolation=cv2.INTER_LINEAR)
        arr = rgb.astype(np.float32) / 255.0
        arr = (arr - np.asarray(self.mean, dtype=np.float32)) / \
            np.asarray(self.std, dtype=np.float32)
        tensor = torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)[None]))

        with self._infer_lock, torch.inference_mode():
            logits = self._model(pixel_values=tensor).logits
        probs = torch.softmax(logits.float(), dim=-1)[0].cpu().numpy()
        return aggregate_to_brands(probs, self.id2label, k=k)
