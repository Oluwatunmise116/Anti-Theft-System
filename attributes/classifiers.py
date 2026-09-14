"""
One MobileNetV3-Large backbone, three linear heads.

WHY ONE BACKBONE
----------------
Colour, body type and brand are three questions about the same crop that
share almost all of their useful features. Three separately fine-tuned
classifiers would mean three forward passes, roughly 40x the parameters,
and a latency budget blown by an order of magnitude on a CPU-only Pi.

HOW THE BACKBONE IS RUN
-----------------------
Twice per frame, not once, because the two inputs differ:

  vehicle crop (224x224) -> backbone -> colour head, type head
  badge crop   (224x224) -> backbone -> brand head

A badge occupies a few dozen pixels at gate distance; feeding the
whole-body crop to the brand head is what makes end-to-end brand
classifiers collapse. The second pass is only paid on frames where the
logo detector actually found a badge.

UNTRAINED WEIGHTS
-----------------
Until scripts/train_vehicle_attributes.py has produced a checkpoint, this
loads a randomly initialised backbone so integration, tests and latency
measurement all work. A stub model NEVER reports a label: `trained` is
False and the pipeline converts every head's output into UNKNOWN with an
explicit reason. Random weights have the same compute cost as trained ones,
so latency measured against a stub is a real measurement of this
architecture; accuracy against a stub is meaningless and is not reported.
"""
from __future__ import annotations

import logging
import os
import threading
from typing import List, Optional, Sequence, Tuple

from .models import BRANDS, COLOURS, TYPES, ScoredLabel

log = logging.getLogger("attributes.classifiers")

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

HEAD_LABELS = {"colour": COLOURS, "type": TYPES, "brand": BRANDS}


class AttributeModelError(RuntimeError):
    """Raised when the configured attribute checkpoint is unusable."""


# ── preprocessing ─────────────────────────────────────────────────────────

def is_valid_crop(image) -> bool:
    if image is None:
        return False
    try:
        if getattr(image, "size", 0) == 0:
            return False
        shape = image.shape
    except AttributeError:
        return False
    return len(shape) == 3 and shape[2] == 3 and shape[0] >= 2 and shape[1] >= 2


def preprocess(crop_bgr, size: int = 224):
    """
    BGR crop -> normalized float32 NCHW array of shape (1, 3, size, size).

    Deterministic: fixed resize, fixed normalization, no augmentation, so
    the same crop always yields the same tensor. Raises ValueError on an
    unusable crop so the caller returns a structured FAILED value.
    """
    import cv2
    import numpy as np

    if not is_valid_crop(crop_bgr):
        raise ValueError("empty or malformed crop")

    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    interp = cv2.INTER_AREA if (h > size or w > size) else cv2.INTER_LINEAR
    rgb = cv2.resize(rgb, (size, size), interpolation=interp)

    arr = np.ascontiguousarray(rgb).astype(np.float32) / 255.0
    arr = (arr - np.asarray(IMAGENET_MEAN, dtype=np.float32)) / \
          np.asarray(IMAGENET_STD, dtype=np.float32)
    return np.ascontiguousarray(arr.transpose(2, 0, 1)[None, ...])


def softmax(logits):
    import numpy as np
    x = np.asarray(logits, dtype=np.float64).reshape(-1)
    x = x - x.max()
    e = np.exp(x)
    return (e / e.sum()).astype("float32")


def top_k(probabilities, labels: Sequence[str], k: int = 3) -> List[ScoredLabel]:
    import numpy as np
    probs = np.asarray(probabilities, dtype="float64").reshape(-1)
    k = max(1, min(k, probs.shape[0]))
    return [ScoredLabel(labels[int(i)] if int(i) < len(labels) else f"class_{int(i)}",
                        float(probs[int(i)]))
            for i in np.argsort(-probs)[:k]]


# ── architecture ──────────────────────────────────────────────────────────

def build_network(num_colours: int, num_types: int, num_brands: int,
                  pretrained: bool = False):
    """
    MobileNetV3-Large feature extractor with three independent linear heads.

    Defined here rather than in the training script so inference and
    training construct byte-identical architectures — a mismatch between
    the two is the classic cause of a checkpoint that loads with
    "unexpected keys" and then predicts noise.
    """
    import torch
    import torch.nn as nn
    from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

    class MultiHeadVehicleNet(nn.Module):
        #: Bumped when the architecture changes in a way that invalidates
        #: existing checkpoints.
        ARCH_VERSION = 1

        def __init__(self):
            super().__init__()
            weights = MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
            base = mobilenet_v3_large(weights=weights)
            self.features = base.features
            self.avgpool = base.avgpool
            feature_dim = base.classifier[0].in_features      # 960
            # One shared projection, then three independent heads. The
            # projection is what the per-head loss masking trains jointly.
            self.shared = nn.Sequential(
                nn.Linear(feature_dim, 512), nn.Hardswish(inplace=True),
                nn.Dropout(0.2))
            self.colour_head = nn.Linear(512, num_colours)
            self.type_head = nn.Linear(512, num_types)
            self.brand_head = nn.Linear(512, num_brands)

        def embed(self, pixel_values):
            x = self.features(pixel_values)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            return self.shared(x)

        def forward(self, pixel_values, heads=("colour", "type", "brand")):
            shared = self.embed(pixel_values)
            out = {}
            if "colour" in heads:
                out["colour"] = self.colour_head(shared)
            if "type" in heads:
                out["type"] = self.type_head(shared)
            if "brand" in heads:
                out["brand"] = self.brand_head(shared)
            return out

    return MultiHeadVehicleNet()


def checkpoint_payload(model, label_spaces: dict, metrics: dict = None) -> dict:
    """The checkpoint shape both the trainer writes and the loader expects."""
    return {
        "arch": "mobilenet_v3_large_multihead",
        "arch_version": getattr(model, "ARCH_VERSION", 1),
        "state_dict": model.state_dict(),
        "label_spaces": label_spaces,
        "input_size": 224,
        "mean": list(IMAGENET_MEAN),
        "std": list(IMAGENET_STD),
        "metrics": metrics or {},
    }


# ── inference wrapper ─────────────────────────────────────────────────────

class AttributeClassifier:
    """
    Thread-safe wrapper around the multi-head network.

    Mirrors anpr.detector.PlateDetector: loaded once behind a lock, reused
    for the life of the process, and never downloaded at runtime.
    """

    def __init__(self, model_path: str, input_size: int = 224,
                 model_format: str = "pytorch", threads: int = 4,
                 allow_stub: bool = True):
        self.model_path = model_path
        self.input_size = input_size
        self.model_format = model_format
        self.threads = threads
        self.allow_stub = allow_stub
        self.trained = False
        self.label_spaces = dict(HEAD_LABELS)
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD
        self.version = "unloaded"
        self._model = None
        self._session = None
        self._lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._load_error: Optional[str] = None

    # ── loading ───────────────────────────────────────────────────────────
    def _load(self):
        if self._model is not None or self._session is not None:
            return True
        with self._lock:
            if self._model is not None or self._session is not None:
                return True
            try:
                if self.model_format == "onnx":
                    self._load_onnx()
                else:
                    self._load_pytorch()
            except Exception as exc:
                self._load_error = f"{type(exc).__name__}: {exc}"
                log.warning("Attribute classifier load failed: %s", self._load_error)
                return False
        return True

    def _load_pytorch(self):
        import torch

        try:
            torch.set_num_threads(int(self.threads))
        except Exception:
            pass

        present = bool(self.model_path) and os.path.exists(self.model_path)
        if not present and not self.allow_stub:
            raise AttributeModelError(
                f"attribute checkpoint not found at '{self.model_path}'. Train it "
                "with scripts/train_vehicle_attributes.py --promote. Nothing is "
                "downloaded at runtime.")

        if present:
            # weights_only=True: a checkpoint is data, never code.
            checkpoint = torch.load(self.model_path, map_location="cpu", weights_only=True)
            spaces = checkpoint.get("label_spaces") or {}
            if spaces:
                self.label_spaces = {k: list(v) for k, v in spaces.items()}
            self.input_size = int(checkpoint.get("input_size", self.input_size))
            if isinstance(checkpoint.get("mean"), (list, tuple)):
                self.mean = tuple(checkpoint["mean"])
            if isinstance(checkpoint.get("std"), (list, tuple)):
                self.std = tuple(checkpoint["std"])

            model = build_network(len(self.label_spaces["colour"]),
                                  len(self.label_spaces["type"]),
                                  len(self.label_spaces["brand"]))
            missing, unexpected = model.load_state_dict(
                checkpoint["state_dict"], strict=False)
            if missing:
                raise AttributeModelError(
                    f"checkpoint does not fit the current architecture: "
                    f"{len(missing)} missing key(s), first={missing[0]}")
            model.eval()
            self._model = model
            self.trained = True
            self.version = (f"{os.path.basename(self.model_path)}"
                            f"@arch{checkpoint.get('arch_version', '?')}")
        else:
            # Stub: real compute, no claims. See the module docstring.
            model = build_network(len(COLOURS), len(TYPES), len(BRANDS))
            model.eval()
            self._model = model
            self.trained = False
            self.version = "untrained-stub"
            log.info("No attribute checkpoint at '%s' — running an untrained stub. "
                     "Every attribute will be reported UNKNOWN.", self.model_path)

    def _load_onnx(self):
        import json

        import onnxruntime as ort

        if not self.model_path or not os.path.exists(self.model_path):
            raise AttributeModelError(
                f"ONNX attribute model not found at '{self.model_path}'. Export it "
                "with scripts/export_plate_model.py --attributes.")
        options = ort.SessionOptions()
        options.intra_op_num_threads = int(self.threads)
        self._session = ort.InferenceSession(
            self.model_path, options, providers=["CPUExecutionProvider"])
        # Label spaces travel in the ONNX metadata; a graph carries no
        # vocabulary of its own and guessing one would mislabel everything.
        meta = self._session.get_modelmeta().custom_metadata_map or {}
        if meta.get("label_spaces"):
            self.label_spaces = json.loads(meta["label_spaces"])
        if meta.get("input_size"):
            self.input_size = int(meta["input_size"])
        self.trained = meta.get("trained", "false").lower() == "true"
        self.version = f"{os.path.basename(self.model_path)}@onnx"

    # ── status ────────────────────────────────────────────────────────────
    def is_ready(self) -> bool:
        return self._load()

    def status(self) -> dict:
        present = bool(self.model_path) and os.path.exists(self.model_path)
        return {
            "model_path": self.model_path,
            "model_format": self.model_format,
            "model_present": present,
            "model_loaded": self._model is not None or self._session is not None,
            "trained": self.trained,
            "version": self.version,
            "input_size": self.input_size,
            "load_error": self._load_error,
        }

    # ── inference ─────────────────────────────────────────────────────────
    def predict(self, crop_bgr, heads: Tuple[str, ...] = ("colour", "type")
                ) -> dict:
        """
        One backbone pass over `crop_bgr`, returning {head: [ScoredLabel, ...]}.

        `heads` selects which linear heads are evaluated; the backbone cost
        is identical either way, so this only avoids three trivial matmuls.
        Raises ValueError on an unusable crop and AttributeModelError when
        no model could be loaded.
        """
        if not self._load():
            raise AttributeModelError(self._load_error or "attribute model unavailable")

        array = preprocess(crop_bgr, self.input_size)

        if self._session is not None:
            name = self._session.get_inputs()[0].name
            with self._infer_lock:
                outputs = self._session.run(None, {name: array})
            names = [o.name for o in self._session.get_outputs()]
            logits = {n: outputs[i] for i, n in enumerate(names)}
            return {h: top_k(softmax(logits[h]), self.label_spaces[h])
                    for h in heads if h in logits}

        import torch
        tensor = torch.from_numpy(array)
        with self._infer_lock, torch.inference_mode():
            out = self._model(tensor, heads=heads)
        return {h: top_k(torch.softmax(v.float(), dim=-1)[0].cpu().numpy(),
                         self.label_spaces[h])
                for h, v in out.items()}
