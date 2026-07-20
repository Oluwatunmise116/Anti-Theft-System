"""
OCR backend abstraction.

Two backends are supported:
  * EasyOCRBackend   — the backend already used by this project. Default,
    because it is already a proven dependency here and needs no native
    build toolchain on Raspberry Pi OS.
  * PaddleOCRBackend — evaluated as a lighter/faster alternative per the
    ANPR audit. Only used if `paddleocr` is installed and explicitly
    selected via config ("plate_ocr_backend": "paddleocr") — it is NOT a
    default dependency because PaddlePaddle's ARM wheels are less uniformly
    available across Raspberry Pi OS versions than EasyOCR/PyTorch's.

Both backends run fully locally — no cloud OCR API is used or required for
normal gate operation, and none should ever be added here.
"""
from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import List, Tuple

PLATE_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


class OCRBackend(ABC):
    name: str = "base"
    # Scene-text OCR benefits from upscaling small crops; plate-specific
    # models resize internally to their own input shape.
    wants_upscale: bool = True
    # Extra context around the detector box, as a fraction of box height.
    # Measured per backend: EasyOCR reads junk from padded regions, while
    # fast-plate-ocr (trained on full plates) fails on tight crops.
    crop_pad_fraction: float = 0.0
    # Reads below this confidence are discarded as candidates. Only raised
    # for backends whose confidence is actually calibrated.
    min_candidate_confidence: float = 0.0

    @abstractmethod
    def recognize(self, image, allowlist: str = PLATE_ALLOWLIST) -> List[Tuple[str, float]]:
        """Return a list of (text, confidence) tuples, confidence in [0, 1]."""
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        raise NotImplementedError


class EasyOCRBackend(OCRBackend):
    name = "easyocr"

    def __init__(self):
        self._reader = None
        self._lock = threading.Lock()

    def _get_reader(self):
        if self._reader is not None:
            return self._reader
        with self._lock:
            if self._reader is None:
                try:
                    import easyocr
                    self._reader = easyocr.Reader(["en"], gpu=False, verbose=False)
                except Exception:
                    self._reader = False
        return self._reader or None

    def is_available(self) -> bool:
        return self._get_reader() is not None

    def warm_up(self):
        threading.Thread(target=self._get_reader, daemon=True).start()

    def recognize(self, image, allowlist: str = PLATE_ALLOWLIST) -> List[Tuple[str, float]]:
        reader = self._get_reader()
        if reader is None:
            return []
        try:
            results = reader.readtext(image, allowlist=allowlist, paragraph=False)
        except Exception:
            return []
        # Left-to-right by box position: EasyOCR's own result order is not
        # reading order, and callers concatenate multi-box plates (e.g.
        # "ABC" + "392BL" read as two boxes).
        results = sorted(results, key=lambda r: min(p[0] for p in r[0]))
        return [(text, float(conf)) for (_bbox, text, conf) in results]


class PaddleOCRBackend(OCRBackend):
    """
    Lightweight alternative backend. Not installed by default — see
    requirements.txt comments and README_ANPR.md for the local benchmark
    procedure used to decide between EasyOCR and PaddleOCR for a given
    deployment.
    """
    name = "paddleocr"

    def __init__(self):
        self._engine = None
        self._lock = threading.Lock()

    def _get_engine(self):
        if self._engine is not None:
            return self._engine
        with self._lock:
            if self._engine is None:
                try:
                    from paddleocr import PaddleOCR
                    self._engine = PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
                except Exception:
                    self._engine = False
        return self._engine or None

    def is_available(self) -> bool:
        return self._get_engine() is not None

    def recognize(self, image, allowlist: str = PLATE_ALLOWLIST) -> List[Tuple[str, float]]:
        engine = self._get_engine()
        if engine is None:
            return []
        try:
            result = engine.ocr(image, cls=False)
        except Exception:
            return []
        out = []
        for line in (result or []):
            for _box, (text, conf) in (line or []):
                out.append((text, float(conf)))
        return out


class FastPlateOCRBackend(OCRBackend):
    """
    Dedicated license-plate recognizer (fast-plate-ocr, CCT-XS global ONNX
    model, ~2MB). Reads the whole plate as one fixed-format string — no box
    splitting, no dropped characters — in ~6ms on a Raspberry Pi 5, vs
    1-2s for EasyOCR. Its per-character probabilities are well calibrated
    (correct reads ~1.0, garbage 0.2-0.4), unlike EasyOCR's, which is why
    this backend can afford a real min_candidate_confidence.
    Trained on full plates including surround, so it needs padded crops
    (crop_pad_fraction) — tight number-line crops misread badly.
    """
    name = "fast_plate_ocr"
    wants_upscale = False
    crop_pad_fraction = 0.3
    min_candidate_confidence = 0.5

    MODEL_NAME = "cct-xs-v1-global-model"

    def __init__(self):
        self._model = None
        self._lock = threading.Lock()

    def _get_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is None:
                try:
                    from fast_plate_ocr import LicensePlateRecognizer
                    self._model = LicensePlateRecognizer(self.MODEL_NAME)
                except Exception:
                    self._model = False
        return self._model or None

    def is_available(self) -> bool:
        return self._get_model() is not None

    def warm_up(self):
        threading.Thread(target=self._get_model, daemon=True).start()

    def recognize(self, image, allowlist: str = PLATE_ALLOWLIST) -> List[Tuple[str, float]]:
        model = self._get_model()
        if model is None:
            return []
        try:
            import numpy as np
            preds = model.run(image, return_confidence=True)
        except Exception:
            return []
        out = []
        for p in preds:
            text = p.plate.replace("_", "")
            if not text:
                continue
            conf = float(np.min(p.char_probs)) if p.char_probs is not None and len(p.char_probs) else 0.0
            out.append((text, conf))
        return out


_backends = {}
_backends_lock = threading.Lock()


def get_ocr_backend(name: str = "easyocr") -> OCRBackend:
    """Lazily construct and cache a single instance per backend name (loaded once, reused)."""
    with _backends_lock:
        if name not in _backends:
            if name == "paddleocr":
                _backends[name] = PaddleOCRBackend()
            elif name == "fast_plate_ocr":
                _backends[name] = FastPlateOCRBackend()
            else:
                _backends[name] = EasyOCRBackend()
        return _backends[name]
