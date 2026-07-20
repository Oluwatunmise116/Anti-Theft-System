"""
Plate crop quality assessment, rectification and a small set of OCR
preprocessing variants.

We deliberately do NOT run every conceivable transform and keep whichever
produces "some" text — each variant is generated because it addresses a
specific, known failure mode (glare, low contrast, skew), and every OCR
result downstream retains which variant produced it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    import cv2
    import numpy as np
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False


@dataclass
class CropQuality:
    width: int
    height: int
    sharpness: float          # variance of Laplacian
    contrast: float           # stddev of grayscale intensity
    mean_brightness: float
    overexposed_fraction: float
    underexposed_fraction: float
    glare_fraction: float
    ok: bool
    rejection_reason: Optional[str] = None


def assess_crop_quality(
    bgr_crop,
    min_width: int = 60,
    min_height: int = 18,
    min_sharpness: float = 25.0,
    max_overexposed_fraction: float = 0.75,
    max_glare_fraction: float = 0.70,
    max_aspect_ratio: float = 7.0,
    min_aspect_ratio: float = 1.3,
) -> CropQuality:
    """Score a plate crop and decide whether it's worth running OCR on."""
    if not CV_AVAILABLE or bgr_crop is None or bgr_crop.size == 0:
        return CropQuality(0, 0, 0, 0, 0, 0, 0, 0, ok=False, rejection_reason="empty_crop")

    h, w = bgr_crop.shape[:2]
    if w < min_width or h < min_height:
        return CropQuality(w, h, 0, 0, 0, 0, 0, 0, ok=False, rejection_reason="crop_too_small")

    aspect = w / max(h, 1)
    if aspect < min_aspect_ratio or aspect > max_aspect_ratio:
        return CropQuality(w, h, 0, 0, 0, 0, 0, 0, ok=False,
                            rejection_reason=f"unrealistic_aspect_ratio:{aspect:.2f}")

    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    brightness = float(gray.mean())

    total_px = gray.size
    overexposed = float(np.count_nonzero(gray >= 250)) / total_px
    underexposed = float(np.count_nonzero(gray <= 8)) / total_px

    # Fraction of near-saturated pixels. Nigerian plates have a white/
    # reflective background, so a well-lit, in-focus plate legitimately runs
    # 40-65% near-white — that is not glare obscuring the text, it's the
    # plate. The threshold above only trips on genuine, near-total blowout.
    _, bright_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
    glare_fraction = float(np.count_nonzero(bright_mask)) / total_px

    reasons = []
    if sharpness < min_sharpness:
        reasons.append(f"blurry:sharpness={sharpness:.1f}")
    if overexposed > max_overexposed_fraction:
        reasons.append(f"overexposed:{overexposed:.2f}")
    if glare_fraction > max_glare_fraction:
        reasons.append(f"glare:{glare_fraction:.2f}")
    if contrast < 12:
        reasons.append(f"low_contrast:{contrast:.1f}")

    ok = len(reasons) == 0
    return CropQuality(
        width=w, height=h, sharpness=sharpness, contrast=contrast,
        mean_brightness=brightness, overexposed_fraction=overexposed,
        underexposed_fraction=underexposed, glare_fraction=glare_fraction,
        ok=ok, rejection_reason=(";".join(reasons) if reasons else None),
    )


def pad_and_clamp_box(box: Tuple[int, int, int, int], pad: int,
                       frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Pad a bounding box and clamp to frame bounds. frame_shape = (h, w)."""
    x1, y1, x2, y2 = box
    h, w = frame_shape[:2]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return x1, y1, x2, y2


def rectify_plate(bgr_crop):
    """
    Attempt perspective correction on the plate's outer quadrilateral;
    fall back to a mild deskew; fall back to the original crop unchanged.
    Never returns None — the unmodified crop is always a safe fallback.
    """
    if not CV_AVAILABLE or bgr_crop is None or bgr_crop.size == 0:
        return bgr_crop

    h, w = bgr_crop.shape[:2]
    crop_area = h * w

    try:
        gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_quad = None
        best_area = 0
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if area > best_area and area > 0.55 * crop_area:
                    best_area = area
                    best_quad = approx

        if best_quad is not None:
            pts = best_quad.reshape(4, 2).astype("float32")
            pts = _order_quad_points(pts)
            (tl, tr, br, bl) = pts
            width_a = np.linalg.norm(br - bl)
            width_b = np.linalg.norm(tr - tl)
            max_w = int(max(width_a, width_b))
            height_a = np.linalg.norm(tr - br)
            height_b = np.linalg.norm(tl - bl)
            max_h = int(max(height_a, height_b))
            if max_w > 20 and max_h > 8:
                dst = np.array([[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]],
                                dtype="float32")
                M = cv2.getPerspectiveTransform(pts, dst)
                return cv2.warpPerspective(bgr_crop, M, (max_w, max_h))
    except Exception:
        pass

    # Fall back: mild deskew via minAreaRect angle (only for small angles —
    # large "corrections" are more likely noise than real skew).
    try:
        gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        coords = cv2.findNonZero(thresh)
        if coords is not None and len(coords) > 20:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            if 0.5 < abs(angle) <= 12:
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                return cv2.warpAffine(bgr_crop, M, (w, h), flags=cv2.INTER_CUBIC,
                                       borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        pass

    return bgr_crop


def _order_quad_points(pts):
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")


def frame_sharpness(bgr_frame) -> float:
    """Variance-of-Laplacian sharpness for a full camera frame (not a plate crop)."""
    if not CV_AVAILABLE or bgr_frame is None or bgr_frame.size == 0:
        return 0.0
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_best_frames(frames: List, count: int = 5,
                        max_overexposed_fraction: float = 0.5) -> List:
    """
    Rank candidate frames captured during a stable-vehicle window by
    sharpness, discard frames that are badly over/underexposed, and return
    the top `count`. We deliberately do NOT pick a frame merely because a
    vehicle/plate detector was confident on it — that is a separate, later
    stage.
    """
    if not CV_AVAILABLE or not frames:
        return list(frames)[:count]

    scored = []
    for f in frames:
        if f is None or f.size == 0:
            continue
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        overexposed = float(np.count_nonzero(gray >= 250)) / gray.size
        if overexposed > max_overexposed_fraction:
            continue
        scored.append((frame_sharpness(f), f))

    if not scored:
        return list(frames)[:count]

    scored.sort(key=lambda t: t[0], reverse=True)
    return [f for _, f in scored[:count]]


def generate_variants(bgr_crop) -> List[Tuple[str, "np.ndarray"]]:
    """
    Return a small, deliberate set of (method_name, image) OCR inputs.
    Each variant targets a specific failure mode rather than being an
    arbitrary transform stacked "just in case".
    """
    if not CV_AVAILABLE or bgr_crop is None or bgr_crop.size == 0:
        return []

    variants: List[Tuple[str, "np.ndarray"]] = [("original_color", bgr_crop)]

    gray = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)

    # Upscale small crops before further processing — improves OCR on tiny plates
    h, w = gray.shape[:2]
    if w < 320:
        scale = 320 / w
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    variants.append(("grayscale", gray))

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    variants.append(("clahe_gray", enhanced))

    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    variants.append(("unsharp_mask", sharpened))

    adaptive = cv2.adaptiveThreshold(
        enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    variants.append(("adaptive_threshold", adaptive))

    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu_threshold", otsu))

    # Dark-background plates need inversion — only add it if Otsu suggests
    # the plate is mostly dark (light text on dark background).
    if float(np.mean(otsu)) < 100:
        variants.append(("otsu_threshold_inverted", cv2.bitwise_not(otsu)))

    return variants
