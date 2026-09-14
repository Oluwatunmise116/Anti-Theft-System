"""
HSV-histogram colour baseline over the vehicle body crop.

WHY THIS EXISTS
---------------
It is the number a learned colour head has to beat before it earns its
place on the Pi. It needs no training data, no checkpoint and no download,
it runs in about a millisecond, and it is auditable — every decision traces
to a pixel count rather than to weights nobody can inspect.

It is also the working colour path today, because no colour head has been
trained yet.

HOW IT DECIDES
--------------
1.  Crop the BODY region out of the vehicle box: drop the top (roof,
    windscreen, sky) and the bottom (wheels, shadow, road), and inset
    horizontally. Those regions are where a naive average goes wrong.
2.  Classify every body pixel as achromatic (white / black / silver / grey,
    by value and saturation) or chromatic (by hue).
3.  The winning colour is the one with the most supporting pixels.
4.  `confidence` is the FRACTION of body pixels that voted for it. That is
    a real, interpretable quantity — unlike a softmax over classes, a
    2-of-3 pixel majority genuinely means the model is unsure.
5.  Below `min_support` the answer is `unknown`, which is a real class in
    this application's label space, not a threshold dressed up as one.

KNOWN LIMITS, STATED UP FRONT
-----------------------------
Colour constancy under sodium and LED gate lighting is exactly what this
cannot do: at night everything drifts toward amber and low saturation, and
a pixel-counting method will call it silver or gold. That is the failure
mode a model trained on UFPR-VCR (which includes nighttime scenes) should
beat. Measure it per lighting bucket with
`tools/build_local_plate_test_set.py evaluate-attributes` before believing
either one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .models import COLOURS, ScoredLabel

#: Body region of the vehicle box, as fractions (top, bottom, left, right).
#: Excludes roof/windscreen and wheels/shadow/road.
BODY_REGION = (0.18, 0.78, 0.08, 0.92)

#: Hue bands in OpenCV's 0-179 scale -> application colour.
#:
#: Every name here MUST be in attributes.models.COLOURS. That closed list
#: has no purple and no pink, so:
#:   * pink hues (160-172) are read as red, which is what a faded or
#:     sun-bleached red car actually is;
#:   * purple hues (135-160) are left UNCLASSIFIED — those pixels simply do
#:     not vote, rather than being forced into a neighbouring colour.
#: Red wraps, so it appears at both ends. `gold` is deliberately narrow: it
#: overlaps yellow and orange and over-claiming it is a common way to look
#: wrong at night.
HUE_BANDS: List[Tuple[int, int, str]] = [
    (0, 8, "red"), (8, 18, "orange"), (18, 24, "gold"), (24, 34, "yellow"),
    (34, 95, "green"), (95, 135, "blue"), (160, 172, "red"), (172, 180, "red"),
]

#: Brown is a dark, moderately saturated red/orange — it has no hue band of
#: its own, which is why a naive hue classifier never predicts it even
#: though it is a common car colour and is in the label space.
BROWN_HUE_RANGE = (3, 30)
BROWN_MAX_VALUE = 145

# Achromatic thresholds on OpenCV's 0-255 S and V scales.
# Achromatic value bands, calibrated against solid patches:
#   V <  55           black
#   55 <= V < 140     grey
#   140 <= V < 200    silver
#   V >= 200          white
# Silver sits where it actually falls on a real car — bright but not paper
# white. An earlier cut at 175 put mid-tone silver into the white band.
ACHROMATIC_SATURATION = 55      # below this a pixel carries no reliable hue
BLACK_VALUE = 55
GREY_VALUE = 140
SILVER_VALUE = 200
CHROMATIC_MIN_VALUE = 45        # too dark for its hue to be trustworthy

#: A pixel with a channel at or above this is sensor-CLIPPED: its true
#: colour was brighter than the sensor could record, so its hue and
#: saturation are fabrications of the clipping, not properties of the paint.
#: Specular highlights on a bonnet, direct sun and headlight bloom all do
#: this, and counting them votes "white" for a red car. Clipped pixels are
#: excluded from the vote rather than classified.
CLIPPED_CHANNEL = 250

#: Integer codes, so pixel classification runs on an int array rather than
#: an object array. Object dtype made this ~4x slower for no benefit.
#: Kept in lockstep with attributes.models.COLOURS (minus `unknown`, which
#: is a verdict rather than a pixel class). test_colour_baseline asserts the
#: subset relation, because emitting a name outside the label space silently
#: turns a correct answer into `unknown`.
PIXEL_CLASSES = ["white", "black", "silver", "grey", "red", "orange", "gold",
                 "yellow", "green", "blue", "brown"]
_CODE = {name: index for index, name in enumerate(PIXEL_CLASSES)}
UNCLASSIFIED = -1


@dataclass
class ColourEstimate:
    """One HSV baseline result, with the pixel evidence behind it."""
    label: str = "unknown"
    confidence: float = 0.0            # fraction of body pixels supporting it
    support_pixels: int = 0
    body_pixels: int = 0
    distribution: dict = field(default_factory=dict)
    #: Fraction of the body region that was sensor-clipped and therefore
    #: excluded. High values mean glare, direct sun or an overexposed
    #: capture, and are the reason to distrust the answer.
    clipped_fraction: float = 0.0
    reason: str = ""

    def top_k(self, k: int = 3) -> List[ScoredLabel]:
        ranked = sorted(self.distribution.items(), key=lambda kv: -kv[1])
        total = max(1, self.body_pixels)
        return [ScoredLabel(label, count / total) for label, count in ranked[:k]]


def body_region(crop_bgr, region: Tuple[float, float, float, float] = BODY_REGION):
    """The paint-bearing part of a vehicle crop."""
    height, width = crop_bgr.shape[:2]
    top, bottom, left, right = region
    body = crop_bgr[int(height * top):int(height * bottom),
                    int(width * left):int(width * right)]
    return body if body.size else crop_bgr


def classify_pixels(hsv):
    """
    Per-pixel colour codes for an HSV image, as an int8 array indexing
    PIXEL_CLASSES. Fully vectorised — no Python loop over pixels.
    """
    import numpy as np

    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]

    codes = np.full(hue.shape, UNCLASSIFIED, dtype=np.int8)

    achromatic = saturation < ACHROMATIC_SATURATION
    codes[achromatic & (value >= SILVER_VALUE)] = _CODE["white"]
    codes[achromatic & (value >= GREY_VALUE) & (value < SILVER_VALUE)] = _CODE["silver"]
    codes[achromatic & (value >= BLACK_VALUE) & (value < GREY_VALUE)] = _CODE["grey"]
    codes[achromatic & (value < BLACK_VALUE)] = _CODE["black"]

    chromatic = (~achromatic) & (value >= CHROMATIC_MIN_VALUE)
    for low, high, name in HUE_BANDS:
        codes[chromatic & (hue >= low) & (hue < high)] = _CODE[name]

    # Brown: a dark red/orange. Checked after the hue bands so it overrides
    # the red/orange/gold assignment for dark pixels.
    low, high = BROWN_HUE_RANGE
    codes[chromatic & (hue >= low) & (hue < high)
          & (value < BROWN_MAX_VALUE)] = _CODE["brown"]

    # Saturated but too dark for its hue to be trusted: read as black, which
    # is what a dark navy or bottle green actually looks like at a gate.
    codes[(~achromatic) & (value < CHROMATIC_MIN_VALUE)] = _CODE["black"]
    return codes


#: Above this clipped fraction the body region is too blown out for any
#: colour verdict to mean anything, and the answer is `unknown`. Chosen
#: because a specular highlight covering a third of the visible body leaves
#: too little true paint to outvote it. Glare is the failure mode that
#: decides whether this survives a real gate in direct sun.
MAX_CLIPPED_FRACTION = 0.35


def estimate_colour(crop_bgr, min_support: float = 0.28,
                    region: Tuple[float, float, float, float] = BODY_REGION,
                    max_side: int = 160,
                    max_clipped: float = MAX_CLIPPED_FRACTION) -> ColourEstimate:
    """
    Estimate a vehicle's colour from its crop. Never raises.

    `min_support` is the fraction of body pixels the winner must hold. 0.28
    is deliberately modest: a two-tone car, a reflective bonnet or a large
    windscreen can legitimately split the vote several ways.

    The crop is downscaled to `max_side` first. Colour is a low-frequency
    property, so this costs nothing in accuracy and makes the whole thing
    roughly a millisecond.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return ColourEstimate(reason="OpenCV/NumPy unavailable")

    if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
        return ColourEstimate(reason="empty crop")
    if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3:
        return ColourEstimate(reason="malformed crop")

    body = body_region(crop_bgr, region)
    height, width = body.shape[:2]
    if height < 4 or width < 4:
        return ColourEstimate(reason="body region too small")

    scale = max_side / float(max(height, width))
    if scale < 1.0:
        body = cv2.resize(body, (max(1, int(width * scale)), max(1, int(height * scale))),
                          interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(body, cv2.COLOR_BGR2HSV)
    codes = classify_pixels(hsv)

    # Drop sensor-clipped pixels before counting. Their colour is an
    # artefact of the clipping, not of the vehicle.
    clipped = body.max(axis=2) >= CLIPPED_CHANNEL
    clipped_fraction = float(clipped.mean())
    codes = np.where(clipped, np.int8(UNCLASSIFIED), codes)

    codes = codes.reshape(-1)
    codes = codes[codes >= 0]
    total = int(codes.size)
    if total < 32:
        return ColourEstimate(
            clipped_fraction=clipped_fraction,
            reason=(f"{clipped_fraction:.0%} of the body region is blown out; "
                    "too few unclipped pixels to judge colour"
                    if clipped_fraction > 0.5 else "too few classifiable pixels"))

    counts = np.bincount(codes, minlength=len(PIXEL_CLASSES))
    distribution = {PIXEL_CLASSES[i]: int(c) for i, c in enumerate(counts) if c}
    winner = max(distribution.items(), key=lambda kv: kv[1])
    label, support = winner[0], winner[1]
    confidence = support / float(total)

    estimate = ColourEstimate(
        label=label if label in COLOURS else "unknown",
        confidence=float(confidence), support_pixels=int(support),
        body_pixels=total, distribution=distribution,
        clipped_fraction=clipped_fraction)

    if clipped_fraction > max_clipped:
        # Say "I cannot tell" rather than report the colour of the glare.
        estimate.label = "unknown"
        estimate.reason = (
            f"{clipped_fraction:.0%} of the vehicle body is blown out "
            f"(over the {max_clipped:.0%} limit) — colour cannot be judged "
            "from this image")
    elif confidence < min_support:
        runner_up = sorted(distribution.items(), key=lambda kv: -kv[1])[:2]
        estimate.label = "unknown"
        estimate.reason = (
            f"no colour holds {min_support:.0%} of the body region "
            f"(best {winner[0]} {confidence:.0%}"
            + (f", then {runner_up[1][0]} {runner_up[1][1] / total:.0%}"
               if len(runner_up) > 1 else "") + ")")
    else:
        estimate.reason = (f"{label} holds {confidence:.0%} of {total} "
                           f"unclipped body pixels"
                           + (f" ({clipped_fraction:.0%} blown out and excluded)"
                              if clipped_fraction > 0.15 else ""))
    return estimate


class HSVColourBackend:
    """
    Colour backend with the same shape the learned head presents, so the
    pipeline can use either without knowing which.
    """

    name = "hsv_baseline"
    trained = True          # it has no weights to train, and it works today
    version = "hsv-baseline@1"

    def __init__(self, min_support: float = 0.28,
                 max_clipped: float = MAX_CLIPPED_FRACTION):
        self.min_support = min_support
        self.max_clipped = max_clipped

    def predict(self, crop_bgr) -> Tuple[Optional[str], float, List[ScoredLabel], str]:
        """-> (label or None, confidence, top_k, reason)."""
        estimate = estimate_colour(crop_bgr, min_support=self.min_support,
                                   max_clipped=self.max_clipped)
        label = estimate.label if estimate.label != "unknown" else None
        return label, estimate.confidence, estimate.top_k(3), estimate.reason
