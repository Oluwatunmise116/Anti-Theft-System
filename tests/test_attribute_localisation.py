"""
Vehicle localisation by plate-box containment.

The rule under test is a correctness rule, not an accuracy rule: with two
vehicles in frame, attaching the wrong body to the right plate produces a
confidently wrong colour and brand that never shows up in aggregate
accuracy. Pure geometry, no model needed.
"""
import numpy as np
import pytest

from attributes.models import AttributeStatus
from attributes.vehicle_detector import (AUTHORITATIVE_TYPES, COCO_VEHICLE_CLASSES,
                                         box_area, contains_fraction,
                                         select_vehicle_for_plate)


def vehicle(box, class_name="car", confidence=0.8):
    return {"box": box, "class_name": class_name, "confidence": confidence}


# The left vehicle is larger and more confident; the plate belongs to the right one.
LEFT_TRUCK = vehicle((0, 0, 400, 300), "truck", 0.95)
RIGHT_CAR = vehicle((420, 120, 620, 280), "car", 0.60)
RIGHT_PLATE = (480, 230, 560, 258)


def test_containment_is_measured_against_the_plate_not_iou():
    # A plate is a tiny fraction of a car, so IoU between them is near zero
    # and would reject every correct pairing.
    assert contains_fraction(RIGHT_CAR["box"], RIGHT_PLATE) == pytest.approx(1.0)
    assert contains_fraction((0, 0, 100, 100), (50, 50, 150, 150)) == pytest.approx(0.25)
    assert contains_fraction((0, 0, 100, 100), (200, 200, 210, 210)) == 0.0


def test_two_vehicles_picks_the_one_containing_the_plate():
    result = select_vehicle_for_plate([LEFT_TRUCK, RIGHT_CAR], RIGHT_PLATE)
    assert result.found is True
    assert result.box == RIGHT_CAR["box"]
    assert result.coco_class == "car"
    assert result.candidate_count == 2


def test_the_larger_more_confident_vehicle_does_not_win():
    """Guards the exact silent-correctness bug this module exists to prevent."""
    result = select_vehicle_for_plate([LEFT_TRUCK, RIGHT_CAR], RIGHT_PLATE)
    assert box_area(LEFT_TRUCK["box"]) > box_area(RIGHT_CAR["box"])
    assert LEFT_TRUCK["confidence"] > RIGHT_CAR["confidence"]
    assert result.box != LEFT_TRUCK["box"]


def test_box_order_does_not_change_the_result():
    forward = select_vehicle_for_plate([LEFT_TRUCK, RIGHT_CAR], RIGHT_PLATE)
    reverse = select_vehicle_for_plate([RIGHT_CAR, LEFT_TRUCK], RIGHT_PLATE)
    assert forward.box == reverse.box


def test_no_containing_box_is_an_explicit_status_not_a_fallback():
    orphan_plate = (700, 400, 780, 428)
    result = select_vehicle_for_plate([LEFT_TRUCK, RIGHT_CAR], orphan_plate)
    assert result.found is False
    assert result.status == AttributeStatus.VEHICLE_NOT_LOCALISED.value
    assert result.box is None                      # no fallback to largest
    assert "Refusing to guess" in result.rejection_reason


def test_partial_containment_below_threshold_is_rejected():
    # Plate straddles the vehicle edge: half in, half out.
    straddling = (380, 140, 460, 168)
    result = select_vehicle_for_plate([RIGHT_CAR], straddling, min_contains_fraction=0.90)
    assert result.found is False
    assert 0.0 < result.contains_fraction < 0.90


def test_containment_threshold_tolerates_detector_jitter():
    # A plate poking a few pixels outside the body box must still link.
    jittered = (415, 200, 495, 228)            # ~94% inside RIGHT_CAR
    assert select_vehicle_for_plate([RIGHT_CAR], jittered, 0.90).found is True
    assert select_vehicle_for_plate([RIGHT_CAR], jittered, 1.00).found is False


def test_nested_boxes_choose_the_tightest_containing_body():
    bus = vehicle((400, 100, 700, 320), "bus", 0.9)
    result = select_vehicle_for_plate([bus, RIGHT_CAR], RIGHT_PLATE)
    assert result.box == RIGHT_CAR["box"]
    assert result.coco_class == "car"


def test_missing_plate_box_cannot_localise():
    result = select_vehicle_for_plate([RIGHT_CAR], None)
    assert result.status == AttributeStatus.VEHICLE_NOT_LOCALISED.value
    assert "No plate box" in result.rejection_reason


def test_no_vehicle_boxes_at_all():
    result = select_vehicle_for_plate([], RIGHT_PLATE)
    assert result.status == AttributeStatus.VEHICLE_NOT_LOCALISED.value
    assert result.candidate_count == 0


def test_coarse_coco_classes_are_authoritative_for_type():
    assert AUTHORITATIVE_TYPES == {"bus": "bus", "truck": "truck",
                                   "motorcycle": "motorcycle"}
    # `car` is deliberately absent: the learned head resolves fine style
    # within car, and there is no "car" entry in the TYPES label space.
    assert "car" not in AUTHORITATIVE_TYPES
    assert set(COCO_VEHICLE_CLASSES.values()) == {"car", "motorcycle", "bus", "truck"}
