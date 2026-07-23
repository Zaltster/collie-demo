from __future__ import annotations

from collie_demo.fruit import FruitDetection
from collie_demo.matcher import FruitClassMatcher
from collie_demo.memory import FruitMemory


def memory_for(label: str) -> FruitMemory:
    return FruitMemory.create(
        label=label,
        reference_jpeg=b"reference",
        reference_bbox_xyxy=(0, 0, 20, 20),
    )


def test_class_matcher_chooses_highest_confidence_saved_class() -> None:
    detections = [
        FruitDetection(1, "banana", 0.72, (0, 0, 40, 40), (20, 20)),
        FruitDetection(2, "pear", 0.99, (50, 0, 90, 40), (70, 20)),
        FruitDetection(1, "banana", 0.91, (100, 0, 140, 40), (120, 20)),
    ]

    result = FruitClassMatcher().match(memory_for("banana"), detections)

    assert result.accepted is True
    assert result.reason == "target_class_detected"
    assert result.candidate_count == 2
    assert result.best is not None
    assert result.best.detection.center == (120, 20)
    assert result.best.detection.confidence == 0.91


def test_class_matcher_accepts_any_instance_of_saved_class() -> None:
    detection = FruitDetection(
        1,
        "banana",
        0.88,
        (200, 100, 260, 180),
        (230, 140),
    )

    result = FruitClassMatcher().match(memory_for("banana"), [detection])

    assert result.accepted is True
    assert result.best is not None
    assert result.best.detection is detection


def test_class_matcher_rejects_other_classes() -> None:
    detections = [
        FruitDetection(0, "apple", 0.97, (0, 0, 40, 40), (20, 20)),
        FruitDetection(2, "pear", 0.96, (50, 0, 90, 40), (70, 20)),
    ]

    result = FruitClassMatcher().match(memory_for("banana"), detections)

    assert result.accepted is False
    assert result.reason == "target_class_not_detected"
    assert result.best is None
    assert result.candidate_count == 0


def test_class_matcher_keeps_the_spatially_selected_class_target() -> None:
    detections = [
        FruitDetection(1, "banana", 0.97, (0, 0, 40, 40), (20, 20)),
        FruitDetection(1, "banana", 0.81, (200, 100, 260, 180), (230, 140)),
    ]

    result = FruitClassMatcher().match(
        memory_for("banana"), detections, preferred_center=(235, 145)
    )

    assert result.accepted is True
    assert result.best is not None
    assert result.best.detection.center == (230, 140)
