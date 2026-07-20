from __future__ import annotations

import cv2
import numpy as np

from collie_demo.fruit import FruitDetection
from collie_demo.matcher import FruitInstanceMatcher
from collie_demo.memory import AppearanceEncoder, FruitMemory, descriptor_similarity, encode_jpeg


def colored_shape(color: tuple[int, int, int], shape: str = "circle") -> np.ndarray:
    image = np.full((120, 120, 3), 28, dtype=np.uint8)
    if shape == "circle":
        cv2.circle(image, (60, 60), 38, color, -1)
    else:
        cv2.rectangle(image, (25, 35), (95, 85), color, -1)
    return image


def memory_for(image: np.ndarray, label: str = "apple") -> FruitMemory:
    encoder = AppearanceEncoder(minimum_crop_side_px=8)
    reference = encoder.encode(image)
    darker = encoder.encode(cv2.convertScaleAbs(image, alpha=0.82, beta=8))
    return FruitMemory.create(
        label=label,
        samples=[reference, darker],
        reference_jpeg=encode_jpeg(image),
        reference_bbox_xyxy=(0, 0, image.shape[1], image.shape[0]),
    )


def test_appearance_embedding_survives_brightness_change() -> None:
    encoder = AppearanceEncoder(minimum_crop_side_px=8)
    apple = colored_shape((20, 20, 220))
    dim_apple = cv2.convertScaleAbs(apple, alpha=0.75, beta=12)
    pear = colored_shape((30, 190, 210), "rectangle")

    same_score = descriptor_similarity(encoder.encode(apple), encoder.encode(dim_apple))[0]
    different_score = descriptor_similarity(encoder.encode(apple), encoder.encode(pear))[0]

    assert same_score > 0.80
    assert same_score > different_score + 0.07


def test_matcher_rejects_two_indistinguishable_same_class_candidates() -> None:
    fruit = colored_shape((20, 20, 220))
    frame = np.hstack((fruit, fruit))
    detections = [
        FruitDetection(0, "apple", 0.95, (0, 0, 120, 120), (60, 60)),
        FruitDetection(0, "apple", 0.94, (120, 0, 240, 120), (180, 60)),
    ]
    result = FruitInstanceMatcher(
        AppearanceEncoder(minimum_crop_side_px=8),
        minimum_score=0.75,
        minimum_margin=0.05,
    ).rank(memory_for(fruit), frame, detections)

    assert result.accepted is False
    assert result.reason == "appearance_match_ambiguous"
    assert result.candidate_count == 2


def test_matcher_accepts_the_saved_instance_and_ignores_other_classes() -> None:
    apple = colored_shape((20, 20, 220))
    pear = colored_shape((30, 190, 210), "rectangle")
    frame = np.hstack((apple, pear))
    detections = [
        FruitDetection(0, "apple", 0.95, (0, 0, 120, 120), (60, 60)),
        FruitDetection(2, "pear", 0.96, (120, 0, 240, 120), (180, 60)),
    ]
    result = FruitInstanceMatcher(
        AppearanceEncoder(minimum_crop_side_px=8),
        minimum_score=0.75,
        minimum_margin=0.05,
    ).rank(memory_for(apple), frame, detections)

    assert result.accepted is True
    assert result.best is not None
    assert result.best.detection.center == (60, 60)
    assert result.candidate_count == 1


def test_different_matcher_rejects_the_saved_banana_when_it_is_alone() -> None:
    banana_a = colored_shape((20, 210, 235), "rectangle")
    detection = FruitDetection(1, "banana", 0.93, (0, 0, 120, 120), (60, 60))
    result = FruitInstanceMatcher(
        AppearanceEncoder(minimum_crop_side_px=8),
        maximum_saved_similarity=0.94,
    ).rank_different(memory_for(banana_a, "banana"), banana_a, [detection])

    assert result.accepted is False
    assert result.reason == "saved_instance_only"
    assert result.best is not None
    assert result.best.score >= 0.94


def test_different_matcher_rejects_banana_a_and_selects_banana_b() -> None:
    banana_a = colored_shape((20, 210, 235), "rectangle")
    banana_b = colored_shape((210, 45, 35), "circle")
    frame = np.hstack((banana_a, banana_b))
    detections = [
        FruitDetection(1, "banana", 0.96, (0, 0, 120, 120), (60, 60)),
        FruitDetection(1, "banana", 0.91, (120, 0, 240, 120), (180, 60)),
    ]
    result = FruitInstanceMatcher(
        AppearanceEncoder(minimum_crop_side_px=8),
        maximum_saved_similarity=0.94,
    ).rank_different(memory_for(banana_a, "banana"), frame, detections)

    assert result.accepted is True
    assert result.reason == "different_instance_match"
    assert result.best is not None
    assert result.best.detection.center == (180, 60)
    assert result.best.score < 0.94
