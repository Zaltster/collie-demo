from __future__ import annotations

import cv2
import numpy as np

from collie_demo.detector import BlueWhaleDetector, YellowWhaleDetector
from collie_demo.types import CameraFrame


def whale_frame(frame_id: int = 1) -> CameraFrame:
    image = np.full((720, 1280, 3), 235, dtype=np.uint8)
    cv2.ellipse(image, (840, 650), (34, 42), 0, 0, 360, (245, 245, 160), -1)
    cv2.ellipse(image, (840, 694), (31, 7), 0, 0, 360, (75, 75, 75), -1)
    return CameraFrame(frame_id, float(frame_id), image)


def yellow_whale_frame(frame_id: int = 1) -> CameraFrame:
    image = np.full((720, 1280, 3), 80, dtype=np.uint8)
    cv2.ellipse(image, (760, 650), (52, 34), 0, 0, 360, (70, 220, 235), -1)
    return CameraFrame(frame_id, float(frame_id), image)


def test_detects_blue_whale_in_floor_region() -> None:
    target = BlueWhaleDetector().detect(whale_frame())
    assert target is not None
    assert abs(target.center[0] - 840) < 8
    assert target.confidence > 0.5


def test_stability_increases_across_nearby_frames() -> None:
    detector = BlueWhaleDetector()
    first = detector.detect(whale_frame(1))
    second = detector.detect(whale_frame(2))
    assert first is not None and second is not None
    assert first.visible_frames == 1
    assert second.visible_frames == 2


def test_nonblue_floor_does_not_detect() -> None:
    image = np.full((720, 1280, 3), 235, dtype=np.uint8)
    assert BlueWhaleDetector().detect(CameraFrame(1, 1.0, image)) is None


def test_detects_yellow_whale_without_confusing_it_for_blue() -> None:
    frame = yellow_whale_frame()
    yellow = YellowWhaleDetector().detect(frame)

    assert yellow is not None
    assert abs(yellow.center[0] - 760) < 8
    assert yellow.confidence > 0.5
    assert BlueWhaleDetector().detect(frame) is None


def test_yellow_detector_ignores_blue_whale() -> None:
    assert YellowWhaleDetector().detect(whale_frame()) is None


def test_yellow_detector_prefers_bright_floor_whale_over_dull_warm_patch() -> None:
    image = np.full((1080, 1920, 3), (80, 96, 119), dtype=np.uint8)
    cv2.rectangle(image, (1666, 772), (1823, 798), (135, 157, 168), -1)
    cv2.ellipse(image, (1140, 880), (38, 25), 0, 0, 360, (186, 233, 230), -1)

    target = YellowWhaleDetector().detect(CameraFrame(1, 1.0, image))

    assert target is not None
    assert abs(target.center[0] - 1140) < 8
    assert abs(target.center[1] - 880) < 8


def test_detects_blue_whale_on_left_side_of_floor() -> None:
    image = np.full((720, 1280, 3), 235, dtype=np.uint8)
    cv2.ellipse(image, (275, 555), (30, 34), 0, 0, 360, (245, 245, 160), -1)
    target = BlueWhaleDetector().detect(CameraFrame(1, 1.0, image))
    assert target is not None
    assert abs(target.center[0] - 275) < 8


def test_ignores_blue_object_above_floor_region() -> None:
    image = np.full((720, 1280, 3), 235, dtype=np.uint8)
    cv2.ellipse(image, (275, 300), (30, 34), 0, 0, 360, (245, 245, 160), -1)
    assert BlueWhaleDetector().detect(CameraFrame(1, 1.0, image)) is None


def test_prefers_large_whale_over_tiny_bright_floor_reflection() -> None:
    image = np.full((720, 1280, 3), 235, dtype=np.uint8)
    cv2.ellipse(image, (640, 600), (34, 42), 0, 0, 360, (245, 245, 160), -1)
    cv2.rectangle(image, (1140, 590), (1148, 598), (255, 255, 80), -1)
    target = BlueWhaleDetector().detect(CameraFrame(1, 1.0, image))
    assert target is not None
    assert abs(target.center[0] - 640) < 8


def test_track_continuity_ignores_new_distant_reflection() -> None:
    detector = BlueWhaleDetector()
    first_image = np.full((720, 1280, 3), 235, dtype=np.uint8)
    cv2.ellipse(first_image, (640, 600), (34, 42), 0, 0, 360, (245, 245, 160), -1)
    first = detector.detect(CameraFrame(1, 1.0, first_image))

    second_image = first_image.copy()
    cv2.rectangle(second_image, (1080, 560), (1140, 620), (255, 255, 80), -1)
    second = detector.detect(CameraFrame(2, 2.0, second_image))

    assert first is not None and second is not None
    assert abs(second.center[0] - first.center[0]) < 8
    assert second.visible_frames == 2
