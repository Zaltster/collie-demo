from __future__ import annotations

from pathlib import Path

import numpy as np

from collie_demo.fruit import FruitDetector, annotate_fruits
from collie_demo.fruit_server import UnitreeFrameSource
from collie_demo.types import CameraFrame


class ArrayValue:
    def __init__(self, value: object) -> None:
        self.value = np.asarray(value)

    def detach(self) -> "ArrayValue":
        return self

    def cpu(self) -> "ArrayValue":
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class FakeBoxes:
    xyxy = ArrayValue(
        [
            [10.2, 20.4, 50.8, 80.6],
            [12.0, 22.0, 49.0, 79.0],
            [60.0, 20.0, 90.0, 80.0],
        ]
    )
    conf = ArrayValue([0.91234, 0.8, 0.25])
    cls = ArrayValue([2, 2, 2])

    def __len__(self) -> int:
        return 3


class FakeResult:
    boxes = FakeBoxes()


class FakeModel:
    names = {2: "banana"}

    def __init__(self) -> None:
        self.last_predict_options: dict[str, object] = {}

    def predict(self, **options: object) -> list[FakeResult]:
        self.last_predict_options = options
        return [FakeResult()]


class FakeUnitreeCamera:
    def __init__(self, frame: np.ndarray) -> None:
        self.frame = frame

    def read(self) -> CameraFrame:
        return CameraFrame(1, 123.0, self.frame)


def test_fruit_detector_returns_label_confidence_box_and_center(tmp_path: Path) -> None:
    model_path = tmp_path / "fruit.pt"
    model_path.write_bytes(b"test")
    model = FakeModel()
    detector = FruitDetector(
        model_path, confidence=0.5, device="cuda:0", model=model
    )
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    detections = detector.detect(frame)

    assert len(detections) == 1
    detection = detections[0]
    assert detection.label == "banana"
    assert detection.confidence == 0.9123
    assert detection.bbox_xyxy == (10, 20, 51, 81)
    assert detection.center == (30, 50)
    assert model.last_predict_options["device"] == "cuda:0"
    assert detector.device_status()["requested"] == "cuda:0"
    assert np.any(annotate_fruits(frame, detections) != frame)


def test_fruit_detector_applies_per_class_threshold_before_returning(
    tmp_path: Path,
) -> None:
    model_path = tmp_path / "fruit.pt"
    model_path.write_bytes(b"test")
    model = FakeModel()
    detector = FruitDetector(
        model_path,
        confidence=0.5,
        class_thresholds={"banana": 0.3},
        model=model,
    )

    detections = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    assert [detection.confidence for detection in detections] == [0.9123]
    assert detector.confidence == 0.3
    assert detector.class_thresholds == {"banana": 0.3}
    assert model.last_predict_options["conf"] == 0.3
    assert model.last_predict_options["classes"] == [2]


def test_unitree_frame_source_returns_video_client_bgr() -> None:
    frame = np.full((12, 16, 3), 42, dtype=np.uint8)
    source = UnitreeFrameSource(None, camera=FakeUnitreeCamera(frame))

    result = source.read()

    assert source.description == "unitree:VideoClient"
    assert np.array_equal(result, frame)
