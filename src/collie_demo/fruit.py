from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class FruitDetection:
    class_id: int
    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]
    center: tuple[int, int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def log_line(self) -> str:
        x1, y1, x2, y2 = self.bbox_xyxy
        return (
            f"fruit={self.label} confidence={self.confidence:.3f} "
            f"center=({self.center[0]},{self.center[1]}) "
            f"bbox=({x1},{y1},{x2},{y2})"
        )


class FruitDetector:
    def __init__(
        self,
        model_path: str | Path,
        *,
        confidence: float = 0.5,
        model: Any | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"fruit model not found: {self.model_path}")
        if not 0.0 < confidence <= 1.0:
            raise ValueError("confidence must be in (0, 1]")
        self.confidence = float(confidence)
        if model is None:
            from ultralytics import YOLO

            model = YOLO(str(self.model_path))
        self.model = model
        names = getattr(model, "names", {})
        self.names = {
            int(index): str(label)
            for index, label in (
                names.items() if isinstance(names, dict) else enumerate(names)
            )
        }

    def detect(self, bgr: NDArray[np.uint8]) -> list[FruitDetection]:
        results = self.model.predict(
            source=bgr,
            conf=self.confidence,
            verbose=False,
        )
        if not results:
            return []
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = _as_numpy(boxes.xyxy)
        confidences = _as_numpy(boxes.conf)
        classes = _as_numpy(boxes.cls)
        detections: list[FruitDetection] = []
        for coordinates, score, class_number in zip(xyxy, confidences, classes):
            x1, y1, x2, y2 = (int(round(float(value))) for value in coordinates)
            class_id = int(class_number)
            detections.append(
                FruitDetection(
                    class_id=class_id,
                    label=self.names.get(class_id, f"class_{class_id}"),
                    confidence=round(float(score), 4),
                    bbox_xyxy=(x1, y1, x2, y2),
                    center=((x1 + x2) // 2, (y1 + y2) // 2),
                )
            )
        return detections


def annotate_fruits(
    bgr: NDArray[np.uint8], detections: list[FruitDetection]
) -> NDArray[np.uint8]:
    annotated = bgr.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox_xyxy
        color = _class_color(detection.class_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        cv2.circle(annotated, detection.center, 5, (255, 255, 255), -1)
        text = f"{detection.label} {detection.confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
        )
        text_top = max(0, y1 - text_height - baseline - 8)
        cv2.rectangle(
            annotated,
            (x1, text_top),
            (x1 + text_width + 10, text_top + text_height + baseline + 8),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            text,
            (x1 + 5, text_top + text_height + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated


def _as_numpy(value: Any) -> NDArray[np.float32]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _class_color(class_id: int) -> tuple[int, int, int]:
    palette = (
        (68, 68, 255),
        (78, 180, 60),
        (30, 220, 245),
        (82, 170, 65),
        (45, 245, 245),
        (20, 150, 255),
        (90, 210, 100),
        (175, 70, 190),
        (140, 90, 255),
        (70, 210, 70),
    )
    return palette[class_id % len(palette)]
