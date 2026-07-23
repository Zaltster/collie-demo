from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from .types import TargetObservation


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
        class_thresholds: dict[str, float] | None = None,
        device: str | int | None = None,
        task: str | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"fruit model not found: {self.model_path}")
        if not 0.0 < confidence <= 1.0:
            raise ValueError("confidence must be in (0, 1]")
        normalized_thresholds: dict[str, float] = {}
        for label, threshold in (class_thresholds or {}).items():
            normalized_label = label.casefold().strip()
            if not normalized_label:
                raise ValueError("class threshold labels cannot be empty")
            if not 0.0 < threshold <= 1.0:
                raise ValueError("class thresholds must be in (0, 1]")
            normalized_thresholds[normalized_label] = float(threshold)
        self.class_thresholds = normalized_thresholds
        self.confidence = min(
            [float(confidence), *normalized_thresholds.values()]
        )
        self.requested_device = device
        self.task = task
        if model is None:
            from ultralytics import YOLO

            model = YOLO(str(self.model_path), task=task)
        self.model = model
        names = getattr(model, "names", {})
        self.names = {
            int(index): str(label)
            for index, label in (
                names.items() if isinstance(names, dict) else enumerate(names)
            )
        }
        self.class_ids = [
            class_id
            for class_id, label in self.names.items()
            if not self.class_thresholds
            or label.casefold().strip() in self.class_thresholds
        ]
        missing_labels = set(self.class_thresholds) - {
            label.casefold().strip() for label in self.names.values()
        }
        if missing_labels:
            rendered = ", ".join(sorted(missing_labels))
            raise ValueError(f"class thresholds not present in model: {rendered}")

    def detect(self, bgr: NDArray[np.uint8]) -> list[FruitDetection]:
        predict_options: dict[str, object] = {
            "source": bgr,
            "conf": self.confidence,
            "verbose": False,
        }
        if self.requested_device is not None:
            predict_options["device"] = self.requested_device
        if self.class_thresholds:
            predict_options["classes"] = self.class_ids
        results = self.model.predict(**predict_options)
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
            label = self.names.get(class_id, f"class_{class_id}")
            threshold = self.class_thresholds.get(
                label.casefold().strip(), self.confidence
            )
            if float(score) < threshold:
                continue
            detections.append(
                FruitDetection(
                    class_id=class_id,
                    label=label,
                    confidence=round(float(score), 4),
                    bbox_xyxy=(x1, y1, x2, y2),
                    center=((x1 + x2) // 2, (y1 + y2) // 2),
                )
            )
        return _suppress_overlapping_detections(detections)

    def device_status(self) -> dict[str, object]:
        try:
            import torch

            cuda_available = bool(torch.cuda.is_available())
            cuda_version = torch.version.cuda
            torch_version = torch.__version__
        except Exception:
            cuda_available = False
            cuda_version = None
            torch_version = "unavailable"
        return {
            "requested": "auto"
            if self.requested_device is None
            else str(self.requested_device),
            "task": self.task,
            "resolved": self._model_device(),
            "cuda_available": cuda_available,
            "cuda_version": cuda_version,
            "torch_version": torch_version,
        }

    def _model_device(self) -> str:
        # PyTorch checkpoints expose ``YOLO.model.device``. Serialized
        # TensorRT engines keep ``YOLO.model`` as the engine path string and
        # expose the real CUDA device through the lazily-created predictor's
        # AutoBackend instead. Inspect both without special-casing the engine
        # suffix so the stage GPU gate remains evidence-based.
        predictor = getattr(self.model, "predictor", None)
        candidates = (
            getattr(self.model, "model", None),
            getattr(predictor, "model", None),
            predictor,
        )
        for candidate in candidates:
            direct_device = getattr(candidate, "device", None)
            if direct_device is not None:
                return str(direct_device)
            parameters = getattr(candidate, "parameters", None)
            if callable(parameters):
                try:
                    return str(next(parameters()).device)
                except (StopIteration, AttributeError, TypeError):
                    pass
        return "unresolved"


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


def annotate_selected_produce(
    bgr: NDArray[np.uint8],
    label: str | None,
    target: TargetObservation | None,
) -> NDArray[np.uint8]:
    if label is None or target is None:
        return bgr
    annotated = bgr.copy()
    x, y, width, height = target.bbox_xywh
    color = (255, 80, 255)
    cv2.rectangle(annotated, (x, y), (x + width, y + height), color, 5)
    cv2.circle(annotated, target.center, 6, (255, 255, 255), -1)
    tracking_text = (
        "TRACKER"
        if target.confidence is None
        else f"YOLO {target.confidence:.2f}"
    )
    cv2.putText(
        annotated,
        f"SELECTED {label.upper()} {tracking_text}",
        (max(8, x), max(24, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
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


def _suppress_overlapping_detections(
    detections: list[FruitDetection], iou_threshold: float = 0.35
) -> list[FruitDetection]:
    """Keep the best same-class box for one physical object."""
    accepted: list[FruitDetection] = []
    for candidate in sorted(
        detections, key=lambda detection: detection.confidence, reverse=True
    ):
        if any(
            existing.label.casefold() == candidate.label.casefold()
            and _box_iou(existing.bbox_xyxy, candidate.bbox_xyxy) >= iou_threshold
            for existing in accepted
        ):
            continue
        accepted.append(candidate)
    return accepted


def _box_iou(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    if intersection == 0:
        return 0.0
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    return intersection / (first_area + second_area - intersection)


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
