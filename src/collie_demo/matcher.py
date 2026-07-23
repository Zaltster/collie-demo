"""Select fresh YOLO detections by the saved fruit class."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .fruit import FruitDetection
from .memory import FruitMemory


@dataclass(frozen=True, slots=True)
class ClassCandidate:
    detection: FruitDetection

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.detection.label,
            "center": self.detection.center,
            "bbox_xyxy": self.detection.bbox_xyxy,
            "detector_confidence": self.detection.confidence,
        }


@dataclass(frozen=True, slots=True)
class ClassMatchResult:
    accepted: bool
    best: ClassCandidate | None
    candidate_count: int
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "best": None if self.best is None else self.best.to_dict(),
            "candidate_count": self.candidate_count,
            "reason": self.reason,
        }


class FruitClassMatcher:
    """Choose the highest-confidence fresh detection of the saved class."""

    def match(
        self,
        memory: FruitMemory,
        detections: list[FruitDetection],
        preferred_center: tuple[int, int] | None = None,
    ) -> ClassMatchResult:
        candidates = [
            ClassCandidate(detection)
            for detection in detections
            if detection.label.casefold() == memory.label.casefold()
        ]
        if not candidates:
            return ClassMatchResult(False, None, 0, "target_class_not_detected")
        if preferred_center is None:
            candidates.sort(
                key=lambda item: item.detection.confidence,
                reverse=True,
            )
        else:
            candidates.sort(
                key=lambda item: (
                    math.hypot(
                        item.detection.center[0] - preferred_center[0],
                        item.detection.center[1] - preferred_center[1],
                    ),
                    -item.detection.confidence,
                )
            )
        return ClassMatchResult(
            True,
            candidates[0],
            len(candidates),
            "target_class_detected",
        )
