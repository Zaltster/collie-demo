from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class CameraFrame:
    frame_id: int
    captured_monotonic_s: float
    bgr: NDArray[np.uint8]

    @property
    def width(self) -> int:
        return int(self.bgr.shape[1])

    @property
    def height(self) -> int:
        return int(self.bgr.shape[0])


@dataclass(frozen=True, slots=True)
class TargetObservation:
    frame_id: int
    captured_monotonic_s: float
    bbox_xywh: tuple[int, int, int, int]
    center: tuple[int, int]
    confidence: float
    visible_frames: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

@dataclass(frozen=True, slots=True)
class VelocityCommand:
    forward_mps: float = 0.0
    yaw_rps: float = 0.0
    reason: str = "idle"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
