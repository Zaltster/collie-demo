"""Camera-bearing controller for a deliberately short, supervised demo burst."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .types import BlueWhaleObservation, VelocityCommand


@dataclass(frozen=True, slots=True)
class ApproachConfig:
    horizontal_fov_deg: float = 100.0
    stable_frames_required: int = 8
    maximum_target_age_s: float = 0.35
    yaw_threshold_deg: float = 7.0
    minimum_yaw_rps: float = 0.30
    maximum_yaw_rps: float = 0.30
    yaw_gain: float = 0.8
    forward_mps: float = 0.08
    forward_budget_s: float = 1.5


class ApproachController:
    def __init__(self, config: ApproachConfig | None = None) -> None:
        self.config = config or ApproachConfig()

    def plan(
        self,
        target: BlueWhaleObservation | None,
        *,
        frame_width: int | None,
        now_monotonic_s: float,
        forward_elapsed_s: float,
        allow_unranged_forward: bool,
    ) -> VelocityCommand:
        if target is None:
            return VelocityCommand(reason="blue_whale_not_found")
        if target.visible_frames < self.config.stable_frames_required:
            return VelocityCommand(reason="blue_whale_not_stable")
        age = now_monotonic_s - target.captured_monotonic_s
        if not math.isfinite(age) or age < 0.0 or age > self.config.maximum_target_age_s:
            return VelocityCommand(reason="blue_whale_stale")
        if frame_width is None or frame_width < 2:
            return VelocityCommand(reason="frame_geometry_missing")
        if not allow_unranged_forward:
            return VelocityCommand(reason="unranged_forward_disabled")
        if forward_elapsed_s >= self.config.forward_budget_s:
            return VelocityCommand(reason="forward_budget_complete")
        half_width = frame_width / 2.0
        focal_px = half_width / math.tan(
            math.radians(self.config.horizontal_fov_deg) / 2.0
        )
        bearing = math.atan2(target.center[0] - half_width, focal_px)
        if abs(math.degrees(bearing)) >= self.config.yaw_threshold_deg:
            direction = -1.0 if bearing > 0.0 else 1.0
            yaw = direction * min(
                self.config.maximum_yaw_rps,
                max(self.config.minimum_yaw_rps, self.config.yaw_gain * abs(bearing)),
            )
            return VelocityCommand(
                forward_mps=self.config.forward_mps,
                yaw_rps=yaw,
                reason="curving_to_blue_whale",
            )
        return VelocityCommand(
            forward_mps=self.config.forward_mps,
            reason="supervised_forward_burst",
        )
