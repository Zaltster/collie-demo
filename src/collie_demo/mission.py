"""Configuration and telemetry for the learn-turn-find-approach demo."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class MissionPhase(str, Enum):
    IDLE = "idle"
    LEARNING = "learning"
    MEMORIZED = "memorized"
    TURNING = "turning"
    SEARCHING = "searching"
    CONFIRMING = "confirming"
    APPROACHING = "approaching"
    SUCCESS = "success"
    ABORTED = "aborted"


@dataclass(frozen=True, slots=True)
class MissionConfig:
    enabled: bool = False
    autonomous_turn_enabled: bool = False
    direct_turn_enabled: bool = False
    capture_samples: int = 5
    capture_timeout_s: float = 2.0
    match_confirmations_required: int = 3
    approach_misses_allowed: int = 2
    turn_angle_rad: float = math.pi
    turn_rate_rps: float = 0.30
    turn_tolerance_rad: float = math.radians(7.0)
    turn_timeout_s: float = 15.0
    turn_stall_timeout_s: float = 2.0
    turn_stall_min_progress_rad: float = math.radians(10.0)
    search_rate_rps: float = 0.20
    search_sweep_rad: float = math.radians(75.0)
    search_timeout_s: float = 9.0
    near_bottom_ratio: float = 0.86
    near_center_ratio: float = 0.72

    def __post_init__(self) -> None:
        if self.capture_samples < 2:
            raise ValueError("capture_samples must be at least two")
        if self.capture_timeout_s <= 0.0:
            raise ValueError("capture_timeout_s must be positive")
        if self.match_confirmations_required < 1:
            raise ValueError("match_confirmations_required must be positive")
        if self.approach_misses_allowed < 1:
            raise ValueError("approach_misses_allowed must be positive")
        for name in (
            "turn_angle_rad",
            "turn_rate_rps",
            "turn_tolerance_rad",
            "turn_timeout_s",
            "turn_stall_timeout_s",
            "turn_stall_min_progress_rad",
            "search_rate_rps",
            "search_sweep_rad",
            "search_timeout_s",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(slots=True)
class MissionTelemetry:
    phase: MissionPhase = MissionPhase.IDLE
    reason: str = "idle"
    started_monotonic_s: float | None = None
    turn_progress_rad: float = 0.0
    match_confirmations: int = 0
    match_failures: int = 0
    last_match: dict[str, object] | None = None
    near_target_seen: bool = False

    def to_dict(self, now: float) -> dict[str, object]:
        return {
            "phase": self.phase.value,
            "reason": self.reason,
            "elapsed_s": None
            if self.started_monotonic_s is None
            else round(max(0.0, now - self.started_monotonic_s), 3),
            "turn_progress_deg": round(math.degrees(self.turn_progress_rad), 1),
            "match_confirmations": self.match_confirmations,
            "match_failures": self.match_failures,
            "last_match": self.last_match,
            "near_target_seen": self.near_target_seen,
        }
