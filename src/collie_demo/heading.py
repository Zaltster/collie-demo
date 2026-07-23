"""Fresh Go2 local-pose samples for measured turns and short returns."""

from __future__ import annotations

from dataclasses import dataclass
import math
from threading import Lock
import time
from typing import Protocol


@dataclass(frozen=True, slots=True)
class HeadingSample:
    yaw_rad: float | None
    age_s: float | None
    healthy: bool
    error: str | None = None
    x_m: float | None = None
    y_m: float | None = None
    position_healthy: bool = False
    position_error: str | None = None

    @property
    def pose_healthy(self) -> bool:
        return bool(
            self.healthy
            and self.position_healthy
            and self.yaw_rad is not None
            and self.x_m is not None
            and self.y_m is not None
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "yaw_rad": None if self.yaw_rad is None else round(self.yaw_rad, 4),
            "age_s": None if self.age_s is None else round(self.age_s, 3),
            "healthy": self.healthy,
            "error": self.error,
            "x_m": None if self.x_m is None else round(self.x_m, 4),
            "y_m": None if self.y_m is None else round(self.y_m, 4),
            "position_healthy": self.position_healthy,
            "pose_healthy": self.pose_healthy,
            "position_error": self.position_error,
        }


class HeadingProviderProtocol(Protocol):
    def start(self) -> None: ...
    def close(self) -> None: ...
    def status(self) -> HeadingSample: ...


class SportModeHeadingProvider:
    def __init__(self, *, maximum_age_s: float = 0.5) -> None:
        if maximum_age_s <= 0.0:
            raise ValueError("maximum_age_s must be positive")
        self.maximum_age_s = float(maximum_age_s)
        self._lock = Lock()
        self._yaw_rad: float | None = None
        self._x_m: float | None = None
        self._y_m: float | None = None
        self._updated_at: float | None = None
        self._error: str | None = "waiting for rt/sportmodestate"
        self._position_error: str | None = "waiting for local position"
        self._subscriber = None

    def start(self) -> None:
        if self._subscriber is not None:
            return
        try:
            from unitree_sdk2py.core.channel import ChannelSubscriber
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_

            subscriber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
            subscriber.Init(self._on_state, 1)
            self._subscriber = subscriber
        except Exception as exc:
            with self._lock:
                self._error = f"heading subscriber failed: {exc}"

    def close(self) -> None:
        subscriber, self._subscriber = self._subscriber, None
        if subscriber is not None:
            try:
                subscriber.Close()
            except Exception:
                pass
        with self._lock:
            self._yaw_rad = None
            self._x_m = None
            self._y_m = None
            self._updated_at = None

    def status(self) -> HeadingSample:
        now = time.monotonic()
        with self._lock:
            yaw = self._yaw_rad
            x_m = self._x_m
            y_m = self._y_m
            updated = self._updated_at
            error = self._error
            position_error = self._position_error
        age = None if updated is None else max(0.0, now - updated)
        healthy = yaw is not None and age is not None and age <= self.maximum_age_s
        position_healthy = bool(
            x_m is not None
            and y_m is not None
            and age is not None
            and age <= self.maximum_age_s
        )
        return HeadingSample(
            yaw,
            age,
            healthy,
            None if healthy else error,
            x_m,
            y_m,
            position_healthy,
            None if position_healthy else position_error,
        )

    def _on_state(self, message: object) -> None:
        try:
            imu_state = getattr(message, "imu_state")
            yaw = normalize_angle(float(getattr(imu_state, "rpy")[2]))
            if not math.isfinite(yaw):
                raise ValueError("non-finite yaw")
        except Exception as exc:
            with self._lock:
                self._error = f"invalid heading sample: {exc}"
            return
        try:
            position = getattr(message, "position")
            x_m = float(position[0])
            y_m = float(position[1])
            if not math.isfinite(x_m) or not math.isfinite(y_m):
                raise ValueError("non-finite local position")
            position_error = None
        except Exception as exc:
            x_m = None
            y_m = None
            position_error = f"invalid local position sample: {exc}"
        with self._lock:
            self._yaw_rad = yaw
            self._x_m = x_m
            self._y_m = y_m
            self._updated_at = time.monotonic()
            self._error = None
            self._position_error = position_error


def normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def directed_progress(start_rad: float, current_rad: float, direction: float) -> float:
    """Return positive angular progress in the commanded direction."""
    delta = normalize_angle(current_rad - start_rad)
    return max(0.0, delta if direction >= 0.0 else -delta)
