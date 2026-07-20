"""Fresh Go2 heading samples for measured autonomous turns."""

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

    def to_dict(self) -> dict[str, object]:
        return {
            "yaw_rad": None if self.yaw_rad is None else round(self.yaw_rad, 4),
            "age_s": None if self.age_s is None else round(self.age_s, 3),
            "healthy": self.healthy,
            "error": self.error,
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
        self._updated_at: float | None = None
        self._error: str | None = "waiting for rt/sportmodestate"
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
            self._updated_at = None

    def status(self) -> HeadingSample:
        now = time.monotonic()
        with self._lock:
            yaw = self._yaw_rad
            updated = self._updated_at
            error = self._error
        age = None if updated is None else max(0.0, now - updated)
        healthy = yaw is not None and age is not None and age <= self.maximum_age_s
        return HeadingSample(yaw, age, healthy, None if healthy else error)

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
        with self._lock:
            self._yaw_rad = yaw
            self._updated_at = time.monotonic()
            self._error = None


def normalize_angle(angle_rad: float) -> float:
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def directed_progress(start_rad: float, current_rad: float, direction: float) -> float:
    """Return positive angular progress in the commanded direction."""
    delta = normalize_angle(current_rad - start_rad)
    return max(0.0, delta if direction >= 0.0 else -delta)
