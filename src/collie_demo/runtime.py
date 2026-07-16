from __future__ import annotations

import asyncio
import time
from typing import Protocol

from .controller import ApproachController
from .detector import BlueWhaleDetector, annotate
from .motion import MotionError, MotionNotReady, UnitreeMotionAdapter
from .types import BlueWhaleObservation, CameraFrame, VelocityCommand


ARM_CONFIRMATION = "WHALE AND PATH CLEAR"


class CameraProtocol(Protocol):
    def read(self) -> CameraFrame: ...


class RuntimeCommandError(RuntimeError):
    pass


class CollieRuntime:
    def __init__(
        self,
        *,
        camera: CameraProtocol,
        detector: BlueWhaleDetector,
        controller: ApproachController,
        motion: UnitreeMotionAdapter | None,
        motion_enabled: bool,
        allow_unranged_forward: bool,
        loop_hz: float = 8.0,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.controller = controller
        self.motion = motion
        self.motion_enabled = bool(motion_enabled)
        self.allow_unranged_forward = bool(allow_unranged_forward)
        self.loop_hz = float(loop_hz)
        self._state_lock = asyncio.Lock()
        self._action_lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._closing = False
        self._jpeg: bytes | None = None
        self._target: BlueWhaleObservation | None = None
        self._frame_width: int | None = None
        self._frame_count = 0
        self._last_frame_at: float | None = None
        self._last_error = "waiting for camera"
        self._lease: str | None = None
        self._last_pulse_at: float | None = None
        self._forward_elapsed_s = 0.0
        self._command = VelocityCommand(reason="disarmed")

    async def start(self) -> None:
        if self.motion_enabled and self.motion is not None:
            try:
                await self.motion.initialize()
            except MotionNotReady as exc:
                self._last_error = str(exc)
        self._closing = False
        self._task = asyncio.create_task(self._camera_loop())

    async def close(self) -> None:
        self._closing = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        await self.stop("shutdown")
        if self.motion is not None:
            await self.motion.close()

    async def arm(self, confirmation: str) -> dict[str, object]:
        async with self._action_lock:
            if confirmation.strip().upper() != ARM_CONFIRMATION:
                raise RuntimeCommandError(f'type exactly "{ARM_CONFIRMATION}"')
            if not self.motion_enabled or self.motion is None:
                raise RuntimeCommandError("motion backend is disabled")
            if not self.allow_unranged_forward:
                raise RuntimeCommandError("unranged demo motion is disabled")
            async with self._state_lock:
                target = self._target
            if target is None or target.visible_frames < self.controller.config.stable_frames_required:
                raise RuntimeCommandError("blue whale is not stably detected")
            try:
                lease = await self.motion.arm()
            except MotionError as exc:
                raise RuntimeCommandError(str(exc)) from exc
            self._lease = lease
            self._last_pulse_at = None
            self._forward_elapsed_s = 0.0
            self._command = VelocityCommand(reason="armed_waiting_for_hold")
            return await self.status()

    async def pulse(self) -> dict[str, object]:
        async with self._action_lock:
            if self.motion is None or self._lease is None or not self.motion.armed:
                self._lease = None
                raise RuntimeCommandError("motion is not armed")
            now = time.monotonic()
            async with self._state_lock:
                target = self._target
                frame_width = self._frame_width
            if self._last_pulse_at is not None and self._command.forward_mps > 0.0:
                self._forward_elapsed_s += min(0.25, max(0.0, now - self._last_pulse_at))
            command = self.controller.plan(
                target,
                frame_width=frame_width,
                now_monotonic_s=now,
                forward_elapsed_s=self._forward_elapsed_s,
                allow_unranged_forward=self.allow_unranged_forward,
            )
            unsafe_reasons = {
                "blue_whale_not_found",
                "blue_whale_not_stable",
                "blue_whale_stale",
                "frame_geometry_missing",
                "unranged_forward_disabled",
                "forward_budget_complete",
            }
            if command.reason in unsafe_reasons:
                await self._release_locked(command.reason)
                raise RuntimeCommandError(command.reason)
            try:
                self._command = await self.motion.send(self._lease, command)
            except MotionError as exc:
                self._lease = None
                self._command = VelocityCommand(reason="motion_fault")
                raise RuntimeCommandError(str(exc)) from exc
            self._last_pulse_at = now
            return await self.status()

    async def stop(self, reason: str = "user_stop") -> dict[str, object]:
        async with self._action_lock:
            if self.motion is not None:
                await self.motion.emergency_stop()
            self._lease = None
            self._last_pulse_at = None
            self._command = VelocityCommand(reason=reason)
            return await self.status()

    async def jpeg(self) -> bytes | None:
        async with self._state_lock:
            return self._jpeg

    async def status(self) -> dict[str, object]:
        async with self._state_lock:
            now = time.monotonic()
            frame_age = None if self._last_frame_at is None else now - self._last_frame_at
            target_age = None if self._target is None else now - self._target.captured_monotonic_s
            target = self._target
            return {
                "ok": frame_age is not None and frame_age < 1.0,
                "frame_count": self._frame_count,
                "frame_age_s": None if frame_age is None else round(frame_age, 3),
                "last_error": self._last_error,
                "blue_whale": None if target is None else target.to_dict(),
                "blue_whale_age_s": None if target_age is None else round(target_age, 3),
                "motion_enabled": self.motion_enabled,
                "allow_unranged_forward": self.allow_unranged_forward,
                "armed": self._lease is not None and self.motion is not None and self.motion.armed,
                "command": self._command.to_dict(),
                "forward_budget_s": self.controller.config.forward_budget_s,
                "forward_elapsed_s": round(self._forward_elapsed_s, 3),
                "forward_remaining_s": round(
                    max(0.0, self.controller.config.forward_budget_s - self._forward_elapsed_s),
                    3,
                ),
                "arm_confirmation": ARM_CONFIRMATION,
                "motion": None if self.motion is None else self.motion.status(),
            }

    async def _camera_loop(self) -> None:
        period = 1.0 / self.loop_hz
        while not self._closing:
            started = time.monotonic()
            lost_while_armed = False
            try:
                frame = await asyncio.to_thread(self.camera.read)
                target = self.detector.detect(frame)
                jpeg = annotate(frame, target)
                async with self._state_lock:
                    self._jpeg = jpeg
                    self._target = target
                    self._frame_width = frame.width
                    self._frame_count += 1
                    self._last_frame_at = time.monotonic()
                    self._last_error = ""
                lost_while_armed = self._lease is not None and (
                    target is None
                    or target.visible_frames < self.controller.config.stable_frames_required
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                async with self._state_lock:
                    self._target = None
                    self._last_error = str(exc)
                lost_while_armed = self._lease is not None
            if lost_while_armed:
                await self.stop("blue_whale_lost")
            await asyncio.sleep(max(0.0, period - (time.monotonic() - started)))

    async def _release_locked(self, reason: str) -> None:
        if self.motion is not None and self._lease is not None:
            try:
                await self.motion.release(self._lease)
            except MotionError:
                await self.motion.emergency_stop()
        self._lease = None
        self._last_pulse_at = None
        self._command = VelocityCommand(reason=reason)

