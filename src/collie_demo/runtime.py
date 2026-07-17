from __future__ import annotations

import asyncio
from pathlib import Path
import time
from typing import Protocol

from .controller import ApproachController
from .detector import BlueWhaleDetector, YellowWhaleDetector, annotate_whales
from .fruit import FruitDetection, annotate_fruits
from .motion import MotionError, MotionNotReady, UnitreeMotionAdapter
from .types import BlueWhaleObservation, CameraFrame, VelocityCommand


ARM_CONFIRMATION = "WHALE AND PATH CLEAR"


class CameraProtocol(Protocol):
    def read(self) -> CameraFrame: ...


class ProduceDetectorProtocol(Protocol):
    model_path: Path
    confidence: float
    names: dict[int, str]

    def detect(self, bgr: object) -> list[FruitDetection]: ...


class RuntimeCommandError(RuntimeError):
    pass


class CollieRuntime:
    def __init__(
        self,
        *,
        camera: CameraProtocol,
        detector: BlueWhaleDetector,
        yellow_detector: YellowWhaleDetector | None = None,
        controller: ApproachController,
        motion: UnitreeMotionAdapter | None,
        motion_enabled: bool,
        allow_unranged_forward: bool,
        produce_detector: ProduceDetectorProtocol | None = None,
        loop_hz: float = 8.0,
    ) -> None:
        self.camera = camera
        self.detectors = {
            "blue": detector,
            "yellow": yellow_detector or YellowWhaleDetector(),
        }
        self.controller = controller
        self.motion = motion
        self.motion_enabled = bool(motion_enabled)
        self.allow_unranged_forward = bool(allow_unranged_forward)
        self.produce_detector = produce_detector
        self.loop_hz = float(loop_hz)
        self._state_lock = asyncio.Lock()
        self._action_lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._produce_task: asyncio.Task[None] | None = None
        self._closing = False
        self._jpeg: bytes | None = None
        self._latest_frame: CameraFrame | None = None
        self._selected_color = "blue"
        self._targets: dict[str, BlueWhaleObservation | None] = {
            color: None for color in self.detectors
        }
        self._target: BlueWhaleObservation | None = None
        self._produce_detections: list[FruitDetection] = []
        self._produce_frame_id: int | None = None
        self._produce_last_at: float | None = None
        self._produce_inference_ms: float | None = None
        self._produce_error = "waiting for first inference" if produce_detector else "disabled"
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
        if self.produce_detector is not None:
            self._produce_task = asyncio.create_task(self._produce_loop())

    async def close(self) -> None:
        self._closing = True
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._produce_task is not None:
            self._produce_task.cancel()
            try:
                await self._produce_task
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
                selected_color = self._selected_color
            if target is None or target.visible_frames < self.controller.config.stable_frames_required:
                raise RuntimeCommandError(
                    f"{selected_color} whale is not stably detected"
                )
            try:
                lease = await self.motion.arm()
            except MotionError as exc:
                raise RuntimeCommandError(str(exc)) from exc
            self._lease = lease
            self._last_pulse_at = None
            self._forward_elapsed_s = 0.0
            self._command = VelocityCommand(reason="armed_waiting_for_hold")
            return await self.status()

    async def select_target(self, color: str) -> dict[str, object]:
        normalized = color.strip().lower()
        if normalized not in self.detectors:
            choices = ", ".join(sorted(self.detectors))
            raise RuntimeCommandError(f"target must be one of: {choices}")
        async with self._action_lock:
            if self.motion is not None and (
                self._lease is not None or self.motion.armed
            ):
                await self.motion.emergency_stop()
            self._lease = None
            self._last_pulse_at = None
            self._forward_elapsed_s = 0.0
            self._command = VelocityCommand(reason=f"{normalized}_whale_selected")
            async with self._state_lock:
                self._selected_color = normalized
                self._target = self._targets.get(normalized)
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
                "selected_whale_not_found",
                "selected_whale_not_stable",
                "selected_whale_stale",
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
            target_ages = {
                color: None
                if target is None
                else round(now - target.captured_monotonic_s, 3)
                for color, target in self._targets.items()
            }
            produce_age = (
                None if self._produce_last_at is None else now - self._produce_last_at
            )
            target = self._target
            return {
                "ok": frame_age is not None and frame_age < 1.0,
                "frame_count": self._frame_count,
                "frame_age_s": None if frame_age is None else round(frame_age, 3),
                "last_error": self._last_error,
                "selected_target_color": self._selected_color,
                "selected_whale": None if target is None else target.to_dict(),
                "selected_whale_age_s": None
                if target_age is None
                else round(target_age, 3),
                "whales": {
                    color: None if item is None else item.to_dict()
                    for color, item in self._targets.items()
                },
                "whale_ages_s": target_ages,
                "blue_whale": None
                if self._targets.get("blue") is None
                else self._targets["blue"].to_dict(),
                "blue_whale_age_s": target_ages.get("blue"),
                "yellow_whale": None
                if self._targets.get("yellow") is None
                else self._targets["yellow"].to_dict(),
                "yellow_whale_age_s": target_ages.get("yellow"),
                "produce": None
                if self.produce_detector is None
                else {
                    "model_path": str(self.produce_detector.model_path),
                    "confidence_threshold": self.produce_detector.confidence,
                    "classes": self.produce_detector.names,
                    "detections": [item.to_dict() for item in self._produce_detections],
                    "frame_id": self._produce_frame_id,
                    "age_s": None if produce_age is None else round(produce_age, 3),
                    "inference_ms": self._produce_inference_ms,
                    "error": self._produce_error,
                },
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
                targets = {
                    color: detector.detect(frame)
                    for color, detector in self.detectors.items()
                }
                async with self._state_lock:
                    self._latest_frame = frame
                    produce_detections = list(self._produce_detections)
                    selected_color = self._selected_color
                produce_image = annotate_fruits(frame.bgr, produce_detections)
                annotated_frame = CameraFrame(
                    frame.frame_id, frame.captured_monotonic_s, produce_image
                )
                jpeg = annotate_whales(
                    annotated_frame,
                    targets,
                    selected_color=selected_color,
                )
                async with self._state_lock:
                    selected_color = self._selected_color
                    selected_target = targets.get(selected_color)
                    self._jpeg = jpeg
                    self._targets = targets
                    self._target = selected_target
                    self._frame_width = frame.width
                    self._frame_count += 1
                    self._last_frame_at = time.monotonic()
                    self._last_error = ""
                lost_while_armed = self._lease is not None and (
                    selected_target is None
                    or selected_target.visible_frames
                    < self.controller.config.stable_frames_required
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                async with self._state_lock:
                    self._targets = {color: None for color in self.detectors}
                    self._target = None
                    self._last_error = str(exc)
                lost_while_armed = self._lease is not None
            if lost_while_armed:
                await self.stop("selected_whale_lost")
            await asyncio.sleep(max(0.0, period - (time.monotonic() - started)))

    async def _produce_loop(self) -> None:
        assert self.produce_detector is not None
        last_frame_id = 0
        while not self._closing:
            async with self._state_lock:
                frame = self._latest_frame
            if frame is None or frame.frame_id == last_frame_id:
                await asyncio.sleep(0.01)
                continue
            last_frame_id = frame.frame_id
            started = time.monotonic()
            try:
                detections = await asyncio.to_thread(
                    self.produce_detector.detect, frame.bgr.copy()
                )
                inference_ms = (time.monotonic() - started) * 1000.0
                for detection in detections:
                    print(detection.log_line(), flush=True)
                async with self._state_lock:
                    self._produce_detections = detections
                    self._produce_frame_id = frame.frame_id
                    self._produce_last_at = time.monotonic()
                    self._produce_inference_ms = round(inference_ms, 1)
                    self._produce_error = ""
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                async with self._state_lock:
                    self._produce_error = str(exc)
            await asyncio.sleep(0)

    async def _release_locked(self, reason: str) -> None:
        if self.motion is not None and self._lease is not None:
            try:
                await self.motion.release(self._lease)
            except MotionError:
                await self.motion.emergency_stop()
        self._lease = None
        self._last_pulse_at = None
        self._command = VelocityCommand(reason=reason)
