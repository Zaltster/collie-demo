from __future__ import annotations

import asyncio
import math
from pathlib import Path
import time
from typing import Callable, Protocol

import cv2

from .controller import ApproachController
from .fruit import FruitDetection, annotate_fruits, annotate_selected_produce
from .heading import HeadingProviderProtocol, directed_progress
from .matcher import CandidateMatch, FruitInstanceMatcher, MatchResult
from .memory import FruitMemory, encode_jpeg
from .mission import MissionConfig, MissionPhase, MissionTelemetry
from .motion import MotionError, MotionNotReady, UnitreeMotionAdapter
from .types import CameraFrame, TargetObservation, VelocityCommand


ARM_CONFIRMATION = "TARGET AND PATH CLEAR"
NAVIGATION_ARM_CONFIRMATION = "MAP AND PATH CLEAR"
DEMO_CONFIRMATION = "TARGET SAVED AND AREA CLEAR"


class CameraProtocol(Protocol):
    def read(self) -> CameraFrame: ...


class ProduceDetectorProtocol(Protocol):
    model_path: Path
    confidence: float
    class_thresholds: dict[str, float]
    names: dict[int, str]

    def detect(self, bgr: object) -> list[FruitDetection]: ...


class ProduceTrackerProtocol(Protocol):
    def update(
        self, bgr: object
    ) -> tuple[bool, tuple[float, float, float, float]]: ...


ProduceTrackerFactory = Callable[
    [object, tuple[int, int, int, int]], ProduceTrackerProtocol
]


def create_produce_tracker(
    bgr: object, bbox_xywh: tuple[int, int, int, int]
) -> ProduceTrackerProtocol:
    tracker = cv2.TrackerMIL_create()
    tracker.init(bgr, bbox_xywh)
    return tracker


class RuntimeCommandError(RuntimeError):
    pass


class CollieRuntime:
    def __init__(
        self,
        *,
        camera: CameraProtocol,
        controller: ApproachController,
        motion: UnitreeMotionAdapter | None,
        motion_enabled: bool,
        allow_unranged_forward: bool,
        produce_detector: ProduceDetectorProtocol | None = None,
        produce_tracker_factory: ProduceTrackerFactory | None = None,
        loop_hz: float = 8.0,
        produce_revalidation_iou: float = 0.15,
        produce_revalidation_misses_required: int = 3,
        maximum_produce_age_s: float = 0.75,
        follow_period_s: float = 0.05,
        follow_start_timeout_s: float = 1.5,
        navigation_idle_arm_s: float = 30.0,
        navigation_command_lease_s: float = 0.75,
        instance_matcher: FruitInstanceMatcher | None = None,
        heading_provider: HeadingProviderProtocol | None = None,
        mission_config: MissionConfig | None = None,
    ) -> None:
        self.camera = camera
        self.controller = controller
        self.motion = motion
        self.motion_enabled = bool(motion_enabled)
        self.allow_unranged_forward = bool(allow_unranged_forward)
        self.produce_detector = produce_detector
        # A local OpenCV tracker is optional. On the Go2 Jetson, MIL tracker
        # updates and repeated tracker initialization can take longer than the
        # 350 ms target-freshness safety window. The production runtime uses
        # the GPU YOLO detections as the authoritative track instead. Tests or
        # other deployments can still opt into a local tracker explicitly.
        self.produce_tracker_factory = produce_tracker_factory
        self.loop_hz = float(loop_hz)
        if not 0.0 <= produce_revalidation_iou <= 1.0:
            raise ValueError("produce_revalidation_iou must be between 0 and 1")
        self.produce_revalidation_iou = float(produce_revalidation_iou)
        if produce_revalidation_misses_required < 1:
            raise ValueError("produce_revalidation_misses_required must be positive")
        self.produce_revalidation_misses_required = int(
            produce_revalidation_misses_required
        )
        if maximum_produce_age_s <= 0.0:
            raise ValueError("maximum_produce_age_s must be positive")
        self.maximum_produce_age_s = float(maximum_produce_age_s)
        if follow_period_s <= 0.0:
            raise ValueError("follow_period_s must be positive")
        self.follow_period_s = float(follow_period_s)
        if follow_start_timeout_s <= 0.0:
            raise ValueError("follow_start_timeout_s must be positive")
        self.follow_start_timeout_s = float(follow_start_timeout_s)
        if navigation_idle_arm_s <= 0.0:
            raise ValueError("navigation_idle_arm_s must be positive")
        self.navigation_idle_arm_s = float(navigation_idle_arm_s)
        if navigation_command_lease_s <= 0.0:
            raise ValueError("navigation_command_lease_s must be positive")
        self.navigation_command_lease_s = float(navigation_command_lease_s)
        self.instance_matcher = instance_matcher or FruitInstanceMatcher()
        self.heading_provider = heading_provider
        self.mission_config = mission_config or MissionConfig()
        self._state_lock = asyncio.Lock()
        self._action_lock = asyncio.Lock()
        self._task: asyncio.Task[None] | None = None
        self._produce_task: asyncio.Task[None] | None = None
        self._closing = False
        self._jpeg: bytes | None = None
        self._latest_frame: CameraFrame | None = None
        self._selected_target_name: str | None = None
        self._selected_target_hint: tuple[int, int] | None = None
        self._target: TargetObservation | None = None
        self._produce_tracker: ProduceTrackerProtocol | None = None
        self._produce_tracker_label: str | None = None
        self._produce_visible_frames = 0
        self._produce_verified_at: float | None = None
        self._produce_revalidation_failures = 0
        self._produce_detections: list[FruitDetection] = []
        self._produce_frame: CameraFrame | None = None
        self._produce_frame_id: int | None = None
        self._produce_last_at: float | None = None
        self._produce_inference_ms: float | None = None
        self._produce_error = "waiting for first inference" if produce_detector else "disabled"
        self._frame_width: int | None = None
        self._frame_count = 0
        self._last_frame_at: float | None = None
        self._last_error = "waiting for camera"
        self._lease: str | None = None
        self._motion_owner: str | None = None
        self._navigation_deadline: float | None = None
        self._navigation_watchdog_task: asyncio.Task[None] | None = None
        self._last_pulse_at: float | None = None
        self._forward_elapsed_s = 0.0
        self._command = VelocityCommand(reason="disarmed")
        self._follow_task: asyncio.Task[None] | None = None
        self._follow_start_generation = 0
        self._fruit_memory: FruitMemory | None = None
        self._mission = MissionTelemetry()
        self._mission_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        if self.heading_provider is not None:
            self.heading_provider.start()
        if self.motion_enabled and self.motion is not None:
            try:
                await self.motion.initialize()
            except MotionNotReady as exc:
                self._last_error = str(exc)
        self._closing = False
        self._task = asyncio.create_task(self._camera_loop())
        if self.produce_detector is not None:
            self._produce_task = asyncio.create_task(self._produce_loop())
        self._navigation_watchdog_task = asyncio.create_task(
            self._navigation_watchdog_loop()
        )

    async def close(self) -> None:
        self._closing = True
        self._follow_start_generation += 1
        if self._mission_task is not None:
            self._mission_task.cancel()
            try:
                await self._mission_task
            except asyncio.CancelledError:
                pass
            self._mission_task = None
        if self._follow_task is not None:
            self._follow_task.cancel()
            try:
                await self._follow_task
            except asyncio.CancelledError:
                pass
            self._follow_task = None
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
        if self._navigation_watchdog_task is not None:
            self._navigation_watchdog_task.cancel()
            try:
                await self._navigation_watchdog_task
            except asyncio.CancelledError:
                pass
            self._navigation_watchdog_task = None
        await self.stop("shutdown")
        if self.motion is not None:
            await self.motion.close()
        if self.heading_provider is not None:
            self.heading_provider.close()

    async def arm(self, confirmation: str) -> dict[str, object]:
        async with self._action_lock:
            if confirmation.strip().upper() != ARM_CONFIRMATION:
                raise RuntimeCommandError(f'type exactly "{ARM_CONFIRMATION}"')
            if not self.motion_enabled or self.motion is None:
                raise RuntimeCommandError("motion backend is disabled")
            if not self.allow_unranged_forward:
                raise RuntimeCommandError("unranged demo motion is disabled")
            async with self._state_lock:
                readiness = self._follow_readiness_locked(time.monotonic())
            if readiness is not None:
                raise RuntimeCommandError(readiness)
            try:
                lease = await self.motion.arm()
            except MotionError as exc:
                raise RuntimeCommandError(str(exc)) from exc
            self._lease = lease
            self._motion_owner = "fruit"
            self._navigation_deadline = None
            self._last_pulse_at = None
            self._forward_elapsed_s = 0.0
            self._command = VelocityCommand(reason="armed_waiting_for_hold")
            return await self.status()

    async def navigation_arm(self, confirmation: str) -> dict[str, object]:
        """Acquire the factory-avoidance lease for map navigation only."""

        async with self._action_lock:
            if confirmation.strip().upper() != NAVIGATION_ARM_CONFIRMATION:
                raise RuntimeCommandError(
                    f'type exactly "{NAVIGATION_ARM_CONFIRMATION}"'
                )
            if not self.motion_enabled or self.motion is None:
                raise RuntimeCommandError("motion backend is disabled")
            if self._lease is not None or self.motion.armed:
                raise RuntimeCommandError("motion is already owned; stop it first")
            async with self._state_lock:
                self._clear_selection_locked()
            try:
                lease = await self.motion.arm()
            except MotionError as exc:
                raise RuntimeCommandError(str(exc)) from exc
            self._lease = lease
            self._motion_owner = "navigation"
            self._navigation_deadline = (
                time.monotonic() + self.navigation_idle_arm_s
            )
            self._last_pulse_at = None
            self._forward_elapsed_s = 0.0
            self._command = VelocityCommand(reason="navigation_armed_zero")
            return await self.navigation_status()

    async def navigation_command(
        self, forward_mps: float, yaw_rps: float
    ) -> dict[str, object]:
        """Send one bounded map-navigation heartbeat for the active lease."""

        async with self._action_lock:
            if self._motion_owner != "navigation":
                raise RuntimeCommandError("navigation motion is not armed")
            if (
                self.motion is None
                or self._lease is None
                or not self.motion.armed
            ):
                self._lease = None
                self._motion_owner = None
                self._navigation_deadline = None
                raise RuntimeCommandError("navigation motion is not armed")
            command = VelocityCommand(
                forward_mps=float(forward_mps),
                yaw_rps=float(yaw_rps),
                reason="map_navigation",
            )
            try:
                self._command = await self.motion.send(self._lease, command)
            except (MotionError, ValueError) as exc:
                self._lease = None
                self._motion_owner = None
                self._navigation_deadline = None
                self._command = VelocityCommand(reason="navigation_motion_fault")
                raise RuntimeCommandError(str(exc)) from exc
            self._navigation_deadline = (
                time.monotonic() + self.navigation_command_lease_s
            )
            return await self.navigation_status()

    async def _direct_turn_arm(self) -> None:
        """Acquire the private yaw-only SportClient lease for the demo turn."""

        async with self._action_lock:
            if not self.motion_enabled or self.motion is None:
                raise RuntimeCommandError("motion backend is disabled")
            if self._lease is not None or self.motion.armed:
                raise RuntimeCommandError("motion is already owned; stop it first")
            try:
                lease = await self.motion.arm_direct_yaw()
            except MotionError as exc:
                raise RuntimeCommandError(str(exc)) from exc
            self._lease = lease
            self._motion_owner = "direct_turn"
            self._navigation_deadline = None
            self._last_pulse_at = None
            self._command = VelocityCommand(reason="direct_turn_armed_zero")

    async def _direct_turn_command(self, yaw_rps: float) -> None:
        """Renew one yaw-only heartbeat; translation is impossible in this mode."""

        async with self._action_lock:
            if self._motion_owner != "direct_turn":
                raise RuntimeCommandError("direct turn motion is not armed")
            if (
                self.motion is None
                or self._lease is None
                or not self.motion.armed
            ):
                self._lease = None
                self._motion_owner = None
                raise RuntimeCommandError("direct turn motion is not armed")
            try:
                self._command = await self.motion.send_direct_yaw(
                    self._lease,
                    yaw_rps,
                    "measured_direct_turn",
                )
            except (MotionError, ValueError) as exc:
                self._lease = None
                self._motion_owner = None
                self._command = VelocityCommand(reason="direct_turn_motion_fault")
                raise RuntimeCommandError(str(exc)) from exc

    async def navigation_status(self) -> dict[str, object]:
        now = time.monotonic()
        motion_status = None if self.motion is None else self.motion.status()
        available = bool(
            self.motion_enabled
            and motion_status is not None
            and motion_status["initialized"]
            and motion_status["fault"] is None
        )
        armed = bool(
            self._motion_owner == "navigation"
            and self._lease is not None
            and self.motion is not None
            and self.motion.armed
        )
        return {
            "available": available,
            "armed": armed,
            "owner": self._motion_owner,
            "fault": None if motion_status is None else motion_status["fault"],
            "deadline_s": None
            if self._navigation_deadline is None
            else round(max(0.0, self._navigation_deadline - now), 3),
            "command": self._command.to_dict(),
            "limits": None if motion_status is None else motion_status["limits"],
        }

    async def start_follow(self, confirmation: str) -> dict[str, object]:
        generation = self._follow_start_generation
        deadline = time.monotonic() + self.follow_start_timeout_s
        print(
            "follow event=start_requested "
            f"generation={generation} "
            f"timeout_s={self.follow_start_timeout_s:.2f}",
            flush=True,
        )
        while True:
            if generation != self._follow_start_generation:
                print(
                    "follow event=start_cancelled "
                    f"expected_generation={generation} "
                    f"actual_generation={self._follow_start_generation}",
                    flush=True,
                )
                raise RuntimeCommandError("follow start cancelled")
            async with self._state_lock:
                selected_name = self._selected_target_name
                readiness = self._follow_readiness_locked(time.monotonic())
            if selected_name is None:
                raise RuntimeCommandError("select a detected fruit first")
            if readiness is None:
                break
            if readiness not in {
                "selected_target_not_stable",
                "selected_target_not_revalidated",
            }:
                raise RuntimeCommandError(readiness)
            if time.monotonic() >= deadline:
                raise RuntimeCommandError(readiness)
            await asyncio.sleep(0.025)

        if generation != self._follow_start_generation:
            raise RuntimeCommandError("follow start cancelled")
        await self.arm(confirmation)
        if generation != self._follow_start_generation:
            await self.stop("follow_start_cancelled")
            raise RuntimeCommandError("follow start cancelled")
        if self._follow_task is not None and not self._follow_task.done():
            await self.stop("follow_already_active")
            raise RuntimeCommandError("follow is already active")
        self._follow_task = asyncio.create_task(self._follow_loop())
        print(
            "follow event=armed "
            f"generation={generation} "
            f"target={selected_name}",
            flush=True,
        )
        return await self.status()

    async def select_target(
        self,
        target_name: str,
        preferred_center: tuple[int, int] | None = None,
        *,
        confirmed_visible_frames: int = 1,
    ) -> dict[str, object]:
        canonical_name = self._canonical_target_name(target_name)
        confirmed_visible_frames = max(1, int(confirmed_visible_frames))
        self._follow_start_generation += 1
        async with self._action_lock:
            if self.motion is not None and (
                self._lease is not None or self.motion.armed
            ):
                await self.motion.emergency_stop()
            self._lease = None
            self._last_pulse_at = None
            self._forward_elapsed_s = 0.0
            self._command = VelocityCommand(reason="fruit_selected")
            frame: CameraFrame | None = None
            detection: FruitDetection | None = None
            selection_error: str | None = None
            async with self._state_lock:
                frame = self._latest_frame
                detection = self._best_produce_detection_locked(
                    canonical_name, preferred_center
                )
                produce_age = (
                    None
                    if self._produce_last_at is None
                    else time.monotonic() - self._produce_last_at
                )
                if (
                    frame is None
                    or detection is None
                    or produce_age is None
                    or produce_age > self.maximum_produce_age_s
                ):
                    self._clear_selection_locked()
                    selection_error = (
                        f"{canonical_name} is no longer freshly detected; select it again"
                    )
                else:
                    self._selected_target_name = canonical_name
                    self._selected_target_hint = preferred_center
                    self._clear_produce_tracker_locked()
                    self._target = None
            if selection_error is not None:
                raise RuntimeCommandError(selection_error)
            if frame is not None and detection is not None:
                tracker: ProduceTrackerProtocol | None = None
                try:
                    if self.produce_tracker_factory is not None:
                        tracker = await asyncio.to_thread(
                            self.produce_tracker_factory,
                            frame.bgr.copy(),
                            self._detection_bbox_xywh(detection),
                        )
                except Exception as exc:
                    async with self._state_lock:
                        self._clear_selection_locked()
                    raise RuntimeCommandError(
                        f"could not initialize {canonical_name} tracker: {exc}"
                    ) from exc
                observation = self._observation_from_bbox(
                    frame,
                    self._detection_bbox_xywh(detection),
                    confidence=detection.confidence,
                    visible_frames=confirmed_visible_frames,
                )
                async with self._state_lock:
                    if self._selected_target_name == canonical_name:
                        self._produce_tracker = tracker
                        self._produce_tracker_label = (
                            canonical_name if tracker is not None else None
                        )
                        self._produce_visible_frames = confirmed_visible_frames
                        self._produce_verified_at = self._produce_last_at
                        self._produce_revalidation_failures = 0
                        self._target = observation
            return await self.status()

    async def remember_target(
        self,
        target_name: str,
        preferred_center: tuple[int, int] | None = None,
    ) -> dict[str, object]:
        """Capture several detector-aligned views without arming motion."""

        if not self.mission_config.enabled:
            raise RuntimeCommandError("fruit-memory demo is disabled")
        canonical_name = self._canonical_target_name(target_name)
        await self.stop("memory_capture")
        async with self._state_lock:
            # A new capture attempt starts a new round. Never leave an older
            # fruit silently armed as the fallback if this capture is poor.
            self._fruit_memory = None
            self._clear_selection_locked()
            self._mission = MissionTelemetry(
                phase=MissionPhase.LEARNING,
                reason="capturing_reference_views",
                started_monotonic_s=time.monotonic(),
            )

        samples = []
        reference_crop = None
        reference_bbox: tuple[int, int, int, int] | None = None
        reference_quality = -1.0
        last_frame_id: int | None = None
        hint = preferred_center
        deadline = time.monotonic() + self.mission_config.capture_timeout_s
        while (
            len(samples) < self.mission_config.capture_samples
            and time.monotonic() < deadline
        ):
            async with self._state_lock:
                frame = self._produce_frame
                frame_id = self._produce_frame_id
                detection = self._best_produce_detection_locked(
                    canonical_name, hint
                )
                produce_age = (
                    None
                    if self._produce_last_at is None
                    else time.monotonic() - self._produce_last_at
                )
            if (
                frame is None
                or frame_id is None
                or frame_id == last_frame_id
                or detection is None
                or produce_age is None
                or produce_age > self.maximum_produce_age_s
            ):
                await asyncio.sleep(0.02)
                continue
            last_frame_id = frame_id
            hint = detection.center
            try:
                descriptor, crop = await asyncio.to_thread(
                    self.instance_matcher.encoder.encode_bbox,
                    frame.bgr.copy(),
                    detection.bbox_xyxy,
                )
            except ValueError:
                await asyncio.sleep(0)
                continue
            samples.append(descriptor)
            if descriptor.quality > reference_quality:
                reference_quality = descriptor.quality
                reference_crop = crop
                reference_bbox = detection.bbox_xyxy
            await asyncio.sleep(0)

        if (
            len(samples) < self.mission_config.capture_samples
            or reference_crop is None
            or reference_bbox is None
        ):
            async with self._state_lock:
                self._mission.phase = MissionPhase.ABORTED
                self._mission.reason = "insufficient_reference_views"
            raise RuntimeCommandError(
                "keep the fruit clearly visible and steady, then save it again"
            )
        memory = FruitMemory.create(
            label=canonical_name,
            samples=samples,
            reference_jpeg=await asyncio.to_thread(encode_jpeg, reference_crop),
            reference_bbox_xyxy=reference_bbox,
        )
        async with self._state_lock:
            self._fruit_memory = memory
            self._clear_selection_locked()
            self._mission = MissionTelemetry(
                phase=MissionPhase.MEMORIZED,
                reason="fruit_saved",
            )
        return await self.status()

    async def clear_memory(self) -> dict[str, object]:
        await self.stop("memory_reset")
        async with self._state_lock:
            self._fruit_memory = None
            self._clear_selection_locked()
            self._mission = MissionTelemetry(
                phase=MissionPhase.IDLE,
                reason="memory_reset",
            )
        return await self.status()

    async def memory_reference_jpeg(self) -> bytes | None:
        async with self._state_lock:
            memory = self._fruit_memory
        return None if memory is None else memory.reference_jpeg

    async def start_demo(self, confirmation: str) -> dict[str, object]:
        if confirmation.strip().upper() != DEMO_CONFIRMATION:
            raise RuntimeCommandError(f'type exactly "{DEMO_CONFIRMATION}"')
        if not self.mission_config.enabled:
            raise RuntimeCommandError("fruit-memory demo is disabled")
        if not self.mission_config.autonomous_turn_enabled:
            raise RuntimeCommandError("autonomous turn is disabled")
        if self._mission_task is not None and not self._mission_task.done():
            raise RuntimeCommandError("fruit-memory demo is already active")
        if not self.motion_enabled or self.motion is None:
            raise RuntimeCommandError("motion backend is disabled")
        if self.heading_provider is None or not self.heading_provider.status().healthy:
            raise RuntimeCommandError("fresh Go2 heading is unavailable")
        async with self._state_lock:
            if self._fruit_memory is None:
                raise RuntimeCommandError("save a fruit first")
            now = time.monotonic()
            camera_fresh = (
                self._last_frame_at is not None
                and now - self._last_frame_at < 1.0
            )
            produce_fresh = (
                self._produce_last_at is not None
                and now - self._produce_last_at < self.maximum_produce_age_s
                and not self._produce_error
            )
        if not camera_fresh or not produce_fresh:
            raise RuntimeCommandError("camera or fruit detector is not fresh")
        await self.stop("demo_start_reset")
        async with self._state_lock:
            self._clear_selection_locked()
            self._mission = MissionTelemetry(
                phase=MissionPhase.TURNING,
                reason="starting_measured_turn",
                started_monotonic_s=time.monotonic(),
            )
        self._mission_task = asyncio.create_task(self._demo_loop())
        return await self.status()

    async def stop_demo(self) -> dict[str, object]:
        await self.stop("demo_operator_stop")
        async with self._state_lock:
            self._mission.phase = MissionPhase.ABORTED
            self._mission.reason = "operator_stop"
        return await self.status()

    async def pulse(self) -> dict[str, object]:
        async with self._action_lock:
            if self._motion_owner != "fruit":
                raise RuntimeCommandError("fruit motion is not armed")
            if (
                self.motion is None
                or self._lease is None
                or not self.motion.armed
            ):
                self._lease = None
                self._motion_owner = None
                if self._command.forward_mps != 0.0 or self._command.yaw_rps != 0.0:
                    self._command = VelocityCommand(reason="motion_not_armed")
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
                "selected_target_not_found",
                "selected_target_not_stable",
                "selected_target_stale",
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
        current_task = asyncio.current_task()
        mission_task = self._mission_task
        # Target-loss stops are the safety brake for an active approach. Keep
        # the mission monitor alive just long enough to classify the stopped
        # run as reached-camera-edge versus lost-before-arrival. All operator,
        # watchdog, navigation, and shutdown stops still cancel it immediately.
        preserve_mission_for_classification = reason in {
            "selected_target_lost",
            "selected_target_not_revalidated",
        }
        cancelled_mission = bool(
            mission_task is not None
            and mission_task is not current_task
            and not mission_task.done()
            and not preserve_mission_for_classification
        )
        if cancelled_mission and mission_task is not None:
            mission_task.cancel()
            try:
                await mission_task
            except asyncio.CancelledError:
                pass
            if self._mission_task is mission_task:
                self._mission_task = None
        self._follow_start_generation += 1
        async with self._action_lock:
            if self.motion is not None:
                await self.motion.emergency_stop()
            self._lease = None
            self._motion_owner = None
            self._navigation_deadline = None
            self._last_pulse_at = None
            self._command = VelocityCommand(reason=reason)
        if cancelled_mission:
            async with self._state_lock:
                if self._mission.phase not in {
                    MissionPhase.SUCCESS,
                    MissionPhase.ABORTED,
                }:
                    self._mission.phase = MissionPhase.ABORTED
                    self._mission.reason = reason
        return await self.status()

    async def jpeg(self) -> bytes | None:
        async with self._state_lock:
            return self._jpeg

    async def raw_jpeg(self) -> bytes | None:
        """Return the latest unannotated frame for evaluation and capture."""
        async with self._state_lock:
            frame = self._latest_frame
        if frame is None:
            return None
        encoded_ok, encoded = await asyncio.to_thread(
            cv2.imencode,
            ".jpg",
            frame.bgr,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        if not encoded_ok:
            raise RuntimeError("could not encode raw camera frame")
        return encoded.tobytes()

    async def status(self) -> dict[str, object]:
        heading = (
            None
            if self.heading_provider is None
            else self.heading_provider.status()
        )
        async with self._state_lock:
            now = time.monotonic()
            frame_age = None if self._last_frame_at is None else now - self._last_frame_at
            target_age = None if self._target is None else now - self._target.captured_monotonic_s
            produce_age = (
                None if self._produce_last_at is None else now - self._produce_last_at
            )
            target = self._target
            device_status = self._produce_device_status()
            camera_live = frame_age is not None and frame_age < 1.0
            produce_live = bool(
                self.produce_detector is not None
                and produce_age is not None
                and produce_age < 1.0
                and not self._produce_error
            )
            gpu_ready = bool(
                device_status.get("cuda_available")
                and str(device_status.get("resolved", "")).startswith("cuda")
            )
            motion_status = None if self.motion is None else self.motion.status()
            motion_ready = bool(
                self.motion_enabled
                and motion_status is not None
                and motion_status["initialized"]
                and motion_status["fault"] is None
            )
            follow_active = bool(
                self._follow_task is not None
                and not self._follow_task.done()
                and self._lease is not None
                and self.motion is not None
                and self.motion.armed
            )
            follow_readiness = self._follow_readiness_locked(now)
            can_follow = (
                not follow_active
                and follow_readiness is None
                and motion_ready
            )
            stage_ready = camera_live and produce_live and gpu_ready and motion_ready
            mission_active = bool(
                self._mission_task is not None and not self._mission_task.done()
            )
            if not self.mission_config.enabled:
                demo_readiness = "fruit-memory demo disabled"
            elif not self.mission_config.autonomous_turn_enabled:
                demo_readiness = "autonomous turn disabled"
            elif self._fruit_memory is None:
                demo_readiness = "save a fruit first"
            elif heading is None or not heading.healthy:
                demo_readiness = "fresh Go2 heading unavailable"
            elif not stage_ready:
                demo_readiness = "stage health is not ready"
            elif mission_active:
                demo_readiness = "demo already active"
            elif self._lease is not None or (self.motion is not None and self.motion.armed):
                demo_readiness = "motion is already armed"
            else:
                demo_readiness = "ready"
            mission_status = self._mission.to_dict(now)
            mission_status.update(
                {
                    "enabled": self.mission_config.enabled,
                    "autonomous_turn_enabled": self.mission_config.autonomous_turn_enabled,
                    "direct_turn_enabled": self.mission_config.direct_turn_enabled,
                    "target_policy": "different_instance",
                    "saved_instance_rejection_score": (
                        self.instance_matcher.maximum_saved_similarity
                    ),
                    "turn_rate_rps": self.mission_config.turn_rate_rps,
                    "turn_timeout_s": self.mission_config.turn_timeout_s,
                    "turn_stall_timeout_s": self.mission_config.turn_stall_timeout_s,
                    "turn_stall_min_progress_deg": round(
                        math.degrees(
                            self.mission_config.turn_stall_min_progress_rad
                        ),
                        1,
                    ),
                    "active": mission_active,
                    "can_remember": bool(
                        self.mission_config.enabled
                        and produce_live
                        and self._produce_detections
                        and not mission_active
                    ),
                    "can_start": demo_readiness == "ready",
                    "readiness": demo_readiness,
                    "confirmation": DEMO_CONFIRMATION,
                    "heading": None if heading is None else heading.to_dict(),
                }
            )
            return {
                "ok": camera_live,
                "stage_ready": stage_ready,
                "health": {
                    "camera_live": camera_live,
                    "produce_live": produce_live,
                    "gpu_ready": gpu_ready,
                    "motion_ready": motion_ready,
                },
                "frame_count": self._frame_count,
                "frame_age_s": None if frame_age is None else round(frame_age, 3),
                "last_error": self._last_error,
                "selected_target_name": self._selected_target_name,
                "selected_target_kind": "produce"
                if self._selected_target_name is not None
                else None,
                "selected_target_hint": self._selected_target_hint,
                "selected_target": None if target is None else target.to_dict(),
                "selected_target_age_s": None
                if target_age is None
                else round(target_age, 3),
                "produce": None
                if self.produce_detector is None
                else {
                    "model_path": str(self.produce_detector.model_path),
                    "confidence_threshold": self.produce_detector.confidence,
                    "class_thresholds": getattr(
                        self.produce_detector, "class_thresholds", {}
                    ),
                    "classes": self.produce_detector.names,
                    "detections": [item.to_dict() for item in self._produce_detections],
                    "frame_id": self._produce_frame_id,
                    "age_s": None if produce_age is None else round(produce_age, 3),
                    "inference_ms": self._produce_inference_ms,
                    "error": self._produce_error,
                    "device": device_status,
                    "tracker": {
                        "mode": "yolo+local"
                        if self.produce_tracker_factory is not None
                        else "yolo",
                        "label": self._produce_tracker_label,
                        "target": None
                        if self._produce_tracker_label is None or target is None
                        else target.to_dict(),
                        "last_verified_age_s": None
                        if self._produce_verified_at is None
                        else round(now - self._produce_verified_at, 3),
                        "revalidation_failures": self._produce_revalidation_failures,
                        "revalidation_failures_required": self.produce_revalidation_misses_required,
                    },
                },
                "motion_enabled": self.motion_enabled,
                "allow_unranged_forward": self.allow_unranged_forward,
                "can_follow": can_follow,
                "follow_readiness": "ready"
                if follow_readiness is None
                else follow_readiness,
                "follow_active": follow_active,
                "armed": self._lease is not None and self.motion is not None and self.motion.armed,
                "motion_owner": self._motion_owner,
                "memory": None
                if self._fruit_memory is None
                else self._fruit_memory.to_status(now),
                "mission": mission_status,
                "navigation": await self.navigation_status(),
                "command": self._command.to_dict(),
                "forward_budget_s": self.controller.config.forward_budget_s,
                "forward_elapsed_s": round(self._forward_elapsed_s, 3),
                "forward_remaining_s": round(
                    max(0.0, self.controller.config.forward_budget_s - self._forward_elapsed_s),
                    3,
                ),
                "arm_confirmation": ARM_CONFIRMATION,
                "motion": motion_status,
            }

    async def _camera_loop(self) -> None:
        period = 1.0 / self.loop_hz
        while not self._closing:
            started = time.monotonic()
            lost_while_armed = False
            try:
                frame = await asyncio.to_thread(self.camera.read)
                async with self._state_lock:
                    self._latest_frame = frame
                    produce_detections = list(self._produce_detections)
                    selected_name = self._selected_target_name
                    tracker = self._produce_tracker
                    tracker_label = self._produce_tracker_label
                    tracker_visible_frames = self._produce_visible_frames
                    current_target = self._target
                tracked_produce: TargetObservation | None = None
                if (
                    selected_name is not None
                    and tracker is not None
                    and tracker_label == selected_name
                ):
                    tracker_ok, tracker_bbox = await asyncio.to_thread(
                        tracker.update, frame.bgr
                    )
                    if tracker_ok:
                        tracked_produce = self._observation_from_bbox(
                            frame,
                            tuple(int(round(value)) for value in tracker_bbox),
                            confidence=None,
                            visible_frames=tracker_visible_frames + 1,
                        )
                produce_image = annotate_fruits(frame.bgr, produce_detections)
                produce_image = annotate_selected_produce(
                    produce_image,
                    selected_name,
                    tracked_produce
                    if tracked_produce is not None
                    else current_target,
                )
                encoded_ok, encoded = cv2.imencode(
                    ".jpg", produce_image, [cv2.IMWRITE_JPEG_QUALITY, 88]
                )
                if not encoded_ok:
                    raise RuntimeError("could not encode annotated frame")
                jpeg = encoded.tobytes()
                async with self._state_lock:
                    current_name = self._selected_target_name
                    if (
                        current_name is not None
                        and current_name == selected_name
                        and self._produce_tracker is tracker
                        and self._produce_tracker_label == current_name
                    ):
                        selected_target = tracked_produce
                        self._produce_visible_frames = (
                            0 if tracked_produce is None else tracked_produce.visible_frames
                        )
                        if tracked_produce is not None:
                            # Keep the reacquisition hint attached to the exact
                            # object as the fast tracker follows it between YOLO
                            # inference frames.
                            self._selected_target_hint = tracked_produce.center
                        if tracked_produce is None:
                            self._clear_selection_locked()
                    else:
                        selected_target = self._target
                    self._jpeg = jpeg
                    self._target = selected_target
                    self._frame_width = frame.width
                    self._frame_count += 1
                    self._last_frame_at = time.monotonic()
                    self._last_error = ""
                lost_while_armed = self._motion_owner == "fruit" and self._lease is not None and (
                    selected_target is None
                    or selected_target.visible_frames
                    < self.controller.config.stable_frames_required
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                async with self._state_lock:
                    self._last_error = str(exc)
                    # Unitree's camera occasionally returns one malformed JPEG.
                    # A single bad sample must not erase an otherwise fresh,
                    # YOLO-verified selection.  Preserve the last observation
                    # until the same bounded target-age rule used by the motion
                    # controller says the camera outage is genuinely stale.
                    now = time.monotonic()
                    last_frame_age = (
                        None
                        if self._last_frame_at is None
                        else now - self._last_frame_at
                    )
                    camera_is_stale = (
                        last_frame_age is None
                        or last_frame_age
                        > self.controller.config.maximum_target_age_s
                    )
                    if camera_is_stale:
                        self._clear_selection_locked()
                lost_while_armed = (
                    camera_is_stale
                    and self._motion_owner == "fruit"
                    and self._lease is not None
                )
            if lost_while_armed:
                await self.stop("selected_target_lost")
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
                refresh_tracker_for: tuple[str, FruitDetection, int] | None = None
                revalidation_failed = False
                async with self._state_lock:
                    self._produce_detections = detections
                    self._produce_frame = frame
                    self._produce_frame_id = frame.frame_id
                    self._produce_last_at = time.monotonic()
                    self._produce_inference_ms = round(inference_ms, 1)
                    self._produce_error = ""
                    selected_name = self._selected_target_name
                    selected_target = self._target
                    selected_detection = None
                    if selected_name is not None:
                        preferred_center = (
                            selected_target.center
                            if selected_target is not None
                            else self._selected_target_hint
                        )
                        selected_detection = self._best_produce_detection_locked(
                            selected_name, preferred_center
                        )
                        tracker_is_active = (
                            self._produce_tracker is not None
                            and self._produce_tracker_label == selected_name
                        )
                        detection_matches_track = (
                            selected_detection is not None
                            and selected_target is not None
                            and self._target_detection_iou(
                                selected_target, selected_detection
                            )
                            >= self.produce_revalidation_iou
                        )
                        if selected_detection is None or (
                            selected_target is not None and not detection_matches_track
                        ):
                            self._produce_revalidation_failures += 1
                            if self._produce_revalidation_expired_locked(
                                time.monotonic()
                            ):
                                self._clear_selection_locked()
                                revalidation_failed = True
                        else:
                            self._produce_revalidation_failures = 0
                            visible_frames = (
                                1
                                if selected_target is None
                                else selected_target.visible_frames + 1
                            )
                            observation = self._observation_from_bbox(
                                frame,
                                self._detection_bbox_xywh(selected_detection),
                                confidence=selected_detection.confidence,
                                visible_frames=visible_frames,
                            )
                            if (
                                self._target is None
                                or observation.captured_monotonic_s
                                >= self._target.captured_monotonic_s
                            ):
                                self._target = observation
                            self._produce_visible_frames = visible_frames
                            self._produce_verified_at = time.monotonic()
                            self._selected_target_hint = selected_detection.center
                            if (
                                self.produce_tracker_factory is not None
                                and not tracker_is_active
                            ):
                                refresh_tracker_for = (
                                    selected_name,
                                    selected_detection,
                                    visible_frames,
                                )
                if revalidation_failed:
                    await self.stop("selected_target_not_revalidated")
                if refresh_tracker_for is not None:
                    refresh_name, selected_detection, visible_frames = refresh_tracker_for
                    tracker_factory = self.produce_tracker_factory
                    if tracker_factory is None:
                        continue
                    tracker = await asyncio.to_thread(
                        tracker_factory,
                        frame.bgr.copy(),
                        self._detection_bbox_xywh(selected_detection),
                    )
                    async with self._state_lock:
                        if (
                            self._selected_target_name == refresh_name
                            and refresh_name is not None
                        ):
                            if self._produce_tracker is None:
                                self._produce_tracker = tracker
                                self._produce_tracker_label = refresh_name
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                revalidation_failed = False
                async with self._state_lock:
                    self._produce_detections = []
                    self._produce_frame = frame
                    self._produce_frame_id = frame.frame_id
                    self._produce_last_at = time.monotonic()
                    self._produce_inference_ms = round(
                        (time.monotonic() - started) * 1000.0, 1
                    )
                    self._produce_error = str(exc)
                    if self._selected_target_name is not None:
                        self._produce_revalidation_failures += 1
                        if self._produce_revalidation_expired_locked(
                            time.monotonic()
                        ):
                            self._clear_selection_locked()
                            revalidation_failed = True
                if revalidation_failed:
                    await self.stop("selected_target_not_revalidated")
            await asyncio.sleep(0)

    async def _demo_loop(self) -> None:
        current_task = asyncio.current_task()
        try:
            await self._run_measured_turn()
            match = await self._search_for_memory()
            await self._start_memory_approach(match)
            await self._monitor_memory_approach()
        except asyncio.CancelledError:
            raise
        except RuntimeCommandError as exc:
            reason = str(exc)
            await self.stop(f"demo_abort:{reason}")
            async with self._state_lock:
                self._mission.phase = MissionPhase.ABORTED
                self._mission.reason = reason
        except Exception as exc:
            reason = f"demo_internal_error:{exc}"
            await self.stop(reason)
            async with self._state_lock:
                self._mission.phase = MissionPhase.ABORTED
                self._mission.reason = reason
                self._last_error = reason
        finally:
            if self._mission_task is current_task:
                self._mission_task = None

    async def _run_measured_turn(self) -> None:
        if self.heading_provider is None:
            raise RuntimeCommandError("fresh Go2 heading is unavailable")
        initial = self.heading_provider.status()
        if not initial.healthy or initial.yaw_rad is None:
            raise RuntimeCommandError("fresh Go2 heading is unavailable")
        async with self._state_lock:
            self._mission.phase = MissionPhase.TURNING
            self._mission.reason = "measured_180_degree_turn"
            self._mission.turn_progress_rad = 0.0
        if self.mission_config.direct_turn_enabled:
            await self._direct_turn_arm()
        else:
            await self.navigation_arm(NAVIGATION_ARM_CONFIRMATION)
        direction = 1.0
        turn_started = time.monotonic()
        deadline = turn_started + self.mission_config.turn_timeout_s
        last_log_at = 0.0
        print(
            "turn event=start "
            f"initial_yaw_rad={initial.yaw_rad:.4f} "
            f"command_yaw_rps={direction * self.mission_config.turn_rate_rps:.3f} "
            f"target_deg={math.degrees(self.mission_config.turn_angle_rad):.1f} "
            f"timeout_s={self.mission_config.turn_timeout_s:.2f} "
            f"mode={'direct_yaw' if self.mission_config.direct_turn_enabled else 'avoidance'}",
            flush=True,
        )
        while time.monotonic() < deadline:
            now = time.monotonic()
            sample = self.heading_provider.status()
            if not sample.healthy or sample.yaw_rad is None:
                raise RuntimeCommandError("Go2 heading became stale during turn")
            progress = directed_progress(initial.yaw_rad, sample.yaw_rad, direction)
            async with self._state_lock:
                self._mission.turn_progress_rad = progress
            if (
                progress
                >= self.mission_config.turn_angle_rad
                - self.mission_config.turn_tolerance_rad
            ):
                print(
                    "turn event=complete "
                    f"elapsed_s={now - turn_started:.3f} "
                    f"current_yaw_rad={sample.yaw_rad:.4f} "
                    f"progress_deg={math.degrees(progress):.1f}",
                    flush=True,
                )
                await self.stop("demo_turn_complete")
                return
            elapsed = now - turn_started
            if (
                self.mission_config.direct_turn_enabled
                and elapsed >= self.mission_config.turn_stall_timeout_s
                and progress < self.mission_config.turn_stall_min_progress_rad
            ):
                print(
                    "turn event=stalled "
                    f"elapsed_s={elapsed:.3f} "
                    f"current_yaw_rad={sample.yaw_rad:.4f} "
                    f"progress_deg={math.degrees(progress):.1f}",
                    flush=True,
                )
                raise RuntimeCommandError(
                    "direct turn stalled at "
                    f"{math.degrees(progress):.1f} degrees"
                )
            if self.mission_config.direct_turn_enabled:
                await self._direct_turn_command(
                    direction * self.mission_config.turn_rate_rps
                )
            else:
                await self.navigation_command(
                    0.0, direction * self.mission_config.turn_rate_rps
                )
            if now - last_log_at >= 0.25:
                last_log_at = now
                motion_status = None if self.motion is None else self.motion.status()
                print(
                    "turn event=progress "
                    f"elapsed_s={elapsed:.3f} "
                    f"current_yaw_rad={sample.yaw_rad:.4f} "
                    f"progress_deg={math.degrees(progress):.1f} "
                    f"command_yaw_rps={direction * self.mission_config.turn_rate_rps:.3f} "
                    f"motion_armed={None if motion_status is None else motion_status['armed']} "
                    f"motion_mode={None if motion_status is None else motion_status['mode']}",
                    flush=True,
                )
            await asyncio.sleep(0.05)
        final = self.heading_provider.status()
        print(
            "turn event=timeout "
            f"elapsed_s={time.monotonic() - turn_started:.3f} "
            f"current_yaw_rad={final.yaw_rad} "
            f"progress_deg={math.degrees(self._mission.turn_progress_rad):.1f}",
            flush=True,
        )
        raise RuntimeCommandError("measured turn timed out")

    async def _search_for_memory(self) -> CandidateMatch:
        if self.heading_provider is None:
            raise RuntimeCommandError("fresh Go2 heading is unavailable")
        initial = self.heading_provider.status()
        if not initial.healthy or initial.yaw_rad is None:
            raise RuntimeCommandError("fresh Go2 heading is unavailable")
        async with self._state_lock:
            self._mission.phase = MissionPhase.SEARCHING
            self._mission.reason = "searching_for_different_fruit"
            self._mission.match_confirmations = 0
            self._mission.match_failures = 0
        await self.navigation_arm(NAVIGATION_ARM_CONFIRMATION)
        deadline = time.monotonic() + self.mission_config.search_timeout_s
        last_frame_id: int | None = None
        last_center: tuple[int, int] | None = None
        confirmations = 0
        direction = 1.0
        while time.monotonic() < deadline:
            result, frame_id = await self._match_latest_frame(last_frame_id)
            if frame_id is not None and frame_id != last_frame_id:
                last_frame_id = frame_id
                accepted = bool(result is not None and result.accepted and result.best)
                if accepted and result is not None and result.best is not None:
                    center = result.best.detection.center
                    box = result.best.detection.bbox_xyxy
                    width = max(1, box[2] - box[0])
                    stable_candidate = bool(
                        last_center is None
                        or math.hypot(
                            center[0] - last_center[0], center[1] - last_center[1]
                        )
                        <= max(90.0, width * 1.75)
                    )
                    confirmations = confirmations + 1 if stable_candidate else 1
                    last_center = center
                else:
                    confirmations = 0
                    last_center = None
                async with self._state_lock:
                    self._mission.last_match = (
                        None if result is None else result.to_dict()
                    )
                    self._mission.match_confirmations = confirmations
                    self._mission.match_failures = (
                        0 if accepted else self._mission.match_failures + 1
                    )
                if result is not None:
                    print(
                        "search event=match "
                        f"accepted={accepted} "
                        f"reason={result.reason} "
                        f"confirmations={confirmations} "
                        f"candidate_count={result.candidate_count} "
                        f"best_score={None if result.best is None else result.best.score}",
                        flush=True,
                    )
                if (
                    accepted
                    and result is not None
                    and result.best is not None
                    and confirmations
                    >= self.mission_config.match_confirmations_required
                ):
                    await self.stop("demo_match_locked")
                    return result.best
                if accepted:
                    # Freeze the camera view as soon as a plausible different
                    # fruit appears. Continuing the sweep while waiting for
                    # another inference pushed live candidates out of frame.
                    await self.navigation_command(0.0, 0.0)
                    await asyncio.sleep(0.05)
                    continue

            sample = self.heading_provider.status()
            if not sample.healthy or sample.yaw_rad is None:
                raise RuntimeCommandError("Go2 heading became stale during search")
            progress = directed_progress(initial.yaw_rad, sample.yaw_rad, direction)
            if progress >= self.mission_config.search_sweep_rad:
                raise RuntimeCommandError("different fruit not found in search sweep")
            await self.navigation_command(
                0.0, direction * self.mission_config.search_rate_rps
            )
            await asyncio.sleep(0.05)
        raise RuntimeCommandError("different fruit search timed out")

    async def _start_memory_approach(self, match: CandidateMatch) -> None:
        async with self._state_lock:
            memory = self._fruit_memory
            self._mission.phase = MissionPhase.CONFIRMING
            self._mission.reason = "locking_different_fruit_track"
        if memory is None:
            raise RuntimeCommandError("saved fruit memory disappeared")
        # The search stage already required several consecutive, spatially
        # stable instance matches. Preserve that evidence across the handoff
        # instead of resetting the target to one visible frame and waiting for
        # another full detector cycle. On Jetson, that redundant wait created
        # a race where the follow request could cancel itself before arming.
        confirmed_frames = max(
            self.controller.config.stable_frames_required,
            self.mission_config.match_confirmations_required,
        )
        print(
            "approach event=target_lock "
            f"label={memory.label} "
            f"center={match.detection.center} "
            f"confirmed_frames={confirmed_frames} "
            f"match_score={match.score:.4f}",
            flush=True,
        )
        await self.select_target(
            memory.label,
            match.detection.center,
            confirmed_visible_frames=confirmed_frames,
        )
        await self.start_follow(ARM_CONFIRMATION)
        async with self._state_lock:
            self._mission.phase = MissionPhase.APPROACHING
            self._mission.reason = "approaching_different_fruit"
            self._mission.match_failures = 0

    async def _monitor_memory_approach(self) -> None:
        deadline = (
            time.monotonic()
            + self.controller.config.forward_budget_s
            + 4.0
        )
        last_frame_id: int | None = None
        failures = 0
        near_target_seen = False
        while time.monotonic() < deadline:
            async with self._state_lock:
                memory = self._fruit_memory
                frame = self._produce_frame
                frame_id = self._produce_frame_id
                detections = list(self._produce_detections)
                target = self._target
                selected_name = self._selected_target_name
                frame_width = self._frame_width
                frame_height = None if self._latest_frame is None else self._latest_frame.height
                follow_active = bool(
                    self._follow_task is not None
                    and not self._follow_task.done()
                    and self._lease is not None
                    and self.motion is not None
                    and self.motion.armed
                )
                command_reason = self._command.reason
            if memory is None:
                raise RuntimeCommandError("saved fruit memory disappeared")
            if target is not None and frame_height:
                x, y, width, height = target.bbox_xywh
                bottom_ratio = (y + height) / frame_height
                center_ratio = target.center[1] / frame_height
                if (
                    bottom_ratio >= self.mission_config.near_bottom_ratio
                    and center_ratio >= self.mission_config.near_center_ratio
                ):
                    near_target_seen = True
                    async with self._state_lock:
                        self._mission.near_target_seen = True

            if frame is not None and frame_id is not None and frame_id != last_frame_id:
                last_frame_id = frame_id
                result = await asyncio.to_thread(
                    self.instance_matcher.rank_different,
                    memory,
                    frame.bgr.copy(),
                    detections,
                )
                associated = False
                if result.accepted and result.best is not None and target is not None:
                    candidate_center = result.best.detection.center
                    associated = math.hypot(
                        candidate_center[0] - target.center[0],
                        candidate_center[1] - target.center[1],
                    ) <= max(90.0, target.bbox_xywh[2] * 1.75)
                failures = 0 if associated else failures + 1
                async with self._state_lock:
                    self._mission.last_match = result.to_dict()
                    self._mission.match_failures = failures
                if near_target_seen and result.best is None:
                    await self._finish_demo_success(
                        "different_fruit_reached_camera_edge"
                    )
                    return
                if failures >= self.mission_config.approach_misses_allowed:
                    raise RuntimeCommandError("different fruit identity lost during approach")

            if selected_name is None:
                if near_target_seen:
                    await self._finish_demo_success("different_fruit_reached_camera_edge")
                    return
                raise RuntimeCommandError("different fruit lost before arrival")
            if not follow_active:
                if near_target_seen and command_reason in {
                    "selected_target_not_revalidated",
                    "selected_target_lost",
                    "forward_budget_complete",
                }:
                    await self._finish_demo_success("different_fruit_reached_camera_edge")
                    return
                raise RuntimeCommandError(f"approach stopped: {command_reason}")
            if frame_width is None:
                raise RuntimeCommandError("camera geometry unavailable during approach")
            await asyncio.sleep(0.025)
        raise RuntimeCommandError("saved fruit approach timed out")

    async def _finish_demo_success(self, reason: str) -> None:
        await self.stop("demo_success")
        async with self._state_lock:
            self._mission.phase = MissionPhase.SUCCESS
            self._mission.reason = reason

    async def _match_latest_frame(
        self, last_frame_id: int | None
    ) -> tuple[MatchResult | None, int | None]:
        async with self._state_lock:
            memory = self._fruit_memory
            frame = self._produce_frame
            frame_id = self._produce_frame_id
            detections = list(self._produce_detections)
            produce_age = (
                None
                if self._produce_last_at is None
                else time.monotonic() - self._produce_last_at
            )
        if (
            memory is None
            or frame is None
            or frame_id is None
            or frame_id == last_frame_id
            or produce_age is None
            or produce_age > self.maximum_produce_age_s
        ):
            return None, frame_id
        result = await asyncio.to_thread(
            self.instance_matcher.rank_different,
            memory,
            frame.bgr.copy(),
            detections,
        )
        return result, frame_id

    async def _follow_loop(self) -> None:
        current_task = asyncio.current_task()
        try:
            while not self._closing:
                try:
                    await self.pulse()
                except RuntimeCommandError:
                    break
                await asyncio.sleep(self.follow_period_s)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            async with self._state_lock:
                self._last_error = f"follow loop failed: {exc}"
            await self.stop("follow_loop_error")
        finally:
            if self._follow_task is current_task:
                self._follow_task = None

    async def _navigation_watchdog_loop(self) -> None:
        while not self._closing:
            await asyncio.sleep(0.10)
            deadline = self._navigation_deadline
            if (
                self._motion_owner == "navigation"
                and deadline is not None
                and time.monotonic() >= deadline
            ):
                await self.stop("navigation_lease_expired")

    async def _release_locked(self, reason: str) -> None:
        if self.motion is not None and self._lease is not None:
            try:
                await self.motion.release(self._lease)
            except MotionError:
                await self.motion.emergency_stop()
        self._lease = None
        self._motion_owner = None
        self._navigation_deadline = None
        self._last_pulse_at = None
        self._command = VelocityCommand(reason=reason)

    def _best_produce_detection_locked(
        self,
        label: str,
        preferred_center: tuple[int, int] | None = None,
    ) -> FruitDetection | None:
        matches = [
            item
            for item in self._produce_detections
            if item.label.casefold() == label.casefold()
        ]
        if not matches:
            return None
        if preferred_center is None:
            return max(matches, key=lambda item: item.confidence)
        target_x, target_y = preferred_center
        return min(
            matches,
            key=lambda item: (
                (item.center[0] - target_x) ** 2
                + (item.center[1] - target_y) ** 2,
                -item.confidence,
            ),
        )

    def _canonical_target_name(self, requested_name: str) -> str:
        if self.produce_detector is None:
            raise RuntimeCommandError("produce detector is disabled")
        requested = requested_name.strip()
        available_names = list(self.produce_detector.names.values())
        if requested in available_names:
            return requested
        folded_names = {name.casefold(): name for name in available_names}
        if not requested or requested.casefold() not in folded_names:
            raise RuntimeCommandError("target is not a class in the produce model")
        return folded_names[requested.casefold()]

    def _clear_produce_tracker_locked(self) -> None:
        self._produce_tracker = None
        self._produce_tracker_label = None
        self._produce_visible_frames = 0
        self._produce_verified_at = None
        self._produce_revalidation_failures = 0

    def _clear_selection_locked(self) -> None:
        self._selected_target_name = None
        self._selected_target_hint = None
        self._target = None
        self._clear_produce_tracker_locked()

    def _produce_device_status(self) -> dict[str, object]:
        if self.produce_detector is None:
            return {"requested": "disabled", "resolved": "disabled"}
        status = getattr(self.produce_detector, "device_status", None)
        if callable(status):
            try:
                return status()
            except Exception as exc:
                return {
                    "requested": "unknown",
                    "resolved": "error",
                    "error": str(exc),
                }
        return {"requested": "test", "resolved": "test"}

    def _follow_readiness_locked(self, now: float) -> str | None:
        if self._selected_target_name is None:
            return "select a detected fruit first"
        if self._target is None:
            return "selected_target_not_found"
        if self._target.visible_frames < self.controller.config.stable_frames_required:
            return "selected_target_not_stable"
        target_age = now - self._target.captured_monotonic_s
        if target_age < 0.0 or target_age > self.controller.config.maximum_target_age_s:
            return "selected_target_stale"
        if (
            self._produce_verified_at is None
            or now - self._produce_verified_at > self.maximum_produce_age_s
            or self._produce_revalidation_failures > 0
        ):
            return "selected_target_not_revalidated"
        return None

    def _produce_revalidation_expired_locked(self, now: float) -> bool:
        """Bound target loss by both misses and elapsed time.

        Jetson inference cadence varies, so three quick low-confidence results
        must not erase a still-fresh click.  Conversely, the elapsed-time bound
        guarantees that a genuinely missing target is cleared independently of
        frame rate.
        """
        if (
            self._produce_revalidation_failures
            < self.produce_revalidation_misses_required
        ):
            return False
        return (
            self._produce_verified_at is None
            or now - self._produce_verified_at > self.maximum_produce_age_s
        )

    @staticmethod
    def _target_detection_iou(
        target: TargetObservation, detection: FruitDetection
    ) -> float:
        target_x, target_y, target_width, target_height = target.bbox_xywh
        target_x2 = target_x + target_width
        target_y2 = target_y + target_height
        detection_x1, detection_y1, detection_x2, detection_y2 = detection.bbox_xyxy
        intersection_width = max(
            0, min(target_x2, detection_x2) - max(target_x, detection_x1)
        )
        intersection_height = max(
            0, min(target_y2, detection_y2) - max(target_y, detection_y1)
        )
        intersection = intersection_width * intersection_height
        target_area = target_width * target_height
        detection_area = max(0, detection_x2 - detection_x1) * max(
            0, detection_y2 - detection_y1
        )
        union = target_area + detection_area - intersection
        return 0.0 if union <= 0 else intersection / union

    @staticmethod
    def _detection_bbox_xywh(
        detection: FruitDetection,
    ) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = detection.bbox_xyxy
        return x1, y1, max(1, x2 - x1), max(1, y2 - y1)

    @staticmethod
    def _observation_from_bbox(
        frame: CameraFrame,
        bbox_xywh: tuple[int, int, int, int],
        *,
        confidence: float | None,
        visible_frames: int,
    ) -> TargetObservation:
        x, y, width, height = bbox_xywh
        x = max(0, min(frame.width - 1, x))
        y = max(0, min(frame.height - 1, y))
        width = max(1, min(frame.width - x, width))
        height = max(1, min(frame.height - y, height))
        return TargetObservation(
            frame_id=frame.frame_id,
            captured_monotonic_s=frame.captured_monotonic_s,
            bbox_xywh=(x, y, width, height),
            center=(x + width // 2, y + height // 2),
            confidence=None if confidence is None else round(confidence, 4),
            visible_frames=visible_frames,
        )
