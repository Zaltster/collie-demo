from __future__ import annotations

import asyncio
from pathlib import Path
import time
from typing import Callable, Protocol

import cv2

from .controller import ApproachController
from .fruit import FruitDetection, annotate_fruits, annotate_selected_produce
from .motion import MotionError, MotionNotReady, UnitreeMotionAdapter
from .types import CameraFrame, TargetObservation, VelocityCommand


ARM_CONFIRMATION = "TARGET AND PATH CLEAR"


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
        self._follow_task: asyncio.Task[None] | None = None
        self._follow_start_generation = 0

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
        self._follow_start_generation += 1
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
                readiness = self._follow_readiness_locked(time.monotonic())
            if readiness is not None:
                raise RuntimeCommandError(readiness)
            try:
                lease = await self.motion.arm()
            except MotionError as exc:
                raise RuntimeCommandError(str(exc)) from exc
            self._lease = lease
            self._last_pulse_at = None
            self._forward_elapsed_s = 0.0
            self._command = VelocityCommand(reason="armed_waiting_for_hold")
            return await self.status()

    async def start_follow(self, confirmation: str) -> dict[str, object]:
        generation = self._follow_start_generation
        deadline = time.monotonic() + self.follow_start_timeout_s
        while True:
            if generation != self._follow_start_generation:
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
        return await self.status()

    async def select_target(
        self,
        target_name: str,
        preferred_center: tuple[int, int] | None = None,
    ) -> dict[str, object]:
        canonical_name = self._canonical_target_name(target_name)
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
                    visible_frames=1,
                )
                async with self._state_lock:
                    if self._selected_target_name == canonical_name:
                        self._produce_tracker = tracker
                        self._produce_tracker_label = (
                            canonical_name if tracker is not None else None
                        )
                        self._produce_visible_frames = 1
                        self._produce_verified_at = self._produce_last_at
                        self._produce_revalidation_failures = 0
                        self._target = observation
            return await self.status()

    async def pulse(self) -> dict[str, object]:
        async with self._action_lock:
            if self.motion is None or self._lease is None or not self.motion.armed:
                self._lease = None
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
        self._follow_start_generation += 1
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
            return {
                "ok": camera_live,
                "stage_ready": camera_live and produce_live and gpu_ready and motion_ready,
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
                lost_while_armed = self._lease is not None and (
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
                lost_while_armed = camera_is_stale and self._lease is not None
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
                            if (
                                self._produce_revalidation_failures
                                >= self.produce_revalidation_misses_required
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
                    self._produce_frame_id = frame.frame_id
                    self._produce_last_at = time.monotonic()
                    self._produce_inference_ms = round(
                        (time.monotonic() - started) * 1000.0, 1
                    )
                    self._produce_error = str(exc)
                    if self._selected_target_name is not None:
                        self._produce_revalidation_failures += 1
                        if (
                            self._produce_revalidation_failures
                            >= self.produce_revalidation_misses_required
                        ):
                            self._clear_selection_locked()
                            revalidation_failed = True
                if revalidation_failed:
                    await self.stop("selected_target_not_revalidated")
            await asyncio.sleep(0)

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

    async def _release_locked(self, reason: str) -> None:
        if self.motion is not None and self._lease is not None:
            try:
                await self.motion.release(self._lease)
            except MotionError:
                await self.motion.emergency_stop()
        self._lease = None
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
