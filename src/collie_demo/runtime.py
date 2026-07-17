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
    ) -> None:
        self.camera = camera
        self.controller = controller
        self.motion = motion
        self.motion_enabled = bool(motion_enabled)
        self.allow_unranged_forward = bool(allow_unranged_forward)
        self.produce_detector = produce_detector
        self.produce_tracker_factory = (
            produce_tracker_factory or create_produce_tracker
        )
        self.loop_hz = float(loop_hz)
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
        self._produce_tracker_confidence = 0.0
        self._produce_visible_frames = 0
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
                selected_name = self._selected_target_name
            if selected_name is None:
                raise RuntimeCommandError("select a detected fruit first")
            if target is None or target.visible_frames < self.controller.config.stable_frames_required:
                raise RuntimeCommandError(
                    f"{selected_name} is not stably tracked"
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

    async def select_target(
        self,
        target_name: str,
        preferred_center: tuple[int, int] | None = None,
    ) -> dict[str, object]:
        canonical_name = self._canonical_target_name(target_name)
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
            async with self._state_lock:
                self._selected_target_name = canonical_name
                self._selected_target_hint = preferred_center
                self._clear_produce_tracker_locked()
                self._target = None
                frame = self._latest_frame
                detection = self._best_produce_detection_locked(
                    canonical_name, preferred_center
                )
                produce_age = (
                    None
                    if self._produce_last_at is None
                    else time.monotonic() - self._produce_last_at
                )
                if produce_age is None or produce_age > 8.0:
                    detection = None
            if frame is not None and detection is not None:
                try:
                    tracker = await asyncio.to_thread(
                        self.produce_tracker_factory,
                        frame.bgr.copy(),
                        self._detection_bbox_xywh(detection),
                    )
                except Exception as exc:
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
                        self._produce_tracker_label = canonical_name
                        self._produce_tracker_confidence = detection.confidence
                        self._produce_visible_frames = 1
                        self._target = observation
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
            produce_age = (
                None if self._produce_last_at is None else now - self._produce_last_at
            )
            target = self._target
            return {
                "ok": frame_age is not None and frame_age < 1.0,
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
                    "classes": self.produce_detector.names,
                    "detections": [item.to_dict() for item in self._produce_detections],
                    "frame_id": self._produce_frame_id,
                    "age_s": None if produce_age is None else round(produce_age, 3),
                    "inference_ms": self._produce_inference_ms,
                    "error": self._produce_error,
                    "tracker": {
                        "label": self._produce_tracker_label,
                        "target": None
                        if self._produce_tracker_label is None or target is None
                        else target.to_dict(),
                    },
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
                async with self._state_lock:
                    self._latest_frame = frame
                    produce_detections = list(self._produce_detections)
                    selected_name = self._selected_target_name
                    tracker = self._produce_tracker
                    tracker_label = self._produce_tracker_label
                    tracker_confidence = self._produce_tracker_confidence
                    tracker_visible_frames = self._produce_visible_frames
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
                            confidence=tracker_confidence,
                            visible_frames=tracker_visible_frames + 1,
                        )
                produce_image = annotate_fruits(frame.bgr, produce_detections)
                produce_image = annotate_selected_produce(
                    produce_image,
                    selected_name,
                    tracked_produce,
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
                            self._produce_tracker = None
                            self._produce_tracker_label = None
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
                    self._clear_produce_tracker_locked()
                    self._target = None
                    self._last_error = str(exc)
                lost_while_armed = self._lease is not None
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
                async with self._state_lock:
                    self._produce_detections = detections
                    self._produce_frame_id = frame.frame_id
                    self._produce_last_at = time.monotonic()
                    self._produce_inference_ms = round(inference_ms, 1)
                    self._produce_error = ""
                    selected_name = self._selected_target_name
                    current_frame = self._latest_frame
                    selected_detection = (
                        self._best_produce_detection_locked(
                            selected_name, self._selected_target_hint
                        )
                        if selected_name is not None
                        and self._produce_tracker is None
                        and self._lease is None
                        else None
                    )
                if current_frame is not None and selected_detection is not None:
                    tracker = await asyncio.to_thread(
                        self.produce_tracker_factory,
                        current_frame.bgr.copy(),
                        self._detection_bbox_xywh(selected_detection),
                    )
                    observation = self._observation_from_bbox(
                        current_frame,
                        self._detection_bbox_xywh(selected_detection),
                        confidence=selected_detection.confidence,
                        visible_frames=1,
                    )
                    async with self._state_lock:
                        if (
                            self._selected_target_name == selected_name
                            and selected_name is not None
                            and self._produce_tracker is None
                            and self._lease is None
                        ):
                            self._produce_tracker = tracker
                            self._produce_tracker_label = selected_name
                            self._produce_tracker_confidence = (
                                selected_detection.confidence
                            )
                            self._produce_visible_frames = 1
                            self._target = observation
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
        self._produce_tracker_confidence = 0.0
        self._produce_visible_frames = 0

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
        confidence: float,
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
            confidence=round(confidence, 4),
            visible_frames=visible_frames,
        )
