from __future__ import annotations

import asyncio
from pathlib import Path
import time

import numpy as np

from collie_demo.controller import ApproachConfig, ApproachController
from collie_demo.fruit import FruitDetection
from collie_demo.motion import UnitreeMotionAdapter
from collie_demo.runtime import ARM_CONFIRMATION, CollieRuntime, RuntimeCommandError
from collie_demo.types import CameraFrame

from test_motion import FakeAvoidance, FakeSport


class StaticFruitCamera:
    def __init__(self) -> None:
        self.frame_id = 0

    def read(self) -> CameraFrame:
        self.frame_id += 1
        image = np.full((720, 1280, 3), 120, dtype=np.uint8)
        return CameraFrame(self.frame_id, time.monotonic(), image)


class FakeProduceDetector:
    model_path = Path("/models/snapstock.pt")
    confidence = 0.5
    names = {1: "apple", 6: "banana", 30: "Strawberry", 57: "strawberry"}

    def device_status(self) -> dict[str, object]:
        return {
            "requested": "0",
            "resolved": "cuda:0",
            "cuda_available": True,
        }

    def detect(self, _image: object) -> list[FruitDetection]:
        return [
            FruitDetection(
                class_id=1,
                label="apple",
                confidence=0.91,
                bbox_xyxy=(100, 200, 180, 300),
                center=(140, 250),
            ),
            FruitDetection(
                class_id=6,
                label="banana",
                confidence=0.88,
                bbox_xyxy=(300, 200, 400, 280),
                center=(350, 240),
            ),
            FruitDetection(
                class_id=1,
                label="apple",
                confidence=0.82,
                bbox_xyxy=(960, 200, 1060, 280),
                center=(1010, 240),
            ),
            FruitDetection(
                class_id=30,
                label="Strawberry",
                confidence=0.77,
                bbox_xyxy=(560, 200, 660, 280),
                center=(610, 240),
            ),
        ]


class ToggleProduceDetector(FakeProduceDetector):
    def __init__(self) -> None:
        self.visible = True

    def detect(self, image: object) -> list[FruitDetection]:
        return super().detect(image) if self.visible else []


class BurstMissProduceDetector(FakeProduceDetector):
    def __init__(self) -> None:
        self.misses_remaining = 0

    def detect(self, image: object) -> list[FruitDetection]:
        if self.misses_remaining > 0:
            self.misses_remaining -= 1
            return []
        return super().detect(image)


class FakeProduceTracker:
    def __init__(self, bbox: tuple[int, int, int, int]) -> None:
        self.bbox = bbox

    def update(
        self, _image: object
    ) -> tuple[bool, tuple[float, float, float, float]]:
        return True, tuple(float(value) for value in self.bbox)


def fake_tracker_factory(
    _image: object, bbox: tuple[int, int, int, int]
) -> FakeProduceTracker:
    return FakeProduceTracker(bbox)


def test_runtime_requires_confirmation_then_pulses_forward() -> None:
    async def scenario() -> None:
        avoidance = FakeAvoidance()
        motion = UnitreeMotionAdapter(FakeSport(), avoidance)
        runtime = CollieRuntime(
            camera=StaticFruitCamera(),
            controller=ApproachController(),
            motion=motion,
            motion_enabled=True,
            allow_unranged_forward=True,
            produce_detector=FakeProduceDetector(),
            produce_tracker_factory=fake_tracker_factory,
            loop_hz=60.0,
        )
        await runtime.start()
        try:
            await asyncio.sleep(0.22)
            try:
                await runtime.arm("wrong")
            except RuntimeCommandError:
                pass
            else:
                raise AssertionError("wrong confirmation unexpectedly armed motion")
            try:
                await runtime.arm(ARM_CONFIRMATION)
            except RuntimeCommandError:
                pass
            else:
                raise AssertionError("motion armed without a selected fruit")
            status = await runtime.status()
            assert status["produce"]["detections"][0]["label"] == "apple"
            assert status["produce"]["inference_ms"] is not None
            assert status["selected_target_name"] is None

            apple = await runtime.select_target("apple", (1010, 240))
            assert apple["selected_target_name"] == "apple"
            assert apple["selected_target_kind"] == "produce"
            assert apple["selected_target_hint"] == (1010, 240)
            assert apple["armed"] is False
            await asyncio.sleep(0.18)
            await runtime.arm(ARM_CONFIRMATION)
            apple_status = await runtime.pulse()
            assert apple_status["command"]["reason"] == "curving_to_selected_target"
            assert avoidance.moves[-1][2] < 0.0

            banana = await runtime.select_target("banana", (350, 240))
            assert banana["selected_target_name"] == "banana"
            assert banana["armed"] is False
            await asyncio.sleep(0.18)
            await runtime.arm(ARM_CONFIRMATION)
            banana_status = await runtime.pulse()
            assert banana_status["command"]["reason"] == "curving_to_selected_target"
            assert avoidance.moves[-1][2] > 0.0

            strawberry = await runtime.select_target("Strawberry", (610, 240))
            assert strawberry["selected_target_name"] == "Strawberry"

            try:
                await runtime.select_target("purple")
            except RuntimeCommandError:
                pass
            else:
                raise AssertionError("invalid target color was accepted")
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_yolo_miss_clears_selected_tracker() -> None:
    async def scenario() -> None:
        detector = ToggleProduceDetector()
        runtime = CollieRuntime(
            camera=StaticFruitCamera(),
            controller=ApproachController(),
            motion=None,
            motion_enabled=False,
            allow_unranged_forward=True,
            produce_detector=detector,
            produce_tracker_factory=fake_tracker_factory,
            loop_hz=60.0,
        )
        await runtime.start()
        try:
            for _ in range(100):
                status = await runtime.status()
                if status["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("produce detector never returned a frame")

            selected = await runtime.select_target("banana", (350, 240))
            assert selected["selected_target_name"] == "banana"
            detector.visible = False

            for _ in range(100):
                status = await runtime.status()
                if status["selected_target_name"] is None:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("missing banana remained selected")

            assert status["selected_target"] is None
            assert status["produce"]["tracker"]["label"] is None
            assert status["produce"]["tracker"]["revalidation_failures"] == 0
            assert (
                status["produce"]["tracker"]["revalidation_failures_required"]
                == 3
            )
            assert status["command"]["reason"] == "selected_target_not_revalidated"
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_two_transient_yolo_misses_do_not_clear_selected_tracker() -> None:
    async def scenario() -> None:
        detector = BurstMissProduceDetector()
        runtime = CollieRuntime(
            camera=StaticFruitCamera(),
            controller=ApproachController(),
            motion=None,
            motion_enabled=False,
            allow_unranged_forward=True,
            produce_detector=detector,
            produce_tracker_factory=fake_tracker_factory,
            loop_hz=60.0,
            produce_revalidation_misses_required=3,
        )
        await runtime.start()
        try:
            for _ in range(100):
                status = await runtime.status()
                if status["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)
            await runtime.select_target("banana", (350, 240))
            detector.misses_remaining = 2
            await asyncio.sleep(0.08)

            status = await runtime.status()
            assert detector.misses_remaining == 0
            assert status["selected_target_name"] == "banana"
            assert status["selected_target"] is not None
            assert status["produce"]["tracker"]["revalidation_failures"] == 0
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_robot_side_follow_continues_without_browser_pulses() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = UnitreeMotionAdapter(sport, avoidance)
        runtime = CollieRuntime(
            camera=StaticFruitCamera(),
            controller=ApproachController(),
            motion=motion,
            motion_enabled=True,
            allow_unranged_forward=True,
            produce_detector=FakeProduceDetector(),
            produce_tracker_factory=fake_tracker_factory,
            loop_hz=60.0,
            follow_period_s=0.02,
        )
        await runtime.start()
        try:
            await asyncio.sleep(0.08)
            await runtime.select_target("banana", (350, 240))
            status = await runtime.start_follow(ARM_CONFIRMATION)
            assert status["follow_active"] is True
            await asyncio.sleep(0.12)

            status = await runtime.status()
            nonzero_moves = [move for move in avoidance.moves if move != (0.0, 0.0, 0.0)]
            assert status["stage_ready"] is True
            assert all(status["health"].values())
            assert status["follow_active"] is True
            assert status["armed"] is True
            assert len(nonzero_moves) >= 2

            await runtime.stop("test_stop")
            await asyncio.sleep(0.05)
            status = await runtime.status()
            assert status["follow_active"] is False
            assert status["armed"] is False
            assert status["command"]["forward_mps"] == 0.0
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_stale_detection_cannot_be_selected() -> None:
    async def scenario() -> None:
        detector = ToggleProduceDetector()
        runtime = CollieRuntime(
            camera=StaticFruitCamera(),
            controller=ApproachController(),
            motion=None,
            motion_enabled=False,
            allow_unranged_forward=True,
            produce_detector=detector,
            produce_tracker_factory=fake_tracker_factory,
            loop_hz=60.0,
        )
        await runtime.start()
        try:
            for _ in range(100):
                status = await runtime.status()
                if status["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)
            detector.visible = False
            for _ in range(100):
                status = await runtime.status()
                if not status["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)

            try:
                await runtime.select_target("banana", (350, 240))
            except RuntimeCommandError as exc:
                assert "no longer freshly detected" in str(exc)
            else:
                raise AssertionError("stale banana detection was selectable")
            assert (await runtime.status())["selected_target_name"] is None
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_stop_cancels_a_follow_request_waiting_for_target_stability() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = UnitreeMotionAdapter(sport, avoidance)
        runtime = CollieRuntime(
            camera=StaticFruitCamera(),
            controller=ApproachController(
                ApproachConfig(stable_frames_required=100)
            ),
            motion=motion,
            motion_enabled=True,
            allow_unranged_forward=True,
            produce_detector=FakeProduceDetector(),
            produce_tracker_factory=fake_tracker_factory,
            loop_hz=60.0,
        )
        await runtime.start()
        try:
            await asyncio.sleep(0.04)
            await runtime.select_target("banana", (350, 240))
            pending_follow = asyncio.create_task(
                runtime.start_follow(ARM_CONFIRMATION)
            )
            await asyncio.sleep(0.04)
            await runtime.stop("operator_stop")
            try:
                await pending_follow
            except RuntimeCommandError as exc:
                assert str(exc) == "follow start cancelled"
            else:
                raise AssertionError("cancelled follow request unexpectedly started")

            status = await runtime.status()
            assert status["follow_active"] is False
            assert status["armed"] is False
            assert status["command"]["reason"] == "operator_stop"
            assert not [
                move for move in avoidance.moves if move != (0.0, 0.0, 0.0)
            ]
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_three_yolo_misses_stop_an_active_robot_side_follow() -> None:
    async def scenario() -> None:
        detector = ToggleProduceDetector()
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = UnitreeMotionAdapter(sport, avoidance)
        runtime = CollieRuntime(
            camera=StaticFruitCamera(),
            controller=ApproachController(),
            motion=motion,
            motion_enabled=True,
            allow_unranged_forward=True,
            produce_detector=detector,
            produce_tracker_factory=fake_tracker_factory,
            loop_hz=60.0,
            follow_period_s=0.02,
        )
        await runtime.start()
        try:
            await asyncio.sleep(0.08)
            await runtime.select_target("banana", (350, 240))
            await runtime.start_follow(ARM_CONFIRMATION)
            detector.visible = False

            for _ in range(100):
                status = await runtime.status()
                if not status["armed"] and status["selected_target_name"] is None:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("active follow did not stop after three misses")

            assert status["follow_active"] is False
            assert status["command"]["reason"] == "selected_target_not_revalidated"
            assert sport.stop_calls >= 1
            assert avoidance.moves[-1] == (0.0, 0.0, 0.0)
        finally:
            await runtime.close()

    asyncio.run(scenario())
