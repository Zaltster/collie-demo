from __future__ import annotations

import asyncio
from pathlib import Path
import time

import numpy as np

from collie_demo.controller import ApproachController
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
