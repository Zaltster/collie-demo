from __future__ import annotations

import asyncio
from pathlib import Path
import time

import cv2
import numpy as np

from collie_demo.controller import ApproachController
from collie_demo.detector import BlueWhaleDetector
from collie_demo.fruit import FruitDetection
from collie_demo.motion import UnitreeMotionAdapter
from collie_demo.runtime import ARM_CONFIRMATION, CollieRuntime, RuntimeCommandError
from collie_demo.types import CameraFrame

from test_motion import FakeAvoidance, FakeSport


class CenteredWhaleCamera:
    def __init__(self) -> None:
        self.frame_id = 0

    def read(self) -> CameraFrame:
        self.frame_id += 1
        image = np.full((720, 1280, 3), 235, dtype=np.uint8)
        cv2.ellipse(image, (640, 650), (34, 42), 0, 0, 360, (245, 245, 160), -1)
        cv2.ellipse(image, (900, 650), (52, 34), 0, 0, 360, (70, 220, 235), -1)
        return CameraFrame(self.frame_id, time.monotonic(), image)


class FakeProduceDetector:
    model_path = Path("/models/snapstock.pt")
    confidence = 0.5
    names = {1: "apple"}

    def detect(self, _image: object) -> list[FruitDetection]:
        return [
            FruitDetection(
                class_id=1,
                label="apple",
                confidence=0.91,
                bbox_xyxy=(100, 200, 180, 300),
                center=(140, 250),
            )
        ]


def test_runtime_requires_confirmation_then_pulses_forward() -> None:
    async def scenario() -> None:
        avoidance = FakeAvoidance()
        motion = UnitreeMotionAdapter(FakeSport(), avoidance)
        runtime = CollieRuntime(
            camera=CenteredWhaleCamera(),
            detector=BlueWhaleDetector(),
            controller=ApproachController(),
            motion=motion,
            motion_enabled=True,
            allow_unranged_forward=True,
            produce_detector=FakeProduceDetector(),
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
            await runtime.arm(ARM_CONFIRMATION)
            status = await runtime.pulse()
            assert status["armed"] is True
            assert status["command"]["reason"] == "supervised_forward_burst"
            assert avoidance.moves[-1] == (0.08, 0.0, 0.0)
            assert status["produce"]["detections"][0]["label"] == "apple"
            assert status["produce"]["inference_ms"] is not None
            assert status["selected_target_color"] == "blue"
            assert status["whales"]["blue"] is not None
            assert status["whales"]["yellow"] is not None

            selected = await runtime.select_target("yellow")
            assert selected["selected_target_color"] == "yellow"
            assert selected["selected_whale"] is not None
            assert selected["armed"] is False
            assert selected["command"]["reason"] == "yellow_whale_selected"

            await runtime.arm(ARM_CONFIRMATION)
            yellow_status = await runtime.pulse()
            assert yellow_status["armed"] is True
            assert yellow_status["command"]["reason"] == "curving_to_selected_whale"
            assert avoidance.moves[-1][0] == 0.08
            assert avoidance.moves[-1][2] < 0.0

            try:
                await runtime.select_target("purple")
            except RuntimeCommandError:
                pass
            else:
                raise AssertionError("invalid target color was accepted")
        finally:
            await runtime.close()

    asyncio.run(scenario())
