from __future__ import annotations

import asyncio
import time

import cv2
import numpy as np

from collie_demo.controller import ApproachController
from collie_demo.detector import BlueWhaleDetector
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
        return CameraFrame(self.frame_id, time.monotonic(), image)


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
        finally:
            await runtime.close()

    asyncio.run(scenario())
