from __future__ import annotations

import asyncio
from pathlib import Path
import time

import cv2
import numpy as np

from collie_demo.controller import ApproachConfig, ApproachController
from collie_demo.fruit import FruitDetection
from collie_demo.heading import HeadingSample, normalize_angle
from collie_demo.matcher import FruitInstanceMatcher
from collie_demo.memory import AppearanceEncoder
from collie_demo.mission import MissionConfig
from collie_demo.motion import UnitreeMotionAdapter
from collie_demo.runtime import (
    ARM_CONFIRMATION,
    DEMO_CONFIRMATION,
    NAVIGATION_ARM_CONFIRMATION,
    CollieRuntime,
    RuntimeCommandError,
)
from collie_demo.types import CameraFrame

from test_motion import FakeAvoidance, FakeSport


class StaticFruitCamera:
    def __init__(self) -> None:
        self.frame_id = 0

    def read(self) -> CameraFrame:
        self.frame_id += 1
        image = np.full((720, 1280, 3), 120, dtype=np.uint8)
        return CameraFrame(self.frame_id, time.monotonic(), image)


class OneBadFrameCamera(StaticFruitCamera):
    def __init__(self) -> None:
        super().__init__()
        self.fail_next = False

    def read(self) -> CameraFrame:
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("camera returned an invalid JPEG")
        return super().read()


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


class SlowProduceDetector(FakeProduceDetector):
    def detect(self, image: object) -> list[FruitDetection]:
        time.sleep(0.4)
        return super().detect(image)


class JetsonSpeedProduceDetector(FakeProduceDetector):
    def detect(self, image: object) -> list[FruitDetection]:
        time.sleep(0.12)
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


class NearBananaCamera(StaticFruitCamera):
    def __init__(self) -> None:
        super().__init__()
        self.show_different_banana = False

    def read(self) -> CameraFrame:
        self.frame_id += 1
        image = np.full((720, 1280, 3), 28, dtype=np.uint8)
        if self.show_different_banana:
            cv2.circle(image, (640, 645), 65, (210, 45, 35), -1)
        else:
            image[580:710, 540:740] = (20, 210, 235)
        return CameraFrame(self.frame_id, time.monotonic(), image)


class NearBananaDetector:
    model_path = Path("/models/collie.pt")
    confidence = 0.2
    class_thresholds = {"banana": 0.2}
    names = {0: "banana"}

    def device_status(self) -> dict[str, object]:
        return {
            "requested": "0",
            "resolved": "cuda:0",
            "cuda_available": True,
        }

    def detect(self, _image: object) -> list[FruitDetection]:
        return [
            FruitDetection(
                class_id=0,
                label="banana",
                confidence=0.92,
                bbox_xyxy=(540, 580, 740, 710),
                center=(640, 645),
            )
        ]


class MotionCoupledHeading:
    def __init__(self, avoidance: FakeAvoidance, sport: FakeSport | None = None) -> None:
        self.avoidance = avoidance
        self.sport = sport
        self.yaw = 0.0

    def start(self) -> None:
        return

    def close(self) -> None:
        return

    def status(self) -> HeadingSample:
        yaw_rate = 0.0
        if self.sport is not None:
            yaw_rate = self.sport.current_move[2]
        if not yaw_rate and self.avoidance.moves:
            yaw_rate = self.avoidance.moves[-1][2]
        if yaw_rate:
            self.yaw = normalize_angle(
                self.yaw + (0.36 if yaw_rate > 0.0 else -0.36)
            )
        return HeadingSample(self.yaw, 0.0, True)


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


def test_navigation_owner_accepts_bounded_commands_without_a_fruit() -> None:
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
            loop_hz=60.0,
            navigation_command_lease_s=0.05,
        )
        await runtime.start()
        try:
            try:
                await runtime.navigation_arm("wrong")
            except RuntimeCommandError:
                pass
            else:
                raise AssertionError("wrong navigation confirmation armed motion")

            armed = await runtime.navigation_arm(NAVIGATION_ARM_CONFIRMATION)
            assert armed["armed"] is True
            assert armed["owner"] == "navigation"
            commanded = await runtime.navigation_command(0.3, -0.2)
            assert commanded["command"]["forward_mps"] == 0.08
            assert commanded["command"]["yaw_rps"] == -0.2
            try:
                await runtime.pulse()
            except RuntimeCommandError:
                pass
            else:
                raise AssertionError("fruit pulse stole the navigation lease")
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_navigation_lease_expires_to_hardware_stop() -> None:
    async def scenario() -> None:
        sport = FakeSport()
        motion = UnitreeMotionAdapter(sport, FakeAvoidance())
        runtime = CollieRuntime(
            camera=StaticFruitCamera(),
            controller=ApproachController(),
            motion=motion,
            motion_enabled=True,
            allow_unranged_forward=True,
            loop_hz=60.0,
            navigation_command_lease_s=0.05,
        )
        await runtime.start()
        try:
            await runtime.navigation_arm(NAVIGATION_ARM_CONFIRMATION)
            await runtime.navigation_command(0.03, 0.0)
            await asyncio.sleep(0.18)
            status = await runtime.navigation_status()
            assert status["armed"] is False
            assert sport.stop_calls >= 1
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_sustained_yolo_loss_clears_selected_tracker() -> None:
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
            maximum_produce_age_s=0.05,
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


def test_one_bad_camera_frame_does_not_clear_fresh_selection() -> None:
    async def scenario() -> None:
        camera = OneBadFrameCamera()
        runtime = CollieRuntime(
            camera=camera,
            controller=ApproachController(
                ApproachConfig(
                    stable_frames_required=3,
                    maximum_target_age_s=0.75,
                )
            ),
            motion=None,
            motion_enabled=False,
            allow_unranged_forward=True,
            produce_detector=FakeProduceDetector(),
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

            await runtime.select_target("banana", (350, 240))
            camera.fail_next = True
            await asyncio.sleep(0.08)

            status = await runtime.status()
            assert status["selected_target_name"] == "banana"
            assert status["selected_target"] is not None
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_three_fast_yolo_misses_do_not_clear_fresh_selection() -> None:
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
            detector.misses_remaining = 3
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


def test_confirmed_mission_lock_arms_without_rewaiting_for_yolo() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        runtime = CollieRuntime(
            camera=StaticFruitCamera(),
            controller=ApproachController(
                ApproachConfig(stable_frames_required=3)
            ),
            motion=UnitreeMotionAdapter(sport, avoidance),
            motion_enabled=True,
            allow_unranged_forward=True,
            produce_detector=FakeProduceDetector(),
            loop_hz=60.0,
            follow_start_timeout_s=0.01,
        )
        await runtime.start()
        try:
            for _ in range(100):
                status = await runtime.status()
                if status["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("test detector did not publish")

            await runtime.select_target(
                "banana",
                (350, 240),
                confirmed_visible_frames=3,
            )
            status = await runtime.start_follow(ARM_CONFIRMATION)

            assert status["follow_active"] is True
            assert status["armed"] is True
            assert status["selected_target"]["visible_frames"] >= 3
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


def test_sustained_yolo_loss_stops_an_active_robot_side_follow() -> None:
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
            maximum_produce_age_s=0.05,
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
                raise AssertionError("active follow did not stop after sustained loss")

            assert status["follow_active"] is False
            assert status["command"]["reason"] == "selected_target_not_revalidated"
            assert sport.stop_calls >= 1
            assert avoidance.moves[-1] == (0.0, 0.0, 0.0)
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_old_yolo_result_does_not_replace_fresh_tracker_observation() -> None:
    async def scenario() -> None:
        runtime = CollieRuntime(
            camera=StaticFruitCamera(),
            controller=ApproachController(
                ApproachConfig(stable_frames_required=3)
            ),
            motion=UnitreeMotionAdapter(FakeSport(), FakeAvoidance()),
            motion_enabled=True,
            allow_unranged_forward=True,
            produce_detector=SlowProduceDetector(),
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
                raise AssertionError("slow detector never returned a frame")

            await runtime.select_target("banana", (350, 240))
            await asyncio.sleep(0.1)
            target_ages: list[float] = []
            readiness: list[str] = []
            for _ in range(60):
                status = await runtime.status()
                if status["selected_target_age_s"] is not None:
                    target_ages.append(status["selected_target_age_s"])
                readiness.append(status["follow_readiness"])
                await asyncio.sleep(0.02)

            assert target_ages
            assert max(target_ages) < 0.35
            assert "selected_target_stale" not in readiness
            assert status["can_follow"] is True
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_saved_fruit_memory_survives_visual_target_loss() -> None:
    async def scenario() -> None:
        detector = ToggleProduceDetector()
        runtime = CollieRuntime(
            camera=StaticFruitCamera(),
            controller=ApproachController(),
            motion=None,
            motion_enabled=False,
            allow_unranged_forward=True,
            produce_detector=detector,
            loop_hz=60.0,
            maximum_produce_age_s=0.1,
            mission_config=MissionConfig(
                enabled=True,
                capture_samples=2,
                capture_timeout_s=0.6,
            ),
        )
        await runtime.start()
        try:
            for _ in range(100):
                if (await runtime.status())["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)
            saved = await runtime.remember_target("banana", (350, 240))
            assert saved["memory"]["label"] == "banana"
            assert saved["memory"]["sample_count"] == 2

            detector.visible = False
            for _ in range(100):
                status = await runtime.status()
                if not status["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)

            assert status["selected_target_name"] is None
            assert status["memory"]["label"] == "banana"
            assert status["mission"]["phase"] == "memorized"
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_memory_demo_turns_searches_and_reuses_guarded_follow() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = UnitreeMotionAdapter(sport, avoidance)
        camera = NearBananaCamera()
        runtime = CollieRuntime(
            camera=camera,
            controller=ApproachController(
                ApproachConfig(
                    stable_frames_required=2,
                    maximum_target_age_s=0.75,
                    forward_mps=0.08,
                    forward_budget_s=0.08,
                )
            ),
            motion=motion,
            motion_enabled=True,
            allow_unranged_forward=True,
            produce_detector=NearBananaDetector(),
            loop_hz=60.0,
            maximum_produce_age_s=0.75,
            follow_period_s=0.02,
            instance_matcher=FruitInstanceMatcher(
                AppearanceEncoder(minimum_crop_side_px=8),
                minimum_score=0.70,
                minimum_margin=0.04,
            ),
            heading_provider=MotionCoupledHeading(avoidance, sport),
            mission_config=MissionConfig(
                enabled=True,
                autonomous_turn_enabled=True,
                direct_turn_enabled=True,
                capture_samples=2,
                capture_timeout_s=0.6,
                match_confirmations_required=2,
                approach_misses_allowed=2,
                turn_angle_rad=0.65,
                turn_rate_rps=0.20,
                turn_tolerance_rad=0.05,
                turn_timeout_s=1.5,
                search_rate_rps=0.10,
                search_sweep_rad=2.5,
                search_timeout_s=1.5,
            ),
        )
        await runtime.start()
        try:
            for _ in range(100):
                if (await runtime.status())["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)
            await runtime.remember_target("banana", (640, 645))
            camera.show_different_banana = True
            started = await runtime.start_demo(DEMO_CONFIRMATION)
            assert started["mission"]["active"] is True
            assert started["mission"]["direct_turn_enabled"] is True

            for _ in range(300):
                status = await runtime.status()
                if status["mission"]["phase"] in {"success", "aborted"}:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("memory demo did not reach a terminal state")

            assert status["mission"]["phase"] == "success", status["mission"]
            assert status["mission"]["near_target_seen"] is True
            assert status["memory"]["label"] == "banana"
            assert status["armed"] is False
            assert status["command"]["forward_mps"] == 0.0
            assert any(move[2] > 0.0 for move in sport.moves)
            assert all(move[0] == 0.0 and move[1] == 0.0 for move in sport.moves)
            assert all(move[2] != 0.20 for move in avoidance.moves)
            assert any(move[0] > 0.0 for move in avoidance.moves)
            assert avoidance.moves[-1] == (0.0, 0.0, 0.0)
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_memory_demo_never_approaches_the_saved_banana() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = UnitreeMotionAdapter(sport, avoidance)
        runtime = CollieRuntime(
            camera=NearBananaCamera(),
            controller=ApproachController(
                ApproachConfig(
                    stable_frames_required=2,
                    maximum_target_age_s=0.75,
                    forward_mps=0.08,
                    forward_budget_s=0.08,
                )
            ),
            motion=motion,
            motion_enabled=True,
            allow_unranged_forward=True,
            produce_detector=NearBananaDetector(),
            loop_hz=60.0,
            maximum_produce_age_s=0.75,
            follow_period_s=0.02,
            instance_matcher=FruitInstanceMatcher(
                AppearanceEncoder(minimum_crop_side_px=8),
                maximum_saved_similarity=0.94,
            ),
            heading_provider=MotionCoupledHeading(avoidance, sport),
            mission_config=MissionConfig(
                enabled=True,
                autonomous_turn_enabled=True,
                direct_turn_enabled=True,
                capture_samples=2,
                capture_timeout_s=0.6,
                match_confirmations_required=2,
                approach_misses_allowed=2,
                turn_angle_rad=0.35,
                turn_rate_rps=0.20,
                turn_tolerance_rad=0.05,
                turn_timeout_s=1.0,
                search_rate_rps=0.10,
                search_sweep_rad=0.50,
                search_timeout_s=1.0,
            ),
        )
        await runtime.start()
        try:
            for _ in range(100):
                if (await runtime.status())["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)
            await runtime.remember_target("banana", (640, 645))
            await runtime.start_demo(DEMO_CONFIRMATION)

            for _ in range(200):
                status = await runtime.status()
                if status["mission"]["phase"] in {"success", "aborted"}:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("saved-only mission did not stop")

            assert status["mission"]["phase"] == "aborted"
            assert "different fruit" in status["mission"]["reason"]
            assert status["mission"]["target_policy"] == "different_instance"
            assert status["mission"]["last_match"]["reason"] == "saved_instance_only"
            assert not any(move[0] > 0.0 for move in avoidance.moves)
            assert status["armed"] is False
        finally:
            await runtime.close()

    asyncio.run(scenario())


def test_detector_only_tracking_stays_fresh_at_jetson_cadence() -> None:
    async def scenario() -> None:
        runtime = CollieRuntime(
            camera=StaticFruitCamera(),
            controller=ApproachController(
                ApproachConfig(stable_frames_required=3)
            ),
            motion=UnitreeMotionAdapter(FakeSport(), FakeAvoidance()),
            motion_enabled=True,
            allow_unranged_forward=True,
            produce_detector=JetsonSpeedProduceDetector(),
            loop_hz=8.0,
        )
        await runtime.start()
        try:
            for _ in range(100):
                status = await runtime.status()
                if status["produce"]["detections"]:
                    break
                await asyncio.sleep(0.01)
            else:
                raise AssertionError("detector never returned a frame")

            await runtime.select_target("banana", (350, 240))
            await asyncio.sleep(0.5)
            statuses = []
            for _ in range(60):
                statuses.append(await runtime.status())
                await asyncio.sleep(0.02)

            assert all(
                item["produce"]["tracker"]["mode"] == "yolo"
                for item in statuses
            )
            assert all(
                item["selected_target"]["confidence"] is not None
                for item in statuses
            )
            assert max(item["selected_target_age_s"] for item in statuses) < 0.35
            assert all(item["follow_readiness"] == "ready" for item in statuses)
            assert all(item["can_follow"] is True for item in statuses)
        finally:
            await runtime.close()

    asyncio.run(scenario())
