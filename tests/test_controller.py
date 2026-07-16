from __future__ import annotations

from collie_demo.controller import ApproachController
from collie_demo.types import BlueWhaleObservation


def target(*, x: int = 640, stable: int = 12, captured: float = 10.0) -> BlueWhaleObservation:
    return BlueWhaleObservation(1, captured, (x - 20, 620, 40, 60), (x, 650), 0.9, stable)


def test_missing_target_is_zero() -> None:
    command = ApproachController().plan(
        None,
        frame_width=1280,
        now_monotonic_s=10.1,
        forward_elapsed_s=0.0,
        allow_unranged_forward=True,
    )
    assert command.forward_mps == command.yaw_rps == 0.0


def test_off_center_target_curves_forward_while_turning() -> None:
    command = ApproachController().plan(
        target(x=900),
        frame_width=1280,
        now_monotonic_s=10.1,
        forward_elapsed_s=0.0,
        allow_unranged_forward=True,
    )
    assert command.forward_mps == 0.08
    assert command.yaw_rps < 0.0
    assert command.reason == "curving_to_blue_whale"


def test_centered_target_requires_explicit_unranged_flag() -> None:
    command = ApproachController().plan(
        target(),
        frame_width=1280,
        now_monotonic_s=10.1,
        forward_elapsed_s=0.0,
        allow_unranged_forward=False,
    )
    assert command.forward_mps == 0.0
    assert command.reason == "unranged_forward_disabled"


def test_centered_target_gets_slow_bounded_forward_burst() -> None:
    command = ApproachController().plan(
        target(),
        frame_width=1280,
        now_monotonic_s=10.1,
        forward_elapsed_s=0.5,
        allow_unranged_forward=True,
    )
    assert command.forward_mps == 0.08
    assert command.reason == "supervised_forward_burst"


def test_budget_completion_zeroes_forward_motion() -> None:
    command = ApproachController().plan(
        target(),
        frame_width=1280,
        now_monotonic_s=10.1,
        forward_elapsed_s=1.5,
        allow_unranged_forward=True,
    )
    assert command.forward_mps == 0.0
    assert command.reason == "forward_budget_complete"
