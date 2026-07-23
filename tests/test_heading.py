import math
from types import SimpleNamespace

from collie_demo.heading import SportModeHeadingProvider, directed_progress, normalize_angle


def test_directed_progress_handles_yaw_wraparound() -> None:
    start = math.radians(170)
    current = math.radians(-170)

    assert math.isclose(directed_progress(start, current, 1.0), math.radians(20))
    assert directed_progress(start, current, -1.0) == 0.0
    assert math.isclose(normalize_angle(3 * math.pi), math.pi, abs_tol=1e-9)


def test_sport_mode_provider_exposes_fresh_local_xy_pose() -> None:
    provider = SportModeHeadingProvider(maximum_age_s=0.5)
    provider._on_state(
        SimpleNamespace(
            imu_state=SimpleNamespace(rpy=[0.0, 0.0, 1.25]),
            position=[2.5, -0.75, 0.31],
        )
    )

    sample = provider.status()

    assert sample.healthy is True
    assert sample.position_healthy is True
    assert sample.pose_healthy is True
    assert sample.x_m == 2.5
    assert sample.y_m == -0.75
    assert sample.to_dict()["pose_healthy"] is True
