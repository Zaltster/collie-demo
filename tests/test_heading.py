import math

from collie_demo.heading import directed_progress, normalize_angle


def test_directed_progress_handles_yaw_wraparound() -> None:
    start = math.radians(170)
    current = math.radians(-170)

    assert math.isclose(directed_progress(start, current, 1.0), math.radians(20))
    assert directed_progress(start, current, -1.0) == 0.0
    assert math.isclose(normalize_angle(3 * math.pi), math.pi, abs_tol=1e-9)
