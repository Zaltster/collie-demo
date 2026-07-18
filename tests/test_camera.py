from __future__ import annotations

import cv2
import numpy as np

from collie_demo import camera as camera_module
from collie_demo.camera import UnitreeCamera


class FakeVideoClient:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.returned_image = False

    def GetImageSample(self) -> tuple[int, bytes]:
        self.returned_image = True
        return 0, self.payload


def test_camera_timestamp_is_taken_after_blocking_sdk_read(monkeypatch) -> None:
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    encoded_ok, encoded = cv2.imencode(".jpg", image)
    assert encoded_ok
    client = FakeVideoClient(encoded.tobytes())

    def receipt_time() -> float:
        assert client.returned_image, "timestamp was taken before GetImageSample returned"
        return 123.5

    monkeypatch.setattr(camera_module.time, "monotonic", receipt_time)

    frame = UnitreeCamera(client).read()

    assert frame.frame_id == 1
    assert frame.captured_monotonic_s == 123.5
    assert frame.bgr.shape == image.shape
