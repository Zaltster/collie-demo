"""Direct Go2 camera access through Unitree's public VideoClient."""

from __future__ import annotations

import time
from typing import Protocol

import cv2
import numpy as np

from .types import CameraFrame


class VideoClientProtocol(Protocol):
    def GetImageSample(self) -> tuple[int, object]: ...


class CameraUnavailable(RuntimeError):
    pass


class UnitreeCamera:
    def __init__(self, client: VideoClientProtocol) -> None:
        self._client = client
        self._frame_id = 0

    def read(self) -> CameraFrame:
        captured = time.monotonic()
        code, payload = self._client.GetImageSample()
        if int(code) != 0:
            raise CameraUnavailable(f"VideoClient returned {code}")
        encoded = np.frombuffer(bytes(payload), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3:
            raise CameraUnavailable("camera returned an invalid JPEG")
        self._frame_id += 1
        return CameraFrame(self._frame_id, captured, image)


def create_camera(timeout_s: float = 3.0) -> UnitreeCamera:
    from unitree_sdk2py.go2.video.video_client import VideoClient

    client = VideoClient()
    client.SetTimeout(float(timeout_s))
    client.Init()
    return UnitreeCamera(client)

