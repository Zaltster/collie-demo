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
        code, payload = self._client.GetImageSample()
        # GetImageSample is a blocking RPC.  Timestamping before it starts makes
        # a newly returned image appear older by the full camera/network wait,
        # which can consume most of the controller's freshness budget before
        # inference even begins.  The SDK payload has no source timestamp, so
        # receipt time is the honest local freshness boundary.
        received = time.monotonic()
        if int(code) != 0:
            raise CameraUnavailable(f"VideoClient returned {code}")
        encoded = np.frombuffer(bytes(payload), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None or image.ndim != 3:
            raise CameraUnavailable("camera returned an invalid JPEG")
        self._frame_id += 1
        return CameraFrame(self._frame_id, received, image)


def create_camera(timeout_s: float = 3.0) -> UnitreeCamera:
    from unitree_sdk2py.go2.video.video_client import VideoClient

    client = VideoClient()
    client.SetTimeout(float(timeout_s))
    client.Init()
    return UnitreeCamera(client)
