"""Direct Go2 camera access through Unitree's public VideoClient."""

from __future__ import annotations

import time
from typing import Protocol

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
        source_jpeg = bytes(payload)
        width, height = _jpeg_dimensions(source_jpeg)
        self._frame_id += 1
        return CameraFrame(
            self._frame_id,
            received,
            None,
            source_jpeg,
            width,
            height,
        )


def _jpeg_dimensions(payload: bytes) -> tuple[int, int]:
    """Read JPEG SOF dimensions without decoding the 1920x1080 image."""

    if len(payload) < 4 or payload[:2] != b"\xff\xd8":
        raise CameraUnavailable("camera returned an invalid JPEG")
    sof_markers = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    offset = 2
    while offset < len(payload):
        if payload[offset] != 0xFF:
            offset += 1
            continue
        while offset < len(payload) and payload[offset] == 0xFF:
            offset += 1
        if offset >= len(payload):
            break
        marker = payload[offset]
        offset += 1
        if marker == 0xDA:  # Start of scan; SOF must already have appeared.
            break
        if marker == 0x01 or 0xD0 <= marker <= 0xD9:
            continue
        if offset + 2 > len(payload):
            break
        segment_length = int.from_bytes(payload[offset : offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(payload):
            break
        if marker in sof_markers and segment_length >= 7:
            height = int.from_bytes(payload[offset + 3 : offset + 5], "big")
            width = int.from_bytes(payload[offset + 5 : offset + 7], "big")
            if width > 0 and height > 0:
                return width, height
            break
        offset += segment_length
    raise CameraUnavailable("camera returned a JPEG without dimensions")


def create_camera(timeout_s: float = 3.0) -> UnitreeCamera:
    from unitree_sdk2py.go2.video.video_client import VideoClient

    client = VideoClient()
    client.SetTimeout(float(timeout_s))
    client.Init()
    return UnitreeCamera(client)
