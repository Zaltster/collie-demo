"""Per-round fruit-instance memory built from local camera crops.

The detector label answers "what kind of fruit is this?". This module keeps an
appearance descriptor for the physical prop shown before the turn. The mission
uses that descriptor as a negative reference: same-class detections that look
like the shown prop are rejected, while a different-looking instance may be
selected.
"""

from __future__ import annotations

from dataclasses import dataclass
import secrets
import time

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class AppearanceDescriptor:
    color: NDArray[np.float32]
    shape: NDArray[np.float32]
    quality: float


@dataclass(frozen=True, slots=True)
class FruitMemory:
    memory_id: str
    label: str
    created_monotonic_s: float
    samples: tuple[AppearanceDescriptor, ...]
    reference_jpeg: bytes
    reference_bbox_xyxy: tuple[int, int, int, int]

    @classmethod
    def create(
        cls,
        *,
        label: str,
        samples: list[AppearanceDescriptor],
        reference_jpeg: bytes,
        reference_bbox_xyxy: tuple[int, int, int, int],
    ) -> "FruitMemory":
        if not samples:
            raise ValueError("fruit memory needs at least one appearance sample")
        return cls(
            memory_id=secrets.token_urlsafe(10),
            label=label,
            created_monotonic_s=time.monotonic(),
            samples=tuple(samples),
            reference_jpeg=reference_jpeg,
            reference_bbox_xyxy=reference_bbox_xyxy,
        )

    def to_status(self, now: float | None = None) -> dict[str, object]:
        current = time.monotonic() if now is None else now
        return {
            "id": self.memory_id,
            "label": self.label,
            "role": "excluded_instance",
            "sample_count": len(self.samples),
            "age_s": round(max(0.0, current - self.created_monotonic_s), 3),
            "reference_bbox_xyxy": self.reference_bbox_xyxy,
        }


class AppearanceEncoder:
    """Small deterministic appearance embedding requiring only OpenCV.

    Color histograms carry most of the signal for the brightly colored stage
    props.  Gradient and normalized thumbnail features add shape/texture
    evidence so two same-class props can be distinguished when their visible
    appearance actually differs.
    """

    def __init__(self, *, minimum_crop_side_px: int = 20) -> None:
        if minimum_crop_side_px < 8:
            raise ValueError("minimum_crop_side_px must be at least 8")
        self.minimum_crop_side_px = int(minimum_crop_side_px)

    def encode_bbox(
        self,
        bgr: NDArray[np.uint8],
        bbox_xyxy: tuple[int, int, int, int],
    ) -> tuple[AppearanceDescriptor, NDArray[np.uint8]]:
        crop = crop_bbox(bgr, bbox_xyxy)
        if min(crop.shape[:2]) < self.minimum_crop_side_px:
            raise ValueError("detected fruit crop is too small to remember reliably")
        return self.encode(crop), crop

    def encode(self, crop: NDArray[np.uint8]) -> AppearanceDescriptor:
        if crop.ndim != 3 or crop.shape[2] != 3 or crop.size == 0:
            raise ValueError("appearance crop must be a non-empty BGR image")
        resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(resized, cv2.COLOR_BGR2LAB)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        mask = np.zeros((64, 64), dtype=np.uint8)
        cv2.ellipse(mask, (32, 32), (29, 29), 0, 0, 360, 255, -1)
        hsv_hist = cv2.calcHist(
            [hsv], [0, 1], mask, [18, 8], [0, 180, 0, 256]
        ).reshape(-1)
        lab_hist = cv2.calcHist(
            [lab], [1, 2], mask, [12, 12], [0, 256, 0, 256]
        ).reshape(-1)
        masked_pixels = lab[mask > 0].astype(np.float32)
        color_moments = np.concatenate(
            (masked_pixels.mean(axis=0), masked_pixels.std(axis=0))
        )
        color = _unit(
            np.concatenate(
                (
                    _unit(hsv_hist),
                    _unit(lab_hist),
                    _unit(color_moments),
                )
            )
        )

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        hog_parts: list[NDArray[np.float32]] = []
        for top in (0, 32):
            for left in (0, 32):
                cell_angle = angle[top : top + 32, left : left + 32]
                cell_magnitude = magnitude[top : top + 32, left : left + 32]
                bins = np.mod((cell_angle / 20.0).astype(np.int32), 9)
                histogram = np.bincount(
                    bins.reshape(-1),
                    weights=cell_magnitude.reshape(-1),
                    minlength=9,
                ).astype(np.float32)
                hog_parts.append(_unit(histogram))

        thumbnail = cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA).astype(
            np.float32
        )
        thumbnail = _unit(thumbnail.reshape(-1) - float(thumbnail.mean()))
        edges = cv2.Canny(gray, 60, 160)
        edge_thumbnail = cv2.resize(
            edges, (16, 16), interpolation=cv2.INTER_AREA
        ).astype(np.float32)
        shape = _unit(
            np.concatenate((*hog_parts, thumbnail, _unit(edge_thumbnail.reshape(-1))))
        )

        sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        area_score = min(1.0, (crop.shape[0] * crop.shape[1]) / 12_000.0)
        sharpness_score = min(1.0, sharpness / 180.0)
        quality = round(0.55 * area_score + 0.45 * sharpness_score, 4)
        return AppearanceDescriptor(color=color, shape=shape, quality=quality)


def descriptor_similarity(
    reference: AppearanceDescriptor, candidate: AppearanceDescriptor
) -> tuple[float, float, float]:
    color = _cosine(reference.color, candidate.color)
    shape = (_cosine(reference.shape, candidate.shape) + 1.0) / 2.0
    combined = 0.55 * color + 0.45 * shape
    return (
        round(float(np.clip(combined, 0.0, 1.0)), 4),
        round(float(np.clip(color, 0.0, 1.0)), 4),
        round(float(np.clip(shape, 0.0, 1.0)), 4),
    )


def encode_jpeg(crop: NDArray[np.uint8]) -> bytes:
    ok, encoded = cv2.imencode(
        ".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 94]
    )
    if not ok:
        raise RuntimeError("could not encode remembered fruit reference")
    return encoded.tobytes()


def crop_bbox(
    bgr: NDArray[np.uint8], bbox_xyxy: tuple[int, int, int, int]
) -> NDArray[np.uint8]:
    if bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("camera frame must be a BGR image")
    x1, y1, x2, y2 = bbox_xyxy
    height, width = bgr.shape[:2]
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(x1 + 1, min(width, int(x2)))
    y2 = max(y1 + 1, min(height, int(y2)))
    return bgr[y1:y2, x1:x2].copy()


def _unit(values: NDArray[np.floating] | NDArray[np.integer]) -> NDArray[np.float32]:
    vector = np.asarray(values, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-8:
        return np.zeros_like(vector, dtype=np.float32)
    return vector / norm


def _cosine(first: NDArray[np.float32], second: NDArray[np.float32]) -> float:
    if first.shape != second.shape:
        raise ValueError("appearance descriptors have incompatible shapes")
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= 1e-8 and second_norm <= 1e-8:
        # Two absent feature components are equivalent, not half-similar. This
        # matters for smooth, textureless fruit crops whose edge descriptor is
        # legitimately all zeros.
        return 1.0
    if first_norm <= 1e-8 or second_norm <= 1e-8:
        return 0.0
    return float(np.dot(first, second) / (first_norm * second_norm))
