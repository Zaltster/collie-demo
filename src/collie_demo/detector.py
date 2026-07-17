"""Small color-whale detectors tuned for Woof's bright floor view."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .types import BlueWhaleObservation, CameraFrame


class ColorWhaleDetector:
    """Detect a color-coded toy and retain a short stable track.

    This is deliberately a named-demo detector, not a general object model.
    It combines chroma with a floor-region and component-size gate.
    """

    def __init__(
        self,
        *,
        color: str,
        minimum_color_score: float = 18.0,
        floor_start_fraction: float = 0.65,
        horizontal_start_fraction: float = 0.08,
        horizontal_end_fraction: float = 0.95,
        maximum_center_jump_px: float = 110.0,
    ) -> None:
        if color not in {"blue", "yellow"}:
            raise ValueError(f"unsupported whale color: {color}")
        self.color = color
        self.minimum_color_score = float(minimum_color_score)
        self.floor_start_fraction = float(floor_start_fraction)
        self.horizontal_start_fraction = float(horizontal_start_fraction)
        self.horizontal_end_fraction = float(horizontal_end_fraction)
        self.maximum_center_jump_px = float(maximum_center_jump_px)
        self._last_center: tuple[int, int] | None = None
        self._visible_frames = 0

    def detect(self, frame: CameraFrame) -> BlueWhaleObservation | None:
        image = frame.bgr
        height, width = image.shape[:2]
        pixels = image.astype(np.float32)
        blue, green, red = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
        color_score = self._color_score(blue, green, red)
        brightness = np.maximum(np.maximum(blue, green), red)
        yy, xx = np.indices((height, width))
        mask = (
            self._color_mask(image, color_score, brightness)
            & (yy >= height * self.floor_start_fraction)
            & (xx >= width * self.horizontal_start_fraction)
            & (xx <= width * self.horizontal_end_fraction)
        ).astype(np.uint8) * 255
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        )

        count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        candidates: list[tuple[float, tuple[int, int, int, int], tuple[int, int]]] = []
        for label in range(1, count):
            x, y, box_width, box_height, area = (int(v) for v in stats[label])
            if not (
                35 <= area <= 8_000
                and 7 <= box_width <= 180
                and 5 <= box_height <= 150
            ):
                continue
            cx, cy = (int(round(v)) for v in centroids[label])
            component = mask[y : y + box_height, x : x + box_width] > 0
            scores = color_score[y : y + box_height, x : x + box_width][component]
            mean_score = float(np.mean(scores)) if scores.size else 0.0
            confidence = min(0.99, 0.45 + area / 1800.0 + mean_score / 180.0)
            candidates.append(
                (confidence, (x, y, box_width, box_height), (cx, cy))
            )

        if not candidates:
            self._last_center = None
            self._visible_frames = 0
            return None
        if self._last_center is None:
            confidence, bbox, center = max(
                candidates,
                key=lambda item: (item[0], item[1][2] * item[1][3]),
            )
        else:
            nearby = [
                item
                for item in candidates
                if math.dist(item[2], self._last_center) <= self.maximum_center_jump_px
            ]
            if nearby:
                confidence, bbox, center = min(
                    nearby,
                    key=lambda item: math.dist(item[2], self._last_center),
                )
            else:
                confidence, bbox, center = max(
                    candidates,
                    key=lambda item: (item[0], item[1][2] * item[1][3]),
                )
        if self._last_center is not None and math.dist(center, self._last_center) <= self.maximum_center_jump_px:
            self._visible_frames += 1
        else:
            self._visible_frames = 1
        self._last_center = center
        return BlueWhaleObservation(
            frame_id=frame.frame_id,
            captured_monotonic_s=frame.captured_monotonic_s,
            bbox_xywh=bbox,
            center=center,
            confidence=round(confidence, 3),
            visible_frames=self._visible_frames,
        )

    def _color_score(
        self,
        blue: np.ndarray,
        green: np.ndarray,
        red: np.ndarray,
    ) -> np.ndarray:
        if self.color == "blue":
            return (blue + green) / 2.0 - red
        return (red + green) / 2.0 - blue

    def _color_mask(
        self,
        image: np.ndarray,
        color_score: np.ndarray,
        brightness: np.ndarray,
    ) -> np.ndarray:
        if self.color == "blue":
            return (color_score >= self.minimum_color_score) & (brightness >= 125.0)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue, saturation, value = cv2.split(hsv)
        return (
            (color_score >= self.minimum_color_score)
            & (hue >= 10)
            & (hue <= 38)
            & (saturation >= 50)
            & (value >= 140)
        )


class BlueWhaleDetector(ColorWhaleDetector):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(color="blue", **kwargs)


class YellowWhaleDetector(ColorWhaleDetector):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(color="yellow", minimum_color_score=18.0, **kwargs)


def annotate(frame: CameraFrame, target: BlueWhaleObservation | None) -> bytes:
    return annotate_whales(frame, {"blue": target}, selected_color="blue")


def annotate_whales(
    frame: CameraFrame,
    targets: dict[str, BlueWhaleObservation | None],
    *,
    selected_color: str,
) -> bytes:
    image = frame.bgr.copy()
    palette = {"blue": (255, 255, 0), "yellow": (0, 220, 255)}
    for color, target in targets.items():
        if target is None:
            continue
        x, y, width, height = target.bbox_xywh
        box_color = palette.get(color, (255, 255, 255))
        thickness = 5 if color == selected_color else 2
        cv2.rectangle(
            image,
            (x, y),
            (x + width, y + height),
            box_color,
            thickness,
        )
        cv2.circle(image, target.center, 5, (0, 255, 0), -1)
        prefix = "SELECTED " if color == selected_color else ""
        cv2.putText(
            image,
            f"{prefix}{color.upper()} WHALE {target.confidence:.2f} stable:{target.visible_frames}",
            (max(8, x), max(24, y - 9)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            box_color,
            2,
            cv2.LINE_AA,
        )
    selected = targets.get(selected_color)
    if selected is None:
        cv2.putText(
            image,
            f"SELECTED {selected_color.upper()} WHALE NOT FOUND - MOTION ZERO",
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 80, 255),
            2,
            cv2.LINE_AA,
        )
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise RuntimeError("could not encode annotated frame")
    return encoded.tobytes()
