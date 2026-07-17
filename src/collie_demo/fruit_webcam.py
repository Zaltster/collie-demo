from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .fruit import FruitDetector, annotate_fruits


DEFAULT_MODEL = Path("models/snapstock/fruit_vegetable_yolov8m.pt")


def main() -> None:
    parser = argparse.ArgumentParser(description="Local YOLO fruit webcam detector")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--confidence", type=float, default=0.5)
    args = parser.parse_args()

    detector = FruitDetector(args.model, confidence=args.confidence)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"could not open camera index {args.camera}")

    print(f"model={detector.model_path}")
    print(f"classes={detector.names}")
    print("Press q in the video window to exit.")
    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                raise RuntimeError(f"camera index {args.camera} stopped returning frames")
            detections = detector.detect(frame)
            for detection in detections:
                print(detection.log_line(), flush=True)
            cv2.imshow("Local Fruit Detector", annotate_fruits(frame, detections))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
