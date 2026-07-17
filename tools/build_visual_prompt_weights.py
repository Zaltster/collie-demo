#!/usr/bin/env python3
"""Bake exact fruit exemplars into a self-contained YOLOE checkpoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        action="append",
        required=True,
        metavar="LABEL=X1,Y1,X2,Y2",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    return parser.parse_args()


def parse_prompts(items: list[str]) -> tuple[list[str], np.ndarray]:
    labels: list[str] = []
    boxes: list[list[float]] = []
    for item in items:
        label, separator, coordinates = item.partition("=")
        if not separator or not label.strip():
            raise ValueError("prompts must use LABEL=X1,Y1,X2,Y2")
        values = [float(value) for value in coordinates.split(",")]
        if len(values) != 4:
            raise ValueError("prompt boxes must contain four coordinates")
        labels.append(label.strip())
        boxes.append(values)
    return labels, np.asarray(boxes, dtype=np.float32)


def main() -> None:
    args = parse_args()
    labels, boxes = parse_prompts(args.prompt)
    model = YOLOE(str(args.weights))
    visual_prompts = {
        "bboxes": boxes,
        "cls": np.arange(len(labels), dtype=np.int32),
    }
    model.predict(
        source=str(args.reference),
        refer_image=str(args.reference),
        visual_prompts=visual_prompts,
        predictor=YOLOEVPSegPredictor,
        conf=0.2,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )
    model.model.names = dict(enumerate(labels))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(args.output))
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"saved={args.output}")
    print(f"size_bytes={args.output.stat().st_size}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
