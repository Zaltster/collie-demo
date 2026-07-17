#!/usr/bin/env python3
"""Benchmark fruit detectors on the fixed three-fruit Woof capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any

from ultralytics import YOLO, YOLOE


TARGETS = ("apple", "banana", "pear")
GROUND_TRUTH = {
    "pear": (812, 775, 884, 855),
    "banana": (1186, 779, 1262, 844),
    "apple": (1542, 758, 1596, 817),
}
THRESHOLDS = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.80, 0.85)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def canonical_label(label: str) -> str | None:
    folded = label.casefold().strip()
    for target in TARGETS:
        if folded == target or folded == f"{target}s":
            return target
    return None


def center_matches_target(
    box: tuple[float, float, float, float], target: str, padding: int = 35
) -> bool:
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    gt_x1, gt_y1, gt_x2, gt_y2 = GROUND_TRUTH[target]
    return (
        gt_x1 - padding <= center_x <= gt_x2 + padding
        and gt_y1 - padding <= center_y <= gt_y2 + padding
    )


def load_model(weights: Path) -> Any:
    lowered = weights.name.casefold()
    model = YOLOE(str(weights)) if "yoloe" in lowered else YOLO(str(weights))
    existing_targets = {
        canonical_label(str(label)) for label in model.names.values()
    }
    if ("world" in lowered or "yoloe" in lowered) and not set(TARGETS).issubset(
        existing_targets
    ):
        model.set_classes(list(TARGETS))
    return model


def relevant_class_ids(names: dict[int, str]) -> list[int]:
    return [class_id for class_id, name in names.items() if canonical_label(name)]


def benchmark(args: argparse.Namespace) -> dict[str, object]:
    images = sorted(args.source.glob("*.jpg"))
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        raise RuntimeError(f"no JPEG images found in {args.source}")

    started = time.perf_counter()
    model = load_model(args.weights)
    load_s = time.perf_counter() - started
    class_ids = relevant_class_ids(model.names)
    if not class_ids:
        raise RuntimeError(f"none of {TARGETS} found in model classes: {model.names}")

    per_frame: list[dict[str, list[float]]] = []
    false_positive_scores = {target: [] for target in TARGETS}
    inference_ms: list[float] = []
    wall_started = time.perf_counter()
    for image in images:
        result = model.predict(
            source=str(image),
            conf=0.01,
            imgsz=args.imgsz,
            device=args.device,
            classes=class_ids,
            verbose=False,
        )[0]
        inference_ms.append(float(result.speed.get("inference", 0.0)))
        matches = {target: [] for target in TARGETS}
        if result.boxes is not None:
            boxes = result.boxes.xyxy.detach().cpu().tolist()
            scores = result.boxes.conf.detach().cpu().tolist()
            classes = result.boxes.cls.detach().cpu().int().tolist()
            for box, score, class_id in zip(boxes, scores, classes, strict=True):
                target = canonical_label(result.names[int(class_id)])
                if target is None:
                    continue
                box_tuple = tuple(float(value) for value in box)
                if center_matches_target(box_tuple, target):
                    matches[target].append(float(score))
                else:
                    false_positive_scores[target].append(float(score))
        per_frame.append(matches)
    wall_s = time.perf_counter() - wall_started

    metrics: dict[str, object] = {}
    for target in TARGETS:
        best_scores = [max(frame[target], default=0.0) for frame in per_frame]
        positive_scores = [score for score in best_scores if score > 0.0]
        longest_miss_streak_by_threshold: dict[str, int] = {}
        for threshold in THRESHOLDS:
            longest = 0
            current = 0
            for score in best_scores:
                current = current + 1 if score < threshold else 0
                longest = max(longest, current)
            longest_miss_streak_by_threshold[f"{threshold:.2f}"] = longest
        metrics[target] = {
            "recall_by_threshold": {
                f"{threshold:.2f}": round(
                    sum(score >= threshold for score in best_scores) / len(best_scores),
                    4,
                )
                for threshold in THRESHOLDS
            },
            "confidence": {
                "min": round(min(positive_scores), 4) if positive_scores else None,
                "mean": round(statistics.fmean(positive_scores), 4)
                if positive_scores
                else None,
                "max": round(max(positive_scores), 4) if positive_scores else None,
            },
            "false_positives_by_threshold": {
                f"{threshold:.2f}": sum(
                    score >= threshold for score in false_positive_scores[target]
                )
                for threshold in THRESHOLDS
            },
            "longest_miss_streak_by_threshold": longest_miss_streak_by_threshold,
        }

    return {
        "weights": str(args.weights),
        "size_bytes": args.weights.stat().st_size,
        "device": args.device,
        "imgsz": args.imgsz,
        "frames": len(images),
        "load_s": round(load_s, 3),
        "wall_ms_per_frame": round(wall_s * 1000.0 / len(images), 2),
        "reported_inference_ms": {
            "min": round(min(inference_ms), 2),
            "mean": round(statistics.fmean(inference_ms), 2),
            "max": round(max(inference_ms), 2),
        },
        "targets": metrics,
    }


def main() -> None:
    args = parse_args()
    result = benchmark(args)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
