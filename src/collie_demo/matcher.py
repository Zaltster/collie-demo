"""Rank live fruit detections against a saved per-round appearance memory."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .fruit import FruitDetection
from .memory import AppearanceEncoder, FruitMemory, descriptor_similarity


@dataclass(frozen=True, slots=True)
class CandidateMatch:
    detection: FruitDetection
    score: float
    color_score: float
    shape_score: float

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.detection.label,
            "center": self.detection.center,
            "bbox_xyxy": self.detection.bbox_xyxy,
            "detector_confidence": self.detection.confidence,
            "score": self.score,
            "color_score": self.color_score,
            "shape_score": self.shape_score,
        }


@dataclass(frozen=True, slots=True)
class MatchResult:
    accepted: bool
    best: CandidateMatch | None
    runner_up_score: float | None
    margin: float | None
    candidate_count: int
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "best": None if self.best is None else self.best.to_dict(),
            "runner_up_score": self.runner_up_score,
            "margin": self.margin,
            "candidate_count": self.candidate_count,
            "reason": self.reason,
        }


class FruitInstanceMatcher:
    def __init__(
        self,
        encoder: AppearanceEncoder | None = None,
        *,
        minimum_score: float = 0.76,
        minimum_margin: float = 0.06,
        maximum_saved_similarity: float = 0.94,
    ) -> None:
        if not 0.0 <= minimum_score <= 1.0:
            raise ValueError("minimum_score must be between zero and one")
        if not 0.0 <= minimum_margin <= 1.0:
            raise ValueError("minimum_margin must be between zero and one")
        if not 0.0 <= maximum_saved_similarity <= 1.0:
            raise ValueError("maximum_saved_similarity must be between zero and one")
        self.encoder = encoder or AppearanceEncoder()
        self.minimum_score = float(minimum_score)
        self.minimum_margin = float(minimum_margin)
        self.maximum_saved_similarity = float(maximum_saved_similarity)

    def rank(
        self,
        memory: FruitMemory,
        bgr: NDArray[np.uint8],
        detections: list[FruitDetection],
    ) -> MatchResult:
        ranked = self._rank_candidates(memory, bgr, detections)
        ranked.sort(key=lambda item: item.score, reverse=True)
        if not ranked:
            return MatchResult(False, None, None, None, 0, "no_same_class_candidate")
        best = ranked[0]
        runner_up = ranked[1].score if len(ranked) > 1 else None
        margin = best.score if runner_up is None else best.score - runner_up
        if best.score < self.minimum_score:
            reason = "appearance_score_too_low"
            accepted = False
        elif margin < self.minimum_margin:
            reason = "appearance_match_ambiguous"
            accepted = False
        else:
            reason = "appearance_match"
            accepted = True
        return MatchResult(
            accepted,
            best,
            runner_up,
            round(margin, 4),
            len(ranked),
            reason,
        )

    def rank_different(
        self,
        memory: FruitMemory,
        bgr: NDArray[np.uint8],
        detections: list[FruitDetection],
    ) -> MatchResult:
        """Choose a same-class fruit only when it is not the saved instance.

        Similarity to the saved reference is exclusion evidence here: a high
        score means "this is banana A" and must never become a motion target.
        Among candidates below the rejection threshold, the least similar
        candidate is preferred. Multiple acceptable candidates are not
        ambiguous because the mission is allowed to approach any banana other
        than the one that was shown.
        """

        ranked = self._rank_candidates(memory, bgr, detections)
        ranked.sort(key=lambda item: (item.score, -item.detection.confidence))
        if not ranked:
            return MatchResult(False, None, None, None, 0, "no_same_class_candidate")
        best = ranked[0]
        runner_up = ranked[1].score if len(ranked) > 1 else None
        rejection_margin = self.maximum_saved_similarity - best.score
        accepted = best.score < self.maximum_saved_similarity
        reason = "different_instance_match" if accepted else "saved_instance_only"
        return MatchResult(
            accepted,
            best,
            runner_up,
            round(rejection_margin, 4),
            len(ranked),
            reason,
        )

    def _rank_candidates(
        self,
        memory: FruitMemory,
        bgr: NDArray[np.uint8],
        detections: list[FruitDetection],
    ) -> list[CandidateMatch]:
        ranked: list[CandidateMatch] = []
        for detection in detections:
            if detection.label.casefold() != memory.label.casefold():
                continue
            try:
                candidate, _ = self.encoder.encode_bbox(bgr, detection.bbox_xyxy)
            except ValueError:
                continue
            similarities = [
                descriptor_similarity(reference, candidate)
                for reference in memory.samples
            ]
            # Use the average of the best two reference views. This tolerates a
            # single weak capture without accepting an unrelated outlier.
            similarities.sort(key=lambda item: item[0], reverse=True)
            selected = similarities[: min(2, len(similarities))]
            score = float(np.mean([item[0] for item in selected]))
            color_score = float(np.mean([item[1] for item in selected]))
            shape_score = float(np.mean([item[2] for item in selected]))
            ranked.append(
                CandidateMatch(
                    detection=detection,
                    score=round(score, 4),
                    color_score=round(color_score, 4),
                    shape_score=round(shape_score, 4),
                )
            )
        return ranked
