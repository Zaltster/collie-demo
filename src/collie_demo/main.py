from __future__ import annotations

import math
import os
from pathlib import Path

import uvicorn

from .app import create_app
from .camera import create_camera
from .controller import ApproachConfig, ApproachController
from .fruit import FruitDetector
from .heading import SportModeHeadingProvider
from .matcher import FruitInstanceMatcher
from .memory import AppearanceEncoder
from .mission import MissionConfig
from .motion import MotionConfig, create_motion, initialize_dds
from .runtime import CollieRuntime


def env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def env_class_thresholds(name: str) -> dict[str, float]:
    rendered = os.environ.get(name, "").strip()
    if not rendered:
        return {}
    thresholds: dict[str, float] = {}
    for item in rendered.split(","):
        label, separator, value = item.partition("=")
        if not separator or not label.strip() or not value.strip():
            raise ValueError(f"{name} must use label=value pairs")
        thresholds[label.strip()] = float(value)
    return thresholds


def build_runtime() -> CollieRuntime:
    network_interface = os.environ.get("GO2_NETWORK_INTERFACE", "").strip() or None
    motion_enabled = env_bool("COLLIE_MOTION_ENABLED")
    allow_unranged = env_bool("COLLIE_ALLOW_UNRANGED_DEMO")
    produce_model = Path(
        os.environ.get(
            "COLLIE_PRODUCE_MODEL",
            "models/snapstock/fruit_vegetable_yolov8m.pt",
        )
    )
    initialize_dds(network_interface)
    mission_config = MissionConfig(
        enabled=env_bool("COLLIE_MEMORY_DEMO_ENABLED"),
        autonomous_turn_enabled=env_bool("COLLIE_AUTONOMOUS_TURN_ENABLED"),
        direct_turn_enabled=env_bool("COLLIE_DIRECT_TURN_ENABLED"),
        capture_samples=int(os.environ.get("COLLIE_MEMORY_SAMPLES", "5")),
        capture_timeout_s=float(
            os.environ.get("COLLIE_MEMORY_CAPTURE_TIMEOUT_S", "2.0")
        ),
        match_confirmations_required=int(
            os.environ.get("COLLIE_MATCH_CONFIRMATIONS", "3")
        ),
        approach_misses_allowed=int(
            os.environ.get("COLLIE_APPROACH_MATCH_MISSES", "2")
        ),
        turn_angle_rad=math.radians(
            float(os.environ.get("COLLIE_TURN_ANGLE_DEG", "180"))
        ),
        turn_rate_rps=float(os.environ.get("COLLIE_TURN_RATE_RPS", "0.30")),
        turn_tolerance_rad=math.radians(
            float(os.environ.get("COLLIE_TURN_TOLERANCE_DEG", "7"))
        ),
        turn_timeout_s=float(os.environ.get("COLLIE_TURN_TIMEOUT_S", "15")),
        turn_stall_timeout_s=float(
            os.environ.get("COLLIE_TURN_STALL_TIMEOUT_S", "2.0")
        ),
        turn_stall_min_progress_rad=math.radians(
            float(os.environ.get("COLLIE_TURN_STALL_MIN_PROGRESS_DEG", "10"))
        ),
        search_rate_rps=float(os.environ.get("COLLIE_SEARCH_RATE_RPS", "0.20")),
        search_sweep_rad=math.radians(
            float(os.environ.get("COLLIE_SEARCH_SWEEP_DEG", "75"))
        ),
        search_timeout_s=float(
            os.environ.get("COLLIE_SEARCH_TIMEOUT_S", "9")
        ),
        near_bottom_ratio=float(
            os.environ.get("COLLIE_NEAR_BOTTOM_RATIO", "0.86")
        ),
        near_center_ratio=float(
            os.environ.get("COLLIE_NEAR_CENTER_RATIO", "0.72")
        ),
    )
    controller_config = ApproachConfig(
        stable_frames_required=int(os.environ.get("COLLIE_STABLE_FRAMES", "3")),
        maximum_target_age_s=float(
            os.environ.get("COLLIE_MAX_TARGET_AGE_S", "0.35")
        ),
        forward_mps=float(os.environ.get("COLLIE_FORWARD_MPS", "0.08")),
        forward_budget_s=float(os.environ.get("COLLIE_FORWARD_BUDGET_S", "1.5")),
    )
    motion = (
        create_motion(
            MotionConfig(
                maximum_forward_mps=controller_config.forward_mps,
                maximum_yaw_rps=max(
                    controller_config.maximum_yaw_rps,
                    mission_config.turn_rate_rps,
                    mission_config.search_rate_rps,
                ),
            )
        )
        if motion_enabled
        else None
    )
    return CollieRuntime(
        camera=create_camera(),
        controller=ApproachController(controller_config),
        motion=motion,
        motion_enabled=motion_enabled,
        allow_unranged_forward=allow_unranged,
        produce_detector=FruitDetector(
            produce_model,
            confidence=float(os.environ.get("COLLIE_PRODUCE_CONFIDENCE", "0.5")),
            class_thresholds=env_class_thresholds(
                "COLLIE_PRODUCE_CLASS_THRESHOLDS"
            ),
            device=os.environ.get("COLLIE_INFERENCE_DEVICE", "").strip() or None,
        ),
        produce_revalidation_misses_required=int(
            os.environ.get("COLLIE_REVALIDATION_MISSES", "3")
        ),
        maximum_produce_age_s=float(
            os.environ.get("COLLIE_MAX_PRODUCE_AGE_S", "0.75")
        ),
        follow_period_s=float(os.environ.get("COLLIE_FOLLOW_PERIOD_S", "0.05")),
        follow_start_timeout_s=float(
            os.environ.get("COLLIE_FOLLOW_START_TIMEOUT_S", "1.5")
        ),
        instance_matcher=FruitInstanceMatcher(
            AppearanceEncoder(
                minimum_crop_side_px=int(
                    os.environ.get("COLLIE_MIN_MEMORY_CROP_PX", "20")
                )
            ),
            minimum_score=float(
                os.environ.get("COLLIE_INSTANCE_MATCH_SCORE", "0.76")
            ),
            minimum_margin=float(
                os.environ.get("COLLIE_INSTANCE_MATCH_MARGIN", "0.06")
            ),
            maximum_saved_similarity=float(
                os.environ.get("COLLIE_SAVED_INSTANCE_REJECTION_SCORE", "0.94")
            ),
        ),
        heading_provider=(
            SportModeHeadingProvider(
                maximum_age_s=float(
                    os.environ.get("COLLIE_HEADING_MAX_AGE_S", "0.5")
                )
            )
            if mission_config.enabled
            else None
        ),
        mission_config=mission_config,
    )


def main() -> None:
    web_directory = Path(
        os.environ.get("COLLIE_WEB_DIRECTORY", str(Path.cwd() / "web"))
    ).resolve()
    uvicorn.run(
        create_app(build_runtime(), web_directory),
        host=os.environ.get("COLLIE_BIND", "0.0.0.0"),
        port=int(os.environ.get("COLLIE_PORT", "8096")),
        log_level="info",
    )


if __name__ == "__main__":
    main()
