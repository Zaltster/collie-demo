from __future__ import annotations

import os
from pathlib import Path

import uvicorn

from .app import create_app
from .camera import create_camera
from .controller import ApproachConfig, ApproachController
from .fruit import FruitDetector
from .motion import MotionConfig, create_motion, initialize_dds
from .runtime import CollieRuntime


def env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() in {
        "1",
        "true",
        "yes",
    }


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
    controller_config = ApproachConfig(
        forward_mps=float(os.environ.get("COLLIE_FORWARD_MPS", "0.08")),
        forward_budget_s=float(os.environ.get("COLLIE_FORWARD_BUDGET_S", "1.5")),
    )
    motion = (
        create_motion(
            MotionConfig(
                maximum_forward_mps=controller_config.forward_mps,
                maximum_yaw_rps=controller_config.maximum_yaw_rps,
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
            device=os.environ.get("COLLIE_INFERENCE_DEVICE", "").strip() or None,
        ),
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
