from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from collie_demo.app import create_app


class FakeRuntime:
    def __init__(self) -> None:
        self.selected: str | None = None
        self.center: tuple[int, int] | None = None

    async def start(self) -> None:
        return

    async def close(self) -> None:
        return

    async def status(self) -> dict[str, object]:
        return {"selected_target_name": self.selected}

    async def jpeg(self) -> bytes:
        return b"annotated-jpeg"

    async def raw_jpeg(self) -> bytes:
        return b"raw-jpeg"

    async def select_target(
        self, target: str, center: tuple[int, int] | None = None
    ) -> dict[str, object]:
        self.selected = target
        self.center = center
        return await self.status()

    async def start_follow(self, confirmation: str) -> dict[str, object]:
        return {
            "selected_target_name": self.selected,
            "confirmation": confirmation,
            "follow_active": True,
        }

    async def navigation_status(self) -> dict[str, object]:
        return {"available": True, "armed": False}

    async def navigation_arm(self, confirmation: str) -> dict[str, object]:
        return {"available": True, "armed": True, "confirmation": confirmation}

    async def navigation_command(
        self, forward_mps: float, yaw_rps: float
    ) -> dict[str, object]:
        return {
            "available": True,
            "armed": True,
            "command": {"forward_mps": forward_mps, "yaw_rps": yaw_rps},
        }

    async def stop(self, reason: str = "user_stop") -> dict[str, object]:
        return {"armed": False, "reason": reason}


def test_target_endpoint_selects_a_specific_detection(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("ok")
    runtime = FakeRuntime()

    with TestClient(create_app(runtime, tmp_path)) as client:  # type: ignore[arg-type]
        response = client.post(
            "/api/target", json={"target": "apple", "center": [1010, 240]}
        )

        assert response.status_code == 200
        assert response.json()["selected_target_name"] == "apple"
        assert runtime.center == (1010, 240)
        assert client.post("/api/target", json={"center": [1, 2]}).status_code == 422
        assert client.post(
            "/api/target", json={"target": "banana", "center": [1]}
        ).status_code == 422

        follow = client.post(
            "/api/follow", json={"confirmation": "TARGET AND PATH CLEAR"}
        )
        assert follow.status_code == 200
        assert follow.json()["follow_active"] is True
        assert client.get("/camera.jpg").content == b"annotated-jpeg"
        assert client.get("/camera-raw.jpg").content == b"raw-jpeg"


def test_navigation_motion_endpoints_are_separate_from_fruit_follow(
    tmp_path: Path,
) -> None:
    (tmp_path / "index.html").write_text("ok")
    runtime = FakeRuntime()

    with TestClient(create_app(runtime, tmp_path)) as client:  # type: ignore[arg-type]
        assert client.get("/api/navigation/status").json()["available"] is True
        armed = client.post(
            "/api/navigation/arm", json={"confirmation": "MAP AND PATH CLEAR"}
        )
        assert armed.status_code == 200
        assert armed.json()["armed"] is True
        command = client.post(
            "/api/navigation/cmd",
            json={"forward_mps": 0.3, "yaw_rps": -0.2},
        )
        assert command.status_code == 200
        assert command.json()["command"] == {
            "forward_mps": 0.3,
            "yaw_rps": -0.2,
        }
        assert client.post("/api/navigation/stop").json()["armed"] is False
