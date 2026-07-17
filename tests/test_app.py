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
