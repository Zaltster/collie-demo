from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from collie_demo.app import create_app


class FakeRuntime:
    def __init__(self) -> None:
        self.selected = "blue"

    async def start(self) -> None:
        return

    async def close(self) -> None:
        return

    async def status(self) -> dict[str, object]:
        return {"selected_target_color": self.selected}

    async def select_target(self, color: str) -> dict[str, object]:
        self.selected = color
        return await self.status()


def test_target_endpoint_selects_blue_or_yellow(tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("ok")
    runtime = FakeRuntime()

    with TestClient(create_app(runtime, tmp_path)) as client:  # type: ignore[arg-type]
        response = client.post("/api/target", json={"color": "yellow"})

        assert response.status_code == 200
        assert response.json()["selected_target_color"] == "yellow"
        assert client.post("/api/target", json={"color": "purple"}).status_code == 422
