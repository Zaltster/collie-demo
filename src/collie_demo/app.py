from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, ConfigDict

from .runtime import CollieRuntime, RuntimeCommandError


class ArmRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confirmation: str


class TargetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: str
    center: tuple[int, int] | None = None


def create_app(runtime: CollieRuntime, web_directory: Path) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await runtime.start()
        try:
            yield
        finally:
            await runtime.close()

    app = FastAPI(title="Collie fruit-follow demo", version="1", lifespan=lifespan)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(web_directory / "index.html")

    @app.get("/api/status")
    async def status() -> dict[str, object]:
        return await runtime.status()

    @app.get("/camera.jpg")
    async def camera() -> Response:
        jpeg = await runtime.jpeg()
        if jpeg is None:
            raise HTTPException(status_code=503, detail="no camera frame yet")
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/camera-raw.jpg")
    async def raw_camera() -> Response:
        jpeg = await runtime.raw_jpeg()
        if jpeg is None:
            raise HTTPException(status_code=503, detail="no camera frame yet")
        return Response(jpeg, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/api/arm")
    async def arm(request: ArmRequest) -> dict[str, object]:
        try:
            return await runtime.arm(request.confirmation)
        except RuntimeCommandError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/follow")
    async def follow(request: ArmRequest) -> dict[str, object]:
        try:
            return await runtime.start_follow(request.confirmation)
        except RuntimeCommandError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/target")
    async def target(request: TargetRequest) -> dict[str, object]:
        try:
            return await runtime.select_target(request.target, request.center)
        except RuntimeCommandError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/pulse")
    async def pulse() -> dict[str, object]:
        try:
            return await runtime.pulse()
        except RuntimeCommandError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/stop")
    async def stop() -> dict[str, object]:
        return await runtime.stop()

    return app
