"""Exclusive, watchdog-protected Unitree motion boundary.

Adapted from go2-follow-clean. Non-zero commands only use Unitree's factory
ObstaclesAvoidClient. SportClient is reserved for the independent StopMove
brake; there is no direct SportClient.Move fallback.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import secrets
import time
from typing import Any, Protocol

from .types import VelocityCommand


class MotionError(RuntimeError):
    pass


class MotionNotReady(MotionError):
    pass


class LeaseMismatch(MotionError):
    pass


class SportClientProtocol(Protocol):
    def SetTimeout(self, timeout_s: float) -> Any: ...
    def Init(self) -> Any: ...
    def StopMove(self) -> int: ...


class AvoidanceClientProtocol(Protocol):
    def SetTimeout(self, timeout_s: float) -> Any: ...
    def Init(self) -> Any: ...
    def SwitchSet(self, enabled: bool) -> int: ...
    def SwitchGet(self) -> tuple[int, bool]: ...
    def UseRemoteCommandFromApi(self, enabled: bool) -> int: ...
    def Move(self, vx: float, vy: float, vyaw: float) -> int: ...


@dataclass(frozen=True, slots=True)
class MotionConfig:
    maximum_forward_mps: float = 0.08
    maximum_yaw_rps: float = 0.25
    command_watchdog_s: float = 0.35
    rpc_timeout_s: float = 0.75
    client_timeout_s: float = 3.0
    avoidance_verify_interval_s: float = 0.30


class UnitreeMotionAdapter:
    def __init__(
        self,
        sport: SportClientProtocol,
        avoidance: AvoidanceClientProtocol,
        config: MotionConfig | None = None,
    ) -> None:
        self.sport = sport
        self.avoidance = avoidance
        self.config = config or MotionConfig()
        self._lock = asyncio.Lock()
        self._rpc_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="collie-sdk")
        self._stop_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="collie-stop")
        self._initialized = False
        self._fault: str | None = None
        self._lease: str | None = None
        self._avoidance_enabled = False
        self._remote_api_enabled = False
        self._last_verify_at: float | None = None
        self._last_command = VelocityCommand(reason="disarmed")
        self._watchdog: asyncio.Task[None] | None = None
        self._watchdog_generation = 0

    @property
    def armed(self) -> bool:
        return bool(
            self._initialized
            and self._fault is None
            and self._lease
            and self._avoidance_enabled
            and self._remote_api_enabled
        )

    def status(self) -> dict[str, object]:
        return {
            "initialized": self._initialized,
            "armed": self.armed,
            "fault": self._fault,
            "avoidance_required": True,
            "avoidance_enabled": self._avoidance_enabled,
            "remote_api_enabled": self._remote_api_enabled,
            "watchdog_s": self.config.command_watchdog_s,
            "limits": {
                "forward_mps": self.config.maximum_forward_mps,
                "yaw_rps": self.config.maximum_yaw_rps,
                "lateral_mps": 0.0,
            },
            "last_command": self._last_command.to_dict(),
        }

    async def initialize(self) -> None:
        async with self._lock:
            if self._initialized:
                return
            try:
                await self._call(self.sport.SetTimeout, self.config.client_timeout_s)
                await self._call(self.sport.Init)
                await self._call(self.avoidance.SetTimeout, self.config.client_timeout_s)
                await self._call(self.avoidance.Init)
            except Exception as exc:
                self._fault = f"initialization failed: {exc}"
                raise MotionNotReady(self._fault) from exc
            self._initialized = True

    async def arm(self) -> str:
        async with self._lock:
            self._require_ready()
            if self._lease is not None:
                raise MotionNotReady("motion lease already active")
            try:
                await self._success(self.avoidance.SwitchSet, True)
                response = await self._call(self.avoidance.SwitchGet)
                if response != (0, True):
                    raise MotionError(f"avoidance not confirmed: {response!r}")
                self._avoidance_enabled = True
                self._last_verify_at = time.monotonic()
                await self._success(self.avoidance.UseRemoteCommandFromApi, True)
                self._remote_api_enabled = True
                await self._success(self.avoidance.Move, 0.0, 0.0, 0.0)
            except Exception as exc:
                await self._release_locked(use_stop=True)
                raise MotionNotReady(f"arm failed: {exc}") from exc
            self._lease = secrets.token_urlsafe(32)
            self._last_command = VelocityCommand(reason="armed_zero")
            return self._lease

    async def send(self, lease: str, command: VelocityCommand) -> VelocityCommand:
        forward = self._bounded_forward(command.forward_mps)
        yaw = self._bounded_yaw(command.yaw_rps)
        async with self._lock:
            self._require_owner(lease)
            # A fresh command has arrived. Do not let the previous command's
            # watchdog race the in-flight avoidance verification/Move RPC.
            # Each RPC remains bounded by rpc_timeout_s and faults to StopMove.
            self._cancel_watchdog()
            try:
                if forward != 0.0 or yaw != 0.0:
                    await self._verify_avoidance_if_due()
                await self._success(self.avoidance.Move, forward, 0.0, yaw)
            except Exception as exc:
                self._fault = f"velocity command failed: {exc}"
                await self._release_locked(use_stop=True)
                raise MotionNotReady(self._fault) from exc
            self._last_command = VelocityCommand(forward, yaw, command.reason)
            if forward != 0.0 or yaw != 0.0:
                self._arm_watchdog()
            else:
                self._cancel_watchdog()
            return self._last_command

    async def release(self, lease: str) -> None:
        async with self._lock:
            self._require_owner(lease)
            errors = await self._release_locked(use_stop=True)
            if errors:
                raise MotionError("; ".join(errors))

    async def emergency_stop(self) -> list[str]:
        self._cancel_watchdog()
        errors: list[str] = []
        if self._initialized:
            try:
                result = await self._call_stop(self.sport.StopMove)
                if result != 0:
                    raise MotionError(f"StopMove returned {result!r}")
            except Exception as exc:
                errors.append(f"StopMove: {exc}")
                if self._fault is None:
                    self._fault = f"StopMove failed: {exc}"
        async with self._lock:
            errors.extend(await self._release_locked(use_stop=False))
        return errors

    async def close(self) -> list[str]:
        errors = await self.emergency_stop()
        self._rpc_executor.shutdown(wait=False, cancel_futures=False)
        self._stop_executor.shutdown(wait=False, cancel_futures=False)
        return errors

    async def _verify_avoidance_if_due(self) -> None:
        if (
            self._last_verify_at is not None
            and time.monotonic() - self._last_verify_at
            < self.config.avoidance_verify_interval_s
        ):
            return
        response = await self._call(self.avoidance.SwitchGet)
        if response != (0, True):
            self._avoidance_enabled = False
            raise MotionError(f"avoidance switched off: {response!r}")
        self._last_verify_at = time.monotonic()

    async def _release_locked(self, *, use_stop: bool) -> list[str]:
        self._cancel_watchdog()
        errors: list[str] = []
        calls: list[tuple[str, Any, tuple[Any, ...], bool]] = []
        if self._initialized and use_stop:
            calls.append(("StopMove", self.sport.StopMove, (), True))
        if self._initialized:
            calls.extend(
                [
                    ("avoidance zero", self.avoidance.Move, (0.0, 0.0, 0.0), False),
                    ("remote API disable", self.avoidance.UseRemoteCommandFromApi, (False,), False),
                    ("avoidance disable", self.avoidance.SwitchSet, (False,), False),
                ]
            )
        for label, method, args, emergency in calls:
            try:
                result = await (self._call_stop(method, *args) if emergency else self._call(method, *args))
                if result != 0:
                    raise MotionError(f"returned {result!r}")
            except Exception as exc:
                errors.append(f"{label}: {exc}")
        self._lease = None
        self._avoidance_enabled = False
        self._remote_api_enabled = False
        self._last_verify_at = None
        self._last_command = VelocityCommand(reason="released")
        return errors

    def _require_ready(self) -> None:
        if not self._initialized:
            raise MotionNotReady("motion adapter is not initialized")
        if self._fault is not None:
            raise MotionNotReady(self._fault)

    def _require_owner(self, lease: str) -> None:
        self._require_ready()
        if not self.armed:
            raise MotionNotReady("motion is disarmed")
        if not isinstance(lease, str) or self._lease is None or not secrets.compare_digest(lease, self._lease):
            raise LeaseMismatch("motion lease is stale or does not match")

    def _arm_watchdog(self) -> None:
        self._cancel_watchdog()
        generation = self._watchdog_generation

        async def expire() -> None:
            try:
                await asyncio.sleep(self.config.command_watchdog_s)
                if generation == self._watchdog_generation:
                    await self.emergency_stop()
            except asyncio.CancelledError:
                return

        self._watchdog = asyncio.create_task(expire())

    def _cancel_watchdog(self) -> None:
        self._watchdog_generation += 1
        task, self._watchdog = self._watchdog, None
        if task is not None and not task.done() and task is not asyncio.current_task():
            task.cancel()

    async def _success(self, method: Any, *args: Any) -> None:
        result = await self._call(method, *args)
        if result != 0:
            raise MotionError(f"{method.__name__} returned {result!r}")

    async def _call(self, method: Any, *args: Any) -> Any:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._rpc_executor, method, *args)
        try:
            return await asyncio.wait_for(asyncio.shield(future), self.config.rpc_timeout_s)
        except asyncio.TimeoutError as exc:
            raise MotionNotReady(f"{method.__name__} timed out") from exc

    async def _call_stop(self, method: Any, *args: Any) -> Any:
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(self._stop_executor, method, *args)
        try:
            return await asyncio.wait_for(asyncio.shield(future), self.config.rpc_timeout_s)
        except asyncio.TimeoutError as exc:
            raise MotionNotReady(f"{method.__name__} timed out") from exc

    def _bounded_forward(self, value: float) -> float:
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError("forward speed must be finite and non-negative")
        return min(number, self.config.maximum_forward_mps)

    def _bounded_yaw(self, value: float) -> float:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("yaw speed must be finite")
        return max(-self.config.maximum_yaw_rps, min(self.config.maximum_yaw_rps, number))


def initialize_dds(network_interface: str | None) -> None:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    if network_interface:
        ChannelFactoryInitialize(0, network_interface)
    else:
        ChannelFactoryInitialize(0)


def create_motion(config: MotionConfig | None = None) -> UnitreeMotionAdapter:
    from unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client import ObstaclesAvoidClient
    from unitree_sdk2py.go2.sport.sport_client import SportClient

    return UnitreeMotionAdapter(SportClient(), ObstaclesAvoidClient(), config)
