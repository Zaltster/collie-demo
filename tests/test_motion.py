from __future__ import annotations

import asyncio
import time

from collie_demo.motion import MotionConfig, UnitreeMotionAdapter
from collie_demo.types import VelocityCommand


class FakeSport:
    def __init__(self) -> None:
        self.stop_calls = 0

    def SetTimeout(self, _value: float) -> None: pass
    def Init(self) -> None: pass
    def StopMove(self) -> int:
        self.stop_calls += 1
        return 0


class FakeAvoidance:
    def __init__(self) -> None:
        self.enabled = False
        self.remote = False
        self.moves: list[tuple[float, float, float]] = []

    def SetTimeout(self, _value: float) -> None: pass
    def Init(self) -> None: pass
    def SwitchSet(self, enabled: bool) -> int:
        self.enabled = enabled
        return 0
    def SwitchGet(self) -> tuple[int, bool]: return (0, self.enabled)
    def UseRemoteCommandFromApi(self, enabled: bool) -> int:
        self.remote = enabled
        return 0
    def Move(self, vx: float, vy: float, vyaw: float) -> int:
        self.moves.append((vx, vy, vyaw))
        return 0


class DelayedAvoidance(FakeAvoidance):
    def __init__(self) -> None:
        super().__init__()
        self.move_delay_s = 0.0

    def Move(self, vx: float, vy: float, vyaw: float) -> int:
        time.sleep(self.move_delay_s)
        return super().Move(vx, vy, vyaw)


def test_motion_uses_avoidance_and_clamps_forward_speed() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = UnitreeMotionAdapter(sport, avoidance)
        await motion.initialize()
        lease = await motion.arm()
        sent = await motion.send(lease, VelocityCommand(1.0, 0.0, "test"))
        assert sent.forward_mps == 0.08
        assert avoidance.moves[-1] == (0.08, 0.0, 0.0)
        await motion.close()

    asyncio.run(scenario())


def test_stale_command_watchdog_brakes_and_revokes_lease() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = UnitreeMotionAdapter(
            sport,
            avoidance,
            MotionConfig(command_watchdog_s=0.03),
        )
        await motion.initialize()
        lease = await motion.arm()
        await motion.send(lease, VelocityCommand(0.05, 0.0, "test"))
        await asyncio.sleep(0.08)
        assert not motion.armed
        assert sport.stop_calls >= 1
        assert avoidance.moves[-1] == (0.0, 0.0, 0.0)
        await motion.close()

    asyncio.run(scenario())


def test_fresh_inflight_command_does_not_race_previous_watchdog() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), DelayedAvoidance()
        motion = UnitreeMotionAdapter(
            sport,
            avoidance,
            MotionConfig(command_watchdog_s=0.03, rpc_timeout_s=0.2),
        )
        await motion.initialize()
        lease = await motion.arm()
        await motion.send(lease, VelocityCommand(0.05, 0.0, "first"))
        avoidance.move_delay_s = 0.06
        await motion.send(lease, VelocityCommand(0.05, 0.0, "fresh"))
        assert motion.armed
        assert sport.stop_calls == 0
        await motion.close()

    asyncio.run(scenario())
