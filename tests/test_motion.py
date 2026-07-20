from __future__ import annotations

import asyncio
import time

from collie_demo.motion import MotionConfig, MotionNotReady, UnitreeMotionAdapter
from collie_demo.types import VelocityCommand


class FakeSport:
    def __init__(self) -> None:
        self.stop_calls = 0
        self.balance_stand_calls = 0
        self.moves: list[tuple[float, float, float]] = []
        self.current_move = (0.0, 0.0, 0.0)

    def SetTimeout(self, _value: float) -> None: pass
    def Init(self) -> None: pass
    def Move(self, vx: float, vy: float, vyaw: float) -> int:
        self.current_move = (vx, vy, vyaw)
        self.moves.append(self.current_move)
        return 0
    def BalanceStand(self) -> int:
        self.balance_stand_calls += 1
        return 0
    def StopMove(self) -> int:
        self.stop_calls += 1
        self.current_move = (0.0, 0.0, 0.0)
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


def test_direct_yaw_bypasses_avoidance_and_hard_locks_translation() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = UnitreeMotionAdapter(sport, avoidance)
        await motion.initialize()
        lease = await motion.arm_direct_yaw()

        sent = await motion.send_direct_yaw(lease, 1.0, "measured_turn")

        assert sent.forward_mps == 0.0
        assert sent.yaw_rps == motion.config.maximum_yaw_rps
        assert sport.moves == [(0.0, 0.0, motion.config.maximum_yaw_rps)]
        assert sport.balance_stand_calls == 1
        assert avoidance.moves == []
        assert motion.status()["mode"] == "direct_yaw"
        try:
            await motion.send(lease, VelocityCommand(0.08, 0.0, "forbidden"))
        except MotionNotReady:
            pass
        else:
            raise AssertionError("direct-yaw lease accepted translation")
        assert avoidance.moves == []
        await motion.close()

    asyncio.run(scenario())


def test_direct_yaw_watchdog_uses_stopmove_and_revokes_lease() -> None:
    async def scenario() -> None:
        sport, avoidance = FakeSport(), FakeAvoidance()
        motion = UnitreeMotionAdapter(
            sport,
            avoidance,
            MotionConfig(command_watchdog_s=0.03),
        )
        await motion.initialize()
        lease = await motion.arm_direct_yaw()
        await motion.send_direct_yaw(lease, 0.2)
        stop_calls_after_arm = sport.stop_calls

        await asyncio.sleep(0.08)

        assert not motion.armed
        assert sport.stop_calls > stop_calls_after_arm
        assert motion.status()["mode"] is None
        await motion.close()

    asyncio.run(scenario())
