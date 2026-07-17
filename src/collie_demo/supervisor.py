"""Out-of-process health watchdog with an independent Unitree StopMove path."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


def emergency_brake() -> None:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py.go2.sport.sport_client import SportClient

    interface = os.environ.get("GO2_NETWORK_INTERFACE", "").strip()
    ChannelFactoryInitialize(0, interface) if interface else ChannelFactoryInitialize(0)
    client = SportClient()
    client.SetTimeout(2.0)
    client.Init()
    result = client.StopMove()
    if result != 0:
        raise RuntimeError(f"StopMove returned {result!r}")


def main() -> None:
    port = int(os.environ.get("COLLIE_PORT", "8096"))
    url = f"http://127.0.0.1:{port}/api/status"
    child = subprocess.Popen([sys.executable, "-m", "collie_demo.main"])
    terminating = False

    def terminate(_signum: int, _frame: object) -> None:
        nonlocal terminating
        terminating = True
        try:
            emergency_brake()
        except Exception as exc:
            print(f"emergency brake during shutdown failed: {exc}", file=sys.stderr)
        if child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    deadline = time.monotonic() + 35.0
    failure_limit = int(os.environ.get("COLLIE_SUPERVISOR_FAILURE_LIMIT", "12"))
    healthy_once = False
    failures = 0
    while child.poll() is None and not terminating:
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:
                payload = json.load(response)
                probe_ok = response.status == 200 and payload.get("stage_ready") is True
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
            probe_ok = False
        if probe_ok:
            healthy_once = True
            failures = 0
        elif healthy_once:
            failures += 1
        elif time.monotonic() >= deadline:
            failures = failure_limit
        if failures >= failure_limit:
            try:
                emergency_brake()
            finally:
                child.terminate()
                try:
                    child.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            raise SystemExit(1)
        time.sleep(0.25)
    if child.poll() is None:
        child.wait()
    raise SystemExit(child.returncode or 0)


if __name__ == "__main__":
    main()
