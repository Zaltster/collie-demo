"""Out-of-process health watchdog with an independent Unitree StopMove path."""

from __future__ import annotations

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
    healthy_once = False
    failures = 0
    while child.poll() is None and not terminating:
        try:
            with urllib.request.urlopen(url, timeout=0.5) as response:
                probe_ok = response.status == 200
        except (urllib.error.URLError, OSError, ValueError):
            probe_ok = False
        if probe_ok:
            healthy_once = True
            failures = 0
        elif healthy_once:
            failures += 1
        elif time.monotonic() >= deadline:
            failures = 3
        if failures >= 3:
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
