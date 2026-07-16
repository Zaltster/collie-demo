# Collie Demo

A self-contained Wendy app for Woof that uses the official Unitree camera,
detects the blue whale on the floor, curves toward it, and stops when the
target leaves the camera view.

## Behavior

- The blue whale must be detected for at least 8 consecutive frames.
- Off-center targets produce forward and yaw commands together so Woof curves
  toward the whale instead of turning in place first.
- The operator must type `WHALE AND PATH CLEAR` and arm one burst.
- Motion continues only while the browser sends hold pulses every 120 ms.
- A 350 ms command watchdog brakes and revokes the motion lease.
- One arm permits at most 4 seconds of continuous movement at 0.60 m/s.
- Losing the whale, losing the camera, releasing the button, closing the page,
  exhausting the time budget, disabling factory avoidance, or crashing the web
  process invokes zero motion and `SportClient.StopMove`.
- Non-zero movement goes through Unitree's `ObstaclesAvoidClient`; there is no
  direct `SportClient.Move` fallback.

The motion boundary and independent process brake are adapted from the
`go2-follow-clean` controller.

## Important scope

This build proves **detect blue whale -> curve toward it -> stop on target
loss**. It has no calibrated target distance or contact sensor. The whale
leaving the camera view is only a visual stopping heuristic; it does not prove
physical contact. Add calibrated range/contact sensing before relying on touch
behavior around people or on a stage.

## Operate the demo

1. Confirm Woof, the target, and the entire path are clear.
2. Wait until the page reports a stable blue-whale detection.
3. Click **Arm one burst**.
4. Press and hold **PRESS AND HOLD** to approach; release to stop immediately.
5. Use **STOP NOW** at any time to disarm and brake.

If the page reports `motion is not armed`, the lease is already disarmed or
expired. Re-check the path and click **Arm one burst** again before holding the
walk button.

## Test

```sh
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
pytest
```

## Deploy to Woof

Deployment is intentionally separate from construction because this app has
real motion authority when armed:

```sh
wendy --device woof.local run --yes --detach --restart-on-failure
```

Then open `http://woof.local:8096/`. Keep the physical remote in hand and test
first with Woof on a clear floor at the lowest configured speed.
