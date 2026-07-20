# Collie Demo

A self-contained Wendy app for Woof that can remember a locally detected fruit,
turn around, reject that physical prop, and safely approach a different instance
of the same fruit class. The
validated manual apple/banana/pear follower remains available as a fallback. It uses a
YOLOE-11m checkpoint whose visual prompt embeddings were baked from the three
physical stage props.
Frames come directly from Unitree's public `VideoClient`; inference,
annotation, motion supervision, and the browser UI all run locally in the
robot container. No Roboflow service, hosted inference API, Hugging Face key,
or internet connection is used at runtime.

## Current behavior

- Loads `models/collie/collie-fruit-yoloe11m.pt` locally.
- Detects only the three visual-prompt classes: apple, banana, and pear.
- Draws labels, confidence scores, bounding boxes, and box centers.
- Prints every detection to the container log.
- Serves the annotated Go2 camera and structured detections on port 8096.
- Uses tested per-class thresholds: apple 80%, banana 20%, and pear 80%.
  The lower banana threshold preserves detection as the robot closes in; the
  60-frame close-view live set scored 28.1-45.5%, the benchmark had no banana
  false positives even at 5%, and three historical non-target frames stayed
  below 6.2%.
- Makes every live detection selectable. Selection sends both the model class
  and bounding-box center, so either of two visible apples can be selected
  independently; one click on `Follow Selected Fruit` then starts the guarded
  approach loop.
- Uses each fresh GPU YOLO result as the authoritative target observation and
  collapses overlapping same-class boxes before they reach the UI or control
  loop.
- Revalidates the selected track against every new YOLO result. If the chosen
  fruit disappears or the tracker drifts to a different object for three
  consecutive results, selection is cleared and motion is stopped. One or two
  transient misses do not interrupt the approach.
- Marks tracker-only frames honestly in the UI instead of repeating an old
  YOLO confidence score as though it were current.
- Continuously steers from the latest observation of the selected object.
- Runs the follow pulse loop inside the robot service after one Follow click;
  browser timer throttling or a slow status refresh cannot interrupt command
  renewal. The independent 350 ms motion watchdog remains the final brake.
- Waits briefly for a freshly selected target to become stable and freshly
  YOLO-verified, so the operator does not need to click Follow twice.
- Rejects detections older than 750 ms and clears stale detector output on
  inference errors.
- Disarms whenever the selected object changes, so changing targets cannot
  redirect an active motion burst.
- Runs produce inference in a separate worker so a slow YOLO frame cannot
  block hold pulses, stop commands, or the independent motion watchdog.
- Keeps the exact arm confirmation, dedicated stop control, factory avoidance,
  forward time budget, target-loss stop, and `StopMove` safety boundary.
- Uses Woof's Jetson GPU for YOLO inference and reports the requested/resolved
  device, CUDA version, Torch version, and current inference latency in
  `/api/status`.
- Reports aggregate `stage_ready` health for the camera, YOLO worker, CUDA GPU,
  and motion adapter. The UI displays verification age and the current miss
  count, and the out-of-process supervisor brakes/restarts if readiness remains
  unhealthy.
- `Save Fruit` captures five detector-aligned appearance embeddings while motion
  remains disarmed. The per-round memory is local to the robot process and is
  cleared only by `Reset Round`, a replacement capture, or service restart.
- Keeps persistent fruit memory separate from the ephemeral visual track. A
  normal target-loss stop therefore cannot erase what Woof was shown before it
  turned around.
- Uses fresh `rt/sportmodestate` IMU yaw to measure the turn. The mission refuses
  to start or aborts if heading becomes stale; it never estimates a 180-degree
  turn from elapsed time alone.
- Runs the initial measured turn through an exclusive yaw-only `SportClient`
  lease, with forward and lateral motion structurally fixed at zero. This turn
  explicitly enters `BalanceStand`, commands a 0.80 rad/s yaw, and aborts if it
  has not rotated at least 10 degrees within two seconds. It does not use
  `ObstaclesAvoidClient`, but it retains the 350 ms heartbeat
  watchdog and independent `StopMove` brake. After the turn, the direct lease is
  released and search/approach reacquire the factory-avoidance motion path.
- Ranks every same-class YOLO candidate against the saved appearance using local
  color, texture, and shape embeddings. Candidates at or above 94% saved-instance
  similarity are rejected. The stage build requires two fresh confirmations of
  a candidate below that threshold before moving forward. Search rotation brakes
  on the first accepted candidate so the next inference confirms a stationary
  view instead of rotating the fruit out of frame.
- Runs `TURNING -> SEARCHING -> CONFIRMING -> APPROACHING` inside the robot
  service. UI refresh timing cannot interrupt command renewal, and leaving the
  page sends the same emergency stop used by the manual follower.
- Treats disappearance as success only after the matched fruit reached the
  lower camera region. Earlier loss or an ambiguous appearance match is an
  abort with zero commanded velocity.

Every class emitted by the local model is selectable from the detection list.
Whale color detection and whale motion targets have been removed.

## Model

Download the official YOLOE base checkpoint, capture a clean reference frame,
and bake the three exact props into a self-contained local checkpoint:

```sh
mkdir -p models/candidates models/collie
curl -L --fail \
  "https://github.com/ultralytics/assets/releases/download/v8.4.0/yoloe-11m-seg.pt" \
  -o models/candidates/yoloe-11m-seg.pt

python tools/build_visual_prompt_weights.py \
  --weights models/candidates/yoloe-11m-seg.pt \
  --reference captures/three-fruit-benchmark/raw/frame_000.jpg \
  --output models/collie/collie-fruit-yoloe11m.pt \
  --device mps \
  --prompt apple=1542,758,1596,817 \
  --prompt banana=1186,779,1262,844 \
  --prompt pear=812,775,884,855
```

The current baked checkpoint is 59,997,395 bytes with SHA-256
`7c75fcc5d449a8b00785dfd0c955cbf11bd6bde6a5ede1ea8d34c097413bc53e`.
Model files and camera captures are excluded from Git, but the Docker build
context includes the baked checkpoint.

## Local test

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[fruit,test]'
pytest
```

Run on a Mac camera after granting camera permission:

```sh
collie-fruit-webcam \
  --model models/collie/collie-fruit-yoloe11m.pt \
  --camera 0 \
  --confidence 0.7
```

Run the browser UI against a JPEG source:

```sh
collie-fruit-ui \
  --model models/collie/collie-fruit-yoloe11m.pt \
  --source http://woof.local:8096/camera.jpg \
  --port 8097
```

## Deploy to Woof

The on-robot image reads the Unitree camera and motion clients over the network
interface selected by `GO2_NETWORK_INTERFACE`, and binds the UI to port 8096:

```sh
wendy --device woof.local run --yes --detach --restart-on-failure
```

Verify the actual deployed runtime before considering it ready:

```sh
wendy --device woof.local device ps --json
curl http://woof.local:8096/api/status
```

Then open `http://woof.local:8096/`. A healthy status response must report the
Collie YOLOE model path, `produce.class_thresholds` of `apple: 0.8`,
`banana: 0.2`, and `pear: 0.8`, the selected fruit and its current observation,
`produce.device.resolved: "cuda:0"`, motion state, and `ok: true`. In this
stage configuration the model only returns apple, banana, and pear detections.

The raw, unannotated camera frame is available at `/camera-raw.jpg` for
repeatable detector evaluation. The fixed-scene benchmark used to tune the
three thresholds can be rerun with:

```sh
python tools/benchmark_fruit_models.py \
  models/collie/collie-fruit-yoloe11m.pt \
  --source captures/three-fruit-benchmark/raw \
  --device mps \
  --output captures/three-fruit-benchmark/results/collie-yoloe.json
```

The stage control sequence is: click `Select` beside the exact fruit, wait for
the Follow button to enable, then click `Follow Selected Fruit` once. `STOP
NOW` remains available throughout motion. Do not run the demo unless the header
shows `STAGE READY`.

## Remember-and-find stage sequence

1. Hold one detected fruit steady and press `Save Fruit` on that detection.
2. Confirm the saved reference image and the `MEMORIZED` mission state.
3. Put a different instance of the same fruit class in the search area behind
   Woof. The shown fruit is a negative reference and will be rejected if seen.
   Clear the full turn and approach path.
4. Press `Run Remember & Find` once. Woof performs a heading-measured turn,
   requires a multi-frame different-instance lock, and then hands the target to the
   existing guarded follower.
5. Use `STOP NOW` at any time. `Reset Round` erases the saved instance.

The mission endpoints are `POST /api/memory/capture`, `DELETE /api/memory`,
`GET /api/memory/reference.jpg`, `POST /api/demo/start`, and
`POST /api/demo/stop`. `/api/status` reports `memory`, `mission`, heading age,
match score/margin, turn progress, and the terminal reason.

The feature is controlled by `COLLIE_MEMORY_DEMO_ENABLED` and
`COLLIE_AUTONOMOUS_TURN_ENABLED`. Matching, turn speed/angle, search limits,
and arrival geometry are environment-configurable in the Dockerfile. Disabling
either feature does not remove or weaken the manual follower and STOP path.
