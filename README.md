# Collie Demo

A self-contained Wendy app for Woof that lets an operator select a locally
detected apple, banana, or pear and safely follow that exact object. It uses a
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
- Uses tested per-class thresholds: apple 80%, banana 35%, and pear 80%.
  The lower banana threshold preserves detection as the robot closes in; the
  close-view live set scored 39.8-52.8%, while the benchmark had no banana
  false positives at 30% and three historical non-target frames stayed below
  6.2%.
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
`banana: 0.35`, and `pear: 0.8`, the selected fruit and its current observation,
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
