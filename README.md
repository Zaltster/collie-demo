# Collie Demo

A self-contained Wendy app for Woof that lets an operator select any locally
detected fruit and safely follow that exact object. It uses the SnapStock
YOLOv8m fruit and vegetable detector.
Frames come directly from Unitree's public `VideoClient`; inference,
annotation, motion supervision, and the browser UI all run locally in the
robot container. No Roboflow service, hosted inference API, Hugging Face key,
or internet connection is used at runtime.

## Current behavior

- Loads `models/snapstock/fruit_vegetable_yolov8m.pt` locally.
- Detects all 63 classes stored in that checkpoint.
- Draws labels, confidence scores, bounding boxes, and box centers.
- Prints every detection to the container log.
- Serves the annotated Go2 camera and structured detections on port 8096.
- Uses a 50% confidence threshold by default.
- Makes every live detection selectable. Selection sends both the model class
  and bounding-box center, so either of two visible apples can be selected
  independently; one click on `Follow Selected Fruit` then starts the guarded
  approach loop.
- Uses YOLO to recognize the selected fruit, then follows only that exact box
  with a fast camera-loop MIL tracker so motion never steers from the
  several-seconds-old inference frame.
- Keeps the tracker's latest center as the reacquisition hint when another YOLO
  result arrives.
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

Download the weight before building:

```sh
mkdir -p models/snapstock
curl -L --fail \
  "https://huggingface.co/Senu-12/snapstock-fruit-vegetable-detector/resolve/main/yolov8/fruit_vegetable_yolov8m.pt?download=true" \
  -o models/snapstock/fruit_vegetable_yolov8m.pt
```

The expected size is 52,089,985 bytes. Model files are excluded from Git but
the Docker build context includes this exact weight.

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
  --model models/snapstock/fruit_vegetable_yolov8m.pt \
  --camera 0 \
  --confidence 0.5
```

Run the browser UI against a JPEG source:

```sh
collie-fruit-ui \
  --model models/snapstock/fruit_vegetable_yolov8m.pt \
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
SnapStock model path, all 63 classes under `produce`, the selected fruit and
its current tracker observation, `produce.device.resolved: "cuda:0"`, motion
state, and `ok: true`.

The stage control sequence is: click `Select` beside the exact fruit, wait for
the Follow button to enable, then click `Follow Selected Fruit` once. `STOP
NOW` remains available throughout motion. Do not run the demo unless the header
shows `STAGE READY`.
