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
- Adds a `Follow` button to every live detection. The button sends both the
  model class and bounding-box center, so either of two visible apples can be
  selected independently.
- Uses YOLO to recognize the selected fruit, then follows only that exact box
  with a fast camera-loop MIL tracker so motion never steers from the
  several-seconds-old inference frame.
- Keeps the tracker's latest center as the reacquisition hint when another YOLO
  result arrives.
- Continuously steers from the latest observation of the selected object.
- Disarms whenever the selected object changes, so changing targets cannot
  redirect an active motion burst.
- Runs produce inference in a separate worker so a slow YOLO frame cannot
  block hold pulses, stop commands, or the independent motion watchdog.
- Keeps the exact arm confirmation, press-and-hold control, factory avoidance,
  forward time budget, target-loss stop, and `StopMove` safety boundary.

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
its current tracker observation, motion state, and `ok: true`.
