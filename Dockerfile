# syntax=docker/dockerfile:1.6

# Wendy injects WENDY_PLATFORM after inspecting the target device. Woof resolves
# to nvidia-jetson; ordinary local builds retain the generic CPU fallback.
ARG WENDY_PLATFORM=generic
ARG CYCLONEDDS_VERSION=0.10.5
ARG UNITREE_SDK2_PYTHON_REF=e4cd91f051aaa77a70600e3d2bf7f50889db1980

# ── Generic CPU fallback ────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS builder-generic

ARG DEBIAN_FRONTEND=noninteractive
ARG CYCLONEDDS_VERSION
ARG UNITREE_SDK2_PYTHON_REF

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates \
    && git clone --depth 1 --branch ${CYCLONEDDS_VERSION} \
      https://github.com/eclipse-cyclonedds/cyclonedds.git /tmp/cyclonedds \
    && cmake -S /tmp/cyclonedds -B /tmp/cyclonedds/build \
      -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF \
    && cmake --build /tmp/cyclonedds/build -j2 \
    && cmake --install /tmp/cyclonedds/build \
    && ldconfig \
    && rm -rf /tmp/cyclonedds /var/lib/apt/lists/*

ENV CYCLONEDDS_HOME=/usr/local

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
RUN python -m venv /opt/collie-venv \
    && /opt/collie-venv/bin/python -m pip install --no-cache-dir --upgrade pip \
    && /opt/collie-venv/bin/python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      'torch==2.13.0+cpu' 'torchvision==0.28.0+cpu' \
    && /opt/collie-venv/bin/python -m pip install --no-cache-dir --no-deps \
      "git+https://github.com/unitreerobotics/unitree_sdk2_python.git@${UNITREE_SDK2_PYTHON_REF}" \
    && /opt/collie-venv/bin/python -m pip install --no-cache-dir '.[robot,fruit]'

FROM python:3.11-slim-bookworm AS runtime-generic
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder-generic /usr/local /usr/local
COPY --from=builder-generic /opt/collie-venv /opt/collie-venv
ENV PATH=/opt/collie-venv/bin:$PATH \
    CYCLONEDDS_HOME=/usr/local \
    COLLIE_INFERENCE_DEVICE=cpu

# ── NVIDIA Jetson / JetPack 6.1 path ───────────────────────────────────────
# This is the CUDA/L4T base used by Wendy's bundled Python YOLO template.
FROM dustynv/pytorch:2.7-r36.4.0-cu128-24.04 AS runtime-nvidia-jetson

ARG DEBIAN_FRONTEND=noninteractive
ARG CYCLONEDDS_VERSION
ARG UNITREE_SDK2_PYTHON_REF

ENV PATH=/opt/venv/bin:$PATH \
    CYCLONEDDS_HOME=/usr/local \
    COLLIE_INFERENCE_DEVICE=0 \
    PIP_INDEX_URL=https://pypi.org/simple \
    PIP_EXTRA_INDEX_URL=""

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake git ca-certificates \
      libgl1 libglib2.0-0 libgomp1 \
    && git clone --depth 1 --branch ${CYCLONEDDS_VERSION} \
      https://github.com/eclipse-cyclonedds/cyclonedds.git /tmp/cyclonedds \
    && cmake -S /tmp/cyclonedds -B /tmp/cyclonedds/build \
      -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF \
    && cmake --build /tmp/cyclonedds/build -j2 \
    && cmake --install /tmp/cyclonedds/build \
    && ldconfig \
    && rm -rf /tmp/cyclonedds /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/
RUN python3 -m pip install --no-cache-dir --upgrade pip \
    && python3 -m pip install --no-cache-dir --no-deps \
      "git+https://github.com/unitreerobotics/unitree_sdk2_python.git@${UNITREE_SDK2_PYTHON_REF}" \
    && python3 -m pip install --no-cache-dir '.[robot,fruit]' \
    && python3 -c "import torch; assert torch.version.cuda; print('CUDA torch', torch.__version__, torch.version.cuda)"

# ── Woof final image ────────────────────────────────────────────────────────
# Woof's current Wendy agent reports an empty deviceType even though it exposes
# JetPack 6.1 and an NVIDIA sm_87 GPU. Pin this robot app to the Jetson stage so
# that the CLI cannot silently choose the generic CPU fallback.
FROM runtime-nvidia-jetson AS final

ARG WENDY_PLATFORM=generic
ARG WENDY_DEVICE_TYPE=""
ARG WENDY_HAS_GPU=""
ARG WENDY_GPU_VENDOR=""
ARG WENDY_JETPACK_VERSION=""
ARG WENDY_CUDA_VERSION=""

ENV WENDY_PLATFORM=nvidia-jetson \
    WENDY_DEVICE_TYPE=${WENDY_DEVICE_TYPE} \
    WENDY_HAS_GPU=${WENDY_HAS_GPU} \
    WENDY_GPU_VENDOR=${WENDY_GPU_VENDOR} \
    WENDY_JETPACK_VERSION=${WENDY_JETPACK_VERSION} \
    WENDY_CUDA_VERSION=${WENDY_CUDA_VERSION} \
    PYTHONUNBUFFERED=1 \
    GO2_NETWORK_INTERFACE=enP8p1s0 \
    COLLIE_PORT=8096 \
    COLLIE_MOTION_ENABLED=1 \
    COLLIE_ALLOW_UNRANGED_DEMO=1 \
    COLLIE_PRODUCE_MODEL=/app/models/snapstock/fruit_vegetable_yolov8m.pt \
    COLLIE_PRODUCE_CONFIDENCE=0.5 \
    COLLIE_WEB_DIRECTORY=/app/web \
    COLLIE_FORWARD_MPS=0.60 \
    COLLIE_FORWARD_BUDGET_S=4.0

WORKDIR /app
COPY web/ web/
COPY models/snapstock/fruit_vegetable_yolov8m.pt models/snapstock/fruit_vegetable_yolov8m.pt

EXPOSE 8096
CMD ["python3", "-m", "collie_demo.supervisor"]
