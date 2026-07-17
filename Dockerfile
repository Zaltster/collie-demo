# syntax=docker/dockerfile:1.6
FROM python:3.11-slim-bookworm AS builder

ARG DEBIAN_FRONTEND=noninteractive
ARG CYCLONEDDS_VERSION=0.10.5
ARG UNITREE_SDK2_PYTHON_REF=e4cd91f051aaa77a70600e3d2bf7f50889db1980

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

FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local /usr/local
COPY --from=builder /opt/collie-venv /opt/collie-venv

ENV CYCLONEDDS_HOME=/usr/local \
    PATH=/opt/collie-venv/bin:$PATH \
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
CMD ["python", "-m", "collie_demo.supervisor"]
