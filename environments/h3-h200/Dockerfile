# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG SOURCE_COMMIT=unknown
ARG SOURCE_URL=https://github.com/Junchao-cs/SolarWM
ARG FLASH_ATTN_MAX_JOBS=4

LABEL org.opencontainers.image.title="SolarWM MiniMax-H3 H200 Runtime" \
      org.opencontainers.image.description="SolarWM H3 runtime for NVIDIA H100/H200 (SM90)" \
      org.opencontainers.image.source="${SOURCE_URL}" \
      org.opencontainers.image.revision="${SOURCE_COMMIT}"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/cuda/bin:${PATH} \
    HF_HOME=/mnt/solar/cache/hf \
    TORCH_HOME=/mnt/solar/cache/torch \
    TORCH_CUDA_ARCH_LIST=9.0 \
    MAX_JOBS=${FLASH_ATTN_MAX_JOBS}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        git-lfs \
        ibverbs-providers \
        iproute2 \
        jq \
        libglib2.0-0 \
        libgl1 \
        libibverbs1 \
        netcat-openbsd \
        ninja-build \
        numactl \
        openssh-client \
        pciutils \
        procps \
        python3.10 \
        python3.10-dev \
        python3.10-venv \
        rdma-core \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m venv "${VIRTUAL_ENV}" \
    && python -m pip install --upgrade \
        pip \
        setuptools \
        wheel \
        packaging \
        ninja

RUN python -m pip install \
        torch==2.6.0 \
        torchvision==0.21.0 \
        --index-url https://download.pytorch.org/whl/cu124

RUN python -m pip install \
        diffusers==0.40.0 \
        transformers==5.12.1 \
        peft==0.20.0 \
        imageio==2.37.4 \
        imageio-ffmpeg==0.6.0

# Build FlashAttention before copying the source so normal SolarWM source
# changes can reuse this expensive layer.
RUN python -m pip install --no-build-isolation flash-attn==2.8.3

WORKDIR /opt/SolarWM
COPY . /opt/SolarWM

RUN python -m pip install ".[train]" \
    && python - <<'PY'
import decord
import diffusers
import flash_attn
import peft
import torch
import transformers

print(
    {
        "decord": decord.__version__,
        "diffusers": diffusers.__version__,
        "flash_attn": flash_attn.__version__,
        "peft": peft.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
)
PY

RUN python -m solarwm environment probe

ENTRYPOINT []
CMD ["sleep", "infinity"]
