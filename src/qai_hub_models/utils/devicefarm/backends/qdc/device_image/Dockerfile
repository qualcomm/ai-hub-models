# Runtime image for run_linux_container.sh: Python + onnxruntime-qnn + CPU torch,
# plus the packages in requirements.txt (build_image.sh can swap in another file).
# Build for the device (linux/arm64) with build_image.sh.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_NO_CACHE=1 \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates python3 libatomic1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.8 /uv /usr/local/bin/uv

# onnxruntime-qnn needs Python >= 3.11 and glibc >= 2.34; Ubuntu 24.04 has 3.12 / 2.39.
RUN uv venv --python /usr/bin/python3 /opt/venv
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH

# torch from the PyTorch CPU index: the default aarch64 PyPI wheels pull in the
# multi-GB CUDA stack.
RUN uv pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision \
    && uv pip install onnxruntime-qnn

COPY requirements.txt /tmp/requirements.txt
RUN uv pip install -r /tmp/requirements.txt

WORKDIR /app
