# Hy-Nomas RunPod image: CUDA 13 runtime + Python 3.12 + pinned deps + patched app.
# Weights are NOT baked in: docker/entrypoint.sh downloads them into the attached
# network volume on first boot (idempotent). Build is GPU-free (CI-safe).

FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Ubuntu 24.04 ships Python 3.12 by default; the flash-attn wheel in
# requirements.txt is cp312-only, so the interpreter must be exactly 3.12.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Pinned cu130 torch first: guarantees the cu130 wheels (a plain PyPI install may
# serve a different CUDA tag) and is present before the flash-attn wheel installs.
# GATE: on the first CI build, verify torch==2.11.0 resolves on this index. If the
# exact pin is unavailable, take the newest cu13x torch and update the pin here
# (the bundled compile cache then misses on torch-version mismatch — first turn
# recompiles, subsequent turns unaffected).
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
RUN pip install --upgrade pip \
 && pip install torch==2.11.0 torchvision==0.26.0 --index-url "${TORCH_INDEX_URL}"

WORKDIR /app
COPY requirements.txt /app/requirements.txt
# torch/torchvision are already satisfied by the step above and are skipped.
# gradio + requests are runtime deps the HF Space host used to provide.
# huggingface_hub[cli] provides the `hf` CLI used by the entrypoint.
RUN pip install -r requirements.txt \
 && pip install "gradio==6.30.0" requests "huggingface_hub[cli]"

COPY . /app
RUN chmod +x /app/docker/entrypoint.sh

EXPOSE 7860
ENTRYPOINT ["/app/docker/entrypoint.sh"]
