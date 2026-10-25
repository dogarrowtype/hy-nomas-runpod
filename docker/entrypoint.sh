#!/usr/bin/env bash
# Hy-Nomas RunPod startup orchestration.
#
# Env contract (set on the RunPod template/pod; build-time changes never needed):
#   PE_API_KEY         (REQUIRED) prompt-enhancer API key — no default
#   PE_API_URL         OpenAI-compatible chat-completions URL (default in app.py)
#   PE_OPENING_MODELS  comma-separated provider model IDs (default in app.py)
#   PE_TURN_MODELS     comma-separated provider model IDs (default in app.py)
#   NOMAS_RESOLUTION   WIDTHxHEIGHT, default 1280x720 (app reads it directly;
#                      use 832x480 to fall back to the Space's 480p shapes)
#   NOMAS_MAX_SESSIONS default 4 (set 1-2 on <64 GB GPUs)
#   NOMAS_GPU_SIZE     default xlarge (compile-cache selection)
#   NOMAS_PARK_DIR     world parking dir (default $VOLUME_DIR/park)
#   NOMAS_DATA_DIR     override the data dir (default: /workspace if mounted)
#   HF_TOKEN           optional; only needed if a weight repo is gated
set -euo pipefail

log() { echo "[nomas-entrypoint] $*"; }

# --- 1. Data dir: prefer the mounted network volume, else ephemeral ----------
VOLUME_DIR="${NOMAS_DATA_DIR:-/workspace}"
if [ -z "${NOMAS_DATA_DIR:-}" ] && ! mountpoint -q /workspace; then
  VOLUME_DIR="/root/.nomas-data"
  log "no volume mounted at /workspace; using ephemeral ${VOLUME_DIR}"
  log "(weights will re-download on every pod recreate without a volume)"
fi
mkdir -p "${VOLUME_DIR}"
log "data dir: ${VOLUME_DIR}"

# --- 2. Defaults (all overridable via pod env) -------------------------------
export HF_HOME="${HF_HOME:-${VOLUME_DIR}/hf-cache}"
export NOMAS_PARK_DIR="${NOMAS_PARK_DIR:-${VOLUME_DIR}/park}"
export NOMAS_GPU_SIZE="${NOMAS_GPU_SIZE:-xlarge}"
export NOMAS_MAX_SESSIONS="${NOMAS_MAX_SESSIONS:-4}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${HF_HOME}" "${NOMAS_PARK_DIR}"

# --- 3. Required env ----------------------------------------------------------
if [ -z "${PE_API_KEY:-}" ]; then
  echo "ERROR: PE_API_KEY is not set. The prompt enhancer is served by an" >&2
  echo "       external OpenAI-compatible API and its key is required." >&2
  echo "       Set PE_API_KEY (and optionally PE_API_URL," >&2
  echo "       PE_OPENING_MODELS, PE_TURN_MODELS) on the pod/template." >&2
  exit 1
fi

# --- 3b. GPU gate -------------------------------------------------------------
python - <<'PY'
import sys

import torch

if not torch.cuda.is_available():
    sys.exit("ERROR: no CUDA GPU is visible to torch. Deploy on an NVIDIA GPU pod.")

major, minor = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
vram_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"[nomas-entrypoint] GPU: {name} (sm_{major}{minor}), {vram_gib:.0f} GiB VRAM", flush=True)

if major < 8 or (major == 8 and minor < 9):
    sys.exit(
        f"ERROR: {name} is sm_{major}{minor}, but the PixelUMM FP8 decoder path "
        "(torch._scaled_mm e4m3) requires sm_89+ (Ada/Hopper/Blackwell: "
        "RTX 4090/5090, L40S, H100/H200, RTX 6000 Pro). A100/A40 cannot run this image."
    )

if vram_gib < 64:
    print(
        f"[nomas-entrypoint] WARNING: only {vram_gib:.0f} GiB VRAM — "
        "set NOMAS_MAX_SESSIONS=1-2 to avoid OOM.",
        file=sys.stderr,
        flush=True,
    )
PY

# --- 4. Weight staging (idempotent; optional HF_TOKEN honored by huggingface_hub)
# Pods can boot before DNS/network is fully up — wait for name resolution first,
# then retry downloads with backoff instead of dying on the first hiccup.
log "waiting for outbound network (huggingface.co)…"
for i in $(seq 1 24); do
  if python -c "import socket; socket.getaddrinfo('huggingface.co', 443)" 2>/dev/null; then
    log "network is up"
    break
  fi
  if [ "${i}" -eq 24 ]; then
    echo "ERROR: huggingface.co still unresolvable after ~2 minutes — pod has no" >&2
    echo "       outbound network. Restart the pod or check RunPod status." >&2
    exit 1
  fi
  sleep 5
done

hf_retry() {
  # $1 = repo id, remaining args = allow_patterns (empty = whole repo)
  local repo="$1"; shift
  python - "${repo}" "$@" <<'PY'
import sys
import time

from huggingface_hub import snapshot_download

repo = sys.argv[1]
patterns = sys.argv[2:] or None  # NOTE: passing each pattern as its own argv keeps
                                 # this immune to CLI-style --include parsing bugs;
                                 # a bare "*" would never reach a URL as %2A.

for attempt in range(1, 6):
    try:
        path = snapshot_download(repo, allow_patterns=patterns)
        print(f"[nomas-entrypoint] cached {repo} at {path}", flush=True)
        sys.exit(0)
    except Exception as exc:
        print(f"[nomas-entrypoint] download of {repo} failed (attempt {attempt}/5): {exc}", flush=True)
        time.sleep(10 * attempt)

sys.exit(f"ERROR: downloading {repo} failed after 5 attempts.")
PY
}

# Presence is not enough: network volumes can leave a blob truncated after an
# unclean pod stop ("existing" file, unreadable header → app crash-loop).
# Validate that every cached .safetensors header actually parses; on failure,
# purge the repo cache and re-download.
validate_hub_repo() {
  # $1 = repo id; exit 0 only if fully cached AND all safetensors headers read
  python - "$1" <<'PY'
import pathlib
import sys

from huggingface_hub.constants import HF_HUB_CACHE
from safetensors import safe_open

org, name = sys.argv[1].split("/", 1)
repo_dir = pathlib.Path(HF_HUB_CACHE) / f"models--{org}--{name}"
snapshots = list((repo_dir / "snapshots").glob("*")) if repo_dir.is_dir() else []
snap_files = [f for snap in snapshots for f in snap.rglob("*") if f.is_file() or f.is_symlink()]
safetensors_files = [f for f in snap_files if f.name.endswith(".safetensors")]
if not snapshots or not safetensors_files:
    sys.exit(1)
try:
    for f in safetensors_files:
        with safe_open(str(f), framework="pt", device="cpu") as reader:
            reader.keys()  # forces header parse — catches truncated/corrupt blobs
except Exception as exc:
    print(f"[nomas-entrypoint] cached file failed validation: {f.name}: {exc}", flush=True)
    sys.exit(1)
sys.exit(0)
PY
}

purge_hub_repo() {
  python - "$1" <<'PY'
import pathlib
import shutil
import sys

from huggingface_hub.constants import HF_HUB_CACHE

org, name = sys.argv[1].split("/", 1)
shutil.rmtree(pathlib.Path(HF_HUB_CACHE) / f"models--{org}--{name}", ignore_errors=True)
PY
}

if validate_hub_repo tencent/Hy-Nomas; then
  log "Hy-Nomas DiT/VAE weights already cached — skipping download"
else
  purge_hub_repo tencent/Hy-Nomas
  log "downloading Hy-Nomas DiT/VAE weights (fresh volume, or cache repair after failed validation)"
  hf_retry tencent/Hy-Nomas "DiT/distilled_int8/*" "VAE/PixelUMM_FT/*"
fi

if validate_hub_repo Qwen/Qwen2.5-VL-7B-Instruct; then
  log "Qwen2.5-VL-7B-Instruct already cached — skipping download"
else
  purge_hub_repo Qwen/Qwen2.5-VL-7B-Instruct
  log "downloading Qwen/Qwen2.5-VL-7B-Instruct (fresh volume, or cache repair after failed validation)"
  hf_retry Qwen/Qwen2.5-VL-7B-Instruct
fi

# --- 5. Launch ----------------------------------------------------------------
log "resolution: ${NOMAS_RESOLUTION:-1280x720} | max sessions: ${NOMAS_MAX_SESSIONS} | gpu size: ${NOMAS_GPU_SIZE} | park dir: ${NOMAS_PARK_DIR}"
exec python app.py
