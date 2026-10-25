# Hy-Nomas Live — RunPod Docker Image

A single, self-contained Docker image for running **Hy-Nomas Live** — the interactive
world-model demo for [tencent/Hy-Nomas](https://huggingface.co/tencent/Hy-Nomas) — on a
RunPod GPU pod, decoupled from HF Spaces / ZeroGPU.

**Zero manual steps after pod creation:** pick the image + network volume + env vars,
click deploy, open the proxy URL.

- **Prompt enhancer** is served by an external hosted OpenAI-compatible API (env-configured)
- **Default output is 1280×720** (the Space ran 480×832; fall back via one env var)
- **Model weights (~40 GB) are NOT in the image** — the entrypoint downloads them into the
  attached network volume on first boot, idempotently

> **Attribution.** This repo is a patched clone of the *Hy-Nomas Live* HF Space.
> `hymm/`, `configs/` and `apps/` are vendored from the official repository
> [Tencent-Hunyuan/Hy-Nomas](https://github.com/Tencent-Hunyuan/Hy-Nomas) (Apache-2.0);
> see [LICENSE](LICENSE). The prompt enhancer runs the authors' Hy-Nomas prompts
> unchanged, only the transport is configurable.

---

## How it fits together

```text
GitHub repo (this)                RunPod
┌─────────────────────┐          ┌────────────────────────────────┐
│ push to main        │   GHCR   │  Pod (RTX 6000 Pro, sm_120)     │
│   → GitHub Actions ─┼─────────►│  ghcr.io/<user>/hy-nomas-runpod │
│     docker build    │  image   │    │ entrypoint.sh               │
│     + push          │          │    ├─ download weights (once) ──┼──► Network volume (/workspace)
└─────────────────────┘          │    ├─ GPU gate (sm_89+ only)     │     hf-cache/  inductor-cache/
                                 │    └─ python app.py :7860       │     park/
                                 └──────┼─────────────────────────┘
                                        ▼
                          https://<pod-id>-7860.proxy.runpod.net
```

- The **image** contains: CUDA 13 runtime, Python 3.12, pinned deps (torch 2.11.0 cu130,
  flash-attn cp312 wheel, gradio 6.30.0), and the patched app. No weights, no secrets.
- The **volume** contains: HF model cache, Inductor compile cache, parked worlds.
- The **GPU gate** hard-fails on anything below sm_89 (the PixelUMM FP8 decoder needs
  `torch._scaled_mm` e4m3). A100/A40 **cannot** run this image.
- There is **no auth** — the proxy URL is the only gate. Keep it private.

## One-time GitHub setup

### 1. Create the repo

1. Go to <https://github.com/new>
2. Name: `hy-nomas-runpod`, visibility **Public**
3. **Do not** initialize with README/.gitignore/license — keep it empty

> Public repo + public GHCR package keeps GitHub-side billing at **$0** (unlimited public
> package storage/transfer + free Actions minutes). No weights or secrets ever enter the
> repo — the PE key lives only in RunPod env vars.

### 2. Push

```bash
cd hy-nomas-runpod
git init -b main
git add -A
git commit -m "Hy-Nomas RunPod image: patched app + Dockerfile + CI"
git remote add origin git@github.com:<your-username>/hy-nomas-runpod.git
git push -u origin main
```

(Or with the `gh` CLI, one shot: `gh repo create hy-nomas-runpod --public --source . --push`)

The push immediately triggers the `build-image` workflow.

### 3. Watch the first CI build (verification gate)

1. Repo → **Actions** tab → watch `build-image` (~10–25 min: CUDA base + cu130 torch +
   flash-attn wheel)
2. **If it fails** at the `pip install torch==2.11.0` layer with *"no matching
   distribution"*: the pin doesn't exist on the cu130 index. Fall back to the newest
   torch/torchvision pair at <https://download.pytorch.org/whl/cu130/torch/>, edit the two
   pinned versions in the `Dockerfile`, and push again. No other impact.

### 4. Make the GHCR package public

After the run is green:

1. GitHub profile → **Packages** → `hy-nomas-runpod`
2. Package settings → **Change visibility → Public**, and **link it to the repo**
3. Copy the digest for `:latest` (or the `:<git-sha>` tag). Your pinned image ref:

   ```text
   ghcr.io/<your-username>/hy-nomas-runpod@sha256:…
   ```

Public package = **no registry credentials** needed in the RunPod template.

## One-time RunPod setup

### 5. Network volume + template

1. **Storage → Network Volumes**: create **~60 GB** in a datacenter that has
   RTX 6000 Pro Blackwell (volume region must match pod region)
2. **Templates → New Template**:
   - Container image: the pinned digest from step 4
   - Container disk: **≥ 30 GB** (the image itself is ~15–20 GB)
   - Volume: attach yours, mount path **`/workspace`**
   - Expose HTTP port: **`7860`**
   - Env vars: see the table below

### Environment variables

| Variable | Required | Default | Notes |
|---|---|---|---|
| `PE_API_KEY` | **yes** | — | Prompt-enhancer API key. No default; entrypoint fails fast without it. Never put it in the repo. |
| `PE_API_URL` | yes* | HF router URL | Provider chat-completions URL, e.g. `https://openrouter.ai/api/v1/chat/completions`. *The default only works with an HF key. |
| `PE_OPENING_MODELS` | yes* | Space defaults | Plain provider model IDs for the opening prompt, comma-separated. *The built-in defaults carry HF `:provider` suffixes that break on other providers. |
| `PE_TURN_MODELS` | yes* | Space defaults | Same, for per-turn enhancement. |
| `NOMAS_RESOLUTION` | no | `1280x720` | `WIDTHxHEIGHT`. Use `832x480` to restore the Space's shapes (env-only, no rebuild). |
| `NOMAS_MAX_SESSIONS` | no | `4` | Concurrent world sessions — a VRAM safety cap, not a sharing limit (each live world holds KV cache). Set `1`–`2` on <64 GB GPUs. |
| `NOMAS_SESSION_SECONDS` | no | `0` (no limit) | Per-world session duration. `0` = a world stays live until you leave/stop it. Set >0 to re-enable time-limited sessions. |
| `TORCHINDUCTOR_CACHE_DIR` | recommended | — | **Set to `/workspace/inductor-cache`.** There is no bundled compile cache (the Space's 480p/H200 megacache can't hit on sm_120/720p, so it was dropped); persisting Inductor's cache on the volume means only the *very first* turn on a *fresh* volume pays the compile time — pod recreates stay warm. |
| `HF_TOKEN` | no | — | Only if a weight repo turns out to be gated. |
| `NOMAS_GPU_SIZE` | no | `xlarge` | Selects the compile-cache filename (currently always "missing" — see above). |
| `NOMAS_PARK_DIR` | no | `$VOLUME_DIR/park` | Where suspended worlds are parked. |
| `NOMAS_DATA_DIR` | no | `/workspace` | Override the data dir; entrypoint falls back to ephemeral `/root/.nomas-data` if no volume is mounted. |

### 6. Deploy

Pods → Deploy → **RTX 6000 Pro** (or any sm_89+: 4090 / 5090 / L40S / H100) → Secure
Cloud, on-demand → select template + volume.

Boot sequence:

- **First boot on a fresh volume:** image pull (minutes) → ~40 GB weight download
  (once per volume) → model load → first turn compiles at 720p (minutes, once).
- **Recreates on the same volume:** weights skip, Inductor cache warm (if the env var
  above is set), no manual steps.

## Validation

1. **Pod logs** should show:
   - `[nomas-entrypoint] GPU: … (sm_120), 96 GiB VRAM`
   - weight staging lines (`downloading …` → `already cached — skipping download` on warm volumes)
   - `compile cache megacache_xlarge_…: missing` — **expected**, the cache was dropped
   - `PixelUMM FP8 rowwise linears: <N>`
   - `Hy-Nomas pipeline ready`
2. `curl https://<pod-id>-7860.proxy.runpod.net/api/status` → `{"free":4,…}`
3. **Browser smoke:** open the proxy URL → open a world → Director's notes populate
   (proves the PE API round-trip) → one say turn → one camera-move turn → Undo →
   MP4 download (verify it's 1280×720; the in-browser screen should render 16:9
   with no stretch)
4. Kill and **recreate the pod** from the same template+volume → confirm zero manual steps
5. `nvidia-smi` with 2 concurrent sessions at 720p: if VRAM is tight,
   `NOMAS_MAX_SESSIONS=2` (env, no rebuild); if per-turn latency feels slow,
   `NOMAS_RESOLUTION=832x480` (env, no rebuild)

## Ongoing loop

Edit code → `git push` → CI rebuilds and pushes `:latest` + a new `:<git-sha>` tag →
optionally bump the digest in the RunPod template.

## Troubleshooting

| Symptom | Fix |
|---|---|
| First turn: `Failed to find C compiler` | Triton/Inductor JIT-builds a C extension at runtime; the image needs `build-essential python3-dev` (present in the current Dockerfile). Rebuild + redeploy |
| `Temporary failure in name resolution` during weight staging | Pod booted before its network was ready. The entrypoint now waits for `huggingface.co` DNS (~2 min) and retries each download 5× with backoff. If it *still* fails, the pod has no outbound network — restart or redeploy |
| CI fails on `torch==2.11.0` pin | Use newest cu13x torch/torchvision (see step 3) |
| Entrypoint: `PE_API_KEY is not set` | Add it to the template/pod env — it has no default |
| Entrypoint: `… is sm_80 … requires sm_89+` | A100/A40 class GPU — redeploy on Ada/Hopper/Blackwell |
| `WARNING: only NN GiB VRAM` | Set `NOMAS_MAX_SESSIONS=1-2` |
| First turn takes minutes | torch.compile at the current resolution/GPU. Expected once per pod boot; with `TORCHINDUCTOR_CACHE_DIR=/workspace/inductor-cache` set it drops to once per fresh volume — set it |
| OOM at 720p / too-slow turns | `NOMAS_MAX_SESSIONS=2` or `NOMAS_RESOLUTION=832x480` (both env-only) |
| PE failures in logs | Check `PE_API_URL` / model IDs — plain IDs, no `:provider` suffix, for non-HF providers |
| Weights download fails with 401/403 | Repo gated → set `HF_TOKEN` |

## What changed vs. the original Space

- `app.py`: `import spaces` / `@spaces.GPU` removed; HF `get_token()` fallback removed
  (`PE_API_KEY` only); resolution read from `NOMAS_RESOLUTION` (default `1280x720`);
  **public session limits removed** — the quick/standard/long time leases are gone,
  sessions run until you leave them (`NOMAS_SESSION_SECONDS` can restore a time limit)
- `index.html`: six render constants 832/480 → 1280/720 (16:9); ZeroGPU quota copy,
  handshake, the GPU-time countdown ticker, and the session-length picker removed;
  remaining copy talks about "session slots", not GPUs/quota
- Added: `Dockerfile`, `docker/entrypoint.sh`, `.github/workflows/build-image.yml`,
  `.dockerignore`, `.gitignore`
- Removed: the Space's `compile_cache/` megacache (H200/480p artifacts double-miss on
  sm_120/720p; Inductor's own cache on the volume replaces it)
