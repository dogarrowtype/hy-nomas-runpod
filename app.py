import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("HY_INFER_PARALLEL_VAE_DECODE", "0")

import torch  # noqa: E402
import torch._dynamo  # noqa: E402,F401

import io
import json
import multiprocessing as mp
import queue
import random
import secrets
import struct
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from gradio import Server
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from loguru import logger  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

MODEL_REPO = "tencent/Hy-Nomas"
FPS = 24
_res = os.environ.get("NOMAS_RESOLUTION", "1280x720").lower().replace("×", "x").split("x")
WIDTH, HEIGHT = int(_res[0]), int(_res[1])
GPU_SIZE = os.environ.get("NOMAS_GPU_SIZE", "xlarge")
MAX_SESSIONS = int(os.environ.get("NOMAS_MAX_SESSIONS", "4"))
SESSION_CHOICES = {"quick": 60, "standard": 90, "long": 150}
LEASE_MARGIN = 30
TURN_SECONDS = (3, 4, 5, 6)
JPEG_QUALITY = 88
RETAIN_SECONDS = 1800
PARK_DIR = os.environ.get("NOMAS_PARK_DIR", "/dev/shm")
PARK_TTL = int(os.environ.get("NOMAS_PARK_TTL", "900"))
PARK_BUDGET_BYTES = int(float(os.environ.get("NOMAS_PARK_BUDGET_GB", "40")) * 2**30)
PE_API_URL = os.environ.get("PE_API_URL", "https://router.huggingface.co/v1/chat/completions")

DIT_PATH = hf_hub_download(
    MODEL_REPO, "DiT/distilled_int8/diffusion_pytorch_model_ema_0_995_int8.safetensors"
)
VAE_PATH = hf_hub_download(MODEL_REPO, "VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors")

import torch.nn.attention.flex_attention as _fa  # noqa: E402
import hymm.models.vae.pixelumm_decoder.autoencoder as _pae  # noqa: E402
import hymm.models.vae.pixelumm_decoder.model as _pmodel  # noqa: E402
from chunked_block_mask import chunked_block_mask  # noqa: E402

_orig_layout = _pmodel.local_attention_layout


def _chunked_layout(*args, **kwargs):
    orig = _fa.create_block_mask

    def _cbm(mask_mod, B, H, q_len, kv_len, device=None, **_):
        return chunked_block_mask(mask_mod, q_len, kv_len, device)

    _fa.create_block_mask = _cbm
    try:
        return _orig_layout(*args, **kwargs)
    finally:
        _fa.create_block_mask = orig


_pmodel.local_attention_layout = _chunked_layout
_pae.local_attention_layout = _chunked_layout

import hymm.models.text_encoder.encoder as _te_mod  # noqa: E402


class _BF16AutoModel:
    @staticmethod
    def from_pretrained(*args, **kwargs):
        from transformers import AutoModel

        kwargs.setdefault("dtype", torch.bfloat16)
        return AutoModel.from_pretrained(*args, **kwargs)


_te_mod.AutoModel = _BF16AutoModel

import hymm.utils.range_attention_config as _rac  # noqa: E402

_rac.validate_range_attention_device = lambda args, device: _rac.validate_range_attention_config(args)

from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402

from configs.config import _flatten_cfg  # noqa: E402

GlobalHydra.instance().clear()
with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
    _cfg = compose(
        config_name="config",
        overrides=[
            "+experiment=single_gpu_low_mem_inference",
            f"dit_weight={DIT_PATH}",
            f"model.vae.checkpoint={VAE_PATH}",
            "model.vae.cpu_offload=false",
            "model.vae.stable_prefix_streaming=true",
            "parallelism.text_encoder_cpu_offload=false",
            "parallelism.dit_cpu_offload=false",
            "inference.torch_compile=true",
            "model.vae.cutlass_fp8=false",
            "inference.save_latents=false",
            "inference.seg_ar_segment_vae_decode=true",
            f"inference.video_size=[{HEIGHT},{WIDTH}]",
            "++inference.interactive_noise_max_seconds=120",
            "++inference.interactive_noise_hard_max_seconds=180",
            "++inference.interactive_noise_growth_seconds=30",
        ],
    )
OmegaConf.resolve(_cfg)
CFG = _flatten_cfg(_cfg)
INF = CFG.inference

from hymm.inference import _build_whole_item_prompts  # noqa: E402
from hymm.models.dit.modules.int8_stored_bf16_linear import resolve_w8_bf16_checkpoint  # noqa: E402
from hymm.pipeline.causal_ar import CPTCausalInferencePipeline  # noqa: E402
from hymm.utils.video_output import video_to_uint8_numpy  # noqa: E402

resolve_w8_bf16_checkpoint(INF, CFG.dit_weight)
torch.set_grad_enabled(False)

import speedups  # noqa: E402
import worldstate  # noqa: E402

COMPILE_CACHE = ROOT / "compile_cache" / f"megacache_{GPU_SIZE}_torch{torch.__version__.split('+')[0]}.bin"
speedups.configure_compile()
logger.info(f"compile cache {COMPILE_CACHE.name}: {'loaded' if speedups.load_compile_cache(COMPILE_CACHE) else 'missing'}")
speedups.fuse_int8_dequant()
speedups.keep_kv_archive_on_gpu()

pipe = CPTCausalInferencePipeline(args=CFG, device=torch.device("cpu"), dit_model_path=CFG.dit_weight)
logger.info(f"PixelUMM FP8 rowwise linears: {speedups.quantize_decoder_fp8(pipe.vae.decoder)}")
speedups.compile_decoder(pipe.vae.decoder)
pipe.transformer.to("cuda")
pipe.text_encoder.to("cuda")
pipe.text_encoder.device = torch.device("cuda")
pipe.vae.to("cuda")
pipe.device = torch.device("cuda")
pipe.enable_parallel_vae_decode = False
logger.info("Hy-Nomas pipeline ready")

from apps.kimi_k3_prompt_enhancer import (  # noqa: E402
    KimiK3Client,
    OnlinePromptEnhancer,
    PromptEnhancementError,
)

_CTX = mp.get_context("fork")
CMD_QS = [_CTX.Queue() for _ in range(MAX_SESSIONS)]


def _reset_pipeline():
    from hymm.models.dit.modules.attention.range_attention import clear_range_cache, reset_range_stats

    if hasattr(pipe, "_finish_interactive_decode_phase"):
        pipe._finish_interactive_decode_phase()
    pipe.kv_cache = None
    pipe.cache_mask_list = None
    pipe.cache_valid_len = 0
    pipe.global_frame_counter = 0
    if hasattr(pipe, "segment_token_lengths"):
        pipe.segment_token_lengths.clear()
    pipe._interactive_cached_action_tails = []
    pipe._interactive_cached_seg_video_ranges = []
    pipe._interactive_segment_count = 0
    pipe._interactive_segment_cache_tokens = []
    pipe._interactive_segment_latent_lengths = []
    pipe._reset_interactive_noise_state()
    pipe._reset_interactive_pixelumm_decode_state()
    clear_range_cache()
    reset_range_stats()


def _encode_jpegs(video: np.ndarray) -> List[bytes]:
    params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    return [
        cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR), params)[1].tobytes()
        for frame in video
    ]


def _lease_duration(slot, lease_id, seconds, *args, **kwargs):
    return int(seconds) + LEASE_MARGIN


WORLD_MODULES = {"vae": pipe.vae, "decoder": pipe.vae.decoder, "transformer": pipe.transformer, "text_encoder": pipe.text_encoder}


def gpu_session(slot: int, lease_id: str, seconds: int, park_path: str = "", restore_path: str = ""):
    import torch._dynamo.utils as _du

    _du.common_constant_types.add(type(torch.empty(0).device))
    torch.set_grad_enabled(False)
    _reset_pipeline()
    commands = CMD_QS[slot]
    if restore_path:
        started = time.monotonic()
        try:
            with torch.inference_mode():
                nbytes = worldstate.restore(pipe, WORLD_MODULES, restore_path)
        except Exception as exc:
            logger.exception("world restore failed")
            _reset_pipeline()
            yield {"type": "restore_failed", "message": str(exc)}
            return
        yield {"type": "restored", "bytes": nbytes, "seconds": time.monotonic() - started}
    deadline = time.monotonic() + seconds
    yield {"type": "ready", "remaining": seconds}
    while True:
        now = time.monotonic()
        if now >= deadline:
            if park_path:
                started = time.monotonic()
                try:
                    with torch.inference_mode():
                        nbytes = worldstate.park(pipe, WORLD_MODULES, park_path)
                    yield {"type": "parked", "bytes": nbytes, "seconds": time.monotonic() - started}
                except Exception:
                    logger.exception("world park failed")
                    worldstate.discard(park_path)
            yield {"type": "expired"}
            break
        try:
            cmd = commands.get(timeout=0.5)
        except queue.Empty:
            yield {"type": "tick", "remaining": deadline - now}
            continue
        if cmd.get("lease") != lease_id:
            continue
        if cmd["op"] == "stop":
            break
        if cmd["op"] == "undo":
            try:
                info = pipe.rollback_last_interactive_segment()
            except Exception as exc:
                logger.exception("rollback failed")
                yield {"type": "undo_failed", "turn": cmd["turn"], "message": str(exc)}
                continue
            meta = dict(pipe._last_pixelumm_streaming_metadata or {})
            yield {
                "type": "undone",
                "turn": cmd["turn"],
                "frames_end": int(meta.get("generated_pixel_frames", 0)),
                "remaining_segments": int(info.get("remaining_segments", 0)),
            }
            continue
        started = time.monotonic()
        yield {"type": "gen_start", "turn": cmd["turn"], "remaining": deadline - started}
        try:
            with torch.inference_mode():
                output = pipe.inference(
                    prompts=[cmd["prompt"]],
                    num_frames=cmd["num_frames"],
                    height=HEIGHT,
                    width=WIDTH,
                    guidance_scale=INF.cfg_scale_video,
                    camera_guidance_scale=INF.camera_cfg_scale,
                    num_inference_steps=cmd["steps"],
                    flow_shift=INF.infer_flow_shift_video,
                    seed=cmd["seed"],
                    negative_prompt=INF.neg_prompt or "",
                    segment_frame_boundaries=[0, cmd["num_frames"]],
                    camera_trajectory=cmd.get("camera"),
                    segment_has_camera=[bool(cmd.get("camera"))],
                    preserve_kv_cache=True,
                )
                video = video_to_uint8_numpy(output[0], chunk_frames=8)
            del output
        except Exception as exc:
            logger.exception("turn generation failed")
            _reset_pipeline()
            torch.cuda.empty_cache()
            yield {"type": "gen_failed", "turn": cmd["turn"], "message": str(exc)}
            break
        torch.cuda.empty_cache()
        meta = dict(pipe._last_pixelumm_streaming_metadata or {})
        yield {
            "type": "frames",
            "turn": cmd["turn"],
            "start": int(meta.get("replace_pixel_start", 0)),
            "end": int(meta.get("generated_pixel_frames", 0)),
            "jpegs": _encode_jpegs(video),
            "gen_seconds": time.monotonic() - started,
            "remaining": deadline - time.monotonic(),
        }
    _reset_pipeline()


def _shape_prompt(prompt: str) -> str:
    item = json.loads(prompt)
    segment = json.loads(json.dumps(item["segments"][0], ensure_ascii=False))
    first = isinstance(item.get("subject_list"), list)
    segment["_segment_idx"] = 0 if first else 1
    if not first and bool(getattr(INF, "no_desc_long_after_first", False)):
        segment.pop("visual_description_long", None)
    shaped, count = _build_whole_item_prompts(
        item={**item, "segments": [segment]},
        segments=[segment],
        keep_keywords=list(INF.keep_keywords) if INF.keep_keywords else None,
        resolve_subject_ids=bool(getattr(INF, "resolve_subject_ids", False)),
        global_only_in_first_segment=bool(getattr(INF, "global_only_in_first_segment", False)),
        no_desc_long_after_first=bool(getattr(INF, "no_desc_long_after_first", False)),
        normalize_action_timestamp=bool(
            getattr(getattr(CFG, "dataset", object()), "json_prompt_normalize_action_timestamp", False)
        ),
    )
    if count != 1:
        raise RuntimeError("prompt shaping changed the segment count")
    return shaped


@dataclass
class Session:
    token: str
    slot: int
    lease_id: str
    seconds: int
    world: Dict[str, Any]
    options: Dict[str, Any]
    events: "queue.Queue[dict]" = field(default_factory=queue.Queue)
    requests: "queue.Queue[Optional[dict]]" = field(default_factory=queue.Queue)
    frames: List[bytes] = field(default_factory=list)
    turns: List[Dict[str, Any]] = field(default_factory=list)
    pe_session: Optional[Dict[str, Any]] = None
    pe_history: List[Optional[Dict[str, Any]]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    gpu_event: threading.Event = field(default_factory=threading.Event)
    gpu_result: Optional[dict] = None
    alive: bool = True
    ready: bool = False
    remaining: float = 0.0
    finished_at: Optional[float] = None
    parked: Optional[str] = None
    parked_bytes: int = 0
    parked_at: float = 0.0

    def emit(self, event: dict):
        self.events.put(event)

    def turn(self, turn_id: int) -> Dict[str, Any]:
        return self.turns[turn_id]


SESSIONS: Dict[str, Session] = {}
SESSIONS_LOCK = threading.Lock()
FREE_SLOTS = list(range(MAX_SESSIONS))


class SessionPromptEnhancer(OnlinePromptEnhancer):
    def _sample_previous_video_history(self, relative_url, segment_frame_lengths=None):
        sess = SESSIONS.get(str(relative_url or ""))
        lengths = [int(v) for v in (segment_frame_lengths or []) if int(v) > 0]
        if sess is None or not lengths:
            return []
        import base64

        with sess.lock:
            frames = list(sess.frames)
        groups = []
        start = 0
        for length in lengths:
            end = min(start + length, len(frames))
            if end <= start:
                raise PromptEnhancementError("previous segment has no frames yet")
            count = min(self.max_condition_frames, end - start, max(1, int(round((end - start) * 2.0 / FPS))))
            indices = (
                [start]
                if count == 1
                else [start + int(i * (end - start - 1) / (count - 1)) for i in range(count)]
            )
            groups.append(
                [
                    (
                        "data:image/jpeg;base64," + base64.b64encode(frames[i]).decode("ascii"),
                        i / FPS,
                        max(0.0, (i - start) / FPS),
                    )
                    for i in dict.fromkeys(indices)
                ]
            )
            start += length
        return groups


class RouterClient(KimiK3Client):
    def __init__(self, models: List[str]):
        super().__init__(
            api_url=PE_API_URL,
            model=models[0],
            timeout=90,
            retries=1,
            api_key=os.environ.get("PE_API_KEY"),
        )
        self.models = models

    def available_models(self):
        return [self.model]

    def request_payload(self, *args, **kwargs):
        payload = super().request_payload(*args, **kwargs)
        payload.pop("chat_template_kwargs", None)
        payload.pop("reasoning_effort", None)
        return payload

    def complete(self, messages, **kwargs):
        errors = []
        for model in self.models:
            self.model = model
            try:
                return super().complete(messages, **kwargs)
            except PromptEnhancementError as exc:
                logger.warning(f"prompt enhancer {model} failed: {exc}")
                errors.append(str(exc))
            finally:
                self.model = self.models[0]
        raise PromptEnhancementError("; ".join(errors))


def _models(env: str, default: str) -> List[str]:
    return [m.strip() for m in os.environ.get(env, default).split(",") if m.strip()]


_pe_dir = Path(tempfile.gettempdir()) / "nomas_pe"
OPENING_ENHANCER = SessionPromptEnhancer(
    output_dir=_pe_dir,
    client=RouterClient(_models("PE_OPENING_MODELS", "Qwen/Qwen3.8-27B:cerebras,moonshotai/Kimi-K3:openrouter")),
    fps=FPS,
)
TURN_ENHANCER = SessionPromptEnhancer(
    output_dir=_pe_dir,
    client=RouterClient(_models("PE_TURN_MODELS", "deepseek-ai/DeepSeek-V4.1-Flash:deepinfra,moonshotai/Kimi-K3:openrouter")),
    fps=FPS,
    max_condition_frames=int(os.environ.get("PE_FRAMES_PER_SEGMENT", "40")),
)

PERSPECTIVES = {"first-person", "third-person follow", "third-person observer", "second-person"}


def _turn_public(turn: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in turn.items() if k not in {"camera", "pe_input"}}


def _scene_text(prompt: str) -> Dict[str, Any]:
    try:
        item = json.loads(prompt)
        seg = item["segments"][0]
    except Exception:
        return {}
    actions = [
        a.get("action_description", "")
        for a in seg.get("contained_actions") or []
        if isinstance(a, dict)
    ]
    return {
        "style": item.get("Style"),
        "scene": seg.get("visual_description_long") or seg.get("description_change") or "",
        "actions": [a for a in actions if a],
    }


def _director(sess: Session):
    while sess.alive:
        req = sess.requests.get()
        if req is None or not sess.alive:
            break
        if req["op"] == "undo":
            _run_undo(sess, req)
            continue
        _run_turn(sess, req)


def _await_gpu(sess: Session, timeout: float) -> Optional[dict]:
    if not sess.gpu_event.wait(timeout):
        return None
    sess.gpu_event.clear()
    result, sess.gpu_result = sess.gpu_result, None
    return result


def _run_turn(sess: Session, req: Dict[str, Any]):
    turn = sess.turn(req["turn"])
    turn["status"] = "directing"
    sess.emit({"type": "turn", "turn": _turn_public(turn)})
    pe_input = dict(req["pe_input"])
    pe_input["session_id"] = sess.token
    t0 = time.monotonic()
    try:
        enhancer = OPENING_ENHANCER if sess.pe_session is None else TURN_ENHANCER
        enhanced = enhancer.enhance(pe_input, sess.pe_session, reasoning_effort="none", seed=req["seed"])
    except Exception as exc:
        logger.exception("prompt enhancement failed")
        turn["status"] = "failed"
        turn["error"] = f"The director could not plan this turn: {exc}"
        sess.emit({"type": "turn", "turn": _turn_public(turn)})
        if turn["id"] == 0:
            CMD_QS[sess.slot].put({"op": "stop", "lease": sess.lease_id})
        return
    turn["pe_seconds"] = round(time.monotonic() - t0, 1)
    turn["director"] = _scene_text(enhanced.prompt)
    turn["num_frames"] = int(enhanced.num_frames)
    turn["status"] = "queued"
    sess.emit({"type": "turn", "turn": _turn_public(turn)})
    if not sess.alive:
        return
    sess.gpu_event.clear()
    CMD_QS[sess.slot].put(
        {
            "op": "turn",
            "lease": sess.lease_id,
            "turn": turn["id"],
            "prompt": _shape_prompt(enhanced.prompt),
            "num_frames": int(enhanced.num_frames),
            "camera": json.dumps(req["camera"], ensure_ascii=False) if req.get("camera") else None,
            "steps": int(sess.options["steps"]),
            "seed": int(req["seed"]),
        }
    )
    result = None
    while sess.alive and result is None:
        result = _await_gpu(sess, 1.0)
    if not result or result["type"] != "frames":
        return
    session = dict(enhanced.session)
    lengths = list(session.get("segment_frame_lengths") or [])
    if lengths:
        lengths[-1] = max(1, turn["end"] - turn["start"])
    session["segment_frame_lengths"] = lengths
    session["video_url"] = sess.token
    sess.pe_history.append(sess.pe_session)
    sess.pe_session = session


def _run_undo(sess: Session, req: Dict[str, Any]):
    sess.gpu_event.clear()
    CMD_QS[sess.slot].put({"op": "undo", "lease": sess.lease_id, "turn": req["turn"]})
    result = None
    while sess.alive and result is None:
        result = _await_gpu(sess, 1.0)
    if result and result["type"] == "undone" and sess.pe_history:
        sess.pe_session = sess.pe_history.pop()


def _claim_slot() -> Optional[int]:
    with SESSIONS_LOCK:
        return FREE_SLOTS.pop(0) if FREE_SLOTS else None


def _release_slot(slot: int):
    while True:
        try:
            CMD_QS[slot].get_nowait()
        except queue.Empty:
            break
        except Exception:
            break
    with SESSIONS_LOCK:
        if slot not in FREE_SLOTS:
            FREE_SLOTS.append(slot)


def _unpark(sess: Session):
    if sess.parked:
        worldstate.discard(sess.parked)
    sess.parked = None
    sess.parked_bytes = 0


def _enforce_park_budget():
    with SESSIONS_LOCK:
        parked = sorted((s for s in SESSIONS.values() if s.parked), key=lambda s: s.parked_at, reverse=True)
    used = 0
    for sess in parked:
        used += sess.parked_bytes
        if used > PARK_BUDGET_BYTES:
            _unpark(sess)


def _janitor():
    while True:
        time.sleep(60)
        now = time.time()
        with SESSIONS_LOCK:
            items = list(SESSIONS.items())
        for token, sess in items:
            if sess.parked and now - sess.parked_at > PARK_TTL:
                _unpark(sess)
            if sess.finished_at and not sess.alive and now - sess.finished_at > RETAIN_SECONDS:
                _unpark(sess)
                with SESSIONS_LOCK:
                    SESSIONS.pop(token, None)


for _stale in Path(PARK_DIR).glob("nomas_world_*"):
    try:
        _stale.unlink()
    except OSError:
        pass


threading.Thread(target=_janitor, daemon=True).start()


def _new_turn(sess: Session, kind: str, label: str, seconds: float, camera=None) -> Dict[str, Any]:
    turn = {
        "id": len(sess.turns),
        "kind": kind,
        "label": label,
        "seconds": seconds,
        "status": "waiting",
        "start": None,
        "end": None,
        "camera": camera,
    }
    sess.turns.append(turn)
    return turn


def _clean_camera(camera: Any, seconds: float) -> Optional[Dict[str, Any]]:
    if not isinstance(camera, dict) or not isinstance(camera.get("frames"), list):
        return None
    frames = []
    for i, f in enumerate(camera["frames"][:2000]):
        try:
            p, q = f["position"], f["quaternion"]
            frames.append(
                {
                    "frame": i,
                    "t": float(f.get("t", 0.0)),
                    "position": {k: float(p[k]) for k in "xyz"},
                    "quaternion": {k: float(q[k]) for k in "xyzw"},
                    "fov": float(f.get("fov", 55.0)),
                }
            )
        except (KeyError, TypeError, ValueError):
            return None
    if len(frames) < 2:
        return None
    keys = []
    for e in camera.get("keyEvents") or []:
        if isinstance(e, dict) and isinstance(e.get("code"), str):
            keys.append({"t": float(e.get("t", 0)), "code": e["code"], "pressed": bool(e.get("pressed"))})
    return {
        "id": "live",
        "name": "Live steering",
        "source": "recorded",
        "sampleRate": int(camera.get("sampleRate") or 30),
        "durationMs": int(seconds * 1000),
        "frames": frames,
        "keyEvents": keys[:500],
    }


def _queue_turn(sess: Session, kind: str, text: str, camera_raw: Any, seconds: float) -> Dict[str, Any]:
    seconds = float(min(TURN_SECONDS, key=lambda s: abs(s - float(seconds or 4))))
    camera = _clean_camera(camera_raw, seconds) if kind == "move" else None
    if kind == "move" and camera is None:
        raise HTTPException(400, "The camera path is empty. Hold a movement key to record one.")
    if kind == "say" and not text.strip():
        raise HTTPException(400, "Type what should happen next.")
    label = text.strip()[:280] if kind == "say" else "Camera move"
    turn = _new_turn(sess, kind, label, seconds, camera)
    if camera:
        pe_input = {"camera_controlled": True, "camera_control": camera, "camera_duration_ms": int(seconds * 1000)}
    else:
        pe_input = {"action": text.strip()[:600], "camera_duration_ms": int(seconds * 1000)}
    sess.emit({"type": "turn", "turn": _turn_public(turn)})
    sess.requests.put(
        {"op": "turn", "turn": turn["id"], "pe_input": pe_input, "camera": camera, "seed": sess.options["seed"]}
    )
    return turn


def _store_frames(sess: Session, ev: dict):
    turn = sess.turn(ev["turn"])
    start, jpegs = ev["start"], ev["jpegs"]
    with sess.lock:
        del sess.frames[start:]
        sess.frames.extend(jpegs)
        total = len(sess.frames)
    prior = [t for t in sess.turns if t["status"] == "ready" and t["id"] != turn["id"]]
    turn["start"] = prior[-1]["end"] if prior else 0
    turn["end"] = total
    turn["gen_seconds"] = round(ev["gen_seconds"], 1)
    turn["status"] = "ready"
    if prior and prior[-1]["end"] > start:
        prior[-1]["end"] = max(prior[-1]["start"], min(prior[-1]["end"], total))
    return {
        "type": "frames",
        "turn": _turn_public(turn),
        "replace_from": start,
        "total": total,
        "remaining": ev["remaining"],
    }


app = Server()


@app.api(name="run_session", concurrency_limit=MAX_SESSIONS)
def run_session(world: str = "{}", options: str = "{}") -> str:
    """Open (or resume) a live Hy-Nomas world session (streams JSON events)."""
    try:
        w = json.loads(world or "{}")
        o = json.loads(options or "{}")
    except json.JSONDecodeError:
        yield json.dumps({"type": "error", "message": "Malformed request."})
        return
    seconds = SESSION_CHOICES.get(str(o.get("length")), SESSION_CHOICES["standard"])
    if o.get("resume"):
        yield from _resume_session(str(o["resume"]), o.get("turn") or {}, seconds)
        return
    subject = str(w.get("subject") or "").strip()[:600]
    environment = str(w.get("environment") or "").strip()[:900]
    perspective = str(w.get("perspective") or "third-person follow")
    if not subject or not environment:
        yield json.dumps({"type": "error", "message": "Describe both the subject and the world."})
        return
    if perspective not in PERSPECTIVES:
        perspective = "third-person follow"
    turn_seconds = float(min(TURN_SECONDS, key=lambda s: abs(s - float(o.get("turn_seconds") or 4))))
    steps = int(o.get("steps") or 8)
    steps = steps if steps in (4, 6, 8) else 8
    seed = int(o["seed"]) if str(o.get("seed", "")).isdigit() else random.randint(0, 2**31 - 1)
    slot = _claim_slot()
    if slot is None:
        yield json.dumps(
            {"type": "error", "message": f"All {MAX_SESSIONS} live worlds are busy right now. Try again in a minute."}
        )
        return
    token = secrets.token_urlsafe(18)
    sess = Session(
        token=token,
        slot=slot,
        lease_id=secrets.token_hex(8),
        seconds=seconds,
        world={"subject": subject, "environment": environment, "perspective": perspective, "title": str(w.get("title") or "")[:80]},
        options={"steps": steps, "turn_seconds": turn_seconds, "seed": seed},
    )
    with SESSIONS_LOCK:
        SESSIONS[token] = sess
    first = _new_turn(sess, "world", "Entering the world", turn_seconds)
    sess.requests.put(
        {
            "op": "turn",
            "turn": first["id"],
            "pe_input": {
                "perspective": perspective,
                "subject_description": subject,
                "environment_description": environment,
                "camera_duration_ms": int(turn_seconds * 1000),
            },
            "camera": None,
            "seed": sess.options["seed"],
        }
    )
    director = threading.Thread(target=_director, args=(sess,), daemon=True)
    director.start()
    yield json.dumps(
        {
            "type": "session",
            "token": token,
            "seconds": seconds,
            "turn_seconds": turn_seconds,
            "world": sess.world,
            "turns": [_turn_public(first)],
        }
    )
    yield from _drive(sess)


def _resume_session(token: str, turn_req: Dict[str, Any], seconds: int):
    sess = SESSIONS.get(token)
    if sess is None or sess.alive or not sess.parked:
        yield json.dumps({"type": "error", "message": "This world is no longer saved. Open a new one."})
        return
    slot = _claim_slot()
    if slot is None:
        yield json.dumps({"type": "ended", "reason": "busy", "resumable": True,
                          "message": f"All {MAX_SESSIONS} live worlds are busy right now. Try again in a minute."})
        return
    sess.slot, sess.lease_id, sess.seconds = slot, secrets.token_hex(8), seconds
    sess.alive, sess.ready, sess.finished_at = True, False, None
    sess.events, sess.requests = queue.Queue(), queue.Queue()
    sess.gpu_event.clear()
    threading.Thread(target=_director, args=(sess,), daemon=True).start()
    try:
        turn = _queue_turn(sess, str(turn_req.get("kind") or "say"), str(turn_req.get("text") or ""),
                           turn_req.get("camera"), turn_req.get("seconds") or sess.options["turn_seconds"])
    except HTTPException as exc:
        _release_slot(slot)
        sess.alive, sess.finished_at = False, time.time()
        sess.requests.put(None)
        yield json.dumps({"type": "ended", "reason": "error", "resumable": True, "message": exc.detail})
        return
    yield json.dumps({"type": "session", "token": token, "seconds": seconds, "turn_seconds": sess.options["turn_seconds"],
                      "world": sess.world, "resumed": True, "turns": [_turn_public(t) for t in sess.turns]})
    while turn["status"] not in ("queued", "failed"):
        try:
            yield json.dumps(sess.events.get(timeout=0.25))
        except queue.Empty:
            pass
    while True:
        try:
            yield json.dumps(sess.events.get_nowait())
        except queue.Empty:
            break
    if turn["status"] == "failed":
        _release_slot(slot)
        sess.alive, sess.finished_at = False, time.time()
        sess.requests.put(None)
        yield json.dumps({"type": "ended", "reason": "error", "resumable": True, "message": turn.get("error")})
        return
    restore_path, sess.parked, sess.parked_bytes = sess.parked, None, 0
    yield from _drive(sess, restore_path)


def _drive(sess: Session, restore_path: str = ""):
    slot = sess.slot
    park_path = f"{PARK_DIR}/nomas_world_{sess.token}_{sess.lease_id}"
    parked = None
    last_tick = 0.0
    try:
        for ev in gpu_session(slot, sess.lease_id, sess.seconds, park_path, restore_path):
            while True:
                try:
                    yield json.dumps(sess.events.get_nowait())
                except queue.Empty:
                    break
            kind = ev["type"]
            if "remaining" in ev:
                sess.remaining = float(ev["remaining"])
            if kind == "ready":
                sess.ready = True
                yield json.dumps({"type": "ready", "remaining": ev["remaining"]})
            elif kind == "tick":
                now = time.monotonic()
                if now - last_tick >= 1.0:
                    last_tick = now
                    yield json.dumps({"type": "tick", "remaining": ev["remaining"]})
            elif kind == "gen_start":
                turn = sess.turn(ev["turn"])
                turn["status"] = "generating"
                yield json.dumps({"type": "turn", "turn": _turn_public(turn), "remaining": ev["remaining"]})
            elif kind == "frames":
                out = _store_frames(sess, ev)
                sess.gpu_result = {"type": "frames"}
                sess.gpu_event.set()
                yield json.dumps(out)
            elif kind == "gen_failed":
                turn = sess.turn(ev["turn"])
                turn["status"] = "failed"
                turn["error"] = "Generation failed, so this world has ended."
                logger.error(ev["message"])
                sess.gpu_result = {"type": "gen_failed"}
                sess.gpu_event.set()
                yield json.dumps({"type": "turn", "turn": _turn_public(turn)})
            elif kind == "undone":
                turn = sess.turn(ev["turn"])
                with sess.lock:
                    del sess.frames[ev["frames_end"]:]
                    total = len(sess.frames)
                turn["status"] = "undone"
                for t in sess.turns:
                    if t["status"] == "ready" and t["end"] is not None and t["end"] > total:
                        t["end"] = total
                sess.gpu_result = {"type": "undone"}
                sess.gpu_event.set()
                yield json.dumps({"type": "undone", "turn": _turn_public(turn), "total": total})
            elif kind == "undo_failed":
                sess.gpu_result = {"type": "undo_failed"}
                sess.gpu_event.set()
                yield json.dumps({"type": "notice", "message": "Could not undo that turn."})
            elif kind == "restored":
                worldstate.discard(restore_path)
                logger.info(f"restored {ev['bytes'] / 2**30:.2f} GiB world in {ev['seconds']:.2f}s")
            elif kind == "restore_failed":
                worldstate.discard(restore_path)
                yield json.dumps({"type": "ended", "reason": "error", "message": "Could not restore this world: " + ev["message"]})
                return
            elif kind == "parked":
                parked = ev
                logger.info(f"parked {ev['bytes'] / 2**30:.2f} GiB world in {ev['seconds']:.2f}s")
            elif kind == "expired":
                if parked:
                    sess.parked, sess.parked_bytes, sess.parked_at = park_path, int(parked["bytes"]), time.time()
                    _enforce_park_budget()
                yield json.dumps({"type": "ended", "reason": "time", "resumable": bool(sess.parked), "keep_minutes": PARK_TTL // 60})
        while True:
            try:
                yield json.dumps(sess.events.get_nowait())
            except queue.Empty:
                break
        yield json.dumps({"type": "ended", "reason": "closed"})
    except Exception as exc:
        logger.exception("session failed")
        message = str(exc) or exc.__class__.__name__
        yield json.dumps({"type": "ended", "reason": "error", "message": message})
    finally:
        sess.alive = False
        sess.finished_at = time.time()
        sess.requests.put(None)
        sess.gpu_event.set()
        _release_slot(slot)


def _session_or_404(token: str) -> Session:
    sess = SESSIONS.get(token or "")
    if sess is None:
        raise HTTPException(404, "Unknown or expired session.")
    return sess


@app.post("/api/turn")
async def api_turn(request: Request):
    body = await request.json()
    sess = _session_or_404(body.get("token"))
    if not sess.alive:
        raise HTTPException(409, "This world has ended.")
    kind = body.get("kind")
    if kind not in {"say", "move"}:
        raise HTTPException(400, "Unknown turn kind.")
    pending = [t for t in sess.turns if t["status"] in {"waiting", "directing", "queued", "generating"}]
    if len(pending) >= 3:
        raise HTTPException(429, "A few turns are already queued. Let them land first.")
    turn = _queue_turn(sess, kind, str(body.get("text") or ""), body.get("camera"), body.get("seconds") or sess.options["turn_seconds"])
    return JSONResponse({"turn": _turn_public(turn)})


@app.post("/api/undo")
async def api_undo(request: Request):
    body = await request.json()
    sess = _session_or_404(body.get("token"))
    if not sess.alive:
        raise HTTPException(409, "This world has ended.")
    if any(t["status"] in {"waiting", "directing", "queued", "generating"} for t in sess.turns):
        raise HTTPException(409, "Wait for the current turn to finish first.")
    ready = [t for t in sess.turns if t["status"] == "ready"]
    if len(ready) <= 1:
        raise HTTPException(409, "Nothing to undo; the opening shot is the root of this world.")
    sess.requests.put({"op": "undo", "turn": ready[-1]["id"]})
    return JSONResponse({"ok": True})


@app.post("/api/stop")
async def api_stop(request: Request):
    body = await request.json()
    sess = _session_or_404(body.get("token"))
    if sess.alive:
        CMD_QS[sess.slot].put({"op": "stop", "lease": sess.lease_id})
    else:
        _unpark(sess)
    return JSONResponse({"ok": True})


@app.get("/api/frames/{token}")
async def api_frames(token: str, start: int = 0, end: int = -1):
    sess = _session_or_404(token)
    with sess.lock:
        frames = sess.frames[max(0, start): (len(sess.frames) if end < 0 else end)]
    buf = io.BytesIO()
    buf.write(struct.pack(">I", len(frames)))
    for jpg in frames:
        buf.write(struct.pack(">I", len(jpg)))
        buf.write(jpg)
    return Response(buf.getvalue(), media_type="application/octet-stream", headers={"Cache-Control": "no-store"})


@app.get("/api/video/{token}.mp4")
def api_video(token: str):
    sess = _session_or_404(token)
    with sess.lock:
        frames = list(sess.frames)
    if not frames:
        raise HTTPException(404, "No frames yet.")
    import imageio

    out = Path(tempfile.gettempdir()) / f"nomas_{token}.mp4"
    with imageio.get_writer(
        out, fps=FPS, codec="libx264", pixelformat="yuv420p", output_params=["-movflags", "+faststart", "-crf", "18"]
    ) as writer:
        for jpg in frames:
            writer.append_data(cv2.cvtColor(cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    return FileResponse(out, media_type="video/mp4", filename="hy-nomas-world.mp4")


@app.get("/api/status")
async def api_status():
    with SESSIONS_LOCK:
        free = len(FREE_SLOTS)
    return JSONResponse({"free": free, "total": MAX_SESSIONS, "gpu": GPU_SIZE})


ASSETS = ROOT / "assets"


@app.get("/assets/{name}")
async def assets(name: str):
    path = (ASSETS / name).resolve()
    if path.parent != ASSETS or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, headers={"Cache-Control": "public, max-age=3600"})


@app.get("/", response_class=HTMLResponse)
async def homepage():
    return (ROOT / "index.html").read_text(encoding="utf-8")


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
