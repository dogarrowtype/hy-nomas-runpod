import inspect
import textwrap

import torch
import torch.nn as nn

E4M3 = torch.float8_e4m3fn
E4M3_MAX = torch.finfo(E4M3).max


def fuse_int8_dequant():
    from hymm.models.dit.modules import int8_stored_bf16_linear as w8

    def dequant(weight_q, weight_scale, in_features: int):
        scales = weight_scale.repeat_interleave(w8.GROUP_SIZE, dim=1)
        return (weight_q.float() * scales[:, :in_features]).to(torch.bfloat16)

    fused = torch.compile(dequant, dynamic=False, fullgraph=True)

    def dequant_weight(self):
        return fused(self.weight_q, self.weight_scale, self.in_features)

    w8.Int8StoredBF16Linear.dequant_weight = dequant_weight


def keep_kv_archive_on_gpu():
    import hymm.models.dit.modules.uniform_history_cache as uhc

    edits = {
        "commit_uniform_history": [
            (
                "segment[key].to(device='cpu', copy=True, non_blocking=False)",
                "segment[key].to(device=current_k.device, copy=True, non_blocking=False)",
            ),
            (
                "device='cpu', dtype=torch.bool, copy=True, non_blocking=False,",
                "device=current_k.device, dtype=torch.bool, copy=True, non_blocking=False,",
            ),
        ],
        "_rebuild_history": [
            (
                "packed[key].append(getattr(old, key).index_select(1, indices))",
                "packed[key].append(getattr(old, key).index_select(1, indices.to(getattr(old, key).device)))",
            ),
            (
                "packed_mask.append(old.mask.index_select(1, indices))",
                "packed_mask.append(old.mask.index_select(1, indices.to(old.mask.device)))",
            ),
        ],
        "prepare_history_rollback": [
            (
                "or record.mask.dtype != torch.bool or record.mask.device.type != 'cpu'):",
                "or record.mask.dtype != torch.bool):",
            ),
            (
                "or tensor.dtype != cache[key].dtype or tensor.device.type != 'cpu'):",
                "or tensor.dtype != cache[key].dtype):",
            ),
        ],
    }
    for name, pairs in edits.items():
        src = textwrap.dedent(inspect.getsource(getattr(uhc, name)))
        for old, new in pairs:
            if old not in src:
                raise RuntimeError(f"history cache source changed: {name}")
            src = src.replace(old, new)
        scope = {}
        exec(compile(src, f"<gpu_kv_archive:{name}>", "exec"), uhc.__dict__, scope)
        setattr(uhc, name, scope[name])


def _quant_rows(x):
    amax = x.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-12)
    scale = amax / E4M3_MAX
    return (x.float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(E4M3), scale


_quant_rows_compiled = torch.compile(_quant_rows, dynamic=True)


class FP8RowwiseLinear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        q, s = _quant_rows(linear.weight.data)
        self.register_buffer("weight_q", q.contiguous())
        self.register_buffer("weight_scale", s.contiguous())
        self.bias = linear.bias

    def forward(self, x):
        shape = x.shape
        xq, sx = _quant_rows_compiled(x.reshape(-1, shape[-1]))
        bias = None if self.bias is None else self.bias.to(torch.bfloat16)
        y = torch._scaled_mm(
            xq, self.weight_q.t(), scale_a=sx, scale_b=self.weight_scale.t(),
            out_dtype=torch.bfloat16, bias=bias,
        )
        return y.reshape(*shape[:-1], self.out_features)


def quantize_decoder_fp8(decoder: nn.Module) -> int:
    count = 0
    for parent in list(decoder.modules()):
        for name, child in list(parent.named_children()):
            if (
                isinstance(child, nn.Linear)
                and child is not getattr(decoder, "embed", None)
                and child.in_features >= 1024
                and child.in_features % 16 == 0
                and child.out_features % 16 == 0
            ):
                setattr(parent, name, FP8RowwiseLinear(child))
                count += 1
    return count


def configure_compile():
    import torch._dynamo.config as dynamo_config

    dynamo_config.automatic_dynamic_shapes = False
    dynamo_config.cache_size_limit = 32


def load_compile_cache(path) -> bool:
    from pathlib import Path

    path = Path(path)
    if not path.is_file():
        return False
    torch.compiler.load_cache_artifacts(path.read_bytes())
    return True


def compile_decoder(decoder: nn.Module):
    from hymm.models.vae.pixelumm_decoder import model as pixelumm_model

    layer_forward = torch.compile(pixelumm_model.GenLayer.forward, fullgraph=True, dynamic=False)

    def forward(self, *args, **kwargs):
        return layer_forward(self, *args, **kwargs)

    pixelumm_model.GenLayer.forward = forward
