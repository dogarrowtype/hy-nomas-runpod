import dataclasses
import importlib
import os
import pickle
from pathlib import Path

import torch
import torch.nn as nn

PIPE_ATTRS = (
    "kv_cache",
    "cache_mask_list",
    "cache_valid_len",
    "global_frame_counter",
    "segment_token_lengths",
    "_interactive_cached_action_tails",
    "_interactive_cached_seg_video_ranges",
    "_interactive_noise_bank",
    "_interactive_noise_generator",
    "_interactive_noise_seed",
    "_interactive_pixelumm_decode_session",
    "_interactive_pixelumm_offloaded",
    "_interactive_segment_cache_tokens",
    "_interactive_segment_count",
    "_interactive_segment_latent_lengths",
    "_interactive_segment_noise_rng_states",
    "_last_pixelumm_streaming_metadata",
)
PRIMITIVES = (int, float, str, bool, bytes, type(None), torch.dtype, torch.device, range, slice)
ALIGN = 512


def _cls_path(obj):
    cls = type(obj)
    return f"{cls.__module__}:{cls.__qualname__}"


def _load_cls(path):
    module, name = path.split(":")
    obj = importlib.import_module(module)
    for part in name.split("."):
        obj = getattr(obj, part)
    return obj


class _Packer:
    def __init__(self, modules):
        self.modules = {id(m): name for name, m in modules.items()}
        self.storages = {}
        self.tensors = {}

    def tensor(self, t):
        key = id(t)
        if key not in self.tensors:
            storage = t.untyped_storage()
            skey = (storage.device.type, storage.device.index, storage.data_ptr())
            self.storages.setdefault(skey, storage)
            extra = {k: self.pack(v) for k, v in vars(t).items()} if getattr(t, "__dict__", None) else {}
            self.tensors[key] = (
                skey, t.dtype, tuple(t.shape), tuple(t.stride()), t.storage_offset(), str(t.device), extra,
            )
        return ("T", key)

    def pack(self, obj):
        if isinstance(obj, torch.Tensor):
            return self.tensor(obj)
        if isinstance(obj, PRIMITIVES):
            return ("P", obj)
        if isinstance(obj, torch.Generator):
            return ("G", str(obj.device), self.tensor(obj.get_state()))
        if id(obj) in self.modules:
            return ("M", self.modules[id(obj)])
        if isinstance(obj, nn.Module):
            raise TypeError(f"unregistered module in world state: {type(obj).__name__}")
        if isinstance(obj, dict):
            return ("D", [(self.pack(k), self.pack(v)) for k, v in obj.items()])
        if isinstance(obj, list):
            return ("L", [self.pack(v) for v in obj])
        if isinstance(obj, tuple):
            if hasattr(obj, "_fields"):
                return ("N", _cls_path(obj), [self.pack(v) for v in obj])
            return ("U", [self.pack(v) for v in obj])
        if isinstance(obj, (set, frozenset)):
            return ("S", type(obj) is frozenset, [self.pack(v) for v in obj])
        if dataclasses.is_dataclass(obj):
            return ("C", _cls_path(obj), {f.name: self.pack(getattr(obj, f.name)) for f in dataclasses.fields(obj)})
        if hasattr(obj, "__dict__"):
            return ("O", _cls_path(obj), {k: self.pack(v) for k, v in vars(obj).items()})
        raise TypeError(f"cannot park world state value of type {type(obj).__name__}")


class _Unpacker:
    def __init__(self, modules, tensors, storages):
        self.modules = modules
        self.meta = tensors
        self.storages = storages
        self.built = {}

    def tensor(self, key):
        if key not in self.built:
            skey, dtype, shape, stride, offset, device, extra = self.meta[key]
            t = torch.empty(0, dtype=dtype, device=device).set_(self.storages[skey], offset, shape, stride)
            self.built[key] = t
            for name, value in extra.items():
                setattr(t, name, self.unpack(value))
        return self.built[key]

    def unpack(self, node):
        kind = node[0]
        if kind == "T":
            return self.tensor(node[1])
        if kind == "P":
            return node[1]
        if kind == "G":
            g = torch.Generator(device=node[1])
            g.set_state(self.unpack(node[2]).cpu())
            return g
        if kind == "M":
            return self.modules[node[1]]
        if kind == "D":
            return {self.unpack(k): self.unpack(v) for k, v in node[1]}
        if kind == "L":
            return [self.unpack(v) for v in node[1]]
        if kind == "U":
            return tuple(self.unpack(v) for v in node[1])
        if kind == "N":
            return _load_cls(node[1])(*[self.unpack(v) for v in node[2]])
        if kind == "S":
            values = [self.unpack(v) for v in node[2]]
            return frozenset(values) if node[1] else set(values)
        if kind in ("C", "O"):
            obj = object.__new__(_load_cls(node[1]))
            for name, value in node[2].items():
                object.__setattr__(obj, name, self.unpack(value))
            return obj
        raise ValueError(f"bad world state node {kind}")


def _register(t):
    torch.cuda.cudart().cudaHostRegister(t.data_ptr(), t.numel(), 0)


def _unregister(t):
    torch.cuda.cudart().cudaHostUnregister(t.data_ptr())


def park(pipe, modules, path_prefix) -> int:
    packer = _Packer(modules)
    tree = {name: packer.pack(getattr(pipe, name)) for name in PIPE_ATTRS if hasattr(pipe, name)}
    layout = {}
    total = 0
    for skey, storage in packer.storages.items():
        layout[skey] = (total, storage.nbytes())
        total += (storage.nbytes() + ALIGN - 1) // ALIGN * ALIGN
    data_path = Path(f"{path_prefix}.bin")
    with open(data_path, "wb") as f:
        f.truncate(max(total, 1))
    blob = torch.from_file(str(data_path), shared=True, size=max(total, 1), dtype=torch.uint8)
    _register(blob)
    try:
        for skey, storage in packer.storages.items():
            start, nbytes = layout[skey]
            if nbytes:
                src = torch.empty(0, dtype=torch.uint8, device=storage.device).set_(storage)
                blob[start:start + nbytes].copy_(src, non_blocking=True)
        torch.cuda.synchronize()
    finally:
        _unregister(blob)
    meta = {
        "tree": tree,
        "tensors": packer.tensors,
        "layout": layout,
        "total": total,
    }
    tmp = Path(f"{path_prefix}.pkl.tmp")
    tmp.write_bytes(pickle.dumps(meta, protocol=pickle.HIGHEST_PROTOCOL))
    os.replace(tmp, f"{path_prefix}.pkl")
    return total


def restore(pipe, modules, path_prefix) -> int:
    meta = pickle.loads(Path(f"{path_prefix}.pkl").read_bytes())
    total = meta["total"]
    blob = torch.from_file(f"{path_prefix}.bin", shared=True, size=max(total, 1), dtype=torch.uint8)
    _register(blob)
    storages = {}
    try:
        for skey, (start, nbytes) in meta["layout"].items():
            device = torch.device(skey[0], skey[1]) if skey[0] == "cuda" else torch.device("cpu")
            if device.type == "cuda":
                device = torch.device("cuda")
            buf = torch.empty(nbytes, dtype=torch.uint8, device=device)
            if nbytes:
                buf.copy_(blob[start:start + nbytes], non_blocking=True)
            storages[skey] = buf.untyped_storage()
        torch.cuda.synchronize()
    finally:
        _unregister(blob)
    unpacker = _Unpacker(modules, meta["tensors"], storages)
    for name, node in meta["tree"].items():
        setattr(pipe, name, unpacker.unpack(node))
    return total


def discard(path_prefix):
    for suffix in (".bin", ".pkl", ".pkl.tmp"):
        try:
            os.remove(f"{path_prefix}{suffix}")
        except FileNotFoundError:
            pass
