"""Load optional Hopper attention kernels only when explicitly selected."""

from functools import lru_cache
import importlib.util


def magi_available():
    try:
        return importlib.util.find_spec("magi_attention") is not None
    except (ImportError, ValueError):
        return False


@lru_cache(maxsize=1)
def magi_kernel():
    try:
        from magi_attention.functional import flex_flash_attn_func
    except ImportError as exc:
        raise RuntimeError("The selected Magi backend requires MagiAttention") from exc
    return flex_flash_attn_func


def magi_attention(*args, **kwargs):
    return magi_kernel()(*args, **kwargs)


def profile_flash_attention(*args, **kwargs):
    try:
        from flash_attn_interface import flash_attn_func
    except ImportError:
        from flash_attn import flash_attn_func
    return flash_attn_func(*args, **kwargs)
