"""Row-chunked, eager construction of a FlexAttention BlockMask.

Equivalent to ``create_block_mask(mask_mod, None, None, n, n, device)`` (128x128
blocks, mask_mod kept for partial blocks) but evaluates the mask a few thousand
query rows at a time, so a 36k-token PixelUMM window (720p) needs ~1 GB instead
of tens of GB, and no torch.compile is involved.
"""
import torch
from torch.nn.attention.flex_attention import BlockMask, _dense_to_ordered

BLOCK = 128


@torch.no_grad()
def chunked_block_mask(mask_mod, q_len, kv_len, device, chunk_rows=2048):
    nq, nkv = -(-q_len // BLOCK), -(-kv_len // BLOCK)
    counts = torch.zeros(nq, nkv, dtype=torch.int32, device=device)
    kv = torch.arange(kv_len, device=device)
    pad_kv = nkv * BLOCK - kv_len
    for r0 in range(0, q_len, chunk_rows):
        r1 = min(r0 + chunk_rows, q_len)
        q = torch.arange(r0, r1, device=device)
        m = mask_mod(None, None, q[:, None], kv[None, :])
        rows = r1 - r0
        pad_q = -(-rows // BLOCK) * BLOCK - rows
        if pad_q or pad_kv:
            m = torch.nn.functional.pad(m, (0, pad_kv, 0, pad_q), value=False)
        blk = m.view(-1, BLOCK, nkv, BLOCK).sum(dim=(1, 3), dtype=torch.int32)
        b0 = r0 // BLOCK
        counts[b0 : b0 + blk.shape[0]] += blk
        del m
    full = counts == BLOCK * BLOCK
    partial = (counts > 0) & ~full
    kv_num, kv_idx = _dense_to_ordered(partial[None, None].to(torch.int32))
    full_num, full_idx = _dense_to_ordered(full[None, None].to(torch.int32))
    return BlockMask.from_kv_blocks(
        kv_num, kv_idx, full_num, full_idx,
        BLOCK_SIZE=BLOCK, mask_mod=mask_mod, seq_lengths=(q_len, kv_len),
    )
