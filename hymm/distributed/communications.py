# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team


from typing import Any, Tuple

import torch
import torch.distributed as dist

from torch.nn import functional as F


def _all_to_all_4D(
    input: torch.tensor,
    scatter_idx: int = 2,
    gather_idx: int = 1,
    group=None,
    use_sync: bool = False,
) -> torch.tensor:
    """
    all-to-all for QKV

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 2
        gather_idx (int): default 1
        group : torch process group
        use_sync (bool): whether to call torch.cuda.synchronize() after all_to_all

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    assert input.dim() == 4, (
        f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"
    )

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 2 and gather_idx == 1:
        # Detect uneven seqlen distribution using tensor-based collective (avoids slow pickle-based all_gather_object)
        local_len = torch.tensor(
            [input.shape[1]], dtype=torch.int64, device=input.device
        )
        all_lens = torch.empty(seq_world_size, dtype=torch.int64, device=input.device)
        dist.all_gather_into_tensor(all_lens, local_len, group=group)
        max_len = int(all_lens.max().item())
        # total_gap is consistent across ALL ranks: it equals the total number of padding tokens
        # added globally, so every rank trims by the same amount after all_to_all.
        total_gap = max_len * seq_world_size - int(all_lens.sum().item())
        local_gap = max_len - input.shape[1]
        if local_gap > 0:
            input = F.pad(input, (0, 0, 0, 0, 0, local_gap))

        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        assert hc % seq_world_size == 0, (
            f"Invalid Head size: {hc}, which should be divisible by spsize {seq_world_size}"
        )
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
        input_t = (
            input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
            .transpose(0, 2)
            .contiguous()
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t
        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(seqlen, bs, shard_hc, hs)

        # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
        output = output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)
        if total_gap > 0:
            output = output[:, :-total_gap].contiguous()

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, shard_hc, hs = input.shape

        hc = shard_hc * seq_world_size
        if seqlen % seq_world_size != 0:
            new_seqlen = (seqlen // seq_world_size + 1) * seq_world_size
            gap = new_seqlen - seqlen
            input = F.pad(input, (0, 0, 0, 0, 0, gap))
            bs, seqlen, shard_hc, hs = input.shape
        else:
            gap = 0

        assert seqlen % seq_world_size == 0

        shard_seqlen = seqlen // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)-> (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
        )

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            if use_sync:
                torch.cuda.synchronize()
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, shard_seqlen, bs, hs)

        # (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)

        if (
            gap > 0
            and dist.get_group_rank(group, dist.get_rank()) == seq_world_size - 1
        ):
            output = output[:, :-gap].contiguous()

        return output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


class SeqAllToAll4D(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: torch.Tensor,
        scatter_idx: int,
        gather_idx: int,
        use_sync: bool = False,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.use_sync = use_sync

        return _all_to_all_4D(
            input, scatter_idx, gather_idx, group=group, use_sync=use_sync
        )

    @staticmethod
    def backward(
        ctx: Any, *grad_output: torch.Tensor
    ) -> Tuple[None, torch.Tensor, None, None, None]:
        grad = grad_output[0]
        original_dtype = grad.dtype
        communication_dtype = (
            torch.float32
            if original_dtype in (torch.float16, torch.bfloat16)
            else original_dtype
        )
        if grad.dtype != communication_dtype:
            grad = grad.to(communication_dtype)
        grad_input = SeqAllToAll4D.apply(
            ctx.group,
            grad,
            ctx.gather_idx,
            ctx.scatter_idx,
            ctx.use_sync,
        )
        if grad_input.dtype != original_dtype:
            grad_input = grad_input.to(original_dtype)
        return (
            None,
            grad_input,
            None,
            None,
            None,
        )


def all_to_all_4D(
    input_: torch.Tensor,
    group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    return SeqAllToAll4D.apply(group, input_, scatter_dim, gather_dim)


class _AllGather(torch.autograd.Function):
    """All-gather communication with autograd support.

    Args:
        input_: input tensor
        dim: dimension along which to concatenate
    """

    @staticmethod
    def forward(ctx, input_, dim, group):
        ctx.dim = dim % input_.dim()
        ctx.group = group
        world_size = dist.get_world_size(group)

        # Use tensor-based collective to get sizes along gather dim, avoiding slow pickle-based all_gather_object
        local_dim_size = torch.tensor(
            [input_.shape[ctx.dim]], dtype=torch.int64, device=input_.device
        )
        all_dim_sizes = torch.empty(world_size, dtype=torch.int64, device=input_.device)
        dist.all_gather_into_tensor(all_dim_sizes, local_dim_size, group=group)
        sizes_list = [int(s) for s in all_dim_sizes.tolist()]

        ctx.sizes_list = sizes_list

        base_shape = list(input_.size())
        max_size = max(sizes_list)
        if all(s == max_size for s in sizes_list):
            tensor_list = [
                torch.empty(
                    base_shape,
                    dtype=input_.dtype,
                    device=input_.device,
                )
                for _ in sizes_list
            ]
            dist.all_gather(tensor_list, input_.contiguous(), group=group)
        else:
            # Some backends require equal tensor sizes for all_gather. Pad only
            # the uneven case, gather equal-size tensors, then remove padding.
            padded_shape = base_shape.copy()
            padded_shape[ctx.dim] = max_size
            padded_input = torch.zeros(
                padded_shape,
                dtype=input_.dtype,
                device=input_.device,
            )
            padded_input.narrow(ctx.dim, 0, input_.shape[ctx.dim]).copy_(input_)
            padded_list = [torch.empty_like(padded_input) for _ in sizes_list]
            dist.all_gather(padded_list, padded_input, group=group)
            tensor_list = [
                tensor.narrow(ctx.dim, 0, size).contiguous()
                for tensor, size in zip(padded_list, sizes_list)
            ]

        output = torch.cat(tensor_list, dim=ctx.dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        group = ctx.group
        world_size = dist.get_world_size(group)
        global_rank = dist.get_rank()
        rank = dist.get_group_rank(group, global_rank)
        dim = ctx.dim

        # Every rank consumes the full all-gathered tensor. Therefore the
        # gradient for rank r's input shard is the SUM of slice-r gradients
        # from every consumer rank. Merely selecting the local slice drops
        # cross-rank activation gradients (and lets FSDP average SP
        # contributions that should be summed).
        grad_chunks = torch.split(grad_output, ctx.sizes_list, dim=dim)
        if world_size == 1:
            return grad_chunks[0].contiguous(), None, None

        # reduce_scatter_tensor requires equal-size chunks. The forward supports
        # uneven gather sizes, so pad each shard along the gather dimension,
        # reduce-scatter with SUM, then trim this rank's result.
        max_size = max(ctx.sizes_list)
        if max_size == 0:
            return grad_chunks[rank].contiguous(), None, None

        # Attention activations are typically bf16 under autocast, which means
        # the old reduce-scatter SUM also accumulated in bf16.  Different SP
        # sizes then use different reduction trees and can introduce avoidable
        # quantization drift before gradients reach fp32 master weights.
        #
        # Accumulate the cross-rank SUM in fp32, then cast the local activation
        # gradient back to its original dtype for the preceding bf16 graph.
        reduce_dtype = (
            torch.float32
            if grad_output.dtype in (torch.float16, torch.bfloat16)
            else grad_output.dtype
        )

        padded_chunks = []
        for chunk in grad_chunks:
            moved = chunk.movedim(dim, 0).contiguous().to(reduce_dtype)
            if moved.shape[0] < max_size:
                padded = torch.zeros(
                    (max_size, *moved.shape[1:]),
                    dtype=moved.dtype,
                    device=moved.device,
                )
                padded[: moved.shape[0]].copy_(moved)
                moved = padded
            padded_chunks.append(moved)

        reduce_input = torch.cat(padded_chunks, dim=0).contiguous()
        reduce_output = torch.empty_like(padded_chunks[rank])
        dist.reduce_scatter_tensor(
            reduce_output,
            reduce_input,
            op=dist.ReduceOp.SUM,
            group=group,
        )

        local_size = ctx.sizes_list[rank]
        grad_input = reduce_output[:local_size].movedim(0, dim).contiguous()
        if grad_input.dtype != grad_output.dtype:
            grad_input = grad_input.to(grad_output.dtype)

        return grad_input, None, None


def all_gather(input_: torch.Tensor, dim: int = 1, group=None):
    """Performs an all-gather operation on the input tensor along the specified dimension.

    Args:
        input_ (torch.Tensor): Input tensor of shape [B, H, S, D].
        dim (int, optional): Dimension along which to concatenate. Defaults to 1.

    Returns:
        torch.Tensor: Output tensor after all-gather operation, concatenated along 'dim'.
    """
    return _AllGather.apply(input_, dim, group)
