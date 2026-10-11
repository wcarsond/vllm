# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Card-to-card collectives for XPU on Windows, relayed through host RAM.

On Linux the XPU backend talks card-to-card through oneCCL (torch's ``xccl``
process-group backend). oneCCL has no Windows build, so on Windows the only
process-group backend that exists is ``gloo``, which moves CPU tensors. Every
operation here is the same shape: copy the XPU tensor to pinned-free host
memory, run the gloo collective on the host copy, copy the result back to the
card. It is correct for any tensor and any group size; it is not fast for the
large all-reduces of tensor parallelism, but the per-token hand-off of
pipeline parallelism is a few kilobytes and costs well under a millisecond.

This module deliberately imports only torch so it can be exercised on its own
(see scripts: server/relay-test.input.py) before the rest of vLLM can load.
"""

import os
import pickle
import sys
import time
from collections import defaultdict
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

# VLLM_XPU_EXCHANGE_STATS=1: count and time every relay operation by name and print a table every 1000
# calls, so the gloo share of a token can be seen next to the shared-memory exchange's share.
_STATS = os.environ.get("VLLM_XPU_EXCHANGE_STATS") == "1"
_calls: dict[str, list[float]] = defaultdict(list)
_total = 0


def _timed(name):
    def deco(fn):
        if not _STATS:
            return fn

        def wrapper(*args, **kwargs):
            global _total
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                _calls[name].append(time.perf_counter() - t0)
                _total += 1
                if _total % 1000 == 0:
                    rows = ", ".join(f"{k}: {len(v)} calls, mean {sum(v)/len(v)*1e6:.0f} us" for k, v in _calls.items())
                    print(f"[gloo relay pid {os.getpid()}] {_total} calls so far: {rows}", file=sys.stderr, flush=True)
        return wrapper
    return deco


def _to_host(t: torch.Tensor) -> torch.Tensor:
    # .contiguous() first: gloo wants dense buffers; .cpu() then copies once.
    return t.detach().contiguous().cpu()


@_timed("send")
def send(t: torch.Tensor, dst: int, group: ProcessGroup) -> None:
    dist.send(_to_host(t), dst, group=group)


@_timed("recv")
def recv(
    size: torch.Size, dtype: torch.dtype, src: int, group: ProcessGroup, device
) -> torch.Tensor:
    host = torch.empty(size, dtype=dtype)
    dist.recv(host, src, group=group)
    return host.to(device, non_blocking=True)


@_timed("all_reduce")
def all_reduce(t: torch.Tensor, group: ProcessGroup) -> torch.Tensor:
    host = _to_host(t)
    dist.all_reduce(host, group=group)
    return host.to(t.device, non_blocking=True)


@_timed("broadcast")
def broadcast(t: torch.Tensor, src: int, group: ProcessGroup) -> torch.Tensor:
    host = _to_host(t)
    dist.broadcast(host, src, group=group)
    t.copy_(host, non_blocking=True)
    return t


@_timed("all_gather")
def all_gather(t: torch.Tensor, world_size: int, group: ProcessGroup) -> torch.Tensor:
    """Concatenation along a new leading dim: result[rank] is rank's tensor."""
    host = _to_host(t)
    scalar = host.dim() == 0
    if scalar:
        host = host.unsqueeze(0)
    # gloo's all_gather_into_tensor splits the output along dim 0 into world_size
    # views and insists each view has the input's shape, so the output must be
    # (world_size * n, ...) and not (world_size, n, ...). The first TP run on
    # Windows (2026-10-11, the logits gather) failed on exactly that check.
    out = torch.empty((world_size * host.shape[0],) + tuple(host.shape[1:]), dtype=host.dtype)
    dist.all_gather_into_tensor(out, host, group=group)
    out = out.view((world_size,) + tuple(host.shape))
    if scalar:
        out = out.squeeze(1)
    return out.to(t.device, non_blocking=True)


@_timed("gather")
def gather(
    t: torch.Tensor, dst: int, is_dst: bool, world_size: int, group: ProcessGroup
) -> list[torch.Tensor] | None:
    host = _to_host(t)
    gather_list = [torch.empty_like(host) for _ in range(world_size)] if is_dst else None
    dist.gather(host, gather_list, dst=dst, group=group)
    if not is_dst:
        return None
    return [g.to(t.device, non_blocking=True) for g in gather_list]


@_timed("reduce_scatter")
def reduce_scatter(
    t: torch.Tensor, chunk: int, world_size: int, group: ProcessGroup
) -> torch.Tensor:
    """Sum across ranks, then each rank keeps its ``chunk`` rows of dim 0."""
    host = _to_host(t)
    assert host.shape[0] == chunk * world_size
    out = torch.empty((chunk,) + tuple(host.shape[1:]), dtype=host.dtype)
    dist.reduce_scatter_tensor(out, host, group=group)
    return out.to(t.device, non_blocking=True)


# --- tensor dict: the pipeline-parallel hand-off ---------------------------------
# Wire format, all on the gloo group: one int64 byte count, the pickled
# metadata (keys in order; for tensors their dtype and shape, for everything
# else the value itself), then each tensor in key order as a host copy.

@_timed("send_tensor_dict")
def send_tensor_dict(
    tensor_dict: dict[str, torch.Tensor | Any], dst: int, group: ProcessGroup
) -> None:
    meta: list[tuple[str, Any]] = []
    tensors: list[torch.Tensor] = []
    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            meta.append((key, ("tensor", value.dtype, tuple(value.shape))))
            tensors.append(value)
        else:
            meta.append((key, ("value", value)))
    payload = torch.frombuffer(bytearray(pickle.dumps(meta)), dtype=torch.uint8)
    dist.send(torch.tensor([payload.numel()], dtype=torch.int64), dst, group=group)
    dist.send(payload, dst, group=group)
    for t in tensors:
        if t.numel() == 0:
            continue
        dist.send(_to_host(t), dst, group=group)


@_timed("recv_tensor_dict")
def recv_tensor_dict(
    src: int, group: ProcessGroup, device
) -> dict[str, torch.Tensor | Any]:
    count = torch.empty(1, dtype=torch.int64)
    dist.recv(count, src, group=group)
    payload = torch.empty(int(count.item()), dtype=torch.uint8)
    dist.recv(payload, src, group=group)
    meta = pickle.loads(payload.numpy().tobytes())
    out: dict[str, Any] = {}
    for key, (kind, *rest) in meta:
        if kind == "value":
            out[key] = rest[0]
            continue
        dtype, shape = rest
        host = torch.empty(shape, dtype=dtype)
        if host.numel() > 0:
            dist.recv(host, src, group=group)
        out[key] = host.to(device, non_blocking=True)
    return out
