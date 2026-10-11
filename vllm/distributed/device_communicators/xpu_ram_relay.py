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

import pickle
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


def _to_host(t: torch.Tensor) -> torch.Tensor:
    # .contiguous() first: gloo wants dense buffers; .cpu() then copies once.
    return t.detach().contiguous().cpu()


def send(t: torch.Tensor, dst: int, group: ProcessGroup) -> None:
    dist.send(_to_host(t), dst, group=group)


def recv(
    size: torch.Size, dtype: torch.dtype, src: int, group: ProcessGroup, device
) -> torch.Tensor:
    host = torch.empty(size, dtype=dtype)
    dist.recv(host, src, group=group)
    return host.to(device, non_blocking=True)


def all_reduce(t: torch.Tensor, group: ProcessGroup) -> torch.Tensor:
    host = _to_host(t)
    dist.all_reduce(host, group=group)
    return host.to(t.device, non_blocking=True)


def broadcast(t: torch.Tensor, src: int, group: ProcessGroup) -> torch.Tensor:
    host = _to_host(t)
    dist.broadcast(host, src, group=group)
    t.copy_(host, non_blocking=True)
    return t


def all_gather(t: torch.Tensor, world_size: int, group: ProcessGroup) -> torch.Tensor:
    """Concatenation along a new leading dim: result[rank] is rank's tensor."""
    host = _to_host(t)
    out = torch.empty((world_size,) + tuple(host.shape), dtype=host.dtype)
    dist.all_gather_into_tensor(out, host, group=group)
    return out.to(t.device, non_blocking=True)


def gather(
    t: torch.Tensor, dst: int, is_dst: bool, world_size: int, group: ProcessGroup
) -> list[torch.Tensor] | None:
    host = _to_host(t)
    gather_list = [torch.empty_like(host) for _ in range(world_size)] if is_dst else None
    dist.gather(host, gather_list, dst=dst, group=group)
    if not is_dst:
        return None
    return [g.to(t.device, non_blocking=True) for g in gather_list]


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
