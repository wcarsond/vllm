# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-card all_reduce for XPU on Windows that returns without waiting.

Wraps ``_xpu_exchange.pyd`` (xpu_exchange_ext/xpu_exchange.cpp, built by
kernels/build-exchange.input.cmd). Per call the C++ side enqueues two small
kernels on the current XPU stream and hands the 7 KB hop between the two
processes to a helper thread; the Python thread, which is what caps the
tensor split at 32 tok/s with the host-synchronous exchange, gets back to
launching the next layer at once. Probe cost: ~11 us per all_reduce, hidden
under the layer (PORT-NOTES.md, 2026-10-11 08:36).

Rank 0 creates the named shared block and sends its name over the gloo group
once; rank 1 opens it. Tensors larger than a slot, or of an unsupported dtype,
are refused by ``accepts`` and the caller falls back to the gloo relay.
"""

import os
import uuid

import torch
import torch.distributed as dist

from vllm.logger import init_logger

logger = init_logger(__name__)

_DTYPES = (torch.float32, torch.bfloat16, torch.float16)
# VLLM_XPU_EXCHANGE_TRACE=1 logs the first 200 all_reduce calls (shape, dtype, bytes) per rank;
# VLLM_XPU_EXCHANGE_STATS=1 logs the helper's stage timings now and then
_TRACE = os.environ.get("VLLM_XPU_EXCHANGE_TRACE") == "1"
_STATS = os.environ.get("VLLM_XPU_EXCHANGE_STATS") == "1"


class GpuExchange:
    # 32 MB slot: vLLM's default prefill chunk is 2048 tokens; at hidden 5120 (14B/32B-class, the models two cards are
    # for) a chunk's all_reduce is 21 MB. Larger tensors still fall back to the gloo relay. Per rank this costs
    # 4 x 32 MB pinned + 2 x 32 MB card memory for the big path, plus 2 x 256 KB for the tagged small path.
    def __init__(self, rank: int, world_size: int, group, device, slot_bytes: int = 32 << 20):
        from . import _xpu_exchange  # the .pyd beside this file

        assert world_size == 2, "the Windows XPU exchange is written for two cards"
        dev = torch.device(device)
        index = dev.index if dev.index is not None else torch.xpu.current_device()
        self.slot_bytes = slot_bytes
        name = [f"Local\\vllm-xpu-exchange-{os.getpid()}-{uuid.uuid4().hex}"] if rank == 0 else [None]
        # rank 0 must have created the block before rank 1 tries to open it, so create first and broadcast after
        if rank == 0:
            self.ext = _xpu_exchange.Exchange(rank, world_size, name[0], slot_bytes, index)
        dist.broadcast_object_list(name, src=0, group=group)
        if rank != 0:
            self.ext = _xpu_exchange.Exchange(rank, world_size, name[0], slot_bytes, index)
        dist.barrier(group)
        self._calls = 0
        # Warm up every kernel variant now, both ranks together, so no first-use JIT compile (hundreds of ms) can land
        # inside a real all_reduce, and check each result: a wrong sum here must stop the engine, not a token later.
        for dtype in _DTYPES:
            for numel in (256, 65536):  # one-work-group path (<= 64 KB) and the wide path
                t = torch.full((numel,), 1.0 if rank == 0 else 2.0, dtype=dtype, device=dev)
                t = self.ext.all_reduce_(t)
                torch.xpu.synchronize(index)
                if not bool((t == 3.0).all()):
                    wrong = (t != 3.0).nonzero().flatten()
                    raise RuntimeError(f"Windows XPU exchange: warm-up all_reduce of {numel} x {dtype} came back wrong: "
                                       f"first values {t[:4].float().tolist()}, {int(wrong.numel())} of {numel} wrong at "
                                       f"{wrong[:12].tolist()}..., state {self.ext.debug()}")
                dist.barrier(group)
        logger.info_once("Windows XPU exchange: GPU-timeline all_reduce up and checked (slot %d MB)", slot_bytes >> 20)

    def accepts(self, t: torch.Tensor) -> bool:
        nbytes = t.numel() * t.element_size()
        return t.dtype in _DTYPES and nbytes <= self.slot_bytes and nbytes % 4 == 0

    def try_all_reduce(self, t: torch.Tensor):
        """The hot path: one C++ call that checks dtype, size and layout itself and returns None (doing nothing) when
        the tensor is not for the exchange, else the reduced tensor (a new one: vLLM's all_reduce is a functional custom
        op and torch.compile reuses the input right after). Raises if an earlier exchange timed out (peer gone)."""
        try:
            return self.ext.try_all_reduce_(t)
        except RuntimeError:
            logger.error("xpu exchange state at failure: %s", self.ext.debug())
            raise

    def all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        """Returns the reduced tensor (new). Raises if an earlier exchange timed out (peer gone)."""
        t = t.contiguous()
        if _TRACE or _STATS:
            self._calls += 1
            if _TRACE and self._calls <= 200:
                logger.info("xpu exchange call %d: %s %s (%d B)", self._calls, tuple(t.shape), t.dtype, t.numel() * t.element_size())
            if _STATS and (self._calls in (50, 60, 100) or self._calls % 2000 == 0):
                logger.info("xpu exchange helper stage times (seq, bytes, wait_for_card, publish, wait_for_peer, write_out) us: %s",
                            [(s, b, round(a), round(c), round(d), round(e)) for s, b, a, c, d, e, _ in self.ext.stats(6)])
        try:
            return self.ext.all_reduce_(t)
        except RuntimeError:
            logger.error("xpu exchange state at failure: %s", self.ext.debug())
            raise

    def status(self):
        return self.ext.status()

    def stats(self, n: int = 16):
        """Helper stage timings (us) for the last n all_reduces: (seq, bytes, wait_for_card, memcpy_in, wait_for_peer, memcpy_out, enqueue_copies)."""
        return self.ext.stats(n)

    def close(self) -> None:
        try:
            self.ext.close()
        except Exception:  # noqa: BLE001
            pass
