# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fast two-card all_reduce for XPU on Windows: host memory shared between the
worker processes, no gloo in the hot path.

Measured on two Arc Pro B60s (2026-10-10, probe/p2p.cpp): the driver offers no
peer access between the cards, but card -> pinned host -> card costs 9.7 us for
7 KB. The gloo relay (xpu_ram_relay.py) spends 233 us on the same all_reduce,
almost all of it in loopback TCP, serialization and Python. A tensor split of
a 7B model does 56 such all_reduces per token, so that difference is the whole
gain of the second card.

Protocol (world size 2, lockstep collectives):
  shared block = header (one 64-byte line per rank holding its sequence number)
                 + per rank two slots (double buffer) of ``slot_bytes``
  all_reduce(t), sequence s:
    1. copy t from the card into my slot[s % 2]            (one DMA)
    2. publish s in my header line
    3. spin until the peer's line reads >= s
    4. sum = my slot + peer slot[s % 2]                     (host add, bf16 exact
       and commutative, so both ranks hold identical bytes)
    5. copy sum back into t                                (one DMA)
Double buffering is enough: a rank only starts s+2 after it has seen the peer
publish s+1, and the peer publishes s+1 only after it finished reading slot
s % 2. Rank 0 creates the block and sends its name over the gloo group once.
"""

import os
import sys
import time
from multiprocessing import shared_memory

# VLLM_XPU_EXCHANGE_STATS=1 prints, every 1000 all_reduces, how many there were, the median/mean cost
# and the bytes moved, so a token's budget can be accounted for from inside a real run.
_STATS = os.environ.get("VLLM_XPU_EXCHANGE_STATS") == "1"
_STATS_EVERY = 1000

import numpy as np
import torch
import torch.distributed as dist

_LINE = 64
_HEADER = 4096


class ShmExchange:
    def __init__(self, rank: int, world_size: int, group, device,
                 slot_bytes: int = 64 << 20):
        assert world_size == 2, "shared-memory exchange is written for two cards"
        self.rank, self.peer, self.world_size = rank, 1 - rank, world_size
        self.device = torch.device(device)
        self.slot_bytes = slot_bytes
        total = _HEADER + world_size * 2 * slot_bytes
        if rank == 0:
            self.shm = shared_memory.SharedMemory(create=True, size=total)
            name = [self.shm.name]
        else:
            name = [None]
        dist.broadcast_object_list(name, src=0, group=group)
        if rank != 0:
            self.shm = shared_memory.SharedMemory(name=name[0])
        buf = self.shm.buf
        # one int64 per rank, each on its own cache line, so the two writers never share a line
        self.lines = np.ndarray((world_size,), dtype=np.int64, buffer=buf, offset=0, strides=(_LINE,))
        if rank == 0:
            self.lines[:] = 0
        self.slots = [
            [np.ndarray((slot_bytes,), dtype=np.uint8, buffer=buf,
                        offset=_HEADER + (r * 2 + b) * slot_bytes) for b in range(2)]
            for r in range(world_size)
        ]
        self.seq = 0
        self._t = []          # per-call seconds when _STATS
        self._bytes = 0
        # The card never touches the shared mapping. Device copies go to and from these private,
        # torch-owned host buffers (pinned when the runtime allows it) and plain memcpys move bytes
        # between them and the shared slots. The first version DMA'd straight into the shared
        # mapping; both workers of a finished run then hung forever in a driver call at teardown
        # (2026-10-11 03:17: one thread each, 100% CPU, mostly kernel mode, unkillable), consistent
        # with the mapping vanishing under a registration the driver still held.
        self._mine = self._host_buffer(slot_bytes)
        self._theirs = self._host_buffer(slot_bytes)
        # both ranks must see the zeroed header before anyone publishes
        dist.barrier(group)

    @staticmethod
    def _host_buffer(nbytes: int) -> torch.Tensor:
        try:
            return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        except Exception:  # noqa: BLE001 - pinned host memory not available on this runtime
            return torch.empty(nbytes, dtype=torch.uint8)

    def _view(self, slot: np.ndarray, t: torch.Tensor) -> torch.Tensor:
        n = t.numel() * t.element_size()
        return torch.frombuffer(slot, dtype=torch.uint8, count=n).view(t.dtype).view(t.shape)

    def all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        if not _STATS:
            return self._all_reduce(t)
        t0 = time.perf_counter()
        out = self._all_reduce(t)
        self._t.append(time.perf_counter() - t0)
        self._bytes += t.numel() * t.element_size()
        if len(self._t) % _STATS_EVERY == 0:
            s = sorted(self._t[-_STATS_EVERY:])
            print(f"[shm exchange rank {self.rank}] {len(self._t)} all_reduces so far; last {_STATS_EVERY}: "
                  f"median {s[len(s)//2]*1e6:.0f} us, mean {sum(s)/len(s)*1e6:.0f} us, p90 {s[int(len(s)*0.9)]*1e6:.0f} us, "
                  f"max {s[-1]*1e6:.0f} us; {self._bytes/1e6:.1f} MB moved in total", file=sys.stderr, flush=True)
        return out

    def _all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        t = t.contiguous()
        nbytes = t.numel() * t.element_size()
        if nbytes > self.slot_bytes:
            raise ValueError(f"tensor of {nbytes} bytes exceeds the exchange slot of {self.slot_bytes}")
        self.seq += 1
        s = self.seq
        b = s & 1
        mine = self._mine[:nbytes].view(t.dtype).view(t.shape)
        mine.copy_(t)                                   # card -> private host buffer, synchronous
        self._view(self.slots[self.rank][b], t).copy_(mine)   # memcpy into my shared slot
        self.lines[self.rank] = s                       # publish
        spins = 0
        while self.lines[self.peer] < s:                # wait for the peer's half
            spins += 1
            if spins > 2000:
                time.sleep(0)
        theirs = self._theirs[:nbytes].view(t.dtype).view(t.shape)
        theirs.copy_(self._view(self.slots[self.peer][b], t))  # memcpy out of the peer's shared slot
        total = mine + theirs                           # host add; same two operands on both ranks -> identical result
        t.copy_(total)                                  # private host buffer -> card
        return t

    def close(self) -> None:
        try:
            self.shm.close()
            if self.rank == 0:
                self.shm.unlink()
        except Exception:  # noqa: BLE001
            pass
