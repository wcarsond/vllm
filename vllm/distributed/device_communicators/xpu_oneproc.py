# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One process, two cards, on Windows XPU: the baton, the per-rank state swap, and the shared exchange.

Why one process (2026-10-13): on Windows a kernel cannot touch another process's memory and IPC handles are refused for
both device and host memory (probe/host-ipc), so the two-process exchange relays every decode all_reduce through a CPU
helper in each process (28 us). With both cards in one process, both cards' kernels read and write the same pinned host
memory directly: 7.0 us for the 7B's decode all_reduce (probe/one-process-allreduce).

How the two ranks share one process. Each rank runs in its own thread (torch's current device is per thread), but only
the thread holding the BATON runs; the baton changes hands
  - at every collective, right after the rank has queued it (yield_): the other rank then runs up to the same collective
    and hands it back. So whenever a rank later blocks its host on its own card (a .tolist() in sampling, a synchronize in
    memory profiling), the other rank has already queued every half that card is waiting for. Without this, one thread
    running the ranks one after the other deadlocks at the first such wait;
  - after every graph replay (the replayed graph holds collectives that Python never sees);
  - around every blocking wait that needs the other rank to make progress (the threaded process group's rendezvous, its
    store barrier): the waiting rank gives the baton up (blocking) and takes it back afterwards.
Two threads running torch at the same time hang in oneDNN matmuls on this driver (probe/threads-hang: elementwise ops run
in parallel fine, any matmul hangs, warm or cold), so the strict alternation is also what keeps the two cards' matmul
launches from ever overlapping.

Module globals that differ per rank (the process groups, the forward context, the workspace manager, ...) are saved when a
rank gives the baton up and restored when it takes it back (SWAP below), so code that reads them sees its own rank's.
"""

import importlib
import os
import sys
import threading
import time
from contextlib import contextmanager

_ENABLED = os.environ.get("VLLM_XPU_ONE_PROCESS") == "1"
# VLLM_XPU_ONEPROC_ALIGN=1: hand the baton over before every graph replay too, so both cards start their graphs together.
# Off by default: with async scheduling it changed nothing (7B decode 56.5 on, 56.6 off, 2026-10-13 18:40) and adds two
# hand-overs per step. The hand-over AFTER a replay is not optional (the graph holds exchanges Python never sees).
ALIGN_REPLAYS = os.environ.get("VLLM_XPU_ONEPROC_ALIGN", "0") == "1"


def enabled() -> bool:
    return _ENABLED


def enable() -> None:
    """Called by the one-process executor before any worker exists."""
    global _ENABLED
    _ENABLED = True
    os.environ["VLLM_XPU_ONE_PROCESS"] = "1"


# (module, attribute) pairs that hold per-rank state. Missing modules/attributes are skipped.
SWAP = [
    ("vllm.distributed.parallel_state", a)
    for a in ("_WORLD", "_INNER_DP_WORLD", "_NODE_COUNT", "_TP", "_ETP", "_ENGRAM_DP", "_DCP", "_PP", "_DP", "_EP",
              "_EPLB", "_PCP")
] + [
    ("vllm.config.vllm", "_current_vllm_config"),
    ("vllm.config.vllm", "_current_prefix"),
    ("vllm.forward_context", "_forward_context"),
    ("vllm.v1.worker.workspace", "_manager"),
    ("vllm.utils.torch_utils", "_aux_stream"),
    ("vllm.compilation.monitor", "cudagraph_capturing_enabled"),
    ("vllm.compilation.backends", "model_tag"),
    ("vllm.compilation.backends", "model_is_encoder"),
]
# Module-level caches of objects that hold card tensors, keyed by something other than the card: each rank starts from
# an EMPTY one of its own. 2026-10-13 17:30: _ROPE_DICT handed rank 1 rank 0's rotary embedding, whose cos/sin table
# then crossed between the cards every layer in eager and failed outright inside a graph capture.
SWAP_FRESH = [
    ("vllm.model_executor.layers.rotary_embedding", "_ROPE_DICT"),
    ("vllm.model_executor.layers.hpc.rope_norm", "_hpc_rope_norm_instances"),
]
SWAP = SWAP + SWAP_FRESH

_tls = threading.local()


def current_rank() -> int | None:
    """The rank of the calling thread, or None outside the rank threads."""
    return getattr(_tls, "rank", None)


def set_thread_rank(rank: int | None) -> None:
    _tls.rank = rank


class Baton:
    """Strict hand-over between the two rank threads; see the module docstring.

    The holder lives in a C++ word (_xpu_exchange1p.Turn) that a waiting thread spins on without the GIL: a Python
    condition variable cost ~85 us per hand-over on Windows. -1 means nobody holds it (the next waiter claims it).
    A rank that is blocked in a wait still counts as the one to hand to: in an SPMD program the other rank can only be
    past that wait if the wait is over, so a 'blocked' flag seen from the other side is stale, and skipping the hand-over
    there would let the running rank block its host on a half the other rank has not queued yet."""

    def __init__(self, n: int = 2):
        from . import _xpu_exchange1p

        assert n == 2, "the baton is written for two ranks"
        self.n = n
        self.turn = _xpu_exchange1p.Turn()
        self.active = [False] * n
        self.saved: list[dict] = [dict() for _ in range(n)]
        self.handoffs = 0
        self.wait_s = [0.0] * n      # time each rank spent waiting for the baton in yield_ (perf_counter seconds)

    # ---- per-rank module state
    def _save(self, r: int) -> None:
        d = self.saved[r]
        for mod, attr in SWAP:
            m = sys.modules.get(mod)
            if m is not None and hasattr(m, attr):
                d[(mod, attr)] = getattr(m, attr)

    def _restore(self, r: int) -> None:
        d = self.saved[r]
        for mod, attr in SWAP:
            m = sys.modules.get(mod)
            if m is None:
                continue
            key = (mod, attr)
            if key in d:
                setattr(m, attr, d[key])
            elif hasattr(m, attr):
                # first time this rank sees a module imported while the other rank held the baton: start from the
                # module's import-time value, not the other rank's
                setattr(m, attr, _initial_value(mod, attr, getattr(m, attr)))

    def _other(self, r: int) -> int:
        o = 1 - r
        return o if self.active[o] else -1

    # ---- one collective_rpc
    def begin(self, ranks) -> None:
        for r in ranks:
            self.active[r] = True
        self.turn.set(min(ranks))

    def acquire(self, r: int) -> None:
        self.turn.wait_for(r)
        self._restore(r)

    def finish(self, r: int) -> None:
        held = self.turn.get() == r
        if held:
            self._save(r)
        self.active[r] = False
        if held:
            self.turn.set(self._other(r))

    def yield_(self, r: int) -> None:
        if self.turn.get() != r:
            return
        o = self._other(r)
        if o < 0:
            return
        self._save(r)
        self.handoffs += 1
        t0 = time.perf_counter()
        self.turn.pass_and_wait(r, o)
        self.wait_s[r] += time.perf_counter() - t0
        self._restore(r)

    @contextmanager
    def blocking(self, r: int):
        held = self.turn.get() == r
        if held:
            self._save(r)
            self.turn.set(self._other(r))
        try:
            yield
        finally:
            if held:
                self.turn.wait_for(r)
                self._restore(r)


_initial: dict = {}


def _initial_value(mod, attr, current):
    if (mod, attr) in SWAP_FRESH:
        return type(current)()
    return _initial.get((mod, attr), current)


def record_initial_values() -> None:
    """Remember every SWAP attribute's value before any rank runs (the 'nothing set yet' state each rank starts from)."""
    for mod, attr in SWAP:
        try:
            m = importlib.import_module(mod)
        except Exception:  # noqa: BLE001
            continue
        if hasattr(m, attr):
            _initial[(mod, attr)] = getattr(m, attr)


BATON: Baton | None = None


def baton() -> Baton:
    global BATON
    if BATON is None:
        BATON = Baton(2)
    return BATON


def yield_baton() -> None:
    """Hand the baton to the other rank (if this is a rank thread holding it); return once it is handed back.

    Not while this rank is recording a graph: nothing runs during a capture, so there is nothing to keep in step, and
    torch refuses to start the other card's capture while this one is open ("Cannot register the state during capturing
    stage", 2026-10-13). Each rank therefore records a whole graph in one turn."""
    r = current_rank()
    if r is not None and BATON is not None:
        if _capturing():
            return
        BATON.yield_(r)


def _capturing() -> bool:
    import torch

    try:
        return torch.xpu.graphs.is_current_stream_capturing()
    except Exception:  # noqa: BLE001
        return False


@contextmanager
def blocking_wait():
    """Give the baton up around a wait that needs the other rank to run."""
    r = current_rank()
    if r is None or BATON is None:
        yield
        return
    with BATON.blocking(r):
        yield


def exchange(t, mode: int):
    """all_reduce (mode 0) or flattened all_gather (mode 1) of `t` with the other card's tensor of the same call.

    Eager: queue my half, hand the baton over (the other rank runs to the same call and queues its half), then queue
    the wait-and-combine - so a card never starts waiting before the other card's half is at least queued (a waiting
    kernel gives up after ~1 s). While recording a graph: one fused kernel and no hand-over (nothing runs during a
    capture; the replays are queued back to back)."""
    p = pair()
    nbytes = t.numel() * t.element_size()
    if nbytes > _SLOT_BYTES:
        # larger than a slot (2026-10-13 17:16: the logits gather of memory profiling, 256 sequences x half the
        # vocabulary = 38.9 MB): in slot-size pieces, the same pieces on both ranks
        import torch

        flat = t.reshape(-1)
        step = (_SLOT_BYTES // t.element_size()) & ~1
        parts = [exchange(flat[i:i + step].contiguous(), mode) for i in range(0, flat.numel(), step)]
        if mode == 0:
            return torch.cat(parts).view(t.shape)
        halves = [(q[: q.numel() // 2], q[q.numel() // 2:]) for q in parts]
        return torch.cat([h[0] for h in halves] + [h[1] for h in halves])
    if current_rank() is None or BATON is None or _capturing():
        out = p.all_reduce(t) if mode == 0 else p.all_gather(t)
        yield_baton()
        return out
    out = p.start(t, mode)
    yield_baton()
    p.finish(t, out, mode)
    return out


_STORE = None


def store():
    """The in-process key-value store both rank threads' process groups rendezvous through."""
    global _STORE
    with _PAIR_LOCK:
        if _STORE is None:
            import tempfile

            import torch.distributed as dist

            # HashStore is not built into torch on Windows; a FileStore on a fresh temp file does the same job
            path = os.path.join(tempfile.gettempdir(), f"vllm-oneproc-store-{os.getpid()}-{id(threading.current_thread())}")
            if os.path.exists(path):
                os.remove(path)
            _STORE = dist.FileStore(path, 2)
        return _STORE


_PAIR = None
_PAIR_LOCK = threading.Lock()
# per card: 2 pinned slots + 2 card-memory slots; 32 MB holds a 4096-token prefill chunk of the 7B's all_reduce (29 MB)
_SLOT_BYTES = int(os.environ.get("VLLM_XPU_EXCHANGE_SLOT_MB", "32")) << 20


def pair():
    """The one exchange both ranks share (built on first use)."""
    global _PAIR
    with _PAIR_LOCK:
        if _PAIR is None:
            from . import _xpu_exchange1p

            _PAIR = _xpu_exchange1p.Pair(_SLOT_BYTES)
        return _PAIR


_patched = False


def install_threaded_process_groups():
    """torch's thread-per-rank process group (its test backend 'threaded'), with its blocking points giving the baton up.

    The rendezvous of every collective (Collective.join) and the store barrier run when a group is created
    (_store_based_barrier) block until the other rank arrives, so they run under blocking_wait()."""
    global _patched
    if _patched:
        return
    from torch.testing._internal.distributed import multi_threaded_pg as mtpg

    mtpg._install_threaded_pg()
    join = mtpg.Collective.join

    def join_giving_up_baton(self, rank, data):
        with blocking_wait():
            return join(self, rank, data)

    mtpg.Collective.join = join_giving_up_baton
    barrier = mtpg._store_based_barrier

    def store_barrier_giving_up_baton(*args, **kwargs):
        with blocking_wait():
            return barrier(*args, **kwargs)

    mtpg._store_based_barrier = store_barrier_giving_up_baton

    # torch also enters every group into one process-wide C++ table by name, and the two rank threads' worlds name their
    # groups identically ("0", "1", ...): the second rank's registration raised "A process group is already registered
    # under the name 0" (2026-10-13 17:06). The table only serves torch's functional collectives (name -> group), which
    # vLLM's own collective ops do not use, so the first rank's entry is kept and the duplicate skipped.
    import torch.distributed.distributed_c10d as c10d

    register = c10d._register_process_group

    def register_once(group_name, pg):
        try:
            register(group_name, pg)
        except RuntimeError as e:
            if "already registered" not in str(e):
                raise

    c10d._register_process_group = register_once
    _patched = True
