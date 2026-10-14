# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two cards, one process (Windows XPU): both tensor-parallel workers as threads of the engine's process.

Select it with distributed_executor_backend="vllm.v1.executor.oneproc_executor.OneProcExecutor" and
tensor_parallel_size=2. Why and how: distributed/device_communicators/xpu_oneproc.py. In short, the decode all_reduce
between two processes on Windows has to be relayed by a CPU helper (28 us), while two cards in one process exchange
through one pinned host buffer directly (7 us), and the ranks are kept in step by a baton that changes hands at every
collective.

Each rank has its own worker thread (torch's current device is per thread), its own copy of the config (each rank's
model registers its layers into the config's static forward context, as in separate processes), and its own copy of the
module globals that hold per-rank state (swapped by the baton). An RPC runs on both rank threads; rank 0 starts with the
baton, and the result of `unique_reply_rank` (or the list of both) comes back as with the multiprocess executor.
"""

import copy
import os
import queue
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import Future
from typing import Any

import torch

from vllm.distributed.device_communicators import xpu_oneproc
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.executor.abstract import Executor
from vllm.v1.executor.uniproc_executor import AsyncOutputFuture
from vllm.v1.outputs import AsyncModelRunnerOutput
from vllm.v1.serial_utils import run_method
from vllm.v1.worker.worker_base import WorkerWrapperBase

logger = init_logger(__name__)

# a rank that has failed leaves the other waiting in a collective that will never complete; give the survivor this long
# to notice (the cards' polls give up after ~1 s) before the RPC raises without it
_SURVIVOR_GRACE_S = 60.0


# VLLM_XPU_ONEPROC_TIMING=1 (2026-10-13): each rank's own time inside every RPC (baton waits taken out), printed per
# method every 500 calls - how much host work the two ranks do in turn. Measured around the whole task, never by waiting
# on a card (a timer that synchronized a card inside a step broke the lockstep and stalled the run, 18:20).
_TIMING = os.environ.get("VLLM_XPU_ONEPROC_TIMING") == "1"
_timing: dict = {}


def _timing_add(rank: int, label: str, seconds: float) -> None:
    t = _timing.setdefault((label, rank), [0.0, 0])
    t[0] += seconds
    t[1] += 1
    if t[1] % 500 == 0:
        print(f"[oneproc timing] {label} rank {rank}: {1e3 * t[0] / t[1]:.2f} ms host per call over {t[1]} calls",
              flush=True)


class _RankThread(threading.Thread):
    def __init__(self, rank: int):
        super().__init__(name=f"oneproc-rank{rank}", daemon=True)
        self.rank = rank
        self.tasks: queue.Queue = queue.Queue()
        self.busy = False      # inside a task: a rank stuck in a collective its failed peer never joins stays busy

    def run(self) -> None:
        xpu_oneproc.set_thread_rank(self.rank)
        torch.xpu.set_device(self.rank)
        baton = xpu_oneproc.baton()
        while True:
            task = self.tasks.get()
            if task is None:
                return
            fn, done, label = task
            self.busy = True
            baton.acquire(self.rank)
            result, err = None, None
            t0, w0 = time.perf_counter(), baton.wait_s[self.rank]
            try:
                result = fn()   # an AsyncModelRunnerOutput stays unresolved: see collective_rpc
            except BaseException as e:  # noqa: BLE001
                err = e
                logger.error("one-process rank %d failed:\n%s", self.rank, traceback.format_exc())
            finally:
                if _TIMING and label:
                    _timing_add(self.rank, label, time.perf_counter() - t0 - (baton.wait_s[self.rank] - w0))
                baton.finish(self.rank)
                self.busy = False
            done(self.rank, result, err)


class OneProcExecutor(Executor):
    def _init_executor(self) -> None:
        pc = self.parallel_config
        if pc.world_size != 2 or pc.tensor_parallel_size != 2:
            raise ValueError("the one-process executor runs tensor_parallel_size=2 on two cards, nothing else "
                             f"(got world_size {pc.world_size}, tp {pc.tensor_parallel_size})")
        if not current_platform.is_xpu():
            raise ValueError("the one-process executor is for XPU")
        xpu_oneproc.enable()
        xpu_oneproc.install_threaded_process_groups()
        xpu_oneproc.record_initial_values()
        xpu_oneproc.baton()
        self.world_size = 2
        self.output_rank = 0
        self.threads = [_RankThread(r) for r in range(2)]
        for t in self.threads:
            t.start()
        self.workers: list[WorkerWrapperBase | None] = [None, None]
        # each rank its own config: the model registers its layers in the config's static forward context by name
        configs = [copy.deepcopy(self.vllm_config) for _ in range(2)]
        lock = threading.Lock()

        def make(r: int) -> Callable[[], None]:
            def init() -> None:
                w = WorkerWrapperBase(rpc_rank=r, global_rank=r)
                all_kwargs: list[dict] = [{}, {}]
                all_kwargs[r] = {
                    "vllm_config": configs[r],
                    "local_rank": r,
                    "rank": r,
                    "distributed_init_method": "threaded://in-process",
                    "is_driver_worker": r == 0,
                    "shared_worker_lock": lock,
                }
                w.init_worker(all_kwargs)
                self.workers[r] = w
            return init

        t0 = time.time()
        self._run_each([make(0), make(1)])
        self._run_all(lambda w: w.init_device())
        logger.info("one-process executor: both workers up, process groups ready (%.1f s)", time.time() - t0)
        self._run_all(lambda w: w.load_model())
        current_platform.update_block_size_for_backend(self.vllm_config)
        self._run_each([lambda r=r: current_platform.update_block_size_for_backend(configs[r]) for r in range(2)])
        logger.info("one-process executor: model loaded on both cards (%.1f s)", time.time() - t0)

    # ---- running work on both rank threads
    def _run_each(self, fns: list[Callable[[], Any]], label: str = "") -> list[Any]:
        """fns[r] on rank r's thread, rank 0 starting with the baton; both results, or the first error raised."""
        results: list[Any] = [None, None]
        errors: list[BaseException | None] = [None, None]
        done_count = [0]
        cv = threading.Condition()

        def done(r: int, result: Any, err: BaseException | None) -> None:
            with cv:
                results[r] = result
                errors[r] = err
                done_count[0] += 1
                cv.notify_all()

        xpu_oneproc.baton().begin([0, 1])
        for r in range(2):
            self.threads[r].tasks.put((fns[r], done, label))
        with cv:
            while done_count[0] < 2:
                cv.wait(timeout=1.0)
                if done_count[0] == 1 and any(e is not None for e in errors):
                    # one rank failed; the other may be stuck waiting for it
                    if not cv.wait_for(lambda: done_count[0] == 2, timeout=_SURVIVOR_GRACE_S):
                        break
        for r in range(2):
            if errors[r] is not None:
                raise RuntimeError(f"one-process executor: rank {r} failed: {errors[r]!r}") from errors[r]
        if done_count[0] < 2:
            raise RuntimeError("one-process executor: a rank never finished")
        return results

    def _run_all(self, fn: Callable[[WorkerWrapperBase], Any], label: str = "") -> list[Any]:
        return self._run_each([lambda r=r: fn(self.workers[r]) for r in range(2)], label)

    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        unique_reply_rank: int | None = None,
        kv_output_aggregator: Any = None,
        ec_output_aggregator: Any = None,
    ) -> Any:
        kwargs = kwargs or {}
        try:
            results = self._run_all(lambda w: run_method(w, method, args, kwargs),
                                    method if isinstance(method, str) else "")
        except BaseException as e:  # noqa: BLE001
            if not non_block:
                raise
            f: Future = Future()
            f.set_exception(e)
            return f
        # Async scheduling: a worker returns an AsyncModelRunnerOutput whose host copy is still in flight. Only the
        # reply rank's is ever read (as with the multiprocess executor), and it is resolved when the engine asks for
        # it, on the engine's thread, so the next step's input preparation overlaps this step's GPU work. Safe to wait
        # on there: both ranks have queued everything for this step, so no card can be waiting on an unqueued half.
        if unique_reply_rank is not None:
            out = results[unique_reply_rank]
            if isinstance(out, AsyncModelRunnerOutput):
                if non_block:
                    return AsyncOutputFuture(out, single_value=True)
                return out.get_output()
        else:
            out = [r.get_output() if isinstance(r, AsyncModelRunnerOutput) else r for r in results]
        if not non_block:
            return out
        f = Future()
        f.set_result(out)
        return f

    def execute_model(self, scheduler_output, non_block: bool = False):  # type: ignore[override]
        return self.collective_rpc("execute_model", args=(scheduler_output,), non_block=non_block,
                                   unique_reply_rank=self.output_rank)

    def sample_tokens(self, grammar_output, non_block: bool = False):  # type: ignore[override]
        return self.collective_rpc("sample_tokens", args=(grammar_output,), non_block=non_block,
                                   unique_reply_rank=self.output_rank)

    def execute_dummy_batch(self) -> None:
        self.collective_rpc("execute_dummy_batch")

    def take_draft_token_ids(self):
        return self.collective_rpc("take_draft_token_ids", unique_reply_rank=self.output_rank)

    def check_health(self) -> None:
        for t in self.threads:
            if not t.is_alive():
                raise RuntimeError(f"one-process executor: {t.name} is gone")

    def shutdown(self) -> None:
        if getattr(self, "_shut", False):
            return
        self._shut = True
        try:
            # after a failure one rank can be stuck for good in a collective the other never joins; asking it to shut
            # down would wait forever (2026-10-13 17:31: the process outlived its error by minutes), so skip the workers
            stuck = [t.name for t in getattr(self, "threads", []) if t.busy]
            if stuck:
                logger.warning("one-process executor: %s still inside a task; not shutting the workers down", stuck)
            elif all(w is not None for w in getattr(self, "workers", [])):
                self._run_all(lambda w: w.shutdown())
        except Exception as e:  # noqa: BLE001
            logger.warning("one-process executor: worker shutdown failed: %s", e)
        for t in getattr(self, "threads", []):
            t.tasks.put(None)

    @classmethod
    def supports_async_scheduling(cls) -> bool:
        # VLLM_XPU_ONEPROC_ASYNC=0 turns it off for comparison
        return os.environ.get("VLLM_XPU_ONEPROC_ASYNC", "1") != "0"
