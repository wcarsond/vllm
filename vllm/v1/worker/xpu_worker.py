# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import gc
import os
import sys

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.mem_utils import MemorySnapshot, format_gib
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.worker.gpu_worker import Worker, init_worker_distributed_environment
from vllm.v1.worker.xpu_model_runner import (
    XPUMMEncoderModelRunner,
    XPUModelRunner,
    XPUModelRunnerV2,
)

from .utils import request_memory

logger = init_logger(__name__)


class XPUWorker(Worker):
    """A XPU worker class."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ):
        super().__init__(
            vllm_config, local_rank, rank, distributed_init_method, is_driver_worker
        )
        device_config = self.device_config
        assert device_config.device_type == "xpu"
        assert current_platform.is_xpu()

    @staticmethod
    def _apply_working_set_ceiling() -> None:
        """VLLM_XPU_WS_LIMIT_GB=<n>: hard-cap this worker's resident pages at n GiB (Windows).

        The concurrent-load fault (0xC0000005 in the driver, 2026-10-11) needs two
        tensor-parallel workers each streaming a checkpoint through host memory beside a
        nearly full card on a 32 GB box; the driver's counters show nothing spilled from
        the cards, so the pressure is on the host. Releasing the file pages between shards
        helped one load in three - not a fix, because within a shard the mapping is
        resident again at once. This is the continuous form of the same idea (Carson:
        "dial back ... to just barely not make paging files"): a working-set ceiling the
        OS enforces every instant, so a worker never holds more than n GiB of resident
        pages no matter how fast it reads. Off unless the variable is set.
        """
        import ctypes
        import sys

        gb = os.environ.get("VLLM_XPU_WS_LIMIT_GB")
        if sys.platform != "win32" or not gb:
            return
        k32 = ctypes.windll.kernel32
        QUOTA_LIMITS_HARDWS_MIN_DISABLE = 0x2
        QUOTA_LIMITS_HARDWS_MAX_ENABLE = 0x4
        max_bytes = ctypes.c_size_t(int(float(gb) * (1 << 30)))
        min_bytes = ctypes.c_size_t(64 << 20)
        ok = k32.SetProcessWorkingSetSizeEx(
            k32.GetCurrentProcess(), min_bytes, max_bytes,
            QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_ENABLE)
        logger.info(
            "Windows XPU: hard working-set ceiling %s GiB -> %s",
            gb, "set" if ok else f"FAILED ({ctypes.get_last_error()})")

    def load_model(self, *args, **kwargs) -> None:
        """Load weights one worker at a time on Windows.

        Two processes copying large amounts of host memory to their own cards at the same
        time kill one of themselves with an access violation (0xC0000005) on this driver.
        Reproduced with no vLLM in it at all (probe/mmap-readers.input.py, 2026-10-11):
        two processes, one card each, reading a 41.6 GB checkpoint and copying every
        tensor across. One process doing the identical work is fine; two processes reading
        the same files WITHOUT the device copies are fine; copying from ordinary heap
        memory instead of a file mapping fails just the same. So it is the concurrency of
        the host-to-device traffic, not the mapping, the files, or the volume.

        It needs a model big enough for the two loads to overlap for long enough, which is
        why Qwen2.5-72B-AWQ (39 GB) hits it every time and smaller models never did.

        Serializing the load costs wall-clock time on a cold start and nothing else. A
        named mutex is used rather than the executor's `shared_worker_lock` because that
        lock is consumed by the multimodal cache before the worker object exists.
        """
        import sys

        # VLLM_XPU_NO_LOAD_LOCK=1 turns the serialization off, to reproduce the fault on
        # purpose while the mechanism is still open.
        # Since 2026-10-11 19:10 the default is the parallel load with a private copy per tensor
        # (weight_utils._xpu_private_copy_wanted). This serialized load is the fallback, opt-in
        # with VLLM_XPU_LOAD_LOCK=1; VLLM_XPU_NO_LOAD_LOCK is kept for the old scripts.
        if (sys.platform != "win32" or os.environ.get("VLLM_XPU_LOAD_LOCK") != "1"
                or os.environ.get("VLLM_XPU_NO_LOAD_LOCK") == "1"):
            return super().load_model(*args, **kwargs)

        import ctypes

        k32 = ctypes.windll.kernel32
        handle = k32.CreateMutexW(None, False, "Local\\vllm-xpu-weight-load")
        if not handle:
            logger.warning(
                "Windows XPU: could not create the weight-loading mutex (%s); loading "
                "concurrently, which can crash a worker with 0xC0000005",
                ctypes.get_last_error(),
            )
            return super().load_model(*args, **kwargs)
        # 0xFFFFFFFF = INFINITE. 0x80 = WAIT_ABANDONED, i.e. a previous holder died while
        # holding it; we still own it now, so carry on.
        waited = k32.WaitForSingleObject(handle, 0xFFFFFFFF)
        if waited == 0x80:
            logger.warning(
                "Windows XPU: the previous holder of the weight-loading mutex died; "
                "continuing."
            )
        try:
            logger.info_once(
                "Windows XPU: loading weights under a cross-worker lock (concurrent "
                "host-to-device copies fault this driver; see xpu_worker.load_model)"
            )
            torch.xpu.reset_peak_memory_stats()
            return super().load_model(*args, **kwargs)
        finally:
            # Does loading actually overflow the card? The standalone reproducer only
            # faults once it retains past the card into system memory, but a worker here
            # holds well under the card's capacity, so either a transient peak crosses
            # the line or that recipe does not describe this case. Measure it rather than
            # assume (2026-10-11 17:15, Carson's look-ahead-carry idea depends on it).
            try:
                peak = torch.xpu.max_memory_allocated() / 2**30
                total = torch.xpu.get_device_properties(
                    self.device).total_memory / 2**30
                logger.info(
                    "Windows XPU rank %s: peak device allocation during load %.2f GiB "
                    "of %.2f GiB on the card (%.0f%%)",
                    getattr(self, "rank", "?"), peak, total, 100.0 * peak / total,
                )
            except Exception:  # noqa: BLE001
                pass
            k32.ReleaseMutex(handle)
            k32.CloseHandle(handle)

    def init_device(self):
        self._apply_working_set_ceiling()
        # 15:52: say out loud what this worker actually inherited. Every reproduction of
        # the concurrent-copy access violation outside vLLM needs ZE_AFFINITY_MASK set,
        # and the claim "vLLM does not set it" was read off the source rather than
        # measured. A value leaking in from a shell or an earlier script would make one
        # bug out of what currently looks like two.
        import os as _os
        import sys as _sys

        if _sys.platform == "win32":
            logger.info(
                "Windows XPU worker rank %s: ZE_AFFINITY_MASK=%r, "
                "ONEAPI_DEVICE_SELECTOR=%r, visible devices=%s",
                getattr(self, "rank", "?"),
                _os.environ.get("ZE_AFFINITY_MASK"),
                _os.environ.get("ONEAPI_DEVICE_SELECTOR"),
                torch.xpu.device_count(),
            )

        # Windows: give every worker its own Triton cache directory.
        #
        # Triton compiles a kernel into a shared cache (~/.triton/cache/<hash>/) and publishes each
        # artifact by writing a temporary file and renaming it into place. On POSIX that rename is
        # atomic and two processes compiling the same kernel simply race harmlessly. On Windows a
        # rename onto an existing, open file raises PermissionError, so two workers compiling the
        # SAME kernel at the SAME moment kill one of each other. With eager execution the model's
        # own Triton kernels (not Inductor's - compiler_interface.py already per-ranks those) go
        # straight to the default directory, so tensor parallel hits this whenever both ranks reach
        # a new kernel together, which is exactly what two symmetric ranks do.
        #
        # Seen 2026-10-11 14:30 on Qwen3.5-27B, whose Gated DeltaNet layers compile Triton kernels
        # at load: "PermissionError: [Errno 13] Permission denied: ...\\_fused_post_conv_kernel.json"
        # from rank 1 while rank 0 wrote the same file. Per-rank directories cost one recompile per
        # rank on a cold cache and remove the race entirely.
        # One process, two cards: the environment is shared by both rank threads and only one runs at a time, so one
        # cache is both safe and right (a per-rank variable would just be overwritten by the other rank).
        from vllm.distributed.device_communicators import xpu_oneproc

        if sys.platform == "win32" and not xpu_oneproc.enabled():
            base = os.environ.get("TRITON_CACHE_DIR") or os.path.join(
                os.path.expanduser("~"), ".triton", "cache"
            )
            if not base.endswith(f"rank{self.rank}"):
                per_rank = os.path.join(base, f"rank{self.rank}")
                os.makedirs(per_rank, exist_ok=True)
                os.environ["TRITON_CACHE_DIR"] = per_rank
                logger.info_once(
                    "Windows XPU: per-rank Triton cache at %s (a shared cache races on "
                    "Windows renames; see xpu_worker.init_device)",
                    per_rank,
                )

        # In DP mode, XPU workers see all visible devices.
        # Offset local_rank by the local DP shard.
        parallel_config = self.parallel_config
        if (
            parallel_config.distributed_executor_backend
            not in ("ray", "external_launcher")
            and parallel_config.data_parallel_backend != "ray"
            and (
                parallel_config.data_parallel_external_lb
                or parallel_config.nnodes_within_dp == 1
            )
        ):
            dp_local_rank = parallel_config.data_parallel_rank_local
            if dp_local_rank is None:
                dp_local_rank = parallel_config.data_parallel_index
            replica_world_size = parallel_config.world_size
            visible_device_count = torch.accelerator.device_count()

            if parallel_config.data_parallel_external_lb:
                if parallel_config.nnodes_within_dp > 1:
                    # The replica spans nodes, so use its node-local shard.
                    replica_world_size = parallel_config.local_world_size
                    if replica_world_size > visible_device_count:
                        raise ValueError(
                            f"Local TP/PP/PCP replica size ({replica_world_size}) "
                            "exceeds the number of visible XPU devices "
                            f"({visible_device_count})."
                        )
                    local_dp_capacity = visible_device_count // replica_world_size
                elif replica_world_size < visible_device_count:
                    # A node can host multiple complete TP/PP/PCP replicas.
                    local_dp_capacity = visible_device_count // replica_world_size
                    if visible_device_count % replica_world_size != 0:
                        logger.warning_once(
                            "XPU external LB cannot evenly divide "
                            "%d visible devices into TP/PP/PCP replicas of "
                            "size %d. This node can host %d complete DP "
                            "replicas, leaving %d visible devices unused.",
                            visible_device_count,
                            replica_world_size,
                            local_dp_capacity,
                            visible_device_count % replica_world_size,
                        )
                elif replica_world_size == visible_device_count:
                    # A node hosts exactly one complete TP/PP/PCP replica.
                    local_dp_capacity = 1
                    logger.warning_once(
                        "XPU external LB sees exactly enough devices for one "
                        "TP/PP/PCP replica. This may be the intended "
                        "configuration, but it may also indicate that device "
                        "visibility is misconfigured. Every DP rank must see "
                        "all XPU devices on its node; consider removing "
                        "ZE_AFFINITY_MASK or setting it to expose the complete "
                        "device set."
                    )
                else:
                    # The topology says single-node, but the replica does not fit.
                    raise ValueError(
                        f"TP/PP/PCP replica size ({replica_world_size}) exceeds "
                        f"the number of visible XPU devices ({visible_device_count}), "
                        "but nnodes_within_dp is 1. Configure the multi-node "
                        "topology, or ensure every DP rank can see all devices "
                        "on its node."
                    )
                # Strip the node component off the global DP index to get this
                # engine's slot on its own node. Assumes the launcher assigns
                # DP ranks to nodes in contiguous blocks (node 0 gets ranks
                # 0..capacity-1, and so on), which is what the usual sequential
                # and one-pod-per-rank deployments do. A round-robin or
                # unbalanced assignment would silently map two engines onto the
                # same device.
                dp_local_rank = parallel_config.data_parallel_index % local_dp_capacity

            self.local_rank += dp_local_rank * replica_world_size

        device = self.device_config.device
        if (
            isinstance(device, torch.device)
            and device.type == "xpu"
            and current_platform.is_xpu()
        ):
            assigned_physical_gpu_ids = self.parallel_config.assigned_physical_gpu_ids
            if assigned_physical_gpu_ids is not None:
                from vllm.platforms.interface import set_assigned_physical_gpu_ids

                set_assigned_physical_gpu_ids(assigned_physical_gpu_ids)
                assert self.local_rank < len(assigned_physical_gpu_ids), (
                    f"local_rank {self.local_rank} is out of bounds for "
                    f"assigned_physical_gpu_ids {assigned_physical_gpu_ids}"
                )
                # NOTE: local_world_size is derived from parallel_config.nnodes,
                # which is only set for the "mp" multi-node backend. With the
                # "ray"/"external_launcher" backends nnodes stays 1, so
                # local_world_size collapses to the full world_size and this
                # check wrongly fires on cross-node deployments.
                # assigned_physical_gpu_ids is already per-node and the
                # local_rank bound above fully validates the mapping for
                # these backends, so skip the check for them.
                if parallel_config.distributed_executor_backend not in (
                    "ray",
                    "external_launcher",
                ):
                    assert parallel_config.local_world_size <= len(
                        assigned_physical_gpu_ids
                    ), (
                        f"local_world_size ({parallel_config.local_world_size})"
                        " exceeds assigned_physical_gpu_ids count "
                        f"({len(assigned_physical_gpu_ids)})"
                    )
            else:
                assert self.local_rank < torch.accelerator.device_count(), (
                    f"DP adjusted local rank {self.local_rank} is out of "
                    f"bounds for {torch.accelerator.device_count()} devices."
                )

            visible_device_index = (
                current_platform.logical_device_id_to_visible_device_id(self.local_rank)
            )
            self.device = torch.device(f"xpu:{visible_device_index}")
            torch.accelerator.set_device_index(self.device)
            current_platform.check_if_supports_dtype(self.model_config.dtype)
            torch.accelerator.empty_cache()
            self.init_gpu_memory = torch.xpu.get_device_properties(
                visible_device_index
            ).total_memory
        else:
            raise RuntimeError(f"Unsupported device type: {self.device_config.device}")

        ENV_CCL_ATL_TRANSPORT = os.getenv("CCL_ATL_TRANSPORT", "ofi")
        ENV_LOCAL_WORLD_SIZE = os.getenv(
            "LOCAL_WORLD_SIZE", str(self.parallel_config.world_size)
        )
        os.environ["CCL_ATL_TRANSPORT"] = ENV_CCL_ATL_TRANSPORT
        os.environ["LOCAL_WORLD_SIZE"] = ENV_LOCAL_WORLD_SIZE
        os.environ["LOCAL_RANK"] = str(self.local_rank)

        init_worker_distributed_environment(
            self.vllm_config,
            self.rank,
            self.distributed_init_method,
            self.local_rank,
            current_platform.dist_backend,
        )

        # oneCCL warm-up; only meaningful for multi-device runs. Requiring it
        # with a single worker breaks platforms where oneCCL cannot enumerate
        # device topology (e.g. paravirtualized GPUs).
        if (
            self.parallel_config.world_size > 1
            and torch.distributed.is_xccl_available()
        ):
            torch.distributed.all_reduce(torch.zeros(1).xpu())

        if self.use_v2_model_runner:
            logger.info_once("Using V2 Model Runner")

        # VLLM_XPU_DETERMINISTIC=1: torch's deterministic algorithms (warn only where an op has none). A 28-layer step
        # of norm + oneDNN matmuls differed from itself in the last bf16 bit in 6-8 runs of 100, eager or graph replay,
        # and in 0 of 100 with this on (probe/graph-determinism, 2026-10-13 18:48): an order-dependent sum, not a race.
        if os.environ.get("VLLM_XPU_DETERMINISTIC") == "1":
            torch.use_deterministic_algorithms(True, warn_only=True)
            logger.info_once("Windows XPU: deterministic algorithms on (VLLM_XPU_DETERMINISTIC=1)")

        # Set random seed.
        set_random_seed(self.model_config.seed)

        # Now take memory snapshot after NCCL is initialized
        gc.collect()
        torch.accelerator.empty_cache()

        # take current memory snapshot
        self.init_snapshot = init_snapshot = MemorySnapshot(device=self.device)
        self.requested_memory = request_memory(init_snapshot, self.cache_config)
        logger.debug("worker init memory snapshot: %r", self.init_snapshot)
        logger.debug(
            "worker requested memory: %sGiB", format_gib(self.requested_memory)
        )

        self._init_workspace_and_model_runner()

    def _make_model_runner(self):
        if self.use_v2_model_runner:
            model_runner = (
                XPUMMEncoderModelRunner
                if self.vllm_config.is_mm_encoder_only
                else XPUModelRunnerV2
            )
        else:
            model_runner = XPUModelRunner  # type: ignore[assignment]
        return model_runner(self.vllm_config, self.device)  # type: ignore

    def shutdown(self) -> None:
        logger.info(
            "XPUWorker shutdown: cleaning up (rank=%d, local_rank=%d)",
            self.rank,
            self.local_rank,
        )
        super().shutdown()
        from vllm.device_allocator.xpumem import XpuMemAllocator

        if XpuMemAllocator.instance is not None:
            XpuMemAllocator.instance.release_pools()
        logger.info(
            "XPUWorker shutdown: done (rank=%d, local_rank=%d)",
            self.rank,
            self.local_rank,
        )
