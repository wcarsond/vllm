# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import os
import sys
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm.logger import init_logger

from .base_device_communicator import DeviceCommunicatorBase

logger = init_logger(__name__)

# Windows: no oneCCL, so no xccl process-group backend; the process groups are
# gloo and every card-to-card operation is relayed through host RAM. See
# xpu_ram_relay.py. GroupCoordinator routes the pipeline hand-off through
# send_tensor_dict/recv_tensor_dict when supports_tensor_dict is True.
_RAM_RELAY = sys.platform == "win32"


class XpuCommunicator(DeviceCommunicatorBase):
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
        use_all2all: bool = False,
    ):
        super().__init__(
            cpu_group, device, device_group, unique_name, use_all2all=use_all2all
        )
        self.supports_tensor_dict = _RAM_RELAY
        if _RAM_RELAY:
            from . import xpu_ram_relay

            self._relay = xpu_ram_relay
            # Two-rank all_reduce goes through shared host memory instead of gloo
            # (xpu_shm_exchange.py): 33 us against 233 us for a 7 KB decode
            # tensor, measured 2026-10-10. VLLM_XPU_EXCHANGE=gloo opts out.
            # Built on first use so the collective that names the block runs in
            # lockstep on both ranks.
            self._exchange = None
            self._exchange_wanted = (
                self.world_size == 2
                and os.environ.get("VLLM_XPU_EXCHANGE", "shm").lower() == "shm"
            )
            logger.info_once(
                "XPU on Windows: card-to-card collectives relayed through host RAM "
                "over gloo (no oneCCL on this platform)%s.",
                "; two-rank all_reduce through shared memory" if self._exchange_wanted else "",
            )
        self.ca_comm: None = None
        if self.use_all2all:
            if self.all2all_backend in ("naive", "allgather_reducescatter"):
                from .all2all import AgRsAll2AllManager

                self.all2all_manager = AgRsAll2AllManager(self.cpu_group)
                logger.info("Using AgRs manager on XPU device.")

            else:  # type: ignore[has-type]
                logger.warning(
                    "`%s` all2all manager is not supported on XPU. "
                    "Falling back to AgRs manager for XPU, "
                    "which is the Default backend",
                    self.all2all_backend,  # type: ignore[has-type]
                )
                from .all2all import AgRsAll2AllManager

                self.all2all_manager = AgRsAll2AllManager(self.cpu_group)
                logger.info("Using AgRs manager on XPU device.")

    def _fixed_rank_sum(self, input_: torch.Tensor) -> torch.Tensor:
        flat_input = input_.reshape(-1)
        gathered = torch.empty(
            (self.world_size, flat_input.numel()),
            dtype=input_.dtype,
            device=input_.device,
        )
        dist.all_gather_single(gathered.view(-1), flat_input, group=self.device_group)
        output = gathered[0].clone()
        for rank in range(1, self.world_size):
            output.add_(gathered[rank])
        return output.view(input_.shape)

    # ---- Windows RAM relay -------------------------------------------------
    # Each method below stages through host memory on the gloo cpu_group when
    # _RAM_RELAY is set and otherwise falls through to the xccl path.

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        if not _RAM_RELAY:
            return super().send(tensor, dst)
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        self._relay.send(tensor, self.ranks[dst], self.cpu_group)

    def recv(
        self, size: torch.Size, dtype: torch.dtype, src: int | None = None
    ) -> torch.Tensor:
        if not _RAM_RELAY:
            return super().recv(size, dtype, src)
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        return self._relay.recv(size, dtype, self.ranks[src], self.cpu_group, self.device)

    def send_tensor_dict(
        self, tensor_dict: dict[str, torch.Tensor | Any], dst: int
    ) -> None:
        assert _RAM_RELAY, "tensor dict fast path is the Windows relay only"
        self._relay.send_tensor_dict(tensor_dict, self.ranks[dst], self.cpu_group)

    def recv_tensor_dict(self, src: int) -> dict[str, torch.Tensor | Any]:
        assert _RAM_RELAY, "tensor dict fast path is the Windows relay only"
        return self._relay.recv_tensor_dict(self.ranks[src], self.cpu_group, self.device)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if not _RAM_RELAY:
            return super().all_gather(input_, dim)
        if dim < 0:
            dim += input_.dim()
        input_size = input_.size()
        gathered = self._relay.all_gather(input_, self.world_size, self.cpu_group)
        gathered = gathered.movedim(0, dim)
        return gathered.reshape(
            input_size[:dim] + (self.world_size * input_size[dim],) + input_size[dim + 1 :]
        )

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if _RAM_RELAY:
            if self._exchange_wanted:
                if self._exchange is None:
                    from .xpu_shm_exchange import ShmExchange

                    self._exchange = ShmExchange(
                        self.rank_in_group, self.world_size, self.cpu_group, self.device
                    )
                nbytes = input_.numel() * input_.element_size()
                if nbytes <= self._exchange.slot_bytes:
                    return self._exchange.all_reduce(input_.clone())
                # larger than a slot (a very long prefill chunk): the gloo relay still works
            return self._relay.all_reduce(input_, self.cpu_group)
        if envs.VLLM_BATCH_INVARIANT:
            # XCCL changes its reduction order with message size. Gather first
            # and accumulate in rank order, independent of the token batch.
            return self._fixed_rank_sum(input_)

        output = input_.clone()
        dist.all_reduce(output, group=self.device_group)
        return output

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1):
        world_size = self.world_size

        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        assert input_tensor.shape[0] % world_size == 0
        chunk_size = input_tensor.shape[0] // world_size
        if _RAM_RELAY:
            output = self._relay.reduce_scatter(
                input_tensor, chunk_size, world_size, self.cpu_group
            )
            return output.movedim(0, dim).contiguous()
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        output = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )

        if envs.VLLM_BATCH_INVARIANT:
            reduced = self._fixed_rank_sum(input_tensor)
            start = self.rank_in_group * chunk_size
            output.copy_(reduced.narrow(0, start, chunk_size))
        else:
            dist.reduce_scatter_tensor(output, input_tensor, group=self.device_group)

        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    def reduce_scatterv(
        self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None
    ):
        world_size = self.world_size

        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()

        # Note: This will produce an incorrect answer if we don't make
        # the input_tensor contiguous. Possible bug in reduce_scatter_tensor?
        input_tensor = input_.movedim(0, dim).contiguous()

        if sizes is not None:
            assert len(sizes) == world_size
            assert input_tensor.shape[0] == sum(sizes)
            chunk_size = sizes[self.rank_in_group]
        else:
            assert input_tensor.shape[0] % world_size == 0
            chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        output = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )
        if envs.VLLM_BATCH_INVARIANT:
            reduced = self._fixed_rank_sum(input_tensor)
            start = (
                self.rank_in_group * chunk_size
                if sizes is None
                else sum(sizes[: self.rank_in_group])
            )
            output.copy_(reduced.narrow(0, start, chunk_size))
        elif sizes is None or sizes.count(sizes[0]) == len(sizes):
            dist.reduce_scatter_tensor(output, input_tensor, group=self.device_group)
        elif torch.xpu.device_count() < self.world_size:
            # uneven reduce_scatter is not supported when the
            # process cannot see every device in the group (
            # e.g. external-LBDP).
            max_size = max(sizes)
            padded = input_tensor.new_zeros(
                (world_size * max_size,) + input_tensor.shape[1:]
            )
            for rank, split in enumerate(input_tensor.split(sizes, dim=0)):
                padded[rank * max_size : rank * max_size + sizes[rank]] = split
            padded_output = input_tensor.new_empty((max_size,) + input_tensor.shape[1:])
            dist.reduce_scatter_tensor(padded_output, padded, group=self.device_group)
            output = padded_output[:chunk_size]
        else:
            input_splits = list(input_tensor.split(sizes, dim=0))
            dist.reduce_scatter(output, input_splits, group=self.device_group)
        # Reshape before returning
        return output.movedim(0, dim).contiguous()

    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ):
        if dim != 0:
            raise NotImplementedError("only dim 0 all-gatherv is supported")
        world_size = self.world_size

        # 'sizes' is not needed if all inputs in the same group have the same
        # shape
        if sizes is not None and all(s == sizes[0] for s in sizes):
            sizes = None

        def _all_gather_single(input_: torch.Tensor, sizes: list[int] | None = None):
            input_size = input_.size()
            if sizes is not None:
                assert len(sizes) == world_size
                assert input_.shape[dim] == sizes[self.rank_in_group], (
                    f"{input_.shape[dim]} != {sizes[self.rank_in_group]}"
                )
                output_size = (sum(sizes),) + input_size[1:]
            else:
                output_size = (input_size[0] * world_size,) + input_size[1:]
            # Allocate output tensor.
            output_tensor = torch.empty(
                output_size, dtype=input_.dtype, device=input_.device
            )

            if sizes is not None:
                all_gather_list = []
                for size in sizes:
                    all_gather_list.append(
                        torch.empty(
                            (size,) + input_.shape[1:],
                            dtype=input_.dtype,
                            device=input_.device,
                        )
                    )
                dist.all_gather(all_gather_list, input_, group=self.device_group)
                output_tensor = torch.cat(all_gather_list, dim=0)
            else:
                dist.all_gather([output_tensor], input_, group=self.device_group)
            return output_tensor

        if isinstance(input_, torch.Tensor):
            return _all_gather_single(input_, sizes)

        output_list = []
        for inp in input_:
            output_list.append(_all_gather_single(inp, sizes=sizes))
        return output_list

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )
        if dim < 0:
            # Convert negative dim to positive.
            dim += input_.dim()
        # For xpu path, gather doesn't work properly together with ray
        # cluster so we use all_gather instead for now.
        input_size = input_.size()
        if _RAM_RELAY:
            output_tensor = self._relay.all_gather(input_, self.world_size, self.cpu_group)
        else:
            # Allocate output tensor.
            output_tensor = torch.empty(
                (self.world_size,) + input_size, dtype=input_.dtype, device=input_.device
            )
            # All-gather.
            dist.all_gather_single(output_tensor, input_, group=self.device_group)
        if self.rank_in_group == dst:
            # Reshape
            output_tensor = output_tensor.movedim(0, dim)
            output_tensor = output_tensor.reshape(
                input_size[:dim]
                + (self.world_size * input_size[dim],)
                + input_size[dim + 1 :]
            )
        else:
            output_tensor = None
        return output_tensor

    def broadcast(self, input_: torch.Tensor, src: int = 0) -> None:
        if _RAM_RELAY:
            self._relay.broadcast(input_, self.ranks[src], self.cpu_group)
            return
        dist.broadcast(input_, src=src, group=self.device_group)

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """Dispatch the hidden states and router logits to the appropriate device.
        This is a no-op in the base class.
        """
        assert self.all2all_manager is not None
        return self.all2all_manager.dispatch_router_logits(
            hidden_states,
            router_logits,
            is_sequence_parallel,
            extra_tensors,
        )

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """Dispatch the hidden states and topk weights/ids to the appropriate device.
        This is a no-op in the base class.
        """
        assert self.all2all_manager is not None
        return self.all2all_manager.dispatch(
            hidden_states,
            topk_weights,
            topk_ids,
            is_sequence_parallel,
            extra_tensors=extra_tensors,
        )

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        """Combine the hidden states and router logits from the appropriate device.
        This is a no-op in the base class.
        """
        assert self.all2all_manager is not None
        return self.all2all_manager.combine(
            hidden_states,
            is_sequence_parallel,
        )
