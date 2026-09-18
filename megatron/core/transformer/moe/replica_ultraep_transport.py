# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""UltraEP PR #2 replica communication with physical home-slot identities."""

from __future__ import annotations

import math
from typing import Any, Callable

import torch
import torch.distributed as dist

from megatron.core.transformer.moe.replica_weight_transport import (
    ReplicaGradDestination,
    ReplicaPlacement,
    ReplicaPreparedPlan,
    ReplicaTransferHandle,
    ReplicaTransportCapabilities,
    ReplicaTransportConfig,
    ReplicaWeightLayout,
    ReplicaWeightSource,
    ReplicaWeightTransport,
)
from megatron.core.transformer.moe.ultraep_backend import create_ultraep_manager


def _compile_placement(
    table: list[list[int]], home_experts: int, nvl_domain_size: int
) -> tuple[list[int], list[list[int]], list[int]]:
    """Lower physical sources to UltraEP's fixed-master maps, master first.

    UltraEP's logical IDs here are *source-home slot IDs*. Model logical identity
    remains private to the planner, including after a home permutation. This
    backend does not itself implement the optimizer-state exchange operation.
    """
    ep_size = len(table)
    if home_experts <= 0 or not ep_size or nvl_domain_size <= 0 or ep_size % nvl_domain_size:
        raise ValueError("UltraEP requires positive uniform home and NVLink domain sizes.")
    slots = len(table[0])
    width = home_experts + slots
    total = ep_size * home_experts
    p2l = [-1] * (ep_size * width)
    l2p = [[-1] * ep_size for _ in range(total)]
    counts = [1] * total
    for source in range(total):
        owner, index = divmod(source, home_experts)
        physical = owner * width + index
        p2l[physical] = source
        l2p[source][0] = physical
    for rank, row in enumerate(table):
        if len(row) != slots:
            raise ValueError("UltraEP requires the same number of replica slots per rank.")
        seen = set(range(rank * home_experts, (rank + 1) * home_experts))
        for slot, source in enumerate(row):
            if source == -1:
                continue
            if not 0 <= source < total:
                raise ValueError("UltraEP replica source is outside the physical home slots.")
            if source in seen:
                raise ValueError("UltraEP permits at most one instance of an expert per rank.")
            if source // home_experts // nvl_domain_size != rank // nvl_domain_size:
                raise ValueError("UltraEP replicas must stay within the source's NVLink domain.")
            seen.add(source)
            physical = rank * width + home_experts + slot
            p2l[physical] = source
            l2p[source][counts[source]] = physical
            counts[source] += 1
    return p2l, l2p, counts


class UltraEPTransport(ReplicaWeightTransport):
    """BF16 weights and FP32 joint FC1/FC2 gradient reduction.

    Each layer owns a manager and stable replica buffers. Planning validates the
    global physical table on the host, so MoE CUDA capture is deliberately rejected.
    NCCL stream fences surround UltraEP operations: PR #2's local completion
    events alone do not establish remote producer readiness or destination
    completion. These conservative fences cost overlap and require all EP ranks
    to call every operation in the same order, even with no active replicas.
    """

    transport_name = "replica_ultraep"
    capabilities = ReplicaTransportCapabilities(
        weight_formats=("bf16",), grad_dtypes=(torch.float32,), split_grad_reduce=False
    )

    def __init__(self, config: ReplicaTransportConfig) -> None:
        super().__init__(config)
        if (
            config.device.type != "cuda"
            or config.device.index != torch.cuda.current_device()
            or config.group is None
            or not dist.is_initialized()
            or dist.get_backend(config.group) != "nccl"
            or dist.get_world_size(config.group) != config.world_size
        ):
            raise ValueError(
                "replica_ultraep requires the current CUDA device and an NCCL EP group."
            )
        if config.num_local_replica_slots <= 0:
            raise ValueError("replica_ultraep requires at least one replica slot per rank.")
        if any(
            math.prod(shape) <= 0 or math.prod(shape) * 2 % 16 for shape in config.member_shapes
        ):
            raise ValueError(
                "UltraEP BF16 expert shards must have positive, 16-byte-aligned sizes."
            )
        self.manager = create_ultraep_manager(
            group=config.group,
            num_layers=1,
            num_local_master_experts=config.num_local_home_experts,
            num_local_redundant_experts=config.num_local_replica_slots,
            expert_fc1_numel=math.prod(config.member_shapes[0]),
            expert_fc2_numel=math.prod(config.member_shapes[1]),
            weight_data_dtype=torch.bfloat16,
            grad_dtype=torch.float32,
            is_train=True,
        )
        # UltraEP's native stream_wait rejects a caller on its own comm stream.
        # Keep the EP fences and completion join on a separate orchestration stream.
        self._stream = torch.cuda.Stream(device=config.device)
        self._native_grads = tuple(
            torch.empty(
                (config.num_local_home_experts, *shape), dtype=torch.float32, device=config.device
            )
            for shape in config.member_shapes
        )
        self._fence = torch.zeros(1, dtype=torch.int32, device=config.device)
        self._sources: tuple[ReplicaWeightSource, ...] | None = None
        self._destroyed = False

    @property
    def grad_dtype(self) -> torch.dtype:
        return torch.float32

    def projection_views(
        self, projection_index: int
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        shape = (self.config.num_local_replica_slots, *self.config.member_shapes[projection_index])
        weights = getattr(self.manager, f"local_replica_fc{projection_index + 1}_weight_buffer")
        grads = getattr(self.manager, f"local_replica_fc{projection_index + 1}_grad_buffer")
        return tuple(weights.view(shape)), grads.view(shape)

    def native_projection_grad_view(self, projection_index: int) -> torch.Tensor:
        return self._native_grads[projection_index]

    def prepare_plan(self, placement: ReplicaPlacement) -> ReplicaPreparedPlan:
        self._check_execution()
        super().prepare_plan(placement)
        table = placement.slot_to_expert.tolist()
        tables = [None] * self.config.world_size
        dist.all_gather_object(tables, table, group=self.config.group)
        if any(other != table for other in tables):
            raise ValueError("UltraEP replica placements differ across EP ranks.")
        maps = _compile_placement(
            table, self.config.num_local_home_experts, self.manager.nvl_domain_size
        )
        metadata = dict(
            zip(
                ("physical_to_logical_map", "logical_to_physical_map", "logical_replica_counts"),
                (torch.tensor(m, dtype=torch.int32, device=self.config.device) for m in maps),
            )
        )
        return ReplicaPreparedPlan(placement, self, metadata)

    def _check_execution(self) -> None:
        if self._destroyed:
            raise RuntimeError("UltraEP transport has been destroyed.")
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("replica_ultraep does not support MoE CUDA graph capture.")

    def _register(
        self,
        sources: tuple[ReplicaWeightSource, ...],
        native_grads: tuple[ReplicaGradDestination, ...],
    ) -> tuple[torch.Tensor, ...]:
        if len(sources) != 2 or len(native_grads) != 2:
            raise ValueError("UltraEP requires both FC1 and FC2 sources and destinations.")
        for index, (source, destination) in enumerate(zip(sources, native_grads)):
            if source.layout is not ReplicaWeightLayout.PLAIN or source.scales is not None:
                raise ValueError("replica_ultraep currently supports plain BF16 weights only.")
            for tensors, dtype in (
                (source.data, torch.bfloat16),
                (destination.tensors, torch.float32),
            ):
                if len(tensors) != self.config.num_local_home_experts or any(
                    tensor.dtype != dtype
                    or tensor.device != self.config.device
                    or not tensor.is_contiguous()
                    or tuple(tensor.shape) != self.config.member_shapes[index]
                    for tensor in tensors
                ):
                    raise ValueError(
                        "UltraEP source weights/gradients must match local home shapes and dtypes."
                    )
        # GTP/FSDP materialization can change source addresses; never cache them.
        self.manager.construct_local_master_ptr_pool(
            0,
            list(sources[0].data),
            list(sources[1].data),
            list(native_grads[0].tensors),
            list(native_grads[1].tensors),
        )
        return tuple(
            getattr(self.manager, f"local_master_fc{index}_{component}_ptr_pool")[0]
            for index in (1, 2)
            for component in ("weight", "weight_scale", "grad")
        )

    def _start(
        self,
        operation: Callable[..., Any],
        plan: ReplicaPreparedPlan,
        tensors: tuple[torch.Tensor, ...],
    ) -> ReplicaTransferHandle:
        self._check_execution()
        self.validate_plan(plan)
        caller = torch.cuda.current_stream(self.config.device)
        stream = self._stream
        stream.wait_stream(caller)
        keepalive = (*tensors, *plan.metadata.values())
        with torch.cuda.stream(stream):
            self._fence.record_stream(stream)
            # Join producers/previous consumers on every rank before remote access.
            dist.all_reduce(self._fence, group=self.config.group)
            event = operation(0, async_finish=True, **plan.metadata)
            event.current_stream_wait()
            # Weight pushes and gradient pulls can finish on a different rank.
            dist.all_reduce(self._fence, group=self.config.group)
            completion = torch.cuda.Event()
            completion.record(stream)
            for tensor in keepalive:
                tensor.record_stream(stream)
        return ReplicaTransferHandle(self, completion, (self.manager, plan, *keepalive))

    def start_weight_sync(
        self, *, sources: tuple[ReplicaWeightSource, ...], plan: ReplicaPreparedPlan
    ) -> ReplicaTransferHandle:
        self._check_execution()
        self.validate_plan(plan)
        grads = tuple(ReplicaGradDestination(tuple(buffer)) for buffer in self._native_grads)
        pointers = self._register(sources, grads)
        self._sources = sources
        return self._start(
            self.manager.weight_sync,
            plan,
            (*pointers, *(t for source in sources for t in source.data)),
        )

    def start_grad_reduce(
        self,
        *,
        native_grads: tuple[ReplicaGradDestination, ...],
        plan: ReplicaPreparedPlan,
        projections: tuple[int, ...],
    ) -> ReplicaTransferHandle:
        self._check_execution()
        self.validate_plan(plan)
        if projections != (0, 1):
            raise ValueError("UltraEP reduces FC1 and FC2 together after both wgrads are ready.")
        if self._sources is None:
            raise RuntimeError("UltraEP weights must be registered before gradient reduction.")
        pointers = self._register(self._sources, native_grads)
        return self._start(
            self.manager.grad_reduce,
            plan,
            (*pointers, *(t for grad in native_grads for t in grad.tensors)),
        )

    def _wait(self, handle: ReplicaTransferHandle) -> None:
        self._check_execution()
        if handle.transport is not self:
            raise ValueError("UltraEP completion belongs to a different transport.")
        handle.completion.wait(torch.cuda.current_stream(self.config.device))

    def wait_weight_sync(self, handle: ReplicaTransferHandle) -> None:
        self._wait(handle)

    def wait_grad_reduce(self, handle: ReplicaTransferHandle) -> None:
        self._wait(handle)

    def destroy(self) -> None:
        """Drain operations and release layer-owned staging; defer NVSHMEM cleanup."""
        if self._destroyed:
            return
        self._stream.synchronize()
        self._sources = None
        self._native_grads = ()
        self._fence = None
        self._destroyed = True
