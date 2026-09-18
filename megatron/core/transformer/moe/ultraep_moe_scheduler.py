# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""UltraEP quota planning adapted to physical source-home replica requests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.distributed as dist

from megatron.core.transformer.moe.moe_scheduler import (
    MoELoadPlanner,
    MoEPlacementResult,
    SchedulerContext,
)
from megatron.core.transformer.moe.ultraep_backend import create_ultraep_manager


@dataclass(frozen=True)
class UltraEPPlacementResult(MoEPlacementResult):
    """Snapshot of placement plus the owner of the pending quota reroute."""

    manager: Any
    physical_to_logical_map: torch.Tensor


class _SnapshotReroute(torch.autograd.Function):
    """Use UltraEP's quota forward; keep its exact token assignment for backward.

    UltraEP PR #2 does not export a self-contained quota plan. Its own backward
    reads mutable manager placement. Instead save the actual dense routing mask
    and a placement snapshot: they completely specify the permutation of probs.
    Later microbatches may then reuse the manager slot without changing backward.
    """

    @staticmethod
    def forward(ctx, probs, routing_map, placement):
        physical_probs, physical_routes = placement.manager.reroute(0, probs, routing_map)
        ctx.save_for_backward(placement.physical_to_logical_map, physical_routes)
        ctx.num_logical_experts = probs.size(1)
        ctx.mark_non_differentiable(physical_routes)
        return physical_probs, physical_routes

    @staticmethod
    def backward(ctx, physical_grad, _route_grad):
        if physical_grad is None:
            return None, None, None
        p2l, routes = ctx.saved_tensors
        valid = routes & (p2l >= 0).unsqueeze(0)
        logical_grad = physical_grad.new_zeros((physical_grad.size(0), ctx.num_logical_experts))
        logical_grad.scatter_add_(
            1,
            p2l.clamp_min(0).long().unsqueeze(0).expand_as(physical_grad),
            physical_grad.masked_fill(~valid, 0),
        )
        return logical_grad, None, None


class UltraEPLoadPlanner(MoELoadPlanner):
    """Plan fixed homes with UltraEP, leaving communication to the dispatcher.

    The planner emits no home migration. Consequently its logical IDs equal
    current physical source-home IDs. The returned replica table is [EP, S],
    rather than the full logical p2l table used by the pre-home-exchange branch.
    """

    planner_name = "ultra_ep"

    def __init__(
        self,
        num_redundant_experts: int,
        group: dist.ProcessGroup,
        manager_provider: Callable[[], Any] | None = None,
    ) -> None:
        super().__init__()
        self.num_redundant_experts = num_redundant_experts
        self.group = group
        self._manager_provider = manager_provider
        self._manager = None
        self._pending: UltraEPPlacementResult | None = None

    def _get_manager(self, context: SchedulerContext) -> Any:
        if self._manager is None:
            if self._manager_provider is not None:
                self._manager = self._manager_provider()
            else:
                # The public Manager always allocates communication buffers.
                # Use minimal nonzero 16-byte FC shards for planner-only use;
                # no expert weights/gradients are communicated by this instance.
                self._manager = create_ultraep_manager(
                    group=self.group,
                    num_layers=1,
                    num_local_master_experts=context.num_local_experts,
                    num_local_redundant_experts=self.num_redundant_experts // context.ep_size,
                    expert_fc1_numel=8,
                    expert_fc2_numel=8,
                    is_train=False,
                )
        return self._manager

    def update_placement(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        context: SchedulerContext,
        *,
        tokens_per_expert: torch.Tensor | None = None,
    ) -> tuple[None, torch.Tensor, UltraEPPlacementResult]:
        del tokens_per_expert
        if self._pending is not None:
            raise RuntimeError("UltraEP placement must be rerouted before planning again.")
        if not probs.is_cuda or torch.cuda.is_current_stream_capturing():
            raise ValueError("UltraEP planner requires CUDA execution outside graph capture.")
        if (
            probs.dtype != torch.float32
            or routing_map.dtype != torch.bool
            or probs.ndim != 2
            or probs.shape != routing_map.shape
            or probs.size(1) != context.num_logical_experts
            or probs.device != routing_map.device
            or probs.device.index != torch.cuda.current_device()
        ):
            raise ValueError("UltraEP requires matching CUDA FP32 probs and bool logical routes.")
        if (
            self.group is None
            or not dist.is_initialized()
            or dist.get_world_size(self.group) != context.ep_size
            or dist.get_rank(self.group) != context.ep_rank
            or context.num_logical_experts != context.num_local_experts * context.ep_size
            or self.num_redundant_experts <= 0
            or self.num_redundant_experts % context.ep_size
            or getattr(context.config, "moe_scheduler_home_update_interval", 0)
        ):
            raise ValueError(
                "UltraEP planner requires a matching EP group and fixed uniform homes."
            )
        manager = self._get_manager(context)
        maps = manager.update_placement(0, routing_map.contiguous())
        if not isinstance(maps, tuple) or len(maps) != 3:
            raise RuntimeError("UltraEP update_placement must return the PR #2 placement maps.")
        # These are borrowed views. Own the map before the next update overwrites it.
        p2l = maps[0].clone()
        placement = UltraEPPlacementResult(manager, p2l)
        self._pending = placement
        replica_sources = p2l.view(context.ep_size, -1)[:, context.num_local_experts :].contiguous()
        return None, replica_sources, placement

    def reroute(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        placement_result: MoEPlacementResult,
        context: SchedulerContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del context
        if placement_result is not self._pending:
            raise ValueError("UltraEP reroute must consume the matching pending placement.")
        physical_probs, physical_routes = _SnapshotReroute.apply(
            probs, routing_map, placement_result
        )
        self._pending = None
        return physical_routes, physical_probs
