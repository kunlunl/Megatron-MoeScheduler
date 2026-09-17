# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""EPLB-style planner for the backend-neutral MoEScheduler contract.

The placement policy follows the two central ideas from DeepSeek's EPLB:

* greedily replicate the expert with the largest per-instance load; and
* place replica instances with longest-processing-time-first (LPT) packing.

By default homes remain fixed. With a home update interval, capacity-constrained
LPT periodically proposes a home permutation. Active logical ownership changes
only after state migration. Every forward distributes logical routes across the
committed homes and transient replicas using a global EP token ordinal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from megatron.core import tensor_parallel
from megatron.core.transformer.moe.moe_scheduler import (
    MoELoadPlanner,
    MoEPlacementResult,
    SchedulerContext,
)
from megatron.core.transformer.moe.replica_weight_transport import HomeExpertPlacement


@dataclass(frozen=True, slots=True)
class EPLBPlacementResult(MoEPlacementResult):
    """Explicit EPLB state consumed by the token-reroute phase."""

    logical_to_physical_map: torch.Tensor
    replica_counts: torch.Tensor
    local_expert_offsets: torch.Tensor
    num_physical_experts: int
    ep_size: int
    ep_rank: int


def _replicate_experts(
    expert_loads: torch.Tensor, num_redundant_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose logical experts to replicate with EPLB's greedy policy.

    At each step, the next replica belongs to the expert with the largest
    ``load / current_replica_count``. PyTorch's first-index tie breaking makes
    the result deterministic on every EP rank.
    """
    if expert_loads.dim() != 1:
        raise ValueError(f"expert_loads must be 1D, got shape {tuple(expert_loads.shape)}.")
    if expert_loads.numel() == 0:
        raise ValueError("expert_loads must contain at least one logical expert.")
    if num_redundant_experts < 0:
        raise ValueError("num_redundant_experts must be non-negative.")

    num_logical_experts = expert_loads.numel()
    replica_counts = torch.ones(num_logical_experts, dtype=torch.int64, device=expert_loads.device)
    replica_experts = torch.empty(
        num_redundant_experts, dtype=torch.int64, device=expert_loads.device
    )
    one = torch.ones(1, dtype=torch.int64, device=expert_loads.device)
    loads = expert_loads.to(dtype=torch.float32)
    for replica_index in range(num_redundant_experts):
        logical_expert = torch.argmax(loads / replica_counts)
        replica_experts[replica_index : replica_index + 1].copy_(logical_expert.reshape(1))
        replica_counts.scatter_add_(0, logical_expert.reshape(1), one)
    return replica_experts, replica_counts


def _pack_replicas_with_fixed_homes(
    expert_loads: torch.Tensor,
    replica_experts: torch.Tensor,
    replica_counts: torch.Tensor,
    ep_size: int,
    home_layout: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """LPT-pack replicas while retaining each logical expert's home rank."""
    num_logical_experts = expert_loads.numel()
    num_redundant_experts = replica_experts.numel()
    if ep_size <= 0:
        raise ValueError("EPLB ep_size must be positive.")
    if num_logical_experts % ep_size != 0:
        raise ValueError("EPLB requires logical experts to be divisible by ep_size.")
    if num_redundant_experts % ep_size != 0:
        raise ValueError("EPLB requires redundant experts to be divisible by ep_size.")

    num_local_home_experts = num_logical_experts // ep_size
    num_local_replica_slots = num_redundant_experts // ep_size
    num_local_physical_experts = num_local_home_experts + num_local_replica_slots
    num_physical_experts = num_logical_experts + num_redundant_experts
    device = expert_loads.device

    physical_to_logical_map = torch.full(
        (num_physical_experts,), -1, dtype=torch.int64, device=device
    )
    rank_layout = physical_to_logical_map.view(ep_size, num_local_physical_experts)
    rank_layout[:, :num_local_home_experts] = torch.arange(
        num_logical_experts, dtype=torch.int64, device=device
    ).view(ep_size, num_local_home_experts)
    if home_layout is not None:
        rank_layout[:, :num_local_home_experts].copy_(home_layout)
    if num_redundant_experts == 0:
        return physical_to_logical_map

    per_instance_load = expert_loads.to(torch.float32) / replica_counts
    rank_loads = per_instance_load[rank_layout[:, :num_local_home_experts]].sum(dim=1)
    replica_loads = per_instance_load.gather(0, replica_experts)
    placement_order = torch.argsort(replica_loads, descending=True, stable=True)
    slots_used = torch.zeros(ep_size, dtype=torch.int64, device=device)
    one = torch.ones(1, dtype=torch.int64, device=device)

    for placement_index in range(num_redundant_experts):
        replica_index = placement_order[placement_index]
        logical_expert = replica_experts.gather(0, replica_index.reshape(1))
        load = replica_loads.gather(0, replica_index.reshape(1))
        has_capacity = slots_used < num_local_replica_slots
        destination_rank = torch.argmin(
            torch.where(has_capacity, rank_loads, torch.full_like(rank_loads, torch.inf))
        )
        destination_slot = slots_used.gather(0, destination_rank.reshape(1))
        physical_expert = (
            destination_rank * num_local_physical_experts
            + num_local_home_experts
            + destination_slot
        )
        physical_to_logical_map.scatter_(0, physical_expert, logical_expert)
        slots_used.scatter_add_(0, destination_rank.reshape(1), one)
        rank_loads.scatter_add_(0, destination_rank.reshape(1), load)

    return physical_to_logical_map


def _logical_to_physical_map(
    physical_to_logical_map: torch.Tensor, num_logical_experts: int, num_redundant_experts: int
) -> torch.Tensor:
    """Invert a physical map and retain physical-instance order per expert."""
    num_physical_experts = physical_to_logical_map.numel()
    max_replica_count = num_redundant_experts + 1
    same_expert = physical_to_logical_map[:, None] == physical_to_logical_map[None, :]
    replica_rank = torch.tril(same_expert).sum(dim=1, dtype=torch.int64) - 1
    flat_indices = physical_to_logical_map * max_replica_count + replica_rank
    physical_experts = torch.arange(
        num_physical_experts, dtype=torch.int64, device=physical_to_logical_map.device
    )
    logical_to_physical_map = torch.full(
        (num_logical_experts * max_replica_count,),
        -1,
        dtype=torch.int64,
        device=physical_to_logical_map.device,
    )
    logical_to_physical_map.scatter_(0, flat_indices, physical_experts)
    return logical_to_physical_map.view(num_logical_experts, max_replica_count)


def build_eplb_placement(
    expert_loads: torch.Tensor, num_redundant_experts: int, ep_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build fixed-home EPLB physical/logical maps from global expert loads."""
    replica_experts, replica_counts = _replicate_experts(expert_loads, num_redundant_experts)
    physical_to_logical_map = _pack_replicas_with_fixed_homes(
        expert_loads, replica_experts, replica_counts, ep_size
    )
    logical_to_physical_map = _logical_to_physical_map(
        physical_to_logical_map, expert_loads.numel(), num_redundant_experts
    )
    return physical_to_logical_map, logical_to_physical_map, replica_counts


def build_eplb_home_layout(expert_loads, num_redundant_experts, ep_size):
    """Capacity-constrained LPT: exactly H unique homes on each rank.

    Use per-instance costs after greedy replication. Unlike unconstrained EPLB,
    this always supplies one canonical training owner for every logical expert.
    Host planning occurs only at a periodic migration boundary.
    """
    _, counts = _replicate_experts(expert_loads, num_redundant_experts)
    costs = (expert_loads.float() / counts).cpu().tolist()
    h = len(costs) // ep_size
    if h == 0 or len(costs) % ep_size:
        raise ValueError("Home layout requires a positive uniform home capacity.")
    ranks = [[] for _ in range(ep_size)]
    loads = [0.0] * ep_size
    for expert in sorted(range(len(costs)), key=lambda i: (-costs[i], i)):
        rank = min((r for r in range(ep_size) if len(ranks[r]) < h), key=lambda r: (loads[r], r))
        ranks[rank].append(expert)
        loads[rank] += costs[expert]
    return torch.tensor(ranks, dtype=torch.int64, device=expert_loads.device)


class EPLBLoadPlanner(MoELoadPlanner):
    """EPLB planner with transient replicas and optional deferred home permutations."""

    planner_name = "eplb"

    def __init__(self, num_redundant_experts: int, home_update_interval: int = 0) -> None:
        super().__init__()
        if num_redundant_experts < 0:
            raise ValueError("num_redundant_experts must be non-negative.")
        self.num_redundant_experts = int(num_redundant_experts)
        if home_update_interval < 0:
            raise ValueError("home_update_interval must be non-negative.")
        self.home_update_interval = home_update_interval
        self.register_buffer(
            "active_home_layout", torch.empty(0, dtype=torch.int64), persistent=False
        )
        self.register_buffer(
            "active_replica_layout", torch.empty(0, dtype=torch.int64), persistent=False
        )
        self.register_buffer("_load_history", torch.empty(0), persistent=False)
        self._pending_replica_layout = None
        self._pending_home_layout = None
        self._pending_version = None
        self._version = 0
        self._steps = 0
        self._last_proposal_step = -1

    def step(self, completed_version=None):
        """Publish new logical ownership only after the dispatcher completed exchange."""
        if completed_version is not None:
            if completed_version != self._pending_version:
                raise RuntimeError("EPLB received a stale or unknown home completion.")
            self.active_home_layout.copy_(self._pending_home_layout)
            self._pending_home_layout = None
            self._pending_version = None
        elif self._pending_version is not None:
            raise RuntimeError("EPLB cannot commit a pending layout without home completion.")
        if self._pending_replica_layout is not None:
            self.active_replica_layout.copy_(self._pending_replica_layout)
            self._pending_replica_layout = None
        self._steps += 1

    def get_extra_state(self):
        """Layout metadata saved with physical parameter slots at a step boundary."""
        return {
            "home_layout": self.active_home_layout.detach().cpu(),
            "replica_layout": self.active_replica_layout.detach().cpu(),
            "steps": self._steps,
            "version": self._version,
            "last_proposal_step": self._last_proposal_step,
            "load_history": self._load_history.detach().cpu(),
        }

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        if not self.home_update_interval:
            destination.pop(prefix + "_extra_state", None)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        if not self.home_update_interval and prefix + "_extra_state" in missing_keys:
            missing_keys.remove(prefix + "_extra_state")

    def set_extra_state(self, state):
        """Restore before the first forward; physical checkpoint topology must match."""
        self.active_home_layout = state["home_layout"].to(self.active_home_layout.device)
        self.active_replica_layout = state["replica_layout"].to(self.active_home_layout.device)
        self._pending_replica_layout = None
        self._load_history = state["load_history"].to(self._load_history.device)
        self._steps = state["steps"]
        self._version = state["version"]
        self._last_proposal_step = state["last_proposal_step"]
        self._pending_version = None
        self._pending_home_layout = None

    def _validate_inputs(
        self, probs: torch.Tensor, routing_map: torch.Tensor, context: SchedulerContext
    ) -> None:
        if routing_map.dim() != 2:
            raise ValueError(f"routing_map must be 2D, got shape {tuple(routing_map.shape)}.")
        if probs.shape != routing_map.shape:
            raise ValueError(
                "probs and routing_map must have the same shape, got "
                f"{tuple(probs.shape)} and {tuple(routing_map.shape)}."
            )
        if routing_map.size(1) != context.num_logical_experts:
            raise ValueError(
                "routing_map logical expert dimension does not match SchedulerContext, "
                f"got {routing_map.size(1)} and {context.num_logical_experts}."
            )
        if routing_map.dtype != torch.bool:
            raise ValueError(f"routing_map must be bool, got {routing_map.dtype}.")
        if probs.device != routing_map.device:
            raise ValueError("probs and routing_map must be on the same device.")

    def _validate_context(self, context: SchedulerContext) -> None:
        if context.num_logical_experts % context.ep_size != 0:
            raise ValueError("EPLB requires num_logical_experts divisible by ep_size.")
        if self.num_redundant_experts % context.ep_size != 0:
            raise ValueError("EPLB requires num_redundant_experts divisible by ep_size.")
        num_local_home_experts = context.num_logical_experts // context.ep_size
        if context.num_local_experts != num_local_home_experts:
            raise ValueError(
                "EPLB SchedulerContext.num_local_experts does not match the even EP layout."
            )
        local_start = context.ep_rank * num_local_home_experts
        expected_local_experts = tuple(range(local_start, local_start + num_local_home_experts))
        if context.local_expert_indices != expected_local_experts:
            raise ValueError("EPLB requires contiguous rank-major logical home experts.")

    def should_plan(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        context: SchedulerContext,
        *,
        tokens_per_expert: Optional[torch.Tensor] = None,
    ) -> bool:
        del tokens_per_expert
        self._validate_inputs(probs, routing_map, context)
        self._validate_context(context)
        return self.num_redundant_experts != 0 or self.home_update_interval > 0

    def _get_count_matrix(
        self,
        routing_map: torch.Tensor,
        context: SchedulerContext,
        *,
        tokens_per_expert: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        local_counts = (
            tokens_per_expert if tokens_per_expert is not None else routing_map.sum(dim=0)
        )
        if local_counts.dim() != 1 or local_counts.numel() != context.num_logical_experts:
            raise ValueError(
                "Expected local token counts to be a 1D logical-expert vector, "
                f"got shape {tuple(local_counts.shape)}."
            )
        local_counts = local_counts.to(device=routing_map.device, dtype=torch.int64)
        if context.ep_size == 1:
            return local_counts.unsqueeze(0)

        ep_group = getattr(context.pg_collection, "ep", None)
        if ep_group is None:
            raise ValueError("EPLBLoadPlanner requires SchedulerContext.pg_collection.ep.")
        return tensor_parallel.gather_from_sequence_parallel_region(
            local_counts, group=ep_group
        ).reshape(context.ep_size, context.num_logical_experts)

    @torch.no_grad()
    def update_placement(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        context: SchedulerContext,
        *,
        tokens_per_expert: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[HomeExpertPlacement], torch.Tensor, EPLBPlacementResult]:
        """Compute a common placement from current global EP token counts."""
        self._validate_inputs(probs, routing_map, context)
        self._validate_context(context)
        counts_from_ep_rank = self._get_count_matrix(
            routing_map, context, tokens_per_expert=tokens_per_expert
        )
        loads = counts_from_ep_rank.sum(dim=0)
        h = context.num_local_experts
        if not self.active_home_layout.numel():
            self.active_home_layout = torch.arange(
                context.num_logical_experts, device=loads.device
            ).reshape(context.ep_size, h)
            self._load_history = torch.zeros_like(loads, dtype=torch.float64)
        if tuple(self.active_home_layout.shape) != (context.ep_size, h):
            raise ValueError("Restored EPLB physical checkpoint has a different EP/home topology.")
        self.active_home_layout = self.active_home_layout.to(loads.device)
        self._load_history = self._load_history.to(loads.device)
        if self.home_update_interval:
            if self.active_replica_layout.ndim != 2:
                initial, initial_counts = _replicate_experts(loads, self.num_redundant_experts)
                layout = _pack_replicas_with_fixed_homes(
                    loads, initial, initial_counts, context.ep_size, self.active_home_layout
                )
                self.active_replica_layout = layout.reshape(context.ep_size, -1)[:, h:].clone()
            self.active_replica_layout = self.active_replica_layout.to(loads.device)
            if tuple(self.active_replica_layout.shape) != (
                context.ep_size,
                self.num_redundant_experts // context.ep_size,
            ):
                raise ValueError("Restored EPLB checkpoint has a different replica slot topology.")
        home_placement = None
        if (
            context.training
            and self.home_update_interval
            and self._steps > 0
            and self._steps % self.home_update_interval == 0
            and self._last_proposal_step != self._steps
            and self._pending_version is None
        ):
            historical_loads = self._load_history.clone()
            # Data replicas must choose identical logical ownership before their
            # sharded optimizer states can move between corresponding EP ranks.
            edp = getattr(context.pg_collection, "expt_dp", None)
            if edp is not None:
                torch.distributed.all_reduce(historical_loads, group=edp)
            candidate = build_eplb_home_layout(
                historical_loads, self.num_redundant_experts, context.ep_size
            )
            candidate_replicas, candidate_counts = _replicate_experts(
                historical_loads, self.num_redundant_experts
            )
            candidate_layout = _pack_replicas_with_fixed_homes(
                historical_loads, candidate_replicas, candidate_counts, context.ep_size, candidate
            )
            self._pending_replica_layout = candidate_layout.reshape(context.ep_size, -1)[
                :, h:
            ].clone()
            self._last_proposal_step = self._steps
            self._load_history.zero_()
            if not torch.equal(candidate, self.active_home_layout):
                self._version += 1
                self._pending_version = self._version
                self._pending_home_layout = candidate
                logical_to_home = torch.argsort(self.active_home_layout.flatten())
                home_placement = HomeExpertPlacement(logical_to_home[candidate], self._version)
        if context.training:
            self._load_history.add_(loads)
        # Every microbatch copies CURRENT home sources, even while an exchange is pending.
        if self.home_update_interval:
            physical_to_logical_map = torch.cat(
                (self.active_home_layout, self.active_replica_layout), dim=1
            ).flatten()
            replica_counts = torch.bincount(
                physical_to_logical_map, minlength=context.num_logical_experts
            )
        else:
            replicas, replica_counts = _replicate_experts(loads, self.num_redundant_experts)
            physical_to_logical_map = _pack_replicas_with_fixed_homes(
                loads, replicas, replica_counts, context.ep_size, self.active_home_layout
            )
        logical_to_physical_map = _logical_to_physical_map(
            physical_to_logical_map, context.num_logical_experts, self.num_redundant_experts
        )
        logical_to_home = torch.argsort(self.active_home_layout.flatten())
        replica_layout = physical_to_logical_map.reshape(context.ep_size, -1)[:, h:]
        replica_placement = logical_to_home[replica_layout].to(torch.int32).contiguous()
        local_expert_offsets = counts_from_ep_rank[: context.ep_rank].sum(dim=0)
        return (
            home_placement,
            replica_placement,
            EPLBPlacementResult(
                logical_to_physical_map=logical_to_physical_map,
                replica_counts=replica_counts,
                local_expert_offsets=local_expert_offsets,
                num_physical_experts=physical_to_logical_map.numel(),
                ep_size=context.ep_size,
                ep_rank=context.ep_rank,
            ),
        )

    def reroute(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        placement_result: MoEPlacementResult,
        context: SchedulerContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Round-robin logical routes across their EPLB physical instances."""
        if not isinstance(placement_result, EPLBPlacementResult):
            raise TypeError(
                "EPLBLoadPlanner.reroute requires the EPLBPlacementResult returned by "
                "EPLBLoadPlanner.update_placement."
            )
        self._validate_inputs(probs, routing_map, context)
        self._validate_context(context)
        if (placement_result.ep_size, placement_result.ep_rank) != (
            context.ep_size,
            context.ep_rank,
        ):
            raise ValueError("EPLB placement result belongs to a different EP context.")

        num_tokens, num_logical_experts = routing_map.shape
        expected_shape = (num_logical_experts, self.num_redundant_experts + 1)
        if tuple(placement_result.logical_to_physical_map.shape) != expected_shape:
            raise ValueError(
                "EPLB logical_to_physical_map has the wrong shape, expected " f"{expected_shape}."
            )
        if placement_result.replica_counts.shape != (num_logical_experts,):
            raise ValueError("EPLB replica_counts has the wrong shape.")
        if placement_result.local_expert_offsets.shape != (num_logical_experts,):
            raise ValueError("EPLB local_expert_offsets has the wrong shape.")

        local_ordinals = routing_map.to(torch.int64).cumsum(dim=0) - 1
        global_ordinals = local_ordinals + placement_result.local_expert_offsets.unsqueeze(0)
        replica_ranks = torch.remainder(
            global_ordinals, placement_result.replica_counts.unsqueeze(0)
        )
        physical_experts = placement_result.logical_to_physical_map.gather(
            1, replica_ranks.transpose(0, 1)
        ).transpose(0, 1)

        output_shape = (num_tokens, placement_result.num_physical_experts)
        physical_routing_map = torch.zeros(
            output_shape, dtype=torch.bool, device=routing_map.device
        ).scatter(1, physical_experts, routing_map)
        selected_probs = torch.where(routing_map, probs, torch.zeros_like(probs))
        physical_probs = probs.new_zeros(output_shape).scatter(1, physical_experts, selected_probs)
        return physical_routing_map, physical_probs


__all__ = ["EPLBLoadPlanner", "EPLBPlacementResult", "build_eplb_placement"]
