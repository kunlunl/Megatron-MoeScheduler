# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Backend-neutral interfaces for MoE expert and token rerouting.

MoEScheduler is intended to run after router output is available and before the
normal MoE token dispatcher starts.  This module only defines the shared
contracts:

* planners own logical placement and lower it to physical copy/exchange requests;
* replica dispatch starts immediately, while home dispatch stages a deferred plan;
* rerouting always uses the currently committed placement;
* backend-specific lowering is kept inside the concrete expert dispatcher.

Concrete Echo, EPLB, UltraEP, and MoonEP planners should produce an opaque
``MoEPlacementResult`` between their placement and reroute phases.
Concrete expert dispatchers consume physical source-slot tables, never logical mappings.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Optional

import torch

from megatron.core.transformer.moe.replica_weight_transport import (
    REPLICA_EXPERT_DISPATCHER_TYPES,
    HomeExpertPlacement,
)

IDENTITY_BACKEND = "identity"
ECHO_BACKEND = "echo"
EPLB_BACKEND = "eplb"
ULTRA_EP_BACKEND = "ultra_ep"
MOON_EP_BACKEND = "moon_ep"


def _rank0_info(message: str) -> None:
    try:
        is_rank0 = (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )
    except RuntimeError:
        is_rank0 = True
    if is_rank0:
        print(f"INFO:MoEScheduler: {message}", flush=True)


def _tensor_shape(tensor: Optional[torch.Tensor]) -> Optional[tuple[int, ...]]:
    return None if tensor is None else tuple(tensor.shape)


def _validate_2d_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.dim() != 2:
        raise ValueError(f"Expected {name} to be 2D, got shape {tuple(tensor.shape)}")


@dataclass(frozen=True)
class SchedulerContext:
    """Static and per-forward context shared by planners and dispatchers."""

    layer_number: Optional[int]
    num_logical_experts: int
    num_local_experts: int
    local_expert_indices: tuple[int, ...]
    ep_size: int
    ep_rank: int
    router_topk: int
    training: bool
    config: Any = None
    pg_collection: Any = None

    def __post_init__(self) -> None:
        if self.num_logical_experts <= 0:
            raise ValueError("num_logical_experts must be positive.")
        if self.num_local_experts <= 0:
            raise ValueError("num_local_experts must be positive.")
        if self.ep_size <= 0:
            raise ValueError("ep_size must be positive.")
        if self.ep_rank < 0 or self.ep_rank >= self.ep_size:
            raise ValueError(f"ep_rank must be in [0, {self.ep_size}), got {self.ep_rank}.")
        if self.router_topk <= 0:
            raise ValueError("router_topk must be positive.")
        if len(self.local_expert_indices) != self.num_local_experts:
            raise ValueError(
                "Expected local_expert_indices to match num_local_experts, "
                f"got {len(self.local_expert_indices)} and {self.num_local_experts}"
            )


class MoEPlacementResult:
    """Opaque planner state passed from ``update_placement`` to ``reroute``.

    The scheduler deliberately does not inspect this object. Concrete planners
    use subclasses to retain the allocation metadata needed to reroute tokens
    without recomputing placement or storing implicit per-forward state.
    """


def _validate_reroute_output(
    num_physical_experts: int, routing_map: torch.Tensor, probs: torch.Tensor
) -> None:
    _validate_2d_tensor("routing_map", routing_map)
    _validate_2d_tensor("probs", probs)
    if routing_map.dtype != torch.bool:
        raise ValueError(f"Expected bool routing_map, got {routing_map.dtype}")
    if probs.shape != routing_map.shape:
        raise ValueError(
            "Expected probs and routing_map to have the same shape, "
            f"got {tuple(probs.shape)} and {tuple(routing_map.shape)}"
        )
    if num_physical_experts != routing_map.size(1):
        raise ValueError(
            "Expected physical expert count to match routing_map expert dimension, "
            f"got {num_physical_experts} and {routing_map.size(1)}"
        )


class MoELoadPlanner(torch.nn.Module, ABC):
    """Base class for backend-neutral MoE load planners."""

    planner_name: ClassVar[str] = "abstract"

    def should_plan(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        context: SchedulerContext,
        *,
        tokens_per_expert: Optional[torch.Tensor] = None,
    ) -> bool:
        """Return whether the planner should run for this router output."""
        del probs, routing_map, context, tokens_per_expert
        return True

    @abstractmethod
    def update_placement(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        context: SchedulerContext,
        *,
        tokens_per_expert: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[HomeExpertPlacement], Optional[torch.Tensor], MoEPlacementResult]:
        """Return optional home exchange, replica copies, and private reroute state."""

    def step(self, completed_version: Optional[int] = None) -> None:
        """Commit completed home placement after a successful optimizer update."""
        if completed_version is not None:
            raise RuntimeError("This planner has no pending home exchange.")

    @abstractmethod
    def reroute(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        placement_result: MoEPlacementResult,
        context: SchedulerContext,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return dense physical ``routing_map`` and ``probs`` tensors."""


class ExpertDispatch(torch.nn.Module, ABC):
    """Base class for ReplicaHybridEP and future placement materializers."""

    dispatcher_name: ClassVar[str] = "abstract"

    def supports(self, expert_placement: torch.Tensor, context: SchedulerContext) -> bool:
        """Return whether this dispatcher can materialize the given physical layout."""
        del expert_placement, context
        return True

    @abstractmethod
    def dispatch(
        self,
        experts: torch.nn.Module,
        expert_placement: torch.Tensor,
        context: SchedulerContext,
    ) -> None:
        """Materialize the planned expert placement before token dispatch begins."""

    def finalize(self, context: SchedulerContext) -> None:
        """Release transient dispatch state after the MoE forward finishes."""
        del context

    def assert_idle(self) -> None:
        """Check that no outstanding dispatch can consume home state being migrated."""

    def bind_experts(self, experts: torch.nn.Module) -> None:
        """Bind expert parameters for dispatchers that own persistent runtime weights."""
        del experts

    def wrap_layer_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Attach lifecycle work that must run after all layer-input consumers."""
        return hidden_states

    def before_token_dispatch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Apply a backend-specific autograd boundary before token dispatch."""
        return hidden_states

    def after_token_dispatch(self, dispatched_hidden: torch.Tensor) -> torch.Tensor:
        """Apply a backend-specific autograd boundary after token dispatch."""
        return dispatched_hidden

    def before_token_combine(self, expert_output: torch.Tensor) -> torch.Tensor:
        """Apply a backend-specific autograd boundary before token combine."""
        return expert_output

    def after_token_combine(self, combined_hidden: torch.Tensor) -> torch.Tensor:
        """Apply a backend-specific autograd boundary after token combine."""
        return combined_hidden


class MoEScheduler(torch.nn.Module):
    """Small orchestrator that connects a planner to an expert dispatcher."""

    _logged_config_signatures: ClassVar[set[tuple[str, str, int, str]]] = set()
    _logged_runtime_summary: ClassVar[bool] = False

    def __init__(
        self,
        planner: MoELoadPlanner,
        expert_dispatch: ExpertDispatch,
        home_expert_dispatch: Optional[ExpertDispatch] = None,
    ) -> None:
        super().__init__()
        self.planner = planner
        self.expert_dispatch = expert_dispatch
        self.home_expert_dispatch = home_expert_dispatch

    @classmethod
    def from_config(cls, config: Any, pg_collection: Any) -> "MoEScheduler":
        """Build the configured MoEScheduler backend stack."""
        planner_type = getattr(config, "moe_scheduler_planner_type", None)
        expert_dispatcher_type = getattr(config, "moe_scheduler_expert_dispatcher_type", None)
        if planner_type not in ("echo", "eplb", "moon_ep"):
            raise ValueError(f"Unsupported MoEScheduler planner: {planner_type}")
        if expert_dispatcher_type not in REPLICA_EXPERT_DISPATCHER_TYPES:
            raise ValueError(
                f"Unsupported MoEScheduler expert dispatcher: {expert_dispatcher_type}"
            )

        num_idle_experts = getattr(config, "moe_scheduler_num_idle_experts", None)
        if num_idle_experts is None:
            raise ValueError(
                "moe_scheduler_num_idle_experts must be set when MoEScheduler is enabled."
            )
        assignment_algorithm = getattr(
            config, "moe_scheduler_assignment_algorithm", "approx_bin_packing"
        )

        from megatron.core.transformer.moe.echo_moe_scheduler import EchoLoadPlanner
        from megatron.core.transformer.moe.replica_expert_dispatch import ReplicaExpertDispatch

        if planner_type == "echo":
            planner = EchoLoadPlanner(num_idle_experts, assignment_algorithm=assignment_algorithm)
        elif planner_type == "eplb":
            from megatron.core.transformer.moe.eplb_moe_scheduler import EPLBLoadPlanner

            planner = EPLBLoadPlanner(
                num_redundant_experts=num_idle_experts,
                home_update_interval=getattr(config, "moe_scheduler_home_update_interval", 0),
            )
        else:
            from megatron.core.transformer.moe.moonep_moe_scheduler import MoonEPLoadPlanner

            ep_size = getattr(config, "expert_model_parallel_size", 1)
            planner = MoonEPLoadPlanner(num_redundant_experts=num_idle_experts // ep_size)
        expert_dispatch = ReplicaExpertDispatch(config=config, pg_collection=pg_collection)
        config_signature = (
            str(planner_type),
            str(expert_dispatcher_type),
            int(num_idle_experts),
            str(assignment_algorithm),
        )
        if config_signature not in cls._logged_config_signatures:
            cls._logged_config_signatures.add(config_signature)
            _rank0_info(
                "configured "
                f"planner={planner_type} "
                f"expert_dispatcher={expert_dispatcher_type} "
                f"num_idle_experts={num_idle_experts} "
                f"assignment_algorithm={assignment_algorithm}"
            )
        home_dispatch = None
        if getattr(config, "moe_scheduler_home_update_interval", 0):
            from megatron.core.transformer.moe.home_expert_dispatch import HomeExpertDispatch

            home_dispatch = HomeExpertDispatch()
        return cls(
            planner=planner, expert_dispatch=expert_dispatch, home_expert_dispatch=home_dispatch
        )

    def bind_experts(self, experts: torch.nn.Module) -> None:
        """Bind the layer's native experts to the configured dispatch backend."""
        self.expert_dispatch.bind_experts(experts)
        if self.home_expert_dispatch is not None:
            self.home_expert_dispatch.bind_transport(self.expert_dispatch.runtime.transport)

    def bind_optimizer(self, optimizer) -> None:
        """Bind authoritative state after DDP/FSDP and optimizer construction."""
        if self.home_expert_dispatch is not None:
            from megatron.core.transformer.moe.home_expert_state import OptimizerHomeExpertState

            self.home_expert_dispatch.bind_state_adapter(
                OptimizerHomeExpertState(optimizer, self.expert_dispatch.runtime)
            )

    def step(self) -> None:
        """Migrate updated home state, then publish mapping, at a drained step boundary."""
        version = None
        if self.home_expert_dispatch is not None:
            self.expert_dispatch.assert_idle()
            version = self.home_expert_dispatch.step()
        self.planner.step(version)

    def wrap_layer_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Attach dispatcher work before routing and shared-expert computation."""
        return self.expert_dispatch.wrap_layer_input(hidden_states)

    def _log_first_schedule(
        self,
        input_routing_map: torch.Tensor,
        num_physical_experts: Optional[int],
        output_routing_map: Optional[torch.Tensor],
        context: SchedulerContext,
        *,
        planning_skipped: bool,
        dispatch_materialized: bool,
    ) -> None:
        if MoEScheduler._logged_runtime_summary:
            return
        MoEScheduler._logged_runtime_summary = True
        if num_physical_experts is None:
            output_routing_map = input_routing_map
            expert_backend = IDENTITY_BACKEND
            num_physical_experts = context.num_logical_experts
            num_transfers = 0
            assignment_backend = None
            reroute_backend = "identity"
        else:
            assert output_routing_map is not None
            expert_backend = self.expert_dispatch.dispatcher_name
            num_transfers = max(0, num_physical_experts - context.num_logical_experts)
            assignment_backend = self.planner.planner_name
            reroute_backend = "planner"
        _rank0_info(
            "first schedule completed "
            f"layer={context.layer_number} "
            f"training={context.training} "
            f"planner={self.planner.planner_name} "
            f"dispatcher={self.expert_dispatch.dispatcher_name} "
            f"input_routing_map_shape={_tensor_shape(input_routing_map)} "
            f"output_routing_map_shape={_tensor_shape(output_routing_map)} "
            f"expert_backend={expert_backend} "
            f"num_physical_experts={num_physical_experts} "
            f"num_transfers={num_transfers} "
            f"assignment_backend={assignment_backend} "
            f"reroute_backend={reroute_backend} "
            f"planning_skipped={planning_skipped} "
            f"dispatch_materialized={dispatch_materialized}"
        )

    def schedule(
        self,
        probs: torch.Tensor,
        routing_map: torch.Tensor,
        experts: torch.nn.Module,
        context: SchedulerContext,
        *,
        tokens_per_expert: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Plan token/expert rerouting and materialize the expert side."""
        if not self.planner.should_plan(
            probs, routing_map, context, tokens_per_expert=tokens_per_expert
        ):
            self._log_first_schedule(
                routing_map, None, None, context, planning_skipped=True, dispatch_materialized=False
            )
            return probs, routing_map

        home_expert_placement, replica_expert_placement, placement_result = (
            self.planner.update_placement(
                probs, routing_map, context, tokens_per_expert=tokens_per_expert
            )
        )
        if home_expert_placement is not None:
            if self.home_expert_dispatch is None:
                raise ValueError("Planner requested home exchange without HomeExpertDispatch.")
            self.home_expert_dispatch.dispatch(experts, home_expert_placement, context)
        if replica_expert_placement is not None:
            if (
                replica_expert_placement.ndim != 2
                or replica_expert_placement.shape[0] != context.ep_size
                or replica_expert_placement.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError("Expected integer replica placement [EP, S].")
            if not self.expert_dispatch.supports(replica_expert_placement, context):
                raise ValueError(
                    "Expert dispatcher does not support planner output replica source-slot table."
                )
            self.expert_dispatch.dispatch(experts, replica_expert_placement, context)
        rerouted_routing_map, rerouted_probs = self.planner.reroute(
            probs, routing_map, placement_result, context
        )
        num_physical_experts = context.num_logical_experts + (
            replica_expert_placement.numel() if replica_expert_placement is not None else 0
        )
        _validate_reroute_output(num_physical_experts, rerouted_routing_map, rerouted_probs)
        self._log_first_schedule(
            routing_map,
            num_physical_experts,
            rerouted_routing_map,
            context,
            planning_skipped=False,
            dispatch_materialized=replica_expert_placement is not None,
        )
        return rerouted_probs, rerouted_routing_map

    def finalize(self, context: SchedulerContext) -> None:
        """Finalize the expert-dispatch portion of a scheduled forward."""
        self.expert_dispatch.finalize(context)

    def before_token_dispatch(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the configured dispatcher's pre-dispatch autograd boundary."""
        return self.expert_dispatch.before_token_dispatch(hidden_states)

    def after_token_dispatch(self, dispatched_hidden: torch.Tensor) -> torch.Tensor:
        """Run the configured dispatcher's post-dispatch autograd boundary."""
        return self.expert_dispatch.after_token_dispatch(dispatched_hidden)

    def before_token_combine(self, expert_output: torch.Tensor) -> torch.Tensor:
        """Run the configured dispatcher's pre-combine autograd boundary."""
        return self.expert_dispatch.before_token_combine(expert_output)

    def after_token_combine(self, combined_hidden: torch.Tensor) -> torch.Tensor:
        """Run the configured dispatcher's post-combine autograd boundary."""
        return self.expert_dispatch.after_token_combine(combined_hidden)
