# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Deferred physical home exchange, independent of logical expert identity."""

from __future__ import annotations

import torch
import torch.distributed as dist

from megatron.core.transformer.moe.moe_scheduler import ExpertDispatch
from megatron.core.transformer.moe.replica_weight_transport import HomeExpertPlacement


class HomeExpertDispatch(ExpertDispatch):
    """Stage a permutation in forward; migrate current state at the optimizer boundary.

    The state adapter owns training-recipe details. The borrowed transport is
    also used by replica dispatch, with separate operation handles and staging.
    Neither component needs logical expert identities.
    """

    dispatcher_name = "home"

    def __init__(self):
        super().__init__()
        self.transport = None
        self.state_adapter = None
        self._pending = None
        self._completed_version = 0
        self._failed = False

    def bind_transport(self, transport):
        """Borrow the existing per-layer transport; Peer-TMA is a reserved path."""
        if not transport.capabilities.home_exchange:
            raise NotImplementedError(
                f"{transport.transport_name} home exchange is not implemented."
            )
        self.transport = transport

    def bind_state_adapter(self, state_adapter):
        """Bind after optimizer construction, without registering its Parameters again."""
        self.state_adapter = state_adapter

    @property
    def has_pending(self):
        """Whether this layer needs the post-update migration window."""
        return self._pending is not None

    def dispatch(self, experts, placement: HomeExpertPlacement, context):
        """Snapshot only the physical plan; never snapshot pre-update parameter values."""
        del experts, context
        if self.transport is None:
            raise RuntimeError("Bind a home transport before dispatch.")
        if placement.version <= self._completed_version:
            raise ValueError("Home exchange received a stale proposal.")
        if self._pending is not None:
            raise RuntimeError("Home exchange already has a pending proposal.")
        self._pending = self.transport.prepare_home_exchange(placement)

    @torch.no_grad()
    def step(self):
        """Install all received state in stable slots and return the completion version.

        Caller must finish backward/gradient communication and defer next-step
        parameter gathers. Exceptions are fatal once installation starts; no
        optimistic mapping commit or partial retry is permitted.
        """
        if self._failed:
            raise RuntimeError(
                "Home state installation failed; restore from a checkpoint before retrying."
            )
        if self._pending is None:
            return None
        if self.state_adapter is None:
            raise RuntimeError("Bind optimizer home state before calling scheduler.step().")
        components = self.state_adapter.snapshot()
        signature = (
            self._pending.placement.version,
            self._pending.placement.source_slots.tolist(),
            getattr(self.state_adapter, "component_names", None),
            [(tuple(t.shape), t.dtype) for t in components],
        )
        signatures = [None] * self.transport.config.world_size
        dist.all_gather_object(signatures, signature, group=self.transport.config.group)
        if any(other != signature for other in signatures):
            raise ValueError("Home plans or state component schemas differ across EP ranks.")
        handle = self.transport.start_home_exchange(sources=components, plan=self._pending)
        received = self.transport.wait_home_exchange(handle)
        try:
            self.state_adapter.install(received)
        except Exception:
            self._failed = True
            raise
        version = self._pending.placement.version
        self._pending = None
        self._completed_version = version
        return version
