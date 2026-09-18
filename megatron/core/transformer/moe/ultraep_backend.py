# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optional UltraEP PR #2 dependency and collective manager lifetime."""

from __future__ import annotations

import inspect
from typing import Any

import torch

from megatron.core.transformer.moe.replica_weight_transport import (
    register_replica_transport_finalizer,
)

ULTRAEP_SOURCE_REVISION = "6f27b25e7f03f4166c3721dffcf01d1051b1206a"
_managers: list = []


def create_ultraep_manager(**kwargs) -> Any:
    """Create a manager with the explicit-placement API, importing UltraEP lazily.

    Creation and finalization are collective and must have the same order on
    every EP rank. Keep native managers alive until the common runtime teardown;
    their NVSHMEM buffers cannot be freed by arbitrary Python garbage collection.
    """
    try:
        from ultra_ep import Manager
    except ImportError as error:
        raise ImportError(
            "UltraEP integration requires UltraEP PR #2 (target API revision "
            f"{ULTRAEP_SOURCE_REVISION}) and its CUDA/NVSHMEM dependencies."
        ) from error
    for name in ("weight_sync", "grad_reduce"):
        parameters = inspect.signature(getattr(Manager, name)).parameters
        if not {
            "physical_to_logical_map",
            "logical_to_physical_map",
            "logical_replica_counts",
        }.issubset(parameters):
            raise ImportError("UltraEP PR #2 explicit placement arguments are required.")
    manager = Manager(explicitly_destroy=True, **kwargs)
    _managers.append(manager)
    register_replica_transport_finalizer("ultra_ep", _finalize_managers)
    return manager


def _finalize_managers() -> None:
    for manager in _managers:
        with torch.cuda.device(manager.device):
            torch.cuda.synchronize()
            manager.destroy()
    _managers.clear()
