# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Optional UltraEP explicit-placement API and collective manager lifetime."""

from __future__ import annotations

import inspect
from typing import Any

import torch

_managers: list[Any] = []


def create_ultraep_manager(**kwargs: Any) -> Any:
    """Create a manager with the explicit-placement API, importing UltraEP lazily.

    Creation and finalization are collective and must have the same order on
    every EP rank. Keep native managers alive until the common runtime teardown;
    their NVSHMEM buffers cannot be freed by arbitrary Python garbage collection.
    """
    try:
        from ultra_ep import Manager
    except ImportError as error:
        raise ImportError(
            "UltraEP integration requires the explicit-placement Manager API "
            "and its CUDA/NVSHMEM dependencies."
        ) from error
    for name in ("weight_sync", "grad_reduce"):
        try:
            parameters = inspect.signature(getattr(Manager, name)).parameters
        except (AttributeError, TypeError, ValueError) as error:
            raise ImportError(
                f"UltraEP explicit placement requires an inspectable Manager.{name} method."
            ) from error
        if not {
            "physical_to_logical_map",
            "logical_to_physical_map",
            "logical_replica_counts",
        }.issubset(parameters):
            raise ImportError(
                f"UltraEP explicit placement arguments are required for Manager.{name}."
            )
    if not getattr(Manager, "supports_multi_manager", False):
        raise ImportError(
            "UltraEP per-layer managers require docker/patches/ultraep-manager-lifetime.patch; "
            "apply it to the pinned UltraEP source and rebuild the extension."
        )
    manager = Manager(explicitly_destroy=True, **kwargs)
    _managers.append(manager)
    return manager


def finalize_ultraep_managers() -> None:
    """Collectively release planner/transport Managers after their consumers drain.

    Called by replica-runtime teardown before EP process-group destruction.
    An empty registry does nothing and does not import the native dependency.
    """
    for manager in _managers:
        with torch.cuda.device(manager.device):
            torch.cuda.synchronize()
            manager.destroy()
    _managers.clear()
