# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""UltraEP adaptation contracts; no native UltraEP import for the host tests."""

import sys
from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.moe import ultraep_backend
from megatron.core.transformer.moe.replica_ultraep_transport import _compile_placement
from megatron.core.transformer.moe.ultraep_moe_scheduler import (
    UltraEPPlacementResult,
    _SnapshotReroute,
)
from tests.unit_tests.transformer.moe.scheduler_test_utils import _scheduler_config

pytestmark = pytest.mark.launch_on_gb200


@pytest.mark.parametrize("method_name", ["weight_sync", "grad_reduce"])
@pytest.mark.parametrize("invalid_api", ["missing", "noncallable", "legacy"])
def test_ultraep_rejects_incompatible_manager_api(monkeypatch, method_name, invalid_api):
    class Manager:
        supports_multi_manager = True

        def __init__(self, **kwargs):
            pytest.fail("An incompatible Manager must be rejected before allocation.")

        def weight_sync(
            self, *, physical_to_logical_map, logical_to_physical_map, logical_replica_counts
        ):
            pass

        grad_reduce = weight_sync

    if invalid_api == "missing":
        delattr(Manager, method_name)
    elif invalid_api == "noncallable":
        setattr(Manager, method_name, None)
    else:
        setattr(Manager, method_name, lambda self, layer_id: None)
    monkeypatch.setitem(sys.modules, "ultra_ep", SimpleNamespace(Manager=Manager))
    with pytest.raises(ImportError, match="UltraEP PR #2"):
        ultraep_backend.create_ultraep_manager(num_layers=1)


@pytest.mark.parametrize("supports_multi_manager", [False, True])
def test_ultraep_requires_native_multi_manager_lifetime_patch(monkeypatch, supports_multi_manager):
    created = []

    class Manager:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def weight_sync(
            self, *, physical_to_logical_map, logical_to_physical_map, logical_replica_counts
        ):
            pass

        grad_reduce = weight_sync

    Manager.supports_multi_manager = supports_multi_manager
    monkeypatch.setitem(sys.modules, "ultra_ep", SimpleNamespace(Manager=Manager))
    monkeypatch.setattr(ultraep_backend, "_managers", [])
    monkeypatch.setattr(ultraep_backend, "register_replica_transport_finalizer", lambda *args: None)
    if supports_multi_manager:
        manager = ultraep_backend.create_ultraep_manager(num_layers=1)
        assert created == [{"explicitly_destroy": True, "num_layers": 1}]
        assert ultraep_backend._managers == [manager]
    else:
        with pytest.raises(ImportError, match="ultraep-manager-lifetime.patch"):
            ultraep_backend.create_ultraep_manager(num_layers=1)
        assert created == []


def test_physical_source_maps_have_master_first_and_inverse_consistency():
    table = [[2, 3], [0, -1], [6, -1], [4, 5]]
    p2l, l2p, counts = _compile_placement(table, 2, 2)
    assert p2l == [0, 1, 2, 3, 2, 3, 0, -1, 4, 5, 6, -1, 6, 7, 4, 5]
    for source in range(8):
        rank, slot = divmod(source, 2)
        assert l2p[source][0] == rank * 4 + slot
        assert counts[source] == p2l.count(source)
        assert sorted(p for p in l2p[source] if p >= 0) == [
            p for p, identity in enumerate(p2l) if identity == source
        ]


@pytest.mark.parametrize(
    "table,domain,error",
    [
        ([[0], [-1]], 2, "one instance"),  # duplicate of the native home
        ([[2, 2], [-1, -1]], 2, "one instance"),
        ([[2], [-1]], 1, "NVLink domain"),
        ([[-2], [-1]], 2, "outside"),
        ([[4], [-1]], 2, "outside"),
        ([[2], []], 2, "same number"),
        ([[2], [-1]], 3, "domain sizes"),
    ],
)
def test_ultraep_rejects_incompatible_physical_placements(table, domain, error):
    with pytest.raises(ValueError, match=error):
        _compile_placement(table, 2, domain)


def test_no_replicas_keeps_fixed_physical_homes():
    p2l, l2p, counts = _compile_placement([[-1], [-1]], 2, 2)
    assert p2l == [0, 1, -1, 2, 3, -1]
    assert l2p == [[0, -1], [1, -1], [3, -1], [4, -1]]
    assert counts == [1, 1, 1, 1]


@pytest.mark.parametrize("contiguous_gradient", [True, False])
def test_reroute_backward_survives_internal_slot_overwrite(contiguous_gradient):
    # Two forwards sharing one manager slot have different placement/quota state.
    # The first backward must follow its actual token choices, including top-k,
    # zero-valued selected probabilities, and arbitrary gradients on unused slots.
    class Manager:
        def reroute(self, slot, probs, routes):
            assert slot == 0
            p2l = self.p2l.clamp_min(0).long()
            return probs[:, p2l].masked_fill(~self.routes, 0), self.routes.clone()

    manager = Manager()
    logical_routes = torch.tensor([[True, True], [True, False], [False, True]])
    first = torch.tensor([[0.0, 0.7], [0.2, 0.0], [0.0, 0.8]], requires_grad=True)
    second = first.detach().clone().requires_grad_()
    manager.p2l = torch.tensor([0, 1, 1, -1], dtype=torch.int32)
    manager.routes = torch.tensor(
        [[True, False, True, False], [True, False, False, False], [False, True, False, False]]
    )
    old_map = manager.p2l.clone()
    out1, routes1 = _SnapshotReroute.apply(
        first, logical_routes, UltraEPPlacementResult(manager, old_map)
    )
    # Overwrite the same internal tensor storage, as the native manager does.
    manager.p2l.copy_(torch.tensor([0, 1, 0, -1], dtype=torch.int32))
    manager.routes = torch.tensor(
        [[False, True, True, False], [False, False, True, False], [False, True, False, False]]
    )
    out2, _ = _SnapshotReroute.apply(
        second, logical_routes, UltraEPPlacementResult(manager, manager.p2l.clone())
    )
    grad = torch.arange(1, 13, dtype=torch.float32).view(3, 4)
    if not contiguous_gradient:
        grad = grad.t().contiguous().t()
        assert not grad.is_contiguous()
    out2.backward(grad)
    out1.backward(grad)
    torch.testing.assert_close(first.grad, torch.tensor([[1.0, 3.0], [5.0, 0.0], [0.0, 10.0]]))
    torch.testing.assert_close(second.grad, torch.tensor([[3.0, 2.0], [7.0, 0.0], [0.0, 10.0]]))
    assert not routes1.requires_grad
    torch.testing.assert_close(out1.sum(dim=1), first.detach().sum(dim=1))


def test_reroute_backward_handles_empty_token_batch():
    class Manager:
        def reroute(self, slot, probs, routes):
            return probs.new_empty((0, 4)), routes.new_empty((0, 4))

    probs = torch.empty(0, 2, requires_grad=True)
    routes = torch.empty(0, 2, dtype=torch.bool)
    placement = UltraEPPlacementResult(Manager(), torch.tensor([0, -1, 1, -1]))
    physical_probs, physical_routes = _SnapshotReroute.apply(probs, routes, placement)
    physical_probs.sum().backward()
    assert physical_routes.shape == (0, 4)
    torch.testing.assert_close(probs.grad, torch.empty_like(probs))


def test_ultraep_configuration_rejects_unsupported_combinations():
    config = _scheduler_config(
        moe_scheduler_planner_type="ultra_ep",
        moe_scheduler_expert_dispatcher_type="replica_ultraep",
    )
    assert config.moe_scheduler_planner_type == "ultra_ep"
    with pytest.raises(ValueError, match="FP32 gradients"):
        _scheduler_config(
            moe_scheduler_expert_dispatcher_type="replica_ultraep", grad_reduce_in_bf16=True
        )
    with pytest.raises(ValueError, match="Periodic home exchange"):
        _scheduler_config(
            moe_scheduler_planner_type="ultra_ep", moe_scheduler_home_update_interval=1
        )
    with pytest.raises(ValueError, match="Home exchange currently requires NCCL"):
        _scheduler_config(
            moe_scheduler_planner_type="eplb",
            moe_scheduler_expert_dispatcher_type="replica_ultraep",
            moe_scheduler_home_update_interval=1,
        )
    with pytest.raises(ValueError, match="UltraEP integration"):
        _scheduler_config(
            moe_scheduler_planner_type="ultra_ep",
            cuda_graph_impl="transformer_engine",
            cuda_graph_modules=["moe"],
        )
