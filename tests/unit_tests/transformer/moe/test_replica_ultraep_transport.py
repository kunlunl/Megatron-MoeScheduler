# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Run with torchrun on one NVLink domain and the pinned UltraEP PR #2 build."""

import os

import pytest
import torch
import torch.distributed as dist

from megatron.core.transformer.moe.moe_scheduler import SchedulerContext
from megatron.core.transformer.moe.replica_ultraep_transport import UltraEPTransport
from megatron.core.transformer.moe.replica_weight_transport import (
    ReplicaGradDestination,
    ReplicaOwnership,
    ReplicaPlacement,
    ReplicaTransportConfig,
    ReplicaWeightSource,
    finalize_replica_weight_transports,
)
from megatron.core.transformer.moe.ultraep_moe_scheduler import UltraEPLoadPlanner

pytestmark = pytest.mark.launch_on_gb200


@pytest.fixture(scope="module")
def ultraep_group():
    pytest.importorskip("ultra_ep")
    if not torch.cuda.is_available():
        pytest.skip("UltraEP requires CUDA/NVLink.")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    # UltraEP's global runtime supports only one EP group per process.
    yield dist.group.WORLD
    torch.cuda.synchronize()
    dist.barrier()
    finalize_replica_weight_transports()


def test_ultraep_weights_gradients_and_saved_forward_plan(ultraep_group):
    group = ultraep_group
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    device = torch.device("cuda", torch.cuda.current_device())
    config = ReplicaTransportConfig(
        group=group,
        device=device,
        world_size=world,
        num_local_home_experts=1,
        num_local_replica_slots=1,
        member_shapes=((128, 128), (128, 128)),
        weight_format="bf16",
        rowwise_scale_shapes=None,
        columnwise_scale_shapes=None,
        grad_dtype=torch.float32,
        num_sms=None,
    )
    transport = UltraEPTransport(config)
    domain = transport.manager.nvl_domain_size
    if domain < 2:
        pytest.skip("Requires at least two NVLink peers.")
    # Copy the next rank's physical home inside each NVLink domain.
    sources = [[r // domain * domain + (r + 1) % domain] for r in range(world)]
    table = torch.tensor(sources, dtype=torch.int32, device=device)
    plan = transport.prepare_plan(ReplicaPlacement(table, ReplicaOwnership(1), 1))
    inputs = tuple(
        ReplicaWeightSource(
            (torch.full((128, 128), rank * 10 + p, dtype=torch.bfloat16, device=device),), None
        )
        for p in range(2)
    )
    handle = transport.start_weight_sync(sources=inputs, plan=plan)
    transport.wait_weight_sync(handle)
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(consumer):
        transport.wait_weight_sync(handle)
        observed = [transport.projection_views(p)[0][0].clone() for p in range(2)]
    torch.cuda.current_stream().wait_stream(consumer)
    for p in range(2):
        torch.testing.assert_close(
            observed[p], torch.full_like(observed[p], sources[rank][0] * 10 + p)
        )
        transport.native_projection_grad_view(p).fill_(3)
        transport.projection_views(p)[1].fill_(rank + p + 1)
    native = tuple(
        ReplicaGradDestination(tuple(transport.native_projection_grad_view(p))) for p in range(2)
    )
    reduction = transport.start_grad_reduce(native_grads=native, plan=plan, projections=(0, 1))
    transport.wait_grad_reduce(reduction)
    replica_rank = next(r for r, row in enumerate(sources) if row[0] == rank)
    for p in range(2):
        torch.testing.assert_close(
            native[p].tensors[0], torch.full_like(native[p].tensors[0], 3 + replica_rank + p + 1)
        )
    # A newer plan must not alter the old plan's maps or use cached weight pointers.
    inactive = transport.prepare_plan(
        ReplicaPlacement(torch.full_like(table, -1), ReplicaOwnership(1), 2)
    )
    transport.wait_weight_sync(transport.start_weight_sync(sources=inputs, plan=inactive))
    fresh = tuple(ReplicaWeightSource((s.data[0] + 2,), None) for s in inputs)
    transport.wait_weight_sync(transport.start_weight_sync(sources=fresh, plan=plan))
    for p in range(2):
        weights = transport.projection_views(p)[0][0]
        torch.testing.assert_close(weights, torch.full_like(weights, sources[rank][0] * 10 + p + 2))

    # Real quota forward with the same Manager and independent router autograd.
    planner = UltraEPLoadPlanner(world, group, lambda: transport.manager)
    context = SchedulerContext(1, world, 1, (rank,), world, rank, 1, True)
    routes = torch.zeros(2048, world, dtype=torch.bool, device=device)
    routes[:, 0] = True
    probabilities = routes.float().requires_grad_()
    home, replicas, placement = planner.update_placement(probabilities, routes, context)
    assert home is None and replicas.shape == (world, 1)
    quota_plan = transport.prepare_plan(ReplicaPlacement(replicas, ReplicaOwnership(1), 3))
    transport.wait_weight_sync(transport.start_weight_sync(sources=fresh, plan=quota_plan))
    source = int(replicas[rank, 0])
    if source >= 0:
        for p in range(2):
            weights = transport.projection_views(p)[0][0]
            torch.testing.assert_close(weights, torch.full_like(weights, source * 10 + p + 2))
    rerouted, expanded = planner.reroute(probabilities, routes, placement, context)
    torch.testing.assert_close(expanded.sum(1), probabilities.sum(1))
    assert torch.equal(rerouted.sum(1), routes.sum(1))
    # Overwrite the manager slot before consuming the first forward's gradient.
    next_routes = torch.roll(routes, shifts=1, dims=1)
    _, _, next_placement = planner.update_placement(next_routes.float(), next_routes, context)
    planner.reroute(next_routes.float(), next_routes, next_placement, context)
    expanded.sum().backward()
    torch.testing.assert_close(probabilities.grad, routes.float())
    transport.destroy()
