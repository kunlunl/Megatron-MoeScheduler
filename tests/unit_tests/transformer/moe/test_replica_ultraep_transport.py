# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Run with torchrun on one NVLink domain and the pinned UltraEP PR #2 build."""

from dataclasses import replace

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
)
from megatron.core.transformer.moe.ultraep_moe_scheduler import UltraEPLoadPlanner

pytestmark = pytest.mark.launch_on_gb200


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
        # Completion must include remote replica-gradient clearing.
        replica_grad = transport.projection_views(p)[1]
        torch.testing.assert_close(replica_grad, torch.zeros_like(replica_grad))
    # Reducing the cleared replica buffers again must preserve accumulated native gradients.
    transport.wait_grad_reduce(
        transport.start_grad_reduce(native_grads=native, plan=plan, projections=(0, 1))
    )
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


def test_ultraep_fused_experts_forward_backward_and_empty_slots(ultraep_group, monkeypatch):
    """Compare dispatcher hooks with explicitly materialized, scheduler-free TE experts."""
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelGroupedLinear,
        TERowParallelGroupedLinear,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, TEGroupedMLP
    from megatron.core.transformer.moe.replica_expert_dispatch import ReplicaExpertDispatch
    from tests.unit_tests.transformer.moe.scheduler_test_utils import _scheduler_config

    group = ultraep_group
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    if world < 2:
        pytest.skip("Requires remote expert replicas.")
    tp_group = None
    for owner in range(world):
        candidate = dist.new_group([owner], backend="nccl")
        if owner == rank:
            tp_group = candidate
    pg = ProcessGroupCollection(ep=group, expt_tp=tp_group)
    config = _scheduler_config(
        num_moe_experts=world,
        expert_model_parallel_size=world,
        moe_scheduler_num_idle_experts=world,
        moe_scheduler_planner_type="ultra_ep",
        moe_scheduler_expert_dispatcher_type="replica_ultraep",
        use_cpu_initialization=False,
    )
    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    experts = TEGroupedMLP(
        num_local_experts=1,
        config=config,
        submodules=GroupedMLPSubmodules(
            linear_fc1=TEColumnParallelGroupedLinear, linear_fc2=TERowParallelGroupedLinear
        ),
        pg_collection=pg,
    )
    assert experts._with_fused_impl
    parameters = (experts.linear_fc1.weight0, experts.linear_fc2.weight0)
    torch.manual_seed(1234 + rank)
    gathered = []
    for parameter in parameters:
        with torch.no_grad():
            parameter.copy_(torch.randn_like(parameter) / parameter.shape[1] ** 0.5)
        parameter.main_grad = torch.zeros_like(parameter, dtype=torch.float32)
        parameter.grad_added_to_main_grad = False
        parameter.zero_out_wgrad = True
        peers = [torch.empty_like(parameter) for _ in range(world)]
        dist.all_gather(peers, parameter.detach(), group=group)
        gathered.append(peers)

    dispatcher = ReplicaExpertDispatch(config=config, pg_collection=pg)
    dispatcher.bind_experts(experts)
    runtime = dispatcher.runtime
    domain = runtime.transport.manager.nvl_domain_size
    sources = [r // domain * domain + (r + 1) % domain for r in range(world)]
    placement = torch.tensor(sources, device="cuda", dtype=torch.int32).view(world, 1)
    context = SchedulerContext(1, world, 1, (rank,), world, rank, 1, True, config, pg)
    reference_experts = TEGroupedMLP(
        num_local_experts=2,
        config=replace(config, moe_enable_scheduler=False, num_moe_experts=2 * world),
        submodules=GroupedMLPSubmodules(
            linear_fc1=TEColumnParallelGroupedLinear, linear_fc2=TERowParallelGroupedLinear
        ),
        pg_collection=pg,
    )
    reference_weights = []
    for slot, source in enumerate((rank, sources[rank])):
        weights = tuple(
            getattr(linear, f"weight{slot}")
            for linear in (reference_experts.linear_fc1, reference_experts.linear_fc2)
        )
        for projection, weight in enumerate(weights):
            with torch.no_grad():
                weight.copy_(gathered[projection][source])
            weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)
            weight.overwrite_main_grad = True
            weight.grad_added_to_main_grad = False
            weight.zero_out_wgrad = True
        reference_weights.append(weights)
    try:
        for counts in ((16, 16), (0, 16), (16, 0), (0, 0)):
            accumulated = 3 if sum(counts) == 0 else 0
            for parameter in parameters:
                parameter.main_grad.fill_(accumulated)
                parameter.grad = None
                parameter.grad_added_to_main_grad = False
            for weights in reference_weights:
                for weight in weights:
                    weight.main_grad.zero_()
                    weight.grad = None
                    weight.grad_added_to_main_grad = False
            hidden = torch.randn(
                sum(counts), 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            probs = torch.rand(sum(counts), device="cuda", requires_grad=True)
            ref_hidden = hidden.detach().clone().requires_grad_()
            ref_probs = probs.detach().clone().requires_grad_()
            # Use the same TE arithmetic without replica dispatch. A hand-written
            # torch MLP rounds different intermediate values in BF16.
            token_counts = torch.tensor(counts, device="cuda", dtype=torch.int32)
            if sum(counts) == 0:
                # The empty oracle has zero expert wgrad without invoking TE.
                reference = ref_hidden + ref_probs.sum().to(ref_hidden.dtype) * 0
            else:
                reference, _ = reference_experts(ref_hidden, token_counts, ref_probs)

            wrapped = dispatcher.wrap_layer_input(hidden)
            dispatcher.dispatch(experts, placement, context)
            wrapped = dispatcher.before_token_dispatch(wrapped)
            output, bias = experts(wrapped, token_counts, probs)
            assert bias is None
            output = dispatcher.before_token_combine(output)
            output = dispatcher.after_token_combine(output)
            torch.testing.assert_close(output, reference, rtol=0.03, atol=0.003)
            grad = torch.randn_like(output) / 10
            output.backward(grad)
            reference.backward(grad)
            torch.testing.assert_close(hidden.grad, ref_hidden.grad, rtol=0.03, atol=0.003)
            torch.testing.assert_close(probs.grad, ref_probs.grad, rtol=0.03, atol=0.003)
            for projection, parameter in enumerate(parameters):
                expected = torch.zeros(
                    (world, *parameter.shape), device="cuda", dtype=torch.float32
                )
                for source, weights in zip((rank, sources[rank]), reference_weights):
                    expected[source].add_(weights[projection].main_grad)
                dist.all_reduce(expected, group=group)
                torch.testing.assert_close(
                    parameter.main_grad, expected[rank] + accumulated, rtol=0.03, atol=0.003
                )
            dispatcher.assert_idle()
    finally:
        runtime.destroy()
        dist.destroy_process_group(tp_group)
