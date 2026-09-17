# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Home migration against independent logical optimizer and shard references."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.core.transformer.moe.eplb_moe_scheduler import EPLBLoadPlanner
from megatron.core.transformer.moe.home_expert_dispatch import HomeExpertDispatch
from megatron.core.transformer.moe.home_expert_state import (
    HomeStateShard,
    OptimizerHomeExpertState,
    ShardedHomeExpertState,
)
from megatron.core.transformer.moe.moe_scheduler import SchedulerContext
from megatron.core.transformer.moe.replica_nccl_transport import NcclP2PTransport
from megatron.core.transformer.moe.replica_weight_transport import (
    HomeExpertPlacement,
    ReplicaTransportConfig,
    ReplicaWeightTransport,
)
from tests.unit_tests.transformer.moe.test_replica_nccl_transport import distributed_world, ep_group

pytestmark = pytest.mark.launch_on_gb200


def _transport(group):
    return NcclP2PTransport(
        ReplicaTransportConfig(
            group=group,
            device=torch.device("cuda", torch.cuda.current_device()),
            world_size=dist.get_world_size(group),
            num_local_home_experts=2,
            num_local_replica_slots=1,
            member_shapes=((2, 3), (3, 2)),
            weight_format="bf16",
            rowwise_scale_shapes=None,
            columnwise_scale_shapes=None,
            grad_dtype=torch.float32,
            num_sms=None,
        )
    )


@pytest.mark.parametrize("overwrite", [False, True])
def test_replica_gradient_handoff_respects_fsdp_overwrite(distributed_world, overwrite):
    from megatron.core.transformer.moe.replica_expert_dispatch import _ReplicaWaitGradReduce

    parameter = torch.nn.Parameter(torch.zeros(2, 3, device="cuda", dtype=torch.bfloat16))
    parameter.main_grad = torch.full(
        (2, 3), float("nan") if overwrite else 7.0, device="cuda", dtype=torch.float32
    )
    parameter.grad_added_to_main_grad = False
    parameter.overwrite_main_grad = overwrite
    source = torch.arange(6, device="cuda", dtype=torch.float32).reshape(2, 3)
    context = SimpleNamespace(
        runtime=SimpleNamespace(
            source_parameters=(parameter,), wait_grad_reduce=lambda plan: (source,)
        ),
        context=SimpleNamespace(plan=None),
        num_source_parameters=1,
    )
    _ReplicaWaitGradReduce.backward(context, torch.ones(1, device="cuda"))
    expected = source if overwrite else source + 7
    torch.testing.assert_close(parameter.main_grad, expected, rtol=0, atol=0)
    assert parameter.grad_added_to_main_grad


def test_deferred_home_cycles_preserve_adam_continuation(ep_group):
    transport = _transport(ep_group)
    rank, size = dist.get_rank(ep_group), dist.get_world_size(ep_group)
    device = transport.config.device
    e = 2 * size
    shapes = transport.config.member_shapes
    reference = [
        torch.nn.Parameter(
            torch.arange(6, device=device).float().reshape(shape) / 9 + expert + projection * 10
        )
        for projection, shape in enumerate(shapes)
        for expert in range(e)
    ]
    physical = [
        torch.nn.Parameter(reference[projection * e + rank * 2 + slot].detach().clone())
        for projection in range(2)
        for slot in range(2)
    ]
    reference_optim = torch.optim.Adam(reference, lr=0.03)
    optim = torch.optim.Adam(physical, lr=0.03)
    runtime = SimpleNamespace(source_parameters=tuple(physical), transport=transport)
    home = HomeExpertDispatch()
    home.bind_transport(transport)
    home.bind_state_adapter(OptimizerHomeExpertState(optim, runtime))
    addresses = [p.data_ptr() for p in physical]
    logical_home = torch.arange(e, device=device)
    version = 0
    try:
        for step in range(5):
            permutation = None
            if step in (1, 3):
                version += 1
                permutation = (
                    torch.arange(e, device=device).roll(1)
                    if step == 1
                    else torch.arange(e, device=device).reshape(size, 2).flip(1).flatten()
                )
                before = [p.detach().clone() for p in physical]
                home.dispatch(
                    None, HomeExpertPlacement(permutation.reshape(size, 2), version), None
                )
                for p, old in zip(physical, before):
                    torch.testing.assert_close(p, old, rtol=0, atol=0)
            for projection in range(2):
                for expert in range(e):
                    p = reference[projection * e + expert]
                    p.grad = p.detach() * 0.02 + (expert + 1) * (step + 1) * 0.01
                for slot in range(2):
                    expert = logical_home[rank * 2 + slot].item()
                    p = physical[projection * 2 + slot]
                    p.grad = p.detach() * 0.02 + (expert + 1) * (step + 1) * 0.01
            reference_optim.step()
            optim.step()
            if permutation is not None:
                assert home.step() == version
                logical_home = logical_home[permutation]
                assert home.step() is None
            for projection in range(2):
                for slot in range(2):
                    p = physical[projection * 2 + slot]
                    expected = reference[projection * e + logical_home[rank * 2 + slot].item()]
                    torch.testing.assert_close(p, expected, rtol=0, atol=0)
                    for key in ("exp_avg", "exp_avg_sq", "step"):
                        torch.testing.assert_close(
                            optim.state[p][key],
                            reference_optim.state[expected][key],
                            rtol=0,
                            atol=0,
                        )
            assert [p.data_ptr() for p in physical] == addresses
    finally:
        transport.destroy()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.uint8])
def test_home_transport_mixed_shapes_and_repeated_stream_wait(ep_group, dtype):
    transport = _transport(ep_group)
    rank, size = dist.get_rank(ep_group), dist.get_world_size(ep_group)
    device = transport.config.device
    permutation = torch.arange(size * 2, device=device).flip(0).reshape(size, 2)
    sources = tuple(
        (torch.arange(2 * n, device=device).reshape(2, n) + rank * 2 * n).to(dtype) for n in (7, 19)
    )
    try:
        plan = transport.prepare_home_exchange(HomeExpertPlacement(permutation, 1))
        handle = transport.start_home_exchange(sources=sources, plan=plan)
        consumers = [torch.cuda.Stream(), torch.cuda.Stream()]
        results = []
        for stream in consumers:
            with torch.cuda.stream(stream):
                results.append(tuple(t.clone() for t in transport.wait_home_exchange(handle)))
        for stream in consumers:
            stream.synchronize()
        for result in results:
            for tensor, n in zip(result, (7, 19)):
                expected = (permutation[rank, :, None] * n + torch.arange(n, device=device)).to(
                    dtype
                )
                torch.testing.assert_close(tensor, expected, rtol=0, atol=0)
    finally:
        transport.destroy()


def test_home_exchange_uneven_and_empty_optimizer_shards(distributed_world):
    size = dist.get_world_size()
    if size < 4 or size % 4:
        pytest.skip("Requires EP2 x expert-DP2.")
    rank = dist.get_rank()
    ep = dp = None
    for base in range(0, size, 4):
        for ranks in ([base, base + 2], [base + 1, base + 3]):
            group = dist.new_group(ranks, backend="nccl")
            if rank in ranks:
                ep = group
        for ranks in ([base, base + 1], [base + 2, base + 3]):
            group = dist.new_group(ranks, backend="nccl")
            if rank in ranks:
                dp = group
    transport = _transport(ep)
    ep_rank, dp_rank = dist.get_rank(ep), dist.get_rank(dp)
    device = transport.config.device
    permutation = torch.tensor([[3, 2], [1, 0]], device=device)
    try:
        for cuts in ((0, 7), (3, 2)):
            shards = []
            for slot, cut in enumerate(cuts):
                start, end = (0, cut) if dp_rank == 0 else (cut, 7)
                value = torch.arange(start, end, device=device).float() + (ep_rank * 2 + slot) * 10
                shards.append(HomeStateShard(value, start, 7))
            pointers = [s.tensor.data_ptr() for s in shards]
            state = ShardedHomeExpertState(lambda: {"moment": shards}, device=device, dp_group=dp)
            home = HomeExpertDispatch()
            home.bind_transport(transport)
            home.bind_state_adapter(state)
            home.dispatch(None, HomeExpertPlacement(permutation, 1), None)
            home.step()
            for slot, shard in enumerate(shards):
                expected = (
                    torch.arange(shard.start, shard.start + shard.tensor.numel(), device=device)
                    + permutation[ep_rank, slot] * 10
                ).float()
                torch.testing.assert_close(shard.tensor, expected, rtol=0, atol=0)
            assert [s.tensor.data_ptr() for s in shards] == pointers
    finally:
        transport.destroy()
        dist.destroy_process_group(ep)
        dist.destroy_process_group(dp)


def test_eplb_pending_mapping_and_checkpoint(monkeypatch):
    context = SchedulerContext(1, 4, 2, (0, 1), 2, 0, 1, True)
    counts = torch.tensor([[100, 80, 3, 1], [0, 0, 0, 0]])
    planner = EPLBLoadPlanner(2, home_update_interval=1)
    monkeypatch.setattr(planner, "_get_count_matrix", lambda *args, **kwargs: counts)
    routes = torch.eye(4, dtype=torch.bool)
    probs = routes.float().requires_grad_()
    home, replica, result = planner.update_placement(probs, routes, context)
    assert home is None
    planner.step()
    home, replica, result = planner.update_placement(probs, routes, context)
    assert home is not None
    torch.testing.assert_close(planner.active_home_layout, torch.tensor([[0, 1], [2, 3]]))
    physical_routes, physical_probs = planner.reroute(probs, routes, result, context)
    assert physical_routes.sum() == routes.sum()
    physical_probs.sum().backward()
    torch.testing.assert_close(probs.grad, routes.float())
    # Additional microbatches neither overwrite the proposal nor switch ownership.
    again, _, _ = planner.update_placement(probs, routes, context)
    assert again is None
    with pytest.raises(RuntimeError, match="without home completion"):
        planner.step()
    with pytest.raises(RuntimeError, match="stale"):
        planner.step(home.version + 1)
    expected = planner.active_home_layout.flatten()[home.source_slots]
    planner.step(home.version)
    torch.testing.assert_close(planner.active_home_layout, expected)
    restored = EPLBLoadPlanner(2, home_update_interval=1)
    restored.load_state_dict(planner.state_dict())
    torch.testing.assert_close(restored.active_home_layout, expected)


def test_peer_tma_home_interface_is_explicit_placeholder():
    # Base implementation is inherited by Peer-TMA; replica APIs remain untouched.
    with pytest.raises(NotImplementedError, match="home expert exchange"):
        ReplicaWeightTransport.prepare_home_exchange(
            SimpleNamespace(transport_name="replica_peer_tma"), None
        )


def test_home_rejects_nonpermutation_and_stale_plans(ep_group):
    transport = _transport(ep_group)
    size = dist.get_world_size(ep_group)
    try:
        with pytest.raises(ValueError, match="permutation"):
            transport.prepare_home_exchange(
                HomeExpertPlacement(torch.zeros(size, 2, dtype=torch.int64), 1)
            )
        home = HomeExpertDispatch()
        home.bind_transport(transport)
        plan = HomeExpertPlacement(torch.arange(size * 2).reshape(size, 2), 1)
        home.dispatch(None, plan, None)
        with pytest.raises(RuntimeError, match="pending"):
            home.dispatch(None, plan, None)
        with pytest.raises(RuntimeError, match="optimizer home state"):
            home.step()
    finally:
        transport.destroy()


@pytest.mark.parametrize("recipe", ["ddp", "distopt", "fsdp", "distopt_overlap", "fsdp_overlap"])
def test_mcore_optimizer_storage_and_continuation(distributed_world, recipe):
    """Exercise real wrapper buffers and optimizer shards, against logical Adam."""
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.moe.home_expert_state import optimizer_leaves
    from megatron.core.transformer.moe.home_expert_training import finish_home_expert_step
    from megatron.core.transformer.transformer_config import TransformerConfig
    from tests.unit_tests.test_utilities import Utils

    if dist.get_world_size() < 4 or dist.get_world_size() % 2:
        pytest.skip("Requires EP2 and at least two optimizer DP ranks.")
    overlap = recipe.endswith("_overlap")
    recipe = recipe.removesuffix("_overlap")
    Utils.initialize_model_parallel(expert_model_parallel_size=2, expert_tensor_parallel_size=1)
    pg = ProcessGroupCollection.use_mpu_process_groups()
    rank = dist.get_rank(pg.ep)
    device = torch.device("cuda", torch.cuda.current_device())
    transport = _transport(pg.ep)
    cfg = TransformerConfig(
        num_layers=1,
        num_attention_heads=1,
        hidden_size=128,
        num_moe_experts=4,
        expert_model_parallel_size=2,
        bf16=True,
        params_dtype=torch.bfloat16,
    )

    class ExpertSlots(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # FSDP classifies experts using the same .experts. module path as MoELayer.
            self.layer = torch.nn.Module()
            self.layer.experts = torch.nn.Module()
            self.layer.experts.fc1 = torch.nn.ModuleList(
                [torch.nn.Linear(3, 2, bias=False) for _ in range(2)]
            )
            self.layer.experts.fc2 = torch.nn.ModuleList(
                [torch.nn.Linear(2, 3, bias=False) for _ in range(2)]
            )
            self.logical = [rank * 2, rank * 2 + 1]
            self.config = cfg
            for p in self.parameters():
                p.allreduce = False

        def forward(self):
            loss = 0
            for layers in (self.layer.experts.fc1, self.layer.experts.fc2):
                for slot, layer in enumerate(layers):
                    x = torch.ones(1, layer.in_features, device=device, dtype=torch.bfloat16)
                    loss = loss + layer(x * (self.logical[slot] + 1)).float().square().sum()
            return loss

    native = ExpertSlots().to(device=device, dtype=torch.bfloat16)
    parameters = tuple(native.parameters())
    reference = [
        torch.nn.Parameter(torch.full(shape, (projection * 4 + expert + 1) / 128, device=device))
        for projection, shape in enumerate(transport.config.member_shapes)
        for expert in range(4)
    ]
    with torch.no_grad():
        for projection in range(2):
            for slot in range(2):
                parameters[projection * 2 + slot].copy_(reference[projection * 4 + rank * 2 + slot])
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=recipe != "ddp",
        use_megatron_fsdp=recipe == "fsdp",
        data_parallel_sharding_strategy="optim_grads_params" if recipe == "fsdp" else "no_shard",
        overlap_grad_reduce=overlap,
        overlap_param_gather=overlap,
    )
    wrapper = FullyShardedDataParallel if recipe == "fsdp" else DistributedDataParallel
    wrapper_kwargs = {"fsdp_unit_modules": [torch.nn.Linear]} if recipe == "fsdp" else {}
    model = wrapper(
        config=cfg, ddp_config=ddp_config, module=native, pg_collection=pg, **wrapper_kwargs
    )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            lr=0.001,
            weight_decay=0,
            bf16=True,
            use_distributed_optimizer=recipe != "ddp",
            overlap_param_gather=overlap,
            clip_grad=0,
        ),
        [model],
        pg_collection=pg,
        use_gloo_process_groups=False,
    )
    ref_optim = torch.optim.Adam(reference, lr=0.001)
    runtime = SimpleNamespace(source_parameters=parameters, transport=transport)
    state = OptimizerHomeExpertState(optimizer, runtime)
    home = HomeExpertDispatch()
    home.bind_transport(transport)
    home.bind_state_adapter(state)
    logical = torch.arange(4, device=device)
    permutation = torch.tensor([[3, 2], [1, 0]], device=device)
    try:
        for step in range(3):
            model.zero_grad_buffer()
            optimizer.zero_grad()
            ref_optim.zero_grad()
            loss = model()
            loss.backward()
            model.finish_grad_sync()
            for projection, shape in enumerate(transport.config.member_shapes):
                for expert in range(4):
                    p = reference[projection * 4 + expert]
                    x = torch.ones(1, shape[1], dtype=torch.bfloat16, device=device) * (expert + 1)
                    # The wrapper sums identical expert-DP contributions and
                    # normalizes by the full DP size, including EP ranks.
                    reference_loss = (
                        torch.nn.functional.linear(x, p.to(torch.bfloat16)).float().square().sum()
                    )
                    (reference_loss * (pg.expt_dp.size() / pg.dp.size())).backward()
            if step == 0:
                home.dispatch(None, HomeExpertPlacement(permutation, 1), None)
                for leaf in optimizer_leaves(optimizer):
                    leaf._home_exchange_defer_param_sync = True
            success, _, _ = optimizer.step()
            assert success
            ref_optim.step()
            # Exercise the same refresh/gather boundary as the training loop.
            finish_home_expert_step([SimpleNamespace(step=home.step)], optimizer, success)
            if step == 0:
                logical = logical[permutation.flatten()]
                native.logical = logical[rank * 2 : rank * 2 + 2].tolist()
            observed = state.snapshot()
            # Component order is sorted: projection, exp_avg, exp_avg_sq, master.
            for projection in range(2):
                for component, key in enumerate(("exp_avg", "exp_avg_sq", "master")):
                    expected = []
                    for slot in range(2):
                        p = reference[projection * 4 + logical[rank * 2 + slot].item()]
                        expected.append(
                            (p if key == "master" else ref_optim.state[p][key]).flatten()
                        )
                    torch.testing.assert_close(
                        observed[projection * 3 + component],
                        torch.stack(expected),
                        rtol=2e-5,
                        atol=2e-6,
                    )
    finally:
        transport.destroy()
        Utils.destroy_model_parallel()
