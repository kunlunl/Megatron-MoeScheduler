# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Full MoELayer numerical parity for UltraEP planning and replica communication."""

from dataclasses import replace

import pytest
import torch
import torch.distributed as dist

pytestmark = pytest.mark.launch_on_gb200


@pytest.mark.parametrize("replicate_hot_expert", [False, True])
@pytest.mark.parametrize("freeze_experts", [False, True])
def test_ultraep_moe_layer_matches_unscheduled_alltoall(
    ultraep_group, monkeypatch, replicate_hot_expert, freeze_experts
):
    """Exercise real routing, HybridEP tokens, TE compute and replica backward together."""
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_submodules,
    )
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.moe.fused_a2a import HAVE_HYBRIDEP, reset_hybrid_ep_buffer
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.spec_utils import get_submodules
    from tests.unit_tests.transformer.moe.scheduler_test_utils import _scheduler_config

    if not HAVE_HYBRIDEP:
        pytest.skip("Requires the HybridEP token dispatcher.")
    group = ultraep_group
    rank, world = dist.get_rank(group), dist.get_world_size(group)
    tp_group = None
    for owner in range(world):
        candidate = dist.new_group([owner], backend="nccl")
        if owner == rank:
            tp_group = candidate
    pg = ProcessGroupCollection(
        ep=group,
        tp=tp_group,
        cp=tp_group,
        expt_tp=tp_group,
        expt_dp=tp_group,
        tp_ep=group,
        tp_cp=tp_group,
        tp_dp_cp=group,
    )
    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    monkeypatch.setenv(
        "ULTRA_EP_QUOTA_MIN_TOKENS_PER_REPLICA", "1" if replicate_hot_expert else "4096"
    )
    # The minimum quota is soft; a relaxed balance target makes the existing
    # master-only placement feasible and deterministically avoids replication.
    monkeypatch.setenv("ULTRA_EP_BALANCE_THRESHOLD", "1" if replicate_hot_expert else str(world))
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    config = _scheduler_config(
        num_moe_experts=world,
        expert_model_parallel_size=world,
        moe_scheduler_num_idle_experts=world,
        moe_scheduler_planner_type="ultra_ep",
        moe_scheduler_expert_dispatcher_type="replica_ultraep",
        moe_router_load_balancing_type="none",
        use_cpu_initialization=False,
    )
    submodules = get_submodules(
        get_gpt_layer_with_transformer_engine_submodules(
            num_experts=world, moe_grouped_gemm=True, use_te_op_fuser=True
        ).mlp
    )
    layer = MoELayer(config, submodules, layer_number=1, pg_collection=pg).cuda()
    reference = MoELayer(
        replace(config, moe_enable_scheduler=False, moe_token_dispatcher_type="alltoall"),
        submodules,
        layer_number=1,
        pg_collection=pg,
    ).cuda()
    torch.manual_seed(4321 + rank)
    with torch.no_grad():
        layer.router.weight.zero_()
        layer.router.weight[0].fill_(0.02)
        reference.router.weight.copy_(layer.router.weight)
    pairs = []
    for name in ("linear_fc1", "linear_fc2"):
        actual = getattr(layer.experts, name).weight0
        expected = getattr(reference.experts, name).weight0
        with torch.no_grad():
            actual.copy_(torch.randn_like(actual) / actual.shape[1] ** 0.5)
            expected.copy_(actual)
        for parameter in (actual, expected):
            parameter.requires_grad_(not freeze_experts)
            parameter.main_grad = torch.zeros_like(parameter, dtype=torch.float32)
            parameter.grad_added_to_main_grad = False
            parameter.zero_out_wgrad = True
        pairs.append((actual, expected))
    try:
        for _ in range(2):
            for actual, expected in pairs:
                for parameter in (actual, expected):
                    parameter.main_grad.zero_()
                    parameter.grad = None
                    parameter.grad_added_to_main_grad = False
            layer.router.weight.grad = None
            reference.router.weight.grad = None
            hidden = torch.rand(
                128, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=not freeze_experts
            )
            ref_hidden = hidden.detach().clone().requires_grad_(not freeze_experts)
            expected, _ = reference(ref_hidden)
            actual, _ = layer(hidden)
            sources = layer.experts._replica_expert_runtime.last_plan.experts_to_copy
            assert bool((sources >= 0).any()) == replicate_hot_expert
            overflow = layer.token_dispatcher.check_over_budget().to(torch.int32)
            dist.all_reduce(overflow, op=dist.ReduceOp.MAX, group=group)
            assert not overflow.item(), "The scheduler's default path must not drop tokens."
            torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.003)
            grad = torch.randn_like(actual) / 10
            actual.backward(grad)
            expected.backward(grad)
            if freeze_experts:
                assert hidden.grad is None and ref_hidden.grad is None
            else:
                torch.testing.assert_close(hidden.grad, ref_hidden.grad, rtol=0.03, atol=0.003)
            torch.testing.assert_close(
                layer.router.weight.grad, reference.router.weight.grad, rtol=0.03, atol=0.003
            )
            for actual, expected in pairs:
                torch.testing.assert_close(
                    actual.main_grad, expected.main_grad, rtol=0.03, atol=0.003
                )
                # A second step must refresh replicas from the updated home weights.
                with torch.no_grad():
                    actual.add_(actual.main_grad.to(actual.dtype), alpha=-0.01)
                    expected.copy_(actual)
            layer.moe_scheduler.expert_dispatch.assert_idle()
    finally:
        layer.experts._replica_expert_runtime.destroy()
        reset_hybrid_ep_buffer()
        dist.destroy_process_group(tp_group)
