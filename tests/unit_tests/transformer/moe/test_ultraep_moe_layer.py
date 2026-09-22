# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Full MoELayer numerical parity for UltraEP planning and replica communication."""

import json
from dataclasses import replace
from statistics import median
from time import perf_counter

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
    num_tokens = 16384
    num_iterations = 20
    num_warmup_iterations = 5
    single_rank_group = None
    for owner in range(world):
        candidate = dist.new_group([owner], backend="nccl")
        if owner == rank:
            single_rank_group = candidate
    pg = ProcessGroupCollection(
        ep=group,
        tp=single_rank_group,
        cp=single_rank_group,
        expt_tp=single_rank_group,
        expt_dp=single_rank_group,
        tp_ep=group,
        tp_cp=single_rank_group,
        tp_dp_cp=group,
    )
    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    monkeypatch.setenv(
        "ULTRA_EP_QUOTA_MIN_TOKENS_PER_REPLICA",
        "1" if replicate_hot_expert else str(num_tokens * world + 1),
    )
    # The minimum quota is soft; a relaxed balance target makes the existing
    # master-only placement feasible and deterministically avoids replication.
    monkeypatch.setenv("ULTRA_EP_BALANCE_THRESHOLD", "1" if replicate_hot_expert else str(world))
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    config = _scheduler_config(
        hidden_size=4096,
        moe_ffn_hidden_size=2048,
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
        # Positive inputs always select expert 0; keep logits moderate as hidden size grows.
        layer.router.weight[0].fill_(2.56 / config.hidden_size)
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

    observed_counts = {}
    timings = {}
    measurements = []
    case = {"replicate_hot_expert": replicate_hot_expert, "freeze_experts": freeze_experts}
    timing_names = (
        "reference_forward_ms",
        "reference_backward_ms",
        "ultraep_forward_ms",
        "ultraep_backward_ms",
    )

    def capture_counts(name):
        def capture(_module, inputs):
            # Read tokens_per_expert after token dispatch, before expert computation.
            # Retain the tensor here; gathering and printing happen outside the timers.
            observed_counts[name] = inputs[1]

        return capture

    hooks = [
        reference.experts.register_forward_pre_hook(capture_counts("reference")),
        layer.experts.register_forward_pre_hook(capture_counts("ultraep")),
    ]

    def timed(name, operation, *args):
        # Measure wall time including CPU scheduling and completion of GPU work.
        # Align ranks before starting; exclude this barrier and all diagnostics.
        dist.barrier(group=group)
        torch.cuda.synchronize()
        start = perf_counter()
        result = operation(*args)
        torch.cuda.synchronize()
        timings[name] = (perf_counter() - start) * 1000
        return result

    try:
        for iteration in range(num_iterations):
            for actual, expected in pairs:
                for parameter in (actual, expected):
                    parameter.main_grad.zero_()
                    parameter.grad = None
                    parameter.grad_added_to_main_grad = False
            layer.router.weight.grad = None
            reference.router.weight.grad = None
            hidden = torch.rand(
                num_tokens,
                1,
                config.hidden_size,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=not freeze_experts,
            )
            ref_hidden = hidden.detach().clone().requires_grad_(not freeze_experts)
            expected, _ = timed("reference_forward_ms", reference, ref_hidden)
            actual, _ = timed("ultraep_forward_ms", layer, hidden)
            sources = layer.experts._replica_expert_runtime.last_plan.experts_to_copy
            assert bool((sources >= 0).any()) == replicate_hot_expert
            overflow = layer.token_dispatcher.check_over_budget().to(torch.int32)
            dist.all_reduce(overflow, op=dist.ReduceOp.MAX, group=group)
            assert not overflow.item(), "The scheduler's default path must not drop tokens."
            torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.003)
            # Keep synthetic gradient magnitudes comparable as the token count grows.
            grad = torch.randn_like(actual) / num_tokens**0.5
            timed("ultraep_backward_ms", actual.backward, grad)
            timed("reference_backward_ms", expected.backward, grad)

            stats = torch.tensor(
                [
                    int(torch.as_tensor(observed_counts["reference"]).sum()),
                    int(torch.as_tensor(observed_counts["ultraep"]).sum()),
                    *(timings[name] for name in timing_names),
                ],
                device="cuda",
                dtype=torch.float64,
            )
            all_stats = [torch.empty_like(stats) for _ in range(world)]
            dist.all_gather(all_stats, stats, group=group)
            if rank == 0:
                rows = [values.cpu().tolist() for values in all_stats]
                report = {
                    **case,
                    "iteration": iteration + 1,
                    "warmup": iteration < num_warmup_iterations,
                    "reference_tokens": [int(row[0]) for row in rows],
                    "ultraep_tokens": [int(row[1]) for row in rows],
                    **{
                        name: [row[index + 2] for row in rows]
                        for index, name in enumerate(timing_names)
                    },
                }
                # Observed on 4x GB200, EP=4, 16384 tokens/rank,
                # hidden=4096, FFN=2048; both freeze_experts variants:
                # reference_tokens = [65536, 0, 0, 0] in every case.
                # ultraep_tokens = [65536, 0, 0, 0] with replicate_hot_expert=False,
                #                  [16548, 16548, 16548, 15892] with replicate_hot_expert=True.
                print("ULTRAEP_MOE_LAYER " + json.dumps(report), flush=True)
                if not report["warmup"]:
                    measurements.append(report)

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
                # Scale the update for positive inputs whose squared norm grows with width.
                # Each iteration must refresh replicas from the updated home weights.
                with torch.no_grad():
                    actual.add_(actual.main_grad.to(actual.dtype), alpha=-0.1 / config.hidden_size)
                    if rank == 0 and not freeze_experts:
                        # The hot expert must actually change, even with BF16 rounding.
                        assert not torch.equal(actual, expected)
                    expected.copy_(actual)
            layer.moe_scheduler.expert_dispatch.assert_idle()
        if rank == 0:
            # Take each phase's slowest rank, then sum forward/backward per iteration.
            # Report timing for observation only; performance is not a pass/fail criterion.
            summary = {
                **case,
                "warmup_iterations": num_warmup_iterations,
                "measured_iterations": len(measurements),
            }
            for name in ("reference", "ultraep"):
                samples = [
                    (max(row[f"{name}_forward_ms"]), max(row[f"{name}_backward_ms"]))
                    for row in measurements
                ]
                summary[f"{name}_median_max_rank_ms"] = {
                    "forward": median(forward for forward, backward in samples),
                    "backward": median(backward for forward, backward in samples),
                    "forward_backward": median(forward + backward for forward, backward in samples),
                }
            # Same GB200 run:
            # Median max-rank wall ms: 5 warmups, 15 measured iterations.
            # freeze_experts | replicate_hot_expert | reference fwd/bwd/total | ultraep fwd/bwd/total
            # False          | False                | 5.078/6.773/11.862      | 6.435/7.856/14.323
            # False          | True                 | 5.049/6.783/11.823      | 4.121/5.536/9.622
            # True           | False                | 5.055/2.715/7.773       | 6.417/7.775/14.184
            # True           | True                 | 5.106/2.739/7.844       | 4.142/5.426/9.551
            # Each total is the median of per-iteration forward+backward sums.
            print("ULTRAEP_MOE_LAYER_SUMMARY " + json.dumps(summary), flush=True)
    finally:
        for hook in hooks:
            hook.remove()
        layer.experts._replica_expert_runtime.destroy()
        reset_hybrid_ep_buffer()
        dist.destroy_process_group(single_rank_group)
