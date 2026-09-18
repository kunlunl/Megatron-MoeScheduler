# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Scheduler configuration shared by host contracts and GPU integration tests."""

import torch

from megatron.core.transformer.transformer_config import TransformerConfig


def _scheduler_config(**overrides) -> TransformerConfig:
    defaults = {
        "num_layers": 1,
        "hidden_size": 128,
        "num_attention_heads": 4,
        "num_moe_experts": 4,
        "moe_ffn_hidden_size": 128,
        "use_cpu_initialization": True,
        "bf16": True,
        "params_dtype": torch.bfloat16,
        "gated_linear_unit": True,
        "activation_func": torch.nn.functional.silu,
        "moe_router_topk": 1,
        "moe_router_pre_softmax": True,
        "moe_router_dtype": "fp32",
        "moe_grouped_gemm": True,
        "use_transformer_engine_op_fuser": True,
        "gradient_accumulation_fusion": True,
        "add_bias_linear": False,
        "moe_token_dispatcher_type": "flex",
        "moe_flex_dispatcher_backend": "hybridep",
        "moe_enable_scheduler": True,
        "moe_scheduler_num_idle_experts": 4,
        "moe_scheduler_expert_dispatcher_type": "replica_peer_tma",
    }
    defaults.update(overrides)
    return TransformerConfig(**defaults)
