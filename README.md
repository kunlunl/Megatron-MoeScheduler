<div align="center">

Megatron-LM & Megatron Core
===========================

<h4>GPU-optimized library for training transformer models at scale</h4>

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://docs.nvidia.com/Megatron-Core/developer-guide/latest/index.html)
[![version](https://img.shields.io/badge/release-0.12.0-green)](./CHANGELOG.md)
[![license](https://img.shields.io/badge/license-Apache-blue)](./LICENSE)

<div align="left">

> ## 🚨 **DEVELOPMENT BRANCH**
> ⚠️ **EXPERIMENTAL FEATURES** - This is the **dev branch** with experimental features. 
>
> **→ For releases and comprehensive documentation, visit the [main branch](https://github.com/NVIDIA/Megatron-LM)**

## ⚡ Quickstart

```bash
# Clone the dev branch
git clone -b dev https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM

# Install from source with dev dependencies (includes transformer_engine)
pip install -e .[mlm,dev]
```

**Megatron Core** is a composable library with GPU-optimized building blocks for custom training frameworks. It provides transformer building blocks, advanced parallelism strategies (TP, PP, DP, EP, CP), mixed precision support (FP16, BF16, FP8, FP4), and model architectures. Best for framework developers and ML engineers building custom training pipelines.

**[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)** provides bidirectional Hugging Face ↔ Megatron checkpoint conversion with production-ready recipes.

## Getting Started

**Install from PyPI:**

```bash
uv pip install megatron-core
```

**Or clone and install from source:**

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
uv pip install -e .
```

> **Note:** Building from source can use a lot of memory. If the build runs out of memory, limit parallel compilation jobs by setting `MAX_JOBS` (e.g. `MAX_JOBS=4 uv pip install -e .`).

For NGC container setup and all installation options, see the **[Installation Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/install.html)**.

- **[Your First Training Run](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/quickstart.html)** - End-to-end training examples with data preparation
- **[Parallelism Strategies](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)** - Scale training across GPUs with TP, PP, DP, EP, and CP
- **[Contribution Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/developer/contribute.html)** - How to contribute to Megatron Core

# Latest News

- **[2026/03]** **Deprecating Python 3.10 support:** We're officially dropping Python 3.10 support with the upcoming 0.17.0 release. Downstream applications must raise their lower boundary to 3.12 to stay compatible with MCore.
- **[2026/01]** **[Dynamic Context Parallelism](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/)** - Up to 1.48x speedup for variable-length sequence training with adaptive CP sizing.
- **[2025/12]** **Megatron Core development has moved to GitHub!** All development and CI now happens in the open. We welcome community contributions.
- **[2025/10]** **[Megatron Dev Branch](https://github.com/NVIDIA/Megatron-LM/tree/dev)** - early access branch with experimental features.
- **[2025/10]** **[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)** - Bidirectional converter for interoperability between Hugging Face and Megatron checkpoints, featuring production-ready recipes for popular models.
- **[2025/08]** **[MoE Q3-Q4 2025 Roadmap](https://github.com/NVIDIA/Megatron-LM/issues/1729)** - Comprehensive roadmap for MoE features including DeepSeek-V3, Qwen3, advanced parallelism strategies, FP8 optimizations, and Blackwell performance enhancements.
- **[2025/08]** **[GPT-OSS Model](https://github.com/NVIDIA/Megatron-LM/issues/1739)** - Advanced features including YaRN RoPE scaling, attention sinks, and custom activation functions are being integrated into Megatron Core.
- **[2025/06]** **[Megatron MoE Model Zoo](https://github.com/yanring/Megatron-MoE-ModelZoo)** - Best practices and optimized configurations for training DeepSeek-V3, Mixtral, and Qwen3 MoE models with performance benchmarking and checkpoint conversion tools.
- **[2025/05]** Megatron Core v0.11.0 brings new capabilities for multi-data center LLM training ([blog](https://developer.nvidia.com/blog/turbocharge-llm-training-across-long-haul-data-center-networks-with-nvidia-nemo-framework/)).

<details>
<summary>Table of Contents</summary>

**Getting Started**
- [⚡ Quick Start](#-quick-start)
- [🧠 Dev Branch Philosophy](#-dev-branch-philosophy)
- [MoE Scheduler](#moe-scheduler-experimental)
- [📊 Performance & Benchmarking](#-performance--benchmarking)
- [👥 Community & Support](#-community--support)

**For Complete Documentation** → [Main Branch](https://github.com/NVIDIA/Megatron-LM) | [Official Docs](https://docs.nvidia.com/Megatron-Core/)

</details>


## Dev Branch Philosophy

# Project Structure

```
Megatron-LM/
├── megatron/
│   ├── core/                    # Megatron Core (kernels, parallelism, building blocks)
│   │   ├── models/              # Transformer models
│   │   ├── transformer/         # Transformer building blocks
│   │   ├── tensor_parallel/     # Tensor parallelism
│   │   ├── pipeline_parallel/   # Pipeline parallelism
│   │   ├── distributed/         # Distributed training (FSDP, DDP)
│   │   ├── optimizer/           # Optimizers
│   │   ├── datasets/            # Dataset loaders
│   │   ├── inference/           # Inference engines and server
│   │   └── export/              # Model export (e.g. TensorRT-LLM)
│   ├── training/                # Training scripts
│   ├── legacy/                  # Legacy components
│   ├── post_training/           # Post-training (quantization, distillation, pruning, etc.)
│   └── rl/                      # Reinforcement learning (RLHF, etc.)
├── examples/                    # Ready-to-use training examples
├── tools/                       # Utility tools
├── tests/                       # Comprehensive test suite
└── docs/                        # Documentation
```

# MoE Scheduler (Experimental)

MoE Scheduler addresses load imbalance in dropless MoE models by replicating
hot logical experts into idle physical expert slots and rerouting tokens to the
replicas. It is designed as a preprocessing stage between router output and the
existing Megatron token dispatcher.

The scheduler does not introduce planner-specific branches into the remaining
MoE execution path. After scheduling, `MoELayer` continues through its normal
token preprocessing, dispatch, expert computation, combine, and backward flow.

## Design Goals

- Decouple load planning from expert-weight movement so planners and expert
  dispatch backends can be combined through one contract.
- Keep backend-native state, such as Echo offloading maps, EPLB replica tables,
  and MoonEP replica plans, private to concrete implementations.
- Split placement from token rerouting so asynchronous expert-weight dispatch
  can overlap reroute work without exposing planner-specific state.
- Make planners return final token reroute tensors so token dispatchers do not
  need to understand the selected planning algorithm.
- Return the original router output unchanged when the planner gate decides
  that no planning is required.
- Keep the public planner/dispatcher boundary small enough to lower efficiently
  into different expert communication backends.

## Architecture

| Component | Responsibility |
| --- | --- |
| `SchedulerContext` | Carries layer, logical/local expert, EP rank/group, router top-k, training mode, and configuration context. |
| `MoEPlacementResult` | Opaque, explicit state passed between a planner's placement and reroute phases. |
| `MoELoadPlanner` | Gates planning, updates the physical layout, and reroutes tokens against that placement. |
| `ExpertDispatch` | Validates/materializes a physical placement and owns forward/backward lifecycle hooks. |
| `MoEScheduler` | Orchestrates placement, asynchronous expert materialization, and token rerouting. |
| `MoELayer` | Invokes the scheduler between routing and token preprocessing. |

The planner contract is split into two phases:

```python
home_expert_placement, replica_expert_placement, placement_result = planner.update_placement(
    probs, routing_map, context
)
routing_map, probs = planner.reroute(
    probs, routing_map, placement_result, context
)
```

Logical mappings are private to the planner. `home_expert_placement` is an
optional `HomeExpertPlacement(source_slots, version)` with a `[EP, H]` source
permutation. `replica_expert_placement` is an optional `[EP, S]` integer table,
with `-1` for inactive replicas. Both tables are destination-major and encode
**physical** sources as `rank * H + local_home_slot`. `placement_result` is
opaque reroute state. `reroute()` returns dense physical route/probability tensors.

`MoEScheduler` calls `HomeExpertDispatch` and `ReplicaExpertDispatch` directly.
Home dispatch stages metadata until `scheduler.step()`; replica dispatch starts
weight prefetch immediately and retains the source table through backward. Both
use the same transport implementation. Replica runtime owns TE/GTP bindings and
gradient handoff. Transport never interprets logical identity or optimizer state.

## Class Diagram

```mermaid
classDiagram
    MoELayer --> MoEScheduler
    MoEScheduler o-- MoELoadPlanner
    MoEScheduler o-- ReplicaExpertDispatch
    MoEScheduler o-- HomeExpertDispatch
    ExpertDispatch <|-- HomeExpertDispatch
    HomeExpertDispatch --> ReplicaWeightTransport
    MoELoadPlanner <|-- EchoLoadPlanner
    MoELoadPlanner <|-- EPLBLoadPlanner
    MoELoadPlanner <|-- MoonEPLoadPlanner
    ExpertDispatch <|-- ReplicaExpertDispatch
    MoEPlacementResult <|-- EchoPlacementResult
    MoEPlacementResult <|-- EPLBPlacementResult
    MoEPlacementResult <|-- MoonEPPlacementResult
    EchoLoadPlanner --> EchoPlacementResult
    EPLBLoadPlanner --> EPLBPlacementResult
    MoonEPLoadPlanner --> MoonEPPlacementResult
    MoEScheduler --> ReplicaExpertDispatch
    ReplicaExpertDispatch o-- ReplicaExpertRuntime
    ReplicaExpertRuntime o-- ReplicaWeightTransport
    ReplicaWeightTransport <|-- PeerTmaTransport
    ReplicaWeightTransport <|-- HybridEPWeightTransport
    ReplicaWeightTransport <|-- NcclP2PTransport
    ReplicaWeightTransport --> ReplicaPreparedPlan
    ReplicaPreparedPlan --> ReplicaPlacement
    ReplicaPlacement --> ReplicaOwnership
```

The same architecture is available as standalone PlantUML sources:

- [MoEScheduler class diagram](docs/diagrams/moe_scheduler_class.puml)
- [MoEScheduler forward/backward activity diagram](docs/diagrams/moe_scheduler_activity.puml)

## Implemented Components

| Type | Config value | Implementation | Status |
| --- | --- | --- | --- |
| Planner | `echo` | `EchoLoadPlanner` | CUDA/Triton assignment and token reroute aligned with Echo PR #2368. |
| Planner | `eplb` | `EPLBLoadPlanner` | Greedy replication, optional periodic home LPT permutations, and global round-robin rerouting. |
| Planner | `moon_ep` | `MoonEPLoadPlanner` | PR #6892 fused per-step placement: one cooperative kernel with symmetric-memory histogram exchange. |
| Planner | `ultra_ep` | `UltraEPLoadPlanner` | PR #2 quota forward with saved routing for backward; fixed homes. |
| Expert dispatch | `replica_peer_tma` | `ReplicaExpertDispatch` | One replica lifecycle implementation; the type currently selects `PeerTmaTransport`. |
| Expert dispatch | `replica_hybridep` | Same dispatcher, `HybridEPWeightTransport` | Placeholder; raises `NotImplementedError` before transport allocation. |
| Expert dispatch | `replica_nccl` | Same dispatcher, `NcclP2PTransport` | Packed NCCL P2P; BF16/MXFP8 weights, BF16/FP32 gradients, host planning. |
| Expert dispatch | `replica_ultraep` | Same dispatcher, `UltraEPTransport` | PR #2 explicit placement; BF16 weights, FP32 joint gradients, host validation and EP fences. |

UltraEP PR #2 is available through the experimental `ultra_ep` planner and
`replica_ultraep` weight transport. See [UltraEP integration](#ultraep-integration-pr-2)
for the pinned API, first-version limits and validation commands.

## Execution Flow

1. The dispatcher wraps the MoE layer input when backward work must complete
   after router, shared-expert, and latent-projection backward.
2. The router produces logical-expert `probs` and `routing_map`.
3. `MoELayer` calls `MoEScheduler.schedule()`.
4. `MoELoadPlanner.should_plan()` decides whether scheduling is needed.
5. If planning is skipped, the original router tensors are returned unchanged.
6. `update_placement()` returns optional home/replica source tables and an
   opaque `MoEPlacementResult`.
7. The scheduler stages a home proposal and asynchronously starts current-home
   replica copies. Pending home ownership is never used by this forward.
8. `reroute()` converts the original logical routes to physical routes while
   expert-weight communication is in flight.
9. The existing token dispatcher consumes the physical routing tensors and
   runs the normal dispatch, expert compute, and combine stages.
10. During backward, transports with separate projection reductions start FC2
    after its wgrad GEMM. UltraEP waits until both FC1/FC2 gradients are ready
    after dispatch backward. The runtime waits at the layer input before
    publishing source gradients.

## Configuration

A minimal Echo configuration is:

```yaml
moe_enable_scheduler: true
moe_scheduler_planner_type: echo
moe_scheduler_expert_dispatcher_type: replica_peer_tma
moe_scheduler_num_idle_experts: 4
moe_scheduler_assignment_algorithm: approx_bin_packing
```

**Configuration migration:** the former `replica_hybridep` value selected
Peer-TMA, despite its name. It has been renamed to `replica_peer_tma`, which
is also the default. Update existing YAML/CLI configurations to that value.
`replica_hybridep` is now reserved for the actual HybridEP weight backend and
fails with a migration message; it never silently selects a different data
path. All replica backends construct `ReplicaExpertDispatch` from
`replica_expert_dispatch.py`; the common dispatcher has no transport-specific
name or compatibility alias. The HybridEP placeholder is accepted by
configuration validation but cannot execute or bind weights.

Set `moe_scheduler_expert_dispatcher_type: replica_nccl` to use packed NCCL
P2P with BF16 or native MXFP8 weights and BF16 or FP32 gradient storage. It requires an
initialized NCCL EP group and the current CUDA device to match the runtime
device. Construction is collective across that group. CUDA graph capture
including MoE is rejected; attention-only graphs remain supported. MXFP8
requires `fp8: e4m3`, `fp8_recipe: mxfp8`, and `fp8_param: true`.

For MoonEP, set `moe_scheduler_planner_type: moon_ep`. The current MoonEP
implementation allocates one replica slot for every home expert, so
`moe_scheduler_num_idle_experts` must equal `num_moe_experts`.

For EPLB, set `moe_scheduler_planner_type: eplb`. `EPLBLoadPlanner` gathers the
current logical-expert loads across the EP group, greedily assigns redundant
instances according to `load / replica_count`, and LPT-packs those instances
into the fixed replica slots. It follows the core replication and packing
policy of [DeepSeek EPLB](https://github.com/deepseek-ai/EPLB). Homes remain fixed
by default; `moe_scheduler_home_update_interval` enables periodic migration.

All planners use the transport-backed `ReplicaExpertRuntime`. It supports an
`E + R` runtime layout, where `R` is positive and divisible by the EP size.
Every rank owns `E / EP` native experts followed by `R / EP` replica slots. It
requires the HybridEP flex token dispatcher, BF16 execution, TE grouped GEMM
with the operation fuser, and fused gradient accumulation. With Peer-TMA or NCCL, weights may be BF16
or native-parameter MXFP8 E4M3; FP4 and other FP8 recipes are rejected. The
current `replica_peer_tma` type selects `PeerTmaTransport`, which requires a
single NVLink domain and a PyTorch/NCCL build with working native NCCL
symmetric-memory support.

Current constraints:

- Dropless MoE only; expert capacity and capacity padding must be disabled.
- `num_moe_experts` and `moe_scheduler_num_idle_experts` must be divisible by
  the expert-model-parallel size.
- `add_bias_linear` must be disabled.
- The `replica_peer_tma` expert dispatcher requires discrete native expert
  weights and the Transformer Engine operation fuser; single grouped expert
  weights are unsupported.
- The HybridEP flex token dispatcher requires a build with HybridEP support.
- The MoonEP planner requires CUDA, initialized EP distributed groups, and one
  replica slot per local home expert. Its current #6892 planner uses one fused
  cooperative kernel and an NCCL symmetric-memory histogram window.
- The EPLB planner currently uses the current step's EP-global token counts and
  global placement rather than a historical moving-average load predictor or
  hierarchical multi-node expert-group placement.
- CUDA graph capture is supported only at the whole-MoE scope for
  `replica_peer_tma`; separate `moe_router` and `moe_preprocess` scopes are
  rejected. MoE activation recompute is also rejected for this lifecycle.
  The physical-expert token-dispatcher config does not own graph capture;
  it clears its layer-level graph policy while the enclosing logical MoE
  layer retains the requested capture scopes. Dispatcher kernels execute
  within that enclosing graph. MoonEP converts replica placement to the common
  physical layout with fixed-shape selection, so changing inactive replica slots
  on graph replay never requires dynamic-size boolean indexing.
- When GTP exposes #6892's non-consuming peek protocol, the weight push peeks
  at gathered weights and the expert GEMMs perform the real consume. Older GTP
  implementations retain the pre-existing consume-at-push compatibility path.

Rank 0 logs the configured planner, dispatcher, idle expert count, and
assignment algorithm at startup. The first scheduled forward also logs routing
shapes, physical expert and transfer counts, and whether planning and expert
materialization ran.

## Code Map

| Path | Purpose |
| --- | --- |
| `megatron/core/transformer/moe/moe_scheduler.py` | Shared two-phase planner interfaces and scheduler orchestration. |
| `megatron/core/transformer/moe/echo_moe_scheduler.py` | Echo planner and Triton reroute path. |
| `megatron/core/transformer/moe/eplb_moe_scheduler.py` | EPLB greedy replication, fixed-home LPT placement, and round-robin reroute. |
| `megatron/core/transformer/moe/moonep_moe_scheduler.py` | MoonEP/PR #6892 planner adapter and common-IR conversion. |
| `megatron/core/transformer/moe/moonep_replica_triton.py` | #6892 histogram/placement and route-mapping Triton kernels. |
| `megatron/core/transformer/moe/ultraep_moe_scheduler.py` | UltraEP quota planner and saved routing for probability gradients. |
| `megatron/core/transformer/moe/replica_expert_dispatch.py` | Common placement adapter and replica forward/backward lifecycle. |
| `megatron/core/transformer/moe/replica_expert_runtime.py` | TE/GTP runtime weights, gradient handoff, and transport coordination. |
| `megatron/core/transformer/moe/replica_weight_transport.py` | Backend-neutral replica weight/gradient transport contract and factory. |
| `megatron/core/transformer/moe/replica_peer_tma_transport.py` | Symmetric-memory peer-TMA transport and shared workspace. |
| `megatron/core/transformer/moe/replica_hybridep_transport.py` | Reserved HybridEP weight backend; not implemented. |
| `megatron/core/transformer/moe/replica_nccl_transport.py` | Packed NCCL P2P, host schedules, and FP32 gradient accumulation. |
| `megatron/core/transformer/moe/replica_ultraep_transport.py` | Explicit UltraEP maps, weight synchronization and joint FP32 gradient reduction. |
| `megatron/core/transformer/moe/ultraep_backend.py` | Optional dependency/API checks and collective Manager lifetime. |
| `megatron/core/transformer/moe/replica_weight_triton.py` | #6892 weight transport and projection-selective gradient-reduction kernels, generalized to variable replica slots. |
| `megatron/core/transformer/moe/moe_layer.py` | Integration between logical routing and the existing token dispatcher. |
| `megatron/core/transformer/transformer_config.py` | Scheduler configuration and compatibility validation. |
| `tests/unit_tests/transformer/moe/test_moe_scheduler.py` | Common contract and orchestration tests. |
| `tests/unit_tests/transformer/moe/test_echo_moe_scheduler.py` | Echo planner, unified replica dispatch, and `MoELayer` integration tests. |
| `tests/unit_tests/transformer/moe/test_eplb_moe_scheduler.py` | EPLB placement, global reroute, gradient, and factory tests. |
| `tests/unit_tests/transformer/moe/test_moonep_moe_scheduler.py` | MoonEP planner and cross-component compatibility tests. |
| `tests/unit_tests/transformer/moe/test_replica_weight_transport.py` | Plan ownership, layout, completion lifetime, and placeholder contracts. |
| `tests/unit_tests/transformer/moe/test_replica_nccl_transport.py` | NCCL weight/gradient parity, strided EP subgroups, empty ranks, and stream completion. |
| `tests/unit_tests/transformer/moe/test_ultraep_moe_scheduler.py` | UltraEP API, placement, configuration and saved-router contracts. |
| `tests/unit_tests/transformer/moe/test_replica_ultraep_transport.py` | Native UltraEP communication, fused TE and full MoELayer parity. |

## Replica Transport Contract

The planner lowers logical choices to physical source slots. Replica dispatch
retains an immutable source table through forward/backward and assigns a
dispatcher-local generation. `ReplicaExpertRuntime` wraps that slot table in
`ReplicaPlacement`, with a separate `ReplicaOwnership` descriptor. The current
owner mapping is uniform and fixed; optional explicit owner tables reserve an
extension point and are rejected by Peer-TMA and NCCL P2P. Periodic home exchange
moves contents among these fixed physical sources, so replica transport needs
no logical ownership table. Each rank still has H home slots followed by S replicas.

`transport.prepare_plan(placement)` produces a `ReplicaPreparedPlan` belonging
to that exact transport instance. Peer-TMA keeps device metadata; NCCL
compiles host peer schedules, while a future HybridEP implementation can
compile chunk routing. The common layer never expands chunk routing or reads the
placement back to the CPU. Runtime caches are weak and keyed by plan identity,
not the address of a recycled slot. A future host-scheduled backend must
document synchronization and reject unsupported graph capture. CUDA graph
replay can update device metadata without executing Python or incrementing a
Python generation: cached host schedules must never assume otherwise.

Each `ReplicaWeightSource` exposes canonical local-expert data and optional
scales, with explicit plain/rowwise/columnwise layout. Tensor shape and dtype
describe the components; wire encoding and packing belong to the transport.
Optional pointer tables remain a Peer-TMA fast path. TE wrappers, GTP peeks,
direction selection and optimizer handoff remain in the runtime.

Each start returns a distinct `ReplicaTransferHandle`, separate from reusable
scheduling metadata. A weight completion covers unpack and final layout;
a gradient completion covers `native += sum(replicas)`, including local
replicas, with FP32 accumulation and one final cast. Native gradients are not
cleared. Reduction order can differ between backends; bitwise equivalence is
not promised. Every wait orders its calling stream, including repeated waits
for resident weights. Handles retain operation inputs, but backends must also
protect allocator lifetimes and callers must not overwrite sources or buffers
while the GPU uses them.

Storage views remain stable until teardown. The existing Peer-TMA shared
workspace and its serialized layer lifecycle are preserved; the interface
does not promise independent concurrent storage per layer. New transports
should first use per-layer buffers before introducing shared pools. Only
initialized backends register finalizers, so cleanup does not import unused
Triton/symmetric-memory dependencies. Capabilities advertise implemented
weight formats, gradient dtypes, device planning, graph support and explicit
ownership; placeholders advertise none. Topology compatibility remains a
backend initialization requirement, not an automatic fallback policy.

NCCL P2P can move weights across nodes without symmetric-memory allocation.
Full scheduler execution also requires compatible planner and token-dispatch paths; current scheduler
configuration still requires HybridEP tokens and the MoonEP planner still
uses symmetric-memory histogram exchange.

## NCCL P2P Transport Design

The NCCL backend supports plain BF16 or MXFP8 weights and either BF16 or FP32
replica/native gradient storage. It does not import Peer-TMA, Triton, or
symmetric-memory helpers. TE parameter wrappers, GTP materialization, backward
weight selection and optimizer handoff remain in `ReplicaExpertRuntime`.

`prepare_plan()` performs one synchronous device-to-host copy of the global
`[EP, replica_slots]` placement per immutable microbatch plan. It validates
physical source IDs, skips `-1`, and derives owners using uniform canonical
ownership. The prepared schedule stores only peer ranks, native indices and
replica slots. It never retains source addresses, so backward may use newly
materialized weights. All ranks must provide the same placement; no extra
placement or count exchange occurs in the steady-state path.

For weights, each owner packs both projections into one contiguous BF16
message per destination, ordered by projection and destination slot. Batched
`isend`/`irecv` operations use global ranks translated from EP ranks and a
consistent directed-edge ordering. Local replicas use device copies. The
receiver unpacks into stable, layer-local replica storage. Duplicate experts
in distinct slots are currently sent separately; unused slots are untouched.

MXFP8 sends raw quantized data and E8M0 scale bytes together in one uint8
message per peer. The wire order is FC1 data, FC1 scales, FC2 data, FC2 scales,
with each component in destination-slot order. Scale extents come from the
runtime's directional metadata, including TE padding; no fixed scale-to-data
size ratio is assumed. Rowwise and columnwise data/scales have separate stable
storage, returned in the TE wrapper contract's four-component order. A forward
push updates only rowwise storage, and a backward push only columnwise storage.
No quantization, dequantization, transpose, or scale swizzle occurs in transport;
the runtime retains TE's dtype, quantizer, and scale-layout metadata. This uses
more replica weight memory than the Peer-TMA direction-aliasing arena.

Gradients follow the same routes in reverse, only for the requested
projections. The owner adds local and received replica gradients to the
existing native gradient in FP32, then casts once into the supplied native
gradient destinations. Unreferenced native experts remain untouched. FC2 and
FC1 can start separately, preserving the runtime's FC2-first backward order.

Replica weights/gradients and the communication stream remain layer-local.
Native-gradient staging is private by default. `ReplicaExpertRuntime` opts into
`share_native_grad_storage`: compatible serialized layers reuse one staging
allocation per EP group, device, native-expert count, matrix shapes, and dtype.
The runtime finishes reduction and consumes/forwards the result at its
layer-input backward boundary before another layer writes staging, matching
Peer-TMA's existing shared-staging lifecycle. A weak cache releases the pool
after its last transport is destroyed. This avoids retaining a full extra
expert-gradient copy for every layer. Independent callers must leave sharing
disabled unless they provide the same consumer ordering.

A start waits for
its producer stream, packs messages, submits batched P2P, waits on NCCL Work
objects on the communication stream, and enqueues unpack or accumulation.
The returned event covers all of that work. Every wait inserts an event wait
on its calling stream, including repeated waits on different streams. Normal
NCCL waits order GPU work; enabling NCCL blocking-wait settings can also block
the host. Temporary buffers and input references are retained internally
until the event completes, even if the caller drops the handle. Tensor stream
tracking protects allocator reuse. Callers must still order consumers before
explicitly overwriting sources or reusable replica storage.

Construction must run in matching order on every EP member. It performs a
small all-reduce on the borrowed NCCL group before any sparse batched P2P, so
ranks without traffic need not participate in subsequent P2P batches. All
layers and other users of the same group must maintain matching communication
submission order. `destroy()` drains this layer's communication stream and
releases layer-local storage, without destroying the borrowed group.

Capabilities are conservative: host planning and dynamic message sizes mean
`device_plan=False`, `cuda_graph=False`, and `explicit_ownership=False`.
Both plan preparation and operation starts reject CUDA capture, even for
plans prepared before capture. `num_sms` is unused because the backend does
not expose a portable per-operation NCCL SM limit. Future optimizations can
add peer-local deduplication, reusable staging pools, and fused pack/reduce
kernels after measuring the baseline. Static CUDA-graph schedules require a
separate implementation and validation.

## Extending MoE Scheduler

To add a planner:

1. Subclass `MoELoadPlanner` and implement `update_placement()`, `reroute()`,
   and, when needed, `should_plan()`.
2. Return `(home_placement, replica_sources, private_state)` from
   `update_placement()`. Either placement may be `None`; replica sources use
   current physical home slots. Keep rerouting maps and backend state in the
   `MoEPlacementResult` carried by `private_state`.
3. Return dense physical `routing_map` and `probs` from `reroute()`.
4. Register the planner in `MoEScheduler.from_config()` and
   `TransformerConfig` validation.
5. Add contract tests and at least one planner/dispatcher compatibility test.

To add a replica communication mode while retaining the single expert dispatcher:

1. Implement `ReplicaWeightTransport`, including transport-owned storage views.
2. Keep TE, GTP, MXFP8, and optimizer semantics in `ReplicaExpertRuntime`.
3. Register the transport against an expert-dispatch type in
   `create_replica_weight_transport()`; all types instantiate
   `ReplicaExpertDispatch`.
4. Compile only backend-private schedules in `prepare_plan()` and preserve
   layout, ownership, storage lifetime and asynchronous completion semantics.
5. Add factory, forward/backward lifecycle, and multi-rank correctness tests.

Run the focused unit tests from the Megatron-LM repository root:

```bash
python -m torch.distributed.run --nproc-per-node 8 -m pytest \
  tests/unit_tests/transformer/moe/test_replica_weight_transport.py \
  tests/unit_tests/transformer/moe/test_replica_nccl_transport.py \
  tests/unit_tests/transformer/moe/test_moe_scheduler.py \
  tests/unit_tests/transformer/moe/test_echo_moe_scheduler.py \
  tests/unit_tests/transformer/moe/test_eplb_moe_scheduler.py \
  tests/unit_tests/transformer/moe/test_moonep_moe_scheduler.py
```

## UltraEP Integration (PR #2)

This integration targets [UltraEP PR #2](https://github.com/Dots-Infra/UltraEP/pull/2),
head `6f27b25e7f03f4166c3721dffcf01d1051b1206a` in `xinming-wei/UltraEP`.
The PR was still open on 2026-09-18; verify the final merged revision before
updating the dependency. Apply
[`docker/patches/ultraep-manager-lifetime.patch`](docker/patches/ultraep-manager-lifetime.patch)
to that checkout before building it and its NVSHMEM dependency **inside the
development container**, following the upstream installation instructions:

```bash
# Run from the pinned UltraEP checkout; use the absolute Megatron checkout path.
git apply /path/to/Megatron-MoeScheduler/docker/patches/ultraep-manager-lifetime.patch
uv pip install --no-build-isolation --no-deps .
```

The patch keeps the shared NVSHMEM runtime alive until the final Manager is
destroyed. Unpatched PR #2 assumes one Manager per EP group; this integration
uses per-layer managers and otherwise exits during multi-layer cleanup. The
adapter checks a capability exported by the rebuilt extension and rejects an
unpatched build before allocating buffers. Remove the patch only when an
upstream revision provides equivalent lifetime support.
UltraEP is imported only when selected; other backends do not require it.

On top of a working BF16/TE-fuser/HybridEP-token scheduler configuration, select:

```yaml
moe_enable_scheduler: true
moe_scheduler_planner_type: ultra_ep
moe_scheduler_expert_dispatcher_type: replica_ultraep
moe_scheduler_num_idle_experts: 4  # global count; divisible by EP size
moe_scheduler_home_update_interval: 0
moe_expert_rank_capacity_factor: null  # dynamic, dropless receive size
grad_reduce_in_bf16: false
```

The integration uses the **home-exchange branch's physical-source contract**:

- `UltraEPLoadPlanner.update_placement()` emits no home proposal and lowers
  the returned p2l map to `[EP, S]` current-home source slots. UltraEP keeps homes
  fixed, so its logical IDs equal source-slot IDs in this planner.
- `UltraEPTransport.prepare_plan()` constructs PR #2's three explicit maps from
  those physical sources. Model logical identity never enters the transport.
  It validates master-first inverse maps, one instance per expert per rank,
  within-domain replication, and identical placement across EP ranks.
- `weight_sync()` and `grad_reduce()` always receive explicit maps. The transport
  does not change the manager's internal quota/reroute placement.
- Router backward uses a snapshot of the actual dense token routes and p2l map.
  This fully specifies the probability permutation after quota rerouting;
  no backward reads an overwritten Manager slot. Simply cloning the three maps
  and then calling UltraEP's native backward would not provide this guarantee.
- A `split_grad_reduce` transport capability defers FC2's early reduction hook
  for UltraEP. FC1/FC2 are reduced jointly after dispatch backward, then handed
  to the existing runtime/GTP gradient path. Other backends keep split reduction.

First-version boundaries:

- BF16 weights and FP32 gradient storage for `replica_ultraep`; FP8/FP4 are rejected.
- UltraEP keeps rank capacity dynamic when no factor is specified. Placement
  does not guarantee an exactly balanced load, so an implicit fixed budget can
  drop tokens. An explicitly selected fixed rank budget requires the caller's
  overflow/retry handling, such as the Megatron training wrapper.
- No MoE CUDA graph capture for either UltraEP component. Attention-only capture
  retains the surrounding scheduler's existing policy.
- Periodic home migration still requires EPLB + NCCL. UltraEP does not expose
  the typed optimizer-state exchange needed by `HomeExpertDispatch`.
- The UltraEP planner can use the existing NCCL or Peer-TMA replica transport
  subject to their topology requirements. Other planners can use UltraEP
  communication only when their physical tables satisfy its stricter constraints;
  invalid placements are rejected, never silently rewritten.
- Managers/buffers are per layer. With both UltraEP components selected, they
  share one Manager. Planner-only use allocates minimal nonzero communication
  scratch because upstream has no allocation-free public solver Manager.
- Host placement validation and EP NCCL fences before/after each UltraEP
  transfer prioritize correctness over overlap. The fences cover remote
  producers, destination completion and replica-buffer reuse; this implementation
  uses a separate orchestration stream because UltraEP rejects communication
  stream self-waits. It does not assume PR #2 resolves cross-rank completion by
  itself. No performance or CUDA-graph equivalence with the upstream full-stack
  integration is claimed.
- Native managers are finalized collectively by the existing replica-runtime
  teardown. Per-layer transport destruction first drains its communication
  stream and releases its native-gradient staging; native NVSHMEM buffers remain
  alive until collective finalization. All ranks must construct and use layers
  in matching order; upstream UltraEP supports one EP process group per process.

Validation in the development container (four NVLink-connected GPUs, matching
UltraEP build installed):

```bash
uv run python -m torch.distributed.run --nproc-per-node 4 -m pytest -q \
  tests/unit_tests/transformer/moe/test_moe_scheduler.py \
  tests/unit_tests/transformer/moe/test_ultraep_moe_scheduler.py \
  tests/unit_tests/transformer/moe/test_replica_weight_transport.py \
  tests/unit_tests/transformer/moe/test_replica_ultraep_transport.py
```

The new tests cover map legality, router gradients after multiple forwards
(including non-contiguous gradients and empty token batches), joint-gradient
scheduling, transport teardown, native weight/gradient round trips and clearing,
fresh source pointers, repeated consumer-stream waits, and reuse of older plans.
The fused TE expert test compares forward results, input/probability gradients,
and source-weight gradients against independently materialized TE experts,
including empty home and replica token slots.

On 2026-09-18, the 33 scheduler/transport contract and UltraEP tests passed on
each of eight GB200 GPUs across two nodes, including collective cleanup of
multiple Managers. The build used UltraEP `6f27b25` plus the lifetime patch,
PyTorch 26.06 and TE batch-GEMM `55d6a453`. Another 186 Echo/EPLB/MoonEP/NCCL and
home-exchange regression cases passed on every rank; neither suite skipped cases.

The complete MoELayer test compares UltraEP with real HybridEP token dispatch
against an unscheduled all-to-all reference for two forward/backward steps,
including updated home weights. Fused experts handle zero-token ranks by running
a zero-weighted padded batch and removing it before combine. This keeps wgrad
overwrite and communication hooks active. Frozen expert weights still participate
in transport collectives without receiving optimizer gradients; plan lifetime
also covers router-only training. Full GPT training, UltraEP with GTP/FSDP, and
performance remain separate validation requirements.

# Performance Benchmarking

For our latest performance benchmarking results, please refer to [NVIDIA Megatron Bridge Performance Summary](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html).

Our codebase efficiently trains models from 2B to 462B parameters across thousands of GPUs, achieving up to **47% Model FLOP Utilization (MFU)** on H100 clusters.

![Model table](images/model_table.png)

**Benchmark Configuration:**

- **Vocabulary size**: 131,072 tokens
- **Sequence length**: 4096 tokens
- **Model scaling**: Varied hidden size, attention heads, and layers to achieve target parameter counts
- **Communication optimizations**: Fine-grained overlapping with DP (`--overlap-grad-reduce`, `--overlap-param-gather`), TP (`--tp-comm-overlap`), and PP (enabled by default)

**Key Results:**

- **6144 H100 GPUs**: Successfully benchmarked 462B parameter model training
- **Superlinear scaling**: MFU increases from 41% to 47-48% with model size
- **End-to-end measurement**: Throughputs include all operations (data loading, optimizer steps, communication, logging)
- **Production ready**: Full training pipeline with checkpointing and fault tolerance
- *Note: Performance results measured without training to convergence*

## Weak Scaling Results

Our weak scaled results show superlinear scaling (MFU increases from 41% for the smallest model considered to 47-48% for the largest models); this is because larger GEMMs have higher arithmetic intensity and are consequently more efficient to execute.

![Weak scaling](images/weak_scaling.png)

## Strong Scaling Results

We also strong scaled the standard GPT-3 model (our version has slightly more than 175 billion parameters due to larger vocabulary size) from 96 H100 GPUs to 4608 GPUs, using the same batch size of 1152 sequences throughout. Communication becomes more exposed at larger scale, leading to a reduction in MFU from 47% to 42%.

![Strong scaling](images/strong_scaling.png)

# Roadmaps

### Fast Iteration
- **Streamlined Review**: 1 code owner + 1 dev approver (can delegate review) + CI/CD

### Feature Lifecycle (Coming Soon)
- **6-Month Timeline**: Experimental features must graduate to stable or be deprecated
- **Migration Support**: Assistance provided for feature transitions

### Stability Expectations
- **Experimental Nature**: Features may change or be removed as development progresses
- **Testing**: All features will pass convergence and performance validation before inclusion
- **Support**: Dev branch issues should include `[DEV]` prefix

# Resources

## Performance & Benchmarking

- 🚀 [2025/11] [Optimizing DeepSeek-V3 Training Performance on NVIDIA GB200 NVL72](docs/discussions/deepseek-v3-gb200-optimization/deepseek-v3-gb200-optimization.md).
- ⚡ [2025/11] [A Guide to Reproduce DeepSeek-V3 Pre-training Performance on GB200](docs/discussions/deepseek-v3-gb200-optimization/deepseek-v3-gb200-reproduce-guide.md).

## Community & Support

### Getting Help
- 📖 **[Documentation](https://docs.nvidia.com/Megatron-Core/)** - Official documentation
- 🐛 **[Issues](https://github.com/NVIDIA/Megatron-LM/issues)** - Bug reports and feature requests

### Contributing
We ❤️ contributions! Ways to contribute:

- 🐛 **Report bugs** - Help us improve reliability
- 💡 **Suggest features** - Shape the future of Megatron Core
- 📝 **Improve docs** - Make Megatron Core more accessible
- 🔧 **Submit PRs** - Contribute code improvements

**→ [Contributing Guide](./CONTRIBUTING.md)**

### Citation
```bibtex
@article{megatron-lm,
  title={Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism},
  author={Shoeybi, Mohammad and Patwary, Mostofa and Puri, Raul and LeGresley, Patrick and Casper, Jared and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:1909.08053},
  year={2019}
}
```


## Periodic Home Expert Exchange (NCCL)

Enable alongside the existing scheduler configuration:

```yaml
moe_enable_scheduler: true
moe_scheduler_planner_type: eplb
moe_scheduler_expert_dispatcher_type: replica_nccl
moe_scheduler_num_idle_experts: 4
moe_scheduler_home_update_interval: 50
```

The first 50 successful updates collect logical load history. The first forward
after that interval proposes a permutation; migration runs after that iteration's
optimizer update. Exactly H unique canonical homes remain on every rank. LPT uses
per-instance load after greedy replica selection. This is a constrained EPLB
policy: unrestricted placements cannot always preserve a fixed home capacity.

EPLB keeps active and pending home/replica mappings. In periodic mode both logical
placements are committed together at the successful-step boundary; unchanged
replica placements still issue fresh weight copies on every microbatch. Each
microbatch reroutes against the active mapping and current home slots.
`HomeExpertDispatch.dispatch` snapshots only the proposal, never stale weights.
At the drained optimizer boundary, `MoEScheduler.step()` calls home dispatch
first and `planner.step(completed_version)` second. Failed optimizer updates
retain both the old mapping and pending proposal.

`prepare_home_expert_step(model, optimizer)` and `finish_home_expert_step(...)`
wire this boundary into the Megatron training loop. Custom training loops must
use the same boundary, or bind a state adapter and arrange communication fences
before calling each scheduler's `.step()`. All replica microbatches must finish.
Next-step DDP/FSDP gathers are deferred during migration, including the chained
optimizer MXFP8 gather path, and restarted only after weights are refreshed.

NCCL exchanges typed state components into separate staging buffers. Cyclic
permutations and same-rank swaps cannot overwrite a source before it is read.
The optimizer adapter moves authoritative parameters and tensor-valued optimizer
states in place. Parameter objects, master/moment addresses and bucket views are
preserved. Uneven or empty DistributedOptimizer/FSDP shards are reconstructed
over their optimizer DP group, exchanged over EP, and sliced into each destination's
own shard range. Existing optimizer copy/cast functions regenerate BF16/MXFP8
model weights and scales. This first implementation uses a full-layer state
snapshot and a device fence; migration is intentionally outside CUDA capture.

Supported adapter storage is resident, uncompressed elementwise optimizer state,
including ordinary Adam masters and moments, DDP mixed-precision masters,
DistributedOptimizer shards and Megatron-FSDP main-weight buffers. Precision-aware
compressed states, CPU/chunked optimizer offload, layer-wise optimizers and non-elementwise optimizer
states require dedicated adapters and are rejected. Peer-TMA exposes explicit
home-exchange placeholders; its existing replica APIs remain supported.

Checkpoints retain physical parameter slots and the EPLB `_extra_state` mapping,
step counter and load history. Resume requires the same EP/home-slot topology.
This format does not yet support expert-parallel resharding or importing into a
model without scheduler mapping metadata. Save/load and inference must retain
the scheduler; converting to canonical logical checkpoint keys is future work.

`test_home_expert_exchange.py` checks delayed reads of post-update weights,
local/cross-rank cycles, exact Adam continuation over repeated updates, mixed
tensor formats, repeated consumer-stream waits, uneven/empty optimizer shards,
active/pending routing, checkpoint metadata and explicit unsupported operations.
