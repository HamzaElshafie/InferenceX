from __future__ import annotations

from dataclasses import dataclass

from operatorx.core.op import OpSpec
from operatorx.core.op_registry import register


@dataclass(frozen=True)
class MoeForwardArgs:
    num_tokens: int
    hidden: int
    intermediate: int
    num_experts: int
    top_k: int
    dtype_act: str = "bf16"
    dtype_weight: str = "bf16"
    world_size: int = 1
    expert_parallel_size: int = 1
    routed_tensor_parallel_size: int = 1
    shared_tensor_parallel_size: int = 1
    n_shared_experts: int = 0
    # How tokens are routed across experts at benchmark time.
    #   "uniform"    — uniform-random topk (best-case load balance, default).
    #   "zipf"       — Zipfian (s=1) bias on routing logits — a few experts hot,
    #                  tail cold. Approximates pre-aux-loss DSv3-style skew.
    #   "single_hot" — every token routes to expert 0 + (top_k-1) random — worst
    #                  case for load imbalance and dispatch.
    expert_distribution: str = "uniform"
    # Model semantics required by framework-native module implementations.
    # Older generic backends may leave these unset.
    model_id: str | None = None
    hidden_act: str | None = None
    score_function: str | None = None
    selection_method: str | None = None
    normalize_selected_weights: bool | None = None
    routed_output_scale: float | None = None
    expert_group_count: int | None = None
    selected_expert_group_count: int | None = None
    weight_block_size_n: int | None = None
    weight_block_size_k: int | None = None
    # Workload identity. ``num_tokens`` remains the tensor's local leading
    # dimension; ``global_num_tokens`` identifies the logical workload.
    phase: str | None = None
    global_num_tokens: int | None = None
    workload_source: str | None = None
    weight_source: str = "synthetic"
    execution_mode: str = "eager"
    input_seed: int = 0
    weight_seed: int = 0


MOE_FORWARD = OpSpec(
    type="moe_forward",
    arg_schema=MoeForwardArgs,
    description="Full MoE forward (gate + dispatch + experts + combine), timed as one unit.",
)

register(MOE_FORWARD)


@dataclass(frozen=True)
class MoeGemmArgs:
    num_tokens: int
    hidden: int
    intermediate: int
    num_experts: int
    top_k: int
    dtype_act: str = "bf16"
    dtype_weight: str = "bf16"
    expert_parallel_size: int = 1
    routed_tensor_parallel_size: int = 1
    shared_tensor_parallel_size: int = 1
    n_shared_experts: int = 0
    expert_distribution: str = "uniform"


MOE_GEMM = OpSpec(
    type="moe_gemm",
    arg_schema=MoeGemmArgs,
    description="Per-rank routed-expert grouped GEMM (gate_up + activation + down, effective single-rank shapes) with no comms. ",
)

register(MOE_GEMM)
