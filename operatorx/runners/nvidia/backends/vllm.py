"""vLLM routed-expert and framework-native MoE kernels on CUDA."""

from operatorx.runners.moe import IMPLS as ROUTED_EXPERT_IMPLS
from operatorx.runners.moe import versions
from operatorx.runners.vllm_moe import IMPLS as MODULE_IMPLS

IMPLS = [*ROUTED_EXPERT_IMPLS, *MODULE_IMPLS]

__all__ = ["IMPLS", "versions"]
