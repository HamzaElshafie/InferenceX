"""Validate the strict vLLM DeepSeek module request before GPU allocation."""

import pytest
import torch

from operatorx.core import UnsupportedOpError
from operatorx.runners.vllm_moe import DeepseekMoeSpec


def _args(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "model_id": "deepseek-ai/DeepSeek-R1-0528",
        "num_tokens": 16,
        "hidden": 7168,
        "intermediate": 2048,
        "num_experts": 256,
        "top_k": 8,
        "n_shared_experts": 1,
        "hidden_act": "silu",
        "score_function": "sigmoid",
        "selection_method": "noaux_tc",
        "normalize_selected_weights": True,
        "routed_output_scale": 2.5,
        "expert_group_count": 8,
        "selected_expert_group_count": 4,
        "dtype_act": "bf16",
        "dtype_weight": "fp8",
        "weight_block_size_n": 128,
        "weight_block_size_k": 128,
        "world_size": 1,
        "expert_parallel_size": 1,
        "routed_tensor_parallel_size": 1,
        "shared_tensor_parallel_size": 1,
        "input_seed": 17,
        "weight_seed": 29,
    }
    values.update(overrides)
    return values


def test_deepseek_spec_maps_complete_module_semantics() -> None:
    spec = DeepseekMoeSpec.from_args(_args())

    assert spec.model_id == "deepseek-ai/DeepSeek-R1-0528"
    assert spec.activation_dtype is torch.bfloat16
    assert spec.weight_block_size == (128, 128)
    assert vars(spec.vllm_model_config()) == {
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "dtype": torch.bfloat16,
        "quantization": "fp8",
        "is_moe": True,
    }
    assert vars(spec.hf_config()) == {
        "hidden_size": 7168,
        "moe_intermediate_size": 2048,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "n_shared_experts": 1,
        "hidden_act": "silu",
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.5,
        "n_group": 8,
        "topk_group": 4,
    }


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"expert_parallel_size": 8, "world_size": 8}, "single-rank"),
        ({"dtype_weight": "bf16"}, "serialized FP8"),
        ({"selection_method": None}, "selection_method"),
        ({"top_k": 257}, "top_k"),
        ({"selected_expert_group_count": 9}, "selected_expert_group_count"),
    ],
)
def test_deepseek_spec_rejects_non_equivalent_requests(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(UnsupportedOpError, match=message):
        DeepseekMoeSpec.from_args(_args(**overrides))
