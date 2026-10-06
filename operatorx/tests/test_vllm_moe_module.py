"""Validate the strict vLLM DeepSeek module request before GPU allocation."""

import pytest
import torch
from operatorx.core import UnsupportedOpError
from operatorx.runners.vllm_moe import DeepseekMoeSpec, _validate_repeated_outputs


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
        "phase": "decode",
        "global_num_tokens": 16,
        "workload_source": "operatorx_curated_deepseek_r1_v1",
        "weight_source": "synthetic",
        "execution_mode": "eager",
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
    assert spec.requested_workload_metadata() == {
        "model_id": "deepseek-ai/DeepSeek-R1-0528",
        "boundary": "moe_module",
        "phase": "decode",
        "global_num_tokens": 16,
        "local_num_tokens": 16,
        "input_placement": "single_rank",
        "routing_input_policy": "model_native",
        "source": "operatorx_curated_deepseek_r1_v1",
        "geometry": {
            "hidden_size": 7168,
            "routed_expert_count": 256,
            "experts_per_token": 8,
            "routed_expert_intermediate_size": 2048,
            "shared_expert_count": 1,
            "shared_expert_intermediate_size": 2048,
        },
        "routing": {
            "score_function": "sigmoid",
            "selection_method": "noaux_tc",
            "normalize_selected_weights": True,
            "routed_output_scale": 2.5,
            "expert_group_count": 8,
            "selected_expert_group_count": 4,
        },
        "precision": {
            "module_input_dtype": "bf16",
            "module_output_dtype": "bf16",
            "weight_precision": "fp8",
            "weight_block_size": [128, 128],
            "weight_source": "synthetic",
        },
        "topology": {
            "world_size": 1,
            "expert_parallel_size": 1,
            "routed_tensor_parallel_size": 1,
            "shared_tensor_parallel_size": 1,
        },
        "execution_mode": "eager",
        "synthetic_fixture": {
            "hidden_states": {
                "distribution": "normal",
                "mean": 0.0,
                "standard_deviation": 1.0,
                "seed": 17,
            },
            "parameters": {
                "floating_point_distribution": "normal",
                "floating_point_mean": 0.0,
                "floating_point_standard_deviation": 0.02,
                "quantization_scale_value": 1.0,
                "router_correction_bias_value": 0.0,
                "seed": 29,
            },
        },
    }
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
        ({"phase": "training"}, "phase"),
        ({"global_num_tokens": 32}, "equal global and local"),
        ({"expert_distribution": "uniform"}, "model_native"),
        ({"weight_source": "checkpoint"}, "synthetic weights"),
        ({"execution_mode": "cuda_graph"}, "eager execution"),
    ],
)
def test_deepseek_spec_rejects_non_equivalent_requests(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(UnsupportedOpError, match=message):
        DeepseekMoeSpec.from_args(_args(**overrides))


def test_correctness_gate_reports_repeatable_finite_output() -> None:
    expected = torch.zeros((2, 3), dtype=torch.bfloat16)
    output = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.bfloat16)

    assert _validate_repeated_outputs(output, output.clone(), expected) == {
        "status": "passed",
        "output_shape": [2, 3],
        "output_dtype": "torch.bfloat16",
        "all_values_finite": True,
        "repeatability": {
            "comparison": "exact",
            "untimed_iteration_count": 2,
            "passed": True,
        },
    }


@pytest.mark.parametrize(
    "second, message",
    [
        (torch.zeros((2, 4), dtype=torch.bfloat16), "shape"),
        (torch.zeros((2, 3), dtype=torch.float32), "dtype"),
        (torch.full((2, 3), float("nan"), dtype=torch.bfloat16), "non-finite"),
        (torch.ones((2, 3), dtype=torch.bfloat16), "different outputs"),
    ],
)
def test_correctness_gate_rejects_invalid_or_nonrepeatable_output(
    second: torch.Tensor, message: str
) -> None:
    expected = torch.zeros((2, 3), dtype=torch.bfloat16)
    first = torch.zeros((2, 3), dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match=message):
        _validate_repeated_outputs(first, second, expected)
