"""Behavioral validation for native vLLM Kimi-K3 module requests."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from operatorx.core import UnsupportedOpError
from operatorx.runners.vllm_moe.kimi_k3 import KimiK3MoeSpec


def _args(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "model_family": "kimi_k3",
        "model_id": "moonshotai/Kimi-K3",
        "model_revision": "f831ab66814297da540d832a5235f8e904f29d06",
        "layer_index": 1,
        "num_tokens": 4,
        "hidden": 7168,
        "routed_hidden": 3584,
        "intermediate": 3072,
        "num_experts": 896,
        "top_k": 16,
        "n_shared_experts": 2,
        "hidden_act": "situ",
        "activation_gate_beta": 4.0,
        "activation_up_beta": 25.0,
        "routed_norm_epsilon": 1e-5,
        "score_function": "sigmoid",
        "selection_method": "noaux_tc",
        "normalize_selected_weights": True,
        "routed_output_scale": 1.0,
        "expert_group_count": 1,
        "selected_expert_group_count": 1,
        "dtype_act": "bf16",
        "dtype_weight": "mxfp4",
        "weight_format": "mxfp4-pack-quantized",
        "weight_group_size": 32,
        "phase": "decode",
        "global_num_tokens": 4,
        "workload_source": "operatorx_curated_kimi_k3_v1",
        "weight_source": "synthetic",
        "execution_mode": "eager",
        "world_size": 1,
        "expert_parallel_size": 1,
        "routed_tensor_parallel_size": 1,
        "shared_tensor_parallel_size": 1,
        "expert_distribution": "model_native",
        "input_seed": 43,
        "weight_seed": 31,
    }
    values.update(overrides)
    return values


def test_kimi_k3_spec_maps_native_module_semantics() -> None:
    spec = KimiK3MoeSpec.from_args(_args())

    assert spec.activation_dtype is torch.bfloat16
    assert vars(spec.vllm_model_config()) == {
        "model": "moonshotai/Kimi-K3",
        "revision": "f831ab66814297da540d832a5235f8e904f29d06",
        "dtype": torch.bfloat16,
        "quantization": "compressed-tensors",
        "is_moe": True,
        "word_embeddings_untied_by_checkpoint": False,
        "hf_text_config": SimpleNamespace(model_type="kimi_linear"),
    }
    assert spec.hf_config_fields() == {
        "hidden_size": 7168,
        "moe_intermediate_size": 3072,
        "num_experts": 896,
        "num_experts_per_token": 16,
        "num_shared_experts": 2,
        "routed_expert_hidden_size": 3584,
        "latent_moe_use_norm": True,
        "rms_norm_eps": 1e-5,
        "hidden_act": "situ",
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "moe_router_activation_func": "sigmoid",
        "topk_method": "noaux_tc",
        "moe_renormalize": True,
        "routed_scaling_factor": 1.0,
        "use_grouped_topk": True,
        "num_expert_group": 1,
        "topk_group": 1,
    }
    quantization = spec.quantization_config()
    group = quantization["config_groups"]["group_0"]
    assert group["weights"] == {
        "num_bits": 4,
        "type": "float",
        "strategy": "group",
        "group_size": 32,
        "symmetric": True,
        "dynamic": False,
        "scale_dtype": "torch.uint8",
    }
    assert "re:.*shared_experts.*" in quantization["ignore"]

    requested = spec.requested_workload_metadata()
    assert requested["geometry"] == {
        "hidden_size": 7168,
        "routed_expert_hidden_size": 3584,
        "routed_expert_count": 896,
        "experts_per_token": 16,
        "routed_expert_intermediate_size": 3072,
        "shared_expert_count": 2,
        "shared_expert_intermediate_size": 3072,
        "routed_output_norm_epsilon": 1e-5,
    }
    assert requested["precision"]["activation_operand_format"] == "framework_selected"
    assert (
        requested["synthetic_fixture"]["parameters"]["routed_weight_generation"]
        == "per_expert_bf16_to_native_packed_mxfp4"
    )


def test_kimi_k3_spec_preserves_prefill_workload_identity() -> None:
    spec = KimiK3MoeSpec.from_args(
        _args(phase="prefill", num_tokens=32768, global_num_tokens=32768)
    )

    requested = spec.requested_workload_metadata()
    assert requested["phase"] == "prefill"
    assert requested["global_num_tokens"] == 32768
    assert requested["local_num_tokens"] == 32768
    assert requested["geometry"]["experts_per_token"] == 16


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"layer_index": 0}, "dense"),
        ({"layer_index": 2}, "layer_index"),
        ({"model_family": "deepseek"}, "model_family"),
        ({"model_revision": "main"}, "model_revision"),
        ({"dtype_weight": "bf16"}, "dtype_weight"),
        ({"weight_group_size": 16}, "weight_group_size"),
        ({"hidden_act": "silu"}, "hidden_act"),
        ({"expert_parallel_size": 8, "world_size": 8}, "single-rank"),
        ({"global_num_tokens": 8}, "equal global and local"),
        ({"expert_distribution": "uniform"}, "model_native"),
    ],
)
def test_kimi_k3_spec_rejects_non_equivalent_requests(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(UnsupportedOpError, match=message):
        KimiK3MoeSpec.from_args(_args(**overrides))
