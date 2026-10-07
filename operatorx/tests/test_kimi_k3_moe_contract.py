"""Behavioral tests for Kimi K3's native MoE configuration mapping."""

import json
from pathlib import Path

import pytest

from operatorx.core.moe import MoeActivation, MoeLayerGeometry, MoePrecision, MoeRouting
from operatorx.scripts.inferencex_testlist.models import load_arch


def _write_config(tmp_path: Path, **text_overrides: object) -> tuple[str, ...]:
    """Write the relevant fields of the pinned K3 text configuration.

    Source: https://huggingface.co/moonshotai/Kimi-K3/blob/
    f831ab66814297da540d832a5235f8e904f29d06/config.json
    """
    text_config = {
        "model_type": "kimi_linear",
        "hidden_size": 7168,
        "num_hidden_layers": 93,
        "num_attention_heads": 96,
        "num_key_value_heads": 96,
        "v_head_dim": 128,
        "intermediate_size": 33792,
        "first_k_dense_replace": 1,
        "num_experts": 896,
        "num_experts_per_token": 16,
        "moe_intermediate_size": 3072,
        "num_shared_experts": 2,
        "routed_expert_hidden_size": 3584,
        "latent_moe_use_norm": True,
        "rms_norm_eps": 1e-5,
        "hidden_act": "situ",
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "dtype": "bfloat16",
        "moe_router_activation_func": "sigmoid",
        "topk_method": "noaux_tc",
        "use_grouped_topk": True,
        "num_expert_group": 1,
        "topk_group": 1,
        "moe_renormalize": True,
        "routed_scaling_factor": 1.0,
        "quantization_config": {
            "quant_method": "compressed-tensors",
            "format": "mxfp4-pack-quantized",
            "ignore": ["re:.*shared_experts.*"],
            "config_groups": {
                "group_0": {
                    "format": "mxfp4-pack-quantized",
                    "targets": ["Linear"],
                    "input_activations": None,
                    "weights": {"num_bits": 4, "group_size": 32},
                }
            },
        },
    }
    text_config.update(text_overrides)
    model_dir = tmp_path / "Kimi-K3"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "kimi_k3", "text_config": text_config}),
        encoding="utf-8",
    )
    return (str(tmp_path),)


def test_maps_latent_moe_without_inventing_activation_operand_dtype(
    tmp_path: Path,
) -> None:
    arch = load_arch("moonshotai/Kimi-K3", extra_dirs=_write_config(tmp_path))

    assert arch.family == "kimi_k3"
    assert arch.attention.kind == "hybrid_kda_mla"
    assert arch.moe.num_dense_layers == 1
    assert arch.moe_geometry == MoeLayerGeometry(
        hidden_size=7168,
        routed_expert_count=896,
        experts_per_token=16,
        routed_expert_intermediate_size=3072,
        shared_expert_count=2,
        shared_expert_intermediate_size=3072,
        routed_expert_hidden_size=3584,
        routed_output_norm_eps=1e-5,
    )
    assert arch.moe_routing == MoeRouting(
        score_function="sigmoid",
        selection_method="noaux_tc",
        normalize_selected_weights=True,
        routed_output_scale=1.0,
        group_count=1,
        selected_group_count=1,
    )
    assert arch.moe_activation == MoeActivation(name="situ", gate_beta=4, up_beta=25)
    assert arch.moe_precision == MoePrecision(
        tensor_dtype="bfloat16",
        weight_quant_method="compressed-tensors",
        weight_format="mxfp4-pack-quantized",
        activation_scheme="framework_selected",
        weight_group_size=32,
        shared_weight_dtype="bfloat16",
    )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"hidden_act": "silu"}, "SiTU"),
        ({"routed_expert_hidden_size": 0}, "routed_expert_hidden_size"),
        ({"latent_moe_use_norm": False}, "RMS normalization"),
        ({"num_experts_per_token": 897}, "experts_per_token"),
    ],
)
def test_rejects_unfaithful_k3_architecture(
    tmp_path: Path, override: dict[str, object], message: str
) -> None:
    search_dirs = _write_config(tmp_path, **override)

    with pytest.raises(ValueError, match=message):
        load_arch("moonshotai/Kimi-K3", extra_dirs=search_dirs)


def test_rejects_missing_shared_expert_quantization_exclusion(tmp_path: Path) -> None:
    search_dirs = _write_config(
        tmp_path,
        quantization_config={
            "quant_method": "compressed-tensors",
            "format": "mxfp4-pack-quantized",
            "ignore": [],
            "config_groups": {
                "group_0": {
                    "format": "mxfp4-pack-quantized",
                    "targets": ["Linear"],
                    "input_activations": None,
                    "weights": {"num_bits": 4, "group_size": 32},
                }
            },
        },
    )

    with pytest.raises(ValueError, match="shared experts"):
        load_arch("moonshotai/Kimi-K3", extra_dirs=search_dirs)


def test_rejects_invalid_mxfp4_group_size(tmp_path: Path) -> None:
    search_dirs = _write_config(
        tmp_path,
        quantization_config={
            "quant_method": "compressed-tensors",
            "format": "mxfp4-pack-quantized",
            "ignore": ["re:.*shared_experts.*"],
            "config_groups": {
                "group_0": {
                    "format": "mxfp4-pack-quantized",
                    "targets": ["Linear"],
                    "input_activations": None,
                    "weights": {"num_bits": 4, "group_size": 0},
                }
            },
        },
    )

    with pytest.raises(ValueError, match="weight_group_size"):
        load_arch("moonshotai/Kimi-K3", extra_dirs=search_dirs)
