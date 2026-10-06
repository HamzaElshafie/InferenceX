"""Test DeepSeek-R1 geometry and routing extraction from published config fields."""

import json
from pathlib import Path

import pytest

from operatorx.core.moe import MoeLayerGeometry, MoePrecision, MoeRouting
from operatorx.scripts.inferencex_testlist.models import load_arch


def _write_deepseek_r1_config(
    tmp_path: Path, **moe_overrides: object
) -> tuple[str, ...]:
    """Write the loader's subset of a pinned DeepSeek-R1-0528 config.

    Source: https://huggingface.co/deepseek-ai/DeepSeek-R1-0528/blob/
    11628360bdbb84a195bb216d98bc724f6af08d57/config.json
    """
    config = {
        "model_type": "deepseek_v3",
        "hidden_size": 7168,
        "num_hidden_layers": 61,
        "num_attention_heads": 128,
        "q_lora_rank": 1536,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048,
        "n_shared_experts": 1,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "norm_topk_prob": True,
        "routed_scaling_factor": 2.5,
        "n_group": 8,
        "topk_group": 4,
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        },
    }
    config.update(moe_overrides)
    model_dir = tmp_path / "DeepSeek-R1-0528"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return (str(tmp_path),)


def test_loader_maps_deepseek_r1_moe_contract(tmp_path: Path) -> None:
    search_dirs = _write_deepseek_r1_config(tmp_path)

    arch = load_arch("deepseek-ai/DeepSeek-R1-0528", extra_dirs=search_dirs)

    assert arch.moe_geometry == MoeLayerGeometry(
        hidden_size=7168,
        routed_expert_count=256,
        experts_per_token=8,
        routed_expert_intermediate_size=2048,
        shared_expert_count=1,
        shared_expert_intermediate_size=2048,
    )
    assert arch.moe_routing == MoeRouting(
        score_function="sigmoid",
        selection_method="noaux_tc",
        normalize_selected_weights=True,
        routed_output_scale=2.5,
        group_count=8,
        selected_group_count=4,
    )
    assert arch.moe.n_shared_experts == 1
    assert arch.moe_precision == MoePrecision(
        tensor_dtype="bfloat16",
        weight_quant_method="fp8",
        weight_format="e4m3",
        activation_scheme="dynamic",
        weight_block_size=(128, 128),
    )


def test_loader_rejects_impossible_topk(tmp_path: Path) -> None:
    search_dirs = _write_deepseek_r1_config(tmp_path, num_experts_per_tok=257)

    with pytest.raises(ValueError, match="experts_per_token cannot exceed"):
        load_arch("deepseek-ai/DeepSeek-R1-0528", extra_dirs=search_dirs)


def test_loader_leaves_unmodeled_quantization_unset(tmp_path: Path) -> None:
    search_dirs = _write_deepseek_r1_config(
        tmp_path, quantization_config={"quant_method": "mxfp4"}
    )

    arch = load_arch("deepseek-ai/DeepSeek-R1-0528", extra_dirs=search_dirs)

    assert arch.moe_precision is None


@pytest.mark.parametrize(
    "weight_block_size",
    ([128], [128, 0], [128, True]),
)
def test_loader_rejects_invalid_weight_blocks(
    tmp_path: Path, weight_block_size: list[int]
) -> None:
    search_dirs = _write_deepseek_r1_config(
        tmp_path,
        quantization_config={
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": weight_block_size,
        },
    )

    with pytest.raises(ValueError, match="weight_block_size"):
        load_arch("deepseek-ai/DeepSeek-R1-0528", extra_dirs=search_dirs)
