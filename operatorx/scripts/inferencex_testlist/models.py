"""HuggingFace model config loader + per-architecture shape descriptors.

`load_arch()` returns an `Arch` dataclass that exposes everything the
enumerator needs about a model's structure (hidden dims, attention shape,
MoE config, attention type). Per-arch wrappers translate the various
HF config layouts (top-level, `text_config`, etc.) into a uniform shape.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from operatorx.core.moe import MoeActivation, MoeLayerGeometry, MoePrecision, MoeRouting

_DEFAULT_LOCAL_DIRS = (
    "/models",
    "/scratch/fsw/models",
)


@dataclass(frozen=True)
class AttentionArch:
    kind: str  # "mha" | "mla" | "hybrid_kda_mla" (not yet enumerable)
    num_heads: int
    num_kv_heads: int
    head_dim: int  # MHA: per-head dim. MLA: not directly used (see MLA fields).
    # MLA-only:
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    kv_lora_rank: int = 0
    q_lora_rank: int = 0
    # Optional: per-layer attention types ("full" vs "sliding"/"linear"). When
    # heterogeneous, the canonical shape set should include all variants.
    sliding_window: int | None = None


@dataclass(frozen=True)
class MoeArch:
    """Empty / disabled when this model has no MoE layers."""

    num_experts: int = 0
    num_experts_per_tok: int = 0
    moe_intermediate_size: int = 0
    n_shared_experts: int = 0
    # If the model has dense (non-MoE) layers in addition to MoE layers, set
    # `dense_intermediate_size` so we also emit MLP-GEMM shapes for those.
    dense_intermediate_size: int = 0
    # Number of leading dense layers (first_k_dense_replace etc.).
    num_dense_layers: int = 0


@dataclass(frozen=True)
class Arch:
    """Model dimensions used to enumerate canonical benchmark shapes."""

    name: str
    family: str  # "deepseek" | "glm" | "kimi" | "minimax" | "gptoss" | "qwen3moe"
    hidden_size: int
    num_layers: int
    attention: AttentionArch
    moe: MoeArch
    # Additional speculative-decode layers (DeepSeek MTP, Qwen MTP).
    mtp_num_layers: int = 0
    # Describes MoE layers only; populated as each family gains a validated mapping.
    moe_geometry: MoeLayerGeometry | None = None
    moe_routing: MoeRouting | None = None
    moe_precision: MoePrecision | None = None
    moe_activation: MoeActivation | None = None


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _candidate_paths(model_id: str, extra_dirs: tuple[str, ...]) -> list[str]:
    """Where to look for a config.json for `model_id`."""
    paths: list[str] = []
    # Bare ids from the master configs are usually "<org>/<name>"; the local
    # caches store them under "<name>".
    short = model_id.split("/")[-1]
    for base in extra_dirs:
        paths.append(os.path.join(base, short, "config.json"))
        paths.append(os.path.join(base, model_id.replace("/", "_"), "config.json"))
    # HF default cache layout:
    hf_home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    paths.append(
        os.path.join(
            hf_home, "hub", f"models--{model_id.replace('/', '--')}", "snapshots"
        )
    )
    return paths


def _load_hf_config(model_id: str, extra_dirs: tuple[str, ...]) -> dict[str, Any]:
    for p in _candidate_paths(model_id, extra_dirs):
        if p.endswith("snapshots") and os.path.isdir(p):
            # pick first snapshot dir
            for d in sorted(os.listdir(p)):
                cand = os.path.join(p, d, "config.json")
                if os.path.isfile(cand):
                    with open(cand) as f:
                        return json.load(f)
        elif os.path.isfile(p):
            with open(p) as f:
                return json.load(f)
    raise FileNotFoundError(
        f"Could not locate config.json for {model_id!r}. Searched: {_candidate_paths(model_id, extra_dirs)}"
    )


def _text_subconfig(raw: dict[str, Any]) -> dict[str, Any]:
    """Some configs nest the LM under `text_config` (Kimi, Qwen3.5 multimodal).

    Top-level architectures/model_type wins so family detection sees the brand
    (e.g. KimiK25ForConditionalGeneration) rather than the LM-class it reuses
    (Kimi K2.5 reuses DeepseekV3ForCausalLM under the hood)."""
    if "text_config" in raw and isinstance(raw["text_config"], dict):
        merged = dict(raw["text_config"])
        if raw.get("architectures"):
            merged["architectures"] = raw["architectures"]
        if raw.get("model_type"):
            merged["model_type"] = raw["model_type"]
        return merged
    return raw


# ---------------------------------------------------------------------------
# Family detection
# ---------------------------------------------------------------------------


def _family(cfg: dict[str, Any]) -> str:
    arch = (cfg.get("architectures") or [""])[0]
    mt = cfg.get("model_type", "")
    if mt == "kimi_k3":
        return "kimi_k3"
    if mt.startswith("kimi_k3"):
        raise ValueError(f"Unsupported Kimi K3 model variant: {mt!r}")
    if "Deepseek" in arch or mt.startswith("deepseek"):
        return "deepseek"
    if "Glm" in arch or mt.startswith("glm"):
        return "glm"
    if "Kimi" in arch or mt.startswith("kimi"):
        return "kimi"
    if "MiniMax" in arch or mt.startswith("minimax"):
        return "minimax"
    if "GptOss" in arch or "gpt_oss" in mt or "gptoss" in arch.lower():
        return "gptoss"
    if "Qwen" in arch or mt.startswith("qwen"):
        return "qwen3moe"
    raise ValueError(f"Unsupported model architecture: arch={arch!r} model_type={mt!r}")


# ---------------------------------------------------------------------------
# Per-family builders
# ---------------------------------------------------------------------------


def _build_deepseek(cfg: dict[str, Any], name: str) -> Arch:
    """Extract DeepSeek dimensions without choosing an execution backend."""

    # DeepSeek V3 / R1: qk_nope_head_dim, qk_rope_head_dim, v_head_dim, kv_lora_rank
    #   are all explicit; `head_dim` is absent.
    # DeepSeek V4: head_dim is set (= v_head_dim), qk_rope_head_dim is set,
    #   qk_nope_head_dim is implicit (= head_dim - qk_rope_head_dim),
    #   kv_lora_rank is named `o_lora_rank`.
    qk_rope = cfg.get("qk_rope_head_dim", 0)
    head_dim_explicit = cfg.get("head_dim") or 0
    v_head_dim = cfg.get("v_head_dim") or head_dim_explicit
    qk_nope = cfg.get("qk_nope_head_dim")
    if qk_nope is None:
        qk_nope = head_dim_explicit - qk_rope if head_dim_explicit else 0
    kv_lora = cfg.get("kv_lora_rank") or cfg.get("o_lora_rank") or 0
    shared_expert_count = cfg.get("n_shared_experts") or 0
    moe_geometry = MoeLayerGeometry(
        hidden_size=cfg["hidden_size"],
        routed_expert_count=cfg["n_routed_experts"],
        experts_per_token=cfg["num_experts_per_tok"],
        routed_expert_intermediate_size=cfg["moe_intermediate_size"],
        shared_expert_count=shared_expert_count,
        shared_expert_intermediate_size=(
            cfg["moe_intermediate_size"] if shared_expert_count else 0
        ),
    )
    routing_fields = (
        "scoring_func",
        "topk_method",
        "norm_topk_prob",
        "routed_scaling_factor",
    )
    moe_routing = (
        MoeRouting(
            score_function=cfg["scoring_func"],
            selection_method=cfg["topk_method"],
            normalize_selected_weights=cfg["norm_topk_prob"],
            routed_output_scale=cfg["routed_scaling_factor"],
            group_count=cfg.get("n_group"),
            selected_group_count=cfg.get("topk_group"),
        )
        if all(field in cfg for field in routing_fields)
        else None
    )
    quantization = cfg.get("quantization_config")
    moe_precision = (
        MoePrecision(
            tensor_dtype=cfg["torch_dtype"],
            weight_quant_method=quantization["quant_method"],
            weight_format=quantization["fmt"],
            activation_scheme=quantization["activation_scheme"],
            weight_block_size=tuple(quantization["weight_block_size"]),
        )
        if isinstance(quantization, dict) and quantization.get("quant_method") == "fp8"
        else None
    )
    return Arch(
        name=name,
        family="deepseek",
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_hidden_layers"],
        attention=AttentionArch(
            kind="mla",
            num_heads=cfg["num_attention_heads"],
            num_kv_heads=cfg.get("num_key_value_heads", 1),
            head_dim=v_head_dim,
            qk_nope_head_dim=qk_nope,
            qk_rope_head_dim=qk_rope,
            v_head_dim=v_head_dim,
            kv_lora_rank=kv_lora,
            q_lora_rank=cfg["q_lora_rank"],
            sliding_window=cfg.get("sliding_window"),
        ),
        moe=MoeArch(
            num_experts=moe_geometry.routed_expert_count,
            num_experts_per_tok=moe_geometry.experts_per_token,
            moe_intermediate_size=moe_geometry.routed_expert_intermediate_size,
            n_shared_experts=moe_geometry.shared_expert_count,
            dense_intermediate_size=cfg.get("intermediate_size", 0),
            num_dense_layers=cfg.get("first_k_dense_replace", 0) or 0,
        ),
        mtp_num_layers=cfg.get("num_nextn_predict_layers", 0) or 0,
        moe_geometry=moe_geometry,
        moe_routing=moe_routing,
        moe_precision=moe_precision,
    )


def _build_glm(cfg: dict[str, Any], name: str) -> Arch:
    # GLM-5 MoE+DSA: it has MLA-style attention (qk_nope/qk_rope/v_head_dim/kv_lora).
    return Arch(
        name=name,
        family="glm",
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_hidden_layers"],
        attention=AttentionArch(
            kind="mla",
            num_heads=cfg["num_attention_heads"],
            num_kv_heads=cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
            head_dim=cfg.get("head_dim", 0),
            qk_nope_head_dim=cfg["qk_nope_head_dim"],
            qk_rope_head_dim=cfg["qk_rope_head_dim"],
            v_head_dim=cfg["v_head_dim"],
            kv_lora_rank=cfg["kv_lora_rank"],
            q_lora_rank=cfg["q_lora_rank"],
        ),
        moe=MoeArch(
            num_experts=cfg["n_routed_experts"],
            num_experts_per_tok=cfg["num_experts_per_tok"],
            moe_intermediate_size=cfg["moe_intermediate_size"],
            n_shared_experts=cfg.get("n_shared_experts", 0),
            dense_intermediate_size=cfg.get("intermediate_size", 0),
            num_dense_layers=cfg.get("first_k_dense_replace", 0) or 0,
        ),
        mtp_num_layers=cfg.get("num_nextn_predict_layers", 0) or 0,
    )


def _build_kimi(cfg: dict[str, Any], name: str) -> Arch:
    # Kimi K2.5: MLA + MoE. Same shape as DeepSeek.
    return Arch(
        name=name,
        family="kimi",
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_hidden_layers"],
        attention=AttentionArch(
            kind="mla",
            num_heads=cfg["num_attention_heads"],
            num_kv_heads=cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
            head_dim=cfg.get("head_dim", 0),
            qk_nope_head_dim=cfg["qk_nope_head_dim"],
            qk_rope_head_dim=cfg["qk_rope_head_dim"],
            v_head_dim=cfg["v_head_dim"],
            kv_lora_rank=cfg["kv_lora_rank"],
            q_lora_rank=cfg["q_lora_rank"],
        ),
        moe=MoeArch(
            num_experts=cfg["n_routed_experts"],
            num_experts_per_tok=cfg["num_experts_per_tok"],
            moe_intermediate_size=cfg["moe_intermediate_size"],
            n_shared_experts=cfg.get("n_shared_experts", 0),
            dense_intermediate_size=cfg.get("intermediate_size", 0),
            num_dense_layers=cfg.get("first_k_dense_replace", 0) or 0,
        ),
        mtp_num_layers=cfg.get("num_nextn_predict_layers", 0) or 0,
    )


def _build_kimi_k3(cfg: dict[str, Any], name: str) -> Arch:
    """Map K3's nested language-model config to a strict MoE contract."""

    if cfg.get("hidden_act") != "situ":
        raise ValueError("Kimi K3 requires SiTU experts")
    if cfg.get("dtype") != "bfloat16":
        raise ValueError("Kimi K3 module requires bfloat16 tensors")
    if cfg.get("topk_method") != "noaux_tc":
        raise ValueError("Kimi K3 requires noaux_tc routing")
    if cfg.get("moe_router_activation_func") != "sigmoid":
        raise ValueError("Kimi K3 requires sigmoid routing scores")
    if cfg.get("use_grouped_topk") is not True:
        raise ValueError("Kimi K3 requires grouped top-k selection")

    quant = cfg.get("quantization_config")
    if not isinstance(quant, dict) or quant.get("quant_method") != "compressed-tensors":
        raise ValueError("Kimi K3 requires compressed-tensors quantization")
    groups = quant.get("config_groups")
    if not isinstance(groups, dict) or len(groups) != 1:
        raise ValueError("Kimi K3 requires one MXFP4 quantization group")
    group = next(iter(groups.values()))
    if not isinstance(group, dict) or group.get("format") != "mxfp4-pack-quantized":
        raise ValueError("Kimi K3 requires packed MXFP4 routed weights")
    if quant.get("format") != group["format"] or group.get("targets") != ["Linear"]:
        raise ValueError("Kimi K3 MXFP4 group must target Linear weights")
    if group.get("input_activations") is not None:
        raise ValueError("Kimi K3 activation operand format must be backend-resolved")
    weights = group.get("weights")
    if not isinstance(weights, dict) or weights.get("num_bits") != 4:
        raise ValueError("Kimi K3 requires four-bit MXFP4 weights")
    if not any("shared_experts" in pattern for pattern in quant.get("ignore", [])):
        raise ValueError("Kimi K3 shared experts must be excluded from MXFP4")

    shared_count = cfg["num_shared_experts"]
    geometry = MoeLayerGeometry(
        hidden_size=cfg["hidden_size"],
        routed_expert_count=cfg["num_experts"],
        experts_per_token=cfg["num_experts_per_token"],
        routed_expert_intermediate_size=cfg["moe_intermediate_size"],
        shared_expert_count=shared_count,
        shared_expert_intermediate_size=cfg["moe_intermediate_size"],
        routed_expert_hidden_size=cfg["routed_expert_hidden_size"],
        routed_output_norm_eps=(
            cfg["rms_norm_eps"] if cfg["latent_moe_use_norm"] is True else None
        ),
    )
    if geometry.routed_expert_hidden_size == geometry.hidden_size:
        raise ValueError("Kimi K3 requires a distinct routed latent width")
    if geometry.routed_output_norm_eps is None:
        raise ValueError("Kimi K3 requires routed-output RMS normalization")

    routing = MoeRouting(
        score_function=cfg["moe_router_activation_func"],
        selection_method=cfg["topk_method"],
        normalize_selected_weights=cfg["moe_renormalize"],
        routed_output_scale=cfg["routed_scaling_factor"],
        group_count=cfg["num_expert_group"],
        selected_group_count=cfg["topk_group"],
    )
    activation = MoeActivation(
        name=cfg["hidden_act"],
        gate_beta=cfg["activation_situ_beta"],
        up_beta=cfg["activation_situ_linear_beta"],
    )
    precision = MoePrecision(
        tensor_dtype=cfg["dtype"],
        weight_quant_method=quant["quant_method"],
        weight_format=quant["format"],
        activation_scheme="framework_selected",
        weight_group_size=weights["group_size"],
        shared_weight_dtype=cfg["dtype"],
    )
    return Arch(
        name=name,
        family="kimi_k3",
        hidden_size=geometry.hidden_size,
        num_layers=cfg["num_hidden_layers"],
        attention=AttentionArch(
            kind="hybrid_kda_mla",
            num_heads=cfg["num_attention_heads"],
            num_kv_heads=cfg["num_key_value_heads"],
            head_dim=cfg["v_head_dim"],
        ),
        moe=MoeArch(
            num_experts=geometry.routed_expert_count,
            num_experts_per_tok=geometry.experts_per_token,
            moe_intermediate_size=geometry.routed_expert_intermediate_size,
            n_shared_experts=geometry.shared_expert_count,
            dense_intermediate_size=cfg["intermediate_size"],
            num_dense_layers=cfg["first_k_dense_replace"],
        ),
        moe_geometry=geometry,
        moe_routing=routing,
        moe_precision=precision,
        moe_activation=activation,
    )


def _build_minimax(cfg: dict[str, Any], name: str) -> Arch:
    # MiniMax-M2.5: GQA-style MHA + MoE (no shared experts in published config).
    return Arch(
        name=name,
        family="minimax",
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_hidden_layers"],
        attention=AttentionArch(
            kind="mha",
            num_heads=cfg["num_attention_heads"],
            num_kv_heads=cfg["num_key_value_heads"],
            head_dim=cfg["head_dim"],
        ),
        moe=MoeArch(
            num_experts=cfg.get("num_local_experts", cfg.get("num_experts", 0)),
            num_experts_per_tok=cfg["num_experts_per_tok"],
            moe_intermediate_size=cfg.get(
                "moe_intermediate_size", cfg.get("intermediate_size", 0)
            ),
            n_shared_experts=cfg.get("n_shared_experts", 0),
            dense_intermediate_size=cfg.get("intermediate_size", 0),
            num_dense_layers=0,
        ),
    )


def _build_gptoss(cfg: dict[str, Any], name: str) -> Arch:
    # GPT-OSS-120B: GQA MHA + MoE. layer_types alternates sliding/full.
    sw = cfg.get("sliding_window")
    return Arch(
        name=name,
        family="gptoss",
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_hidden_layers"],
        attention=AttentionArch(
            kind="mha",
            num_heads=cfg["num_attention_heads"],
            num_kv_heads=cfg["num_key_value_heads"],
            head_dim=cfg["head_dim"],
            sliding_window=sw,
        ),
        moe=MoeArch(
            num_experts=cfg.get("num_local_experts", cfg.get("num_experts", 0)),
            num_experts_per_tok=cfg["num_experts_per_tok"],
            moe_intermediate_size=cfg.get("intermediate_size", 0),
            n_shared_experts=0,
            dense_intermediate_size=0,
            num_dense_layers=0,
        ),
    )


def _build_qwen3moe(cfg: dict[str, Any], name: str) -> Arch:
    # Qwen3.5-MoE: hybrid linear/full attention, MoE with `num_experts` and
    # `shared_expert_intermediate_size`. We model the full-attention layers
    # only (the linear-attention layers don't fit the canonical attention op
    # cleanly yet — TODO).
    return Arch(
        name=name,
        family="qwen3moe",
        hidden_size=cfg["hidden_size"],
        num_layers=cfg["num_hidden_layers"],
        attention=AttentionArch(
            kind="mha",
            num_heads=cfg["num_attention_heads"],
            num_kv_heads=cfg["num_key_value_heads"],
            head_dim=cfg["head_dim"],
        ),
        moe=MoeArch(
            num_experts=cfg.get("num_experts", cfg.get("num_local_experts", 0)),
            num_experts_per_tok=cfg["num_experts_per_tok"],
            moe_intermediate_size=cfg.get("moe_intermediate_size", 0),
            n_shared_experts=1 if cfg.get("shared_expert_intermediate_size") else 0,
            dense_intermediate_size=cfg.get("shared_expert_intermediate_size", 0),
            num_dense_layers=0,
        ),
        mtp_num_layers=cfg.get("mtp_num_hidden_layers", 0) or 0,
    )


_BUILDERS = {
    "deepseek": _build_deepseek,
    "glm": _build_glm,
    "kimi": _build_kimi,
    "kimi_k3": _build_kimi_k3,
    "minimax": _build_minimax,
    "gptoss": _build_gptoss,
    "qwen3moe": _build_qwen3moe,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def load_arch(model_id: str, extra_dirs: tuple[str, ...] = _DEFAULT_LOCAL_DIRS) -> Arch:
    """Load the architecture descriptor for a HuggingFace model id."""
    raw = _load_hf_config(model_id, extra_dirs)
    cfg = _text_subconfig(raw)
    family = _family(cfg)
    try:
        return _BUILDERS[family](cfg, name=model_id)
    except KeyError as error:
        if family == "kimi_k3":
            raise ValueError(f"Kimi K3 config is missing {error.args[0]!r}") from error
        raise
