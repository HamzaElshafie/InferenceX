"""Framework-native Kimi-K3 MoE module execution through vLLM.

This adapter constructs vLLM's native ``KimiMoE`` at a representative MoE
layer, supplies deterministic synthetic weights in K3's checkpoint formats,
and validates a real single-rank forward before returning control to the
common OperatorX timing lifecycle.
"""

from __future__ import annotations

import copy
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch

from operatorx.core import Op, UnsupportedOpError
from operatorx.core.moe import MoeWorkload
from operatorx.runners.vllm_moe.common import (
    capture_routing_ids,
    ensure_single_rank_vllm,
    linear_execution_metadata,
    qualified_class_name,
    routing_workload_metadata,
    validate_repeated_outputs,
)

_KIMI_K3_MODEL_ID = "moonshotai/Kimi-K3"
_KIMI_K3_REVISION = "f831ab66814297da540d832a5235f8e904f29d06"
_MXFP4_GROUP_SIZE = 32


@dataclass(frozen=True)
class KimiK3MoeSpec:
    """Validated semantics for one native, single-rank Kimi-K3 MoE case."""

    model_id: str
    model_revision: str
    layer_index: int
    num_tokens: int
    hidden_size: int
    routed_hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    num_shared_experts: int
    activation: str
    activation_gate_beta: float
    activation_up_beta: float
    norm_epsilon: float
    score_function: str
    selection_method: str
    normalize_selected_weights: bool
    routed_output_scale: float
    expert_group_count: int
    selected_expert_group_count: int
    activation_dtype: torch.dtype
    activation_dtype_name: str
    weight_format: str
    weight_group_size: int
    workload: MoeWorkload
    weight_source: str
    execution_mode: str
    weight_seed: int

    @classmethod
    def from_args(cls, args: Mapping[str, Any]) -> KimiK3MoeSpec:
        """Validate an OperatorX request without importing vLLM."""

        required = (
            "model_id",
            "model_revision",
            "layer_index",
            "hidden_act",
            "activation_gate_beta",
            "activation_up_beta",
            "routed_norm_epsilon",
            "score_function",
            "selection_method",
            "normalize_selected_weights",
            "routed_output_scale",
            "expert_group_count",
            "selected_expert_group_count",
            "weight_format",
            "weight_group_size",
            "phase",
            "global_num_tokens",
            "workload_source",
        )
        missing = [name for name in required if args.get(name) is None]
        if missing:
            raise UnsupportedOpError(
                "vLLM Kimi-K3 moe_forward requires explicit model semantics: "
                + ", ".join(missing)
            )

        integer_fields = (
            "layer_index",
            "num_tokens",
            "hidden",
            "routed_hidden",
            "intermediate",
            "num_experts",
            "top_k",
            "n_shared_experts",
            "expert_group_count",
            "selected_expert_group_count",
            "weight_group_size",
        )
        for name in integer_fields:
            value = args.get(name)
            minimum = 0 if name == "layer_index" else 1
            if type(value) is not int or value < minimum:
                qualifier = "non-negative" if minimum == 0 else "positive"
                raise UnsupportedOpError(
                    f"vLLM Kimi-K3 moe_forward requires {qualifier} {name}"
                )

        if args["layer_index"] == 0:
            raise UnsupportedOpError("Kimi-K3 layer 0 is dense, not MoE")
        if args["top_k"] > args["num_experts"]:
            raise UnsupportedOpError("MoE top_k cannot exceed num_experts")
        if args["selected_expert_group_count"] > args["expert_group_count"]:
            raise UnsupportedOpError(
                "selected_expert_group_count cannot exceed expert_group_count"
            )

        topology = {
            "world_size": args.get("world_size", 1),
            "expert_parallel_size": args.get("expert_parallel_size", 1),
            "routed_tensor_parallel_size": args.get("routed_tensor_parallel_size", 1),
            "shared_tensor_parallel_size": args.get("shared_tensor_parallel_size", 1),
        }
        if any(type(value) is not int or value != 1 for value in topology.values()):
            rendered = ", ".join(f"{key}={value}" for key, value in topology.items())
            raise UnsupportedOpError(
                "initial Kimi-K3 moe_forward supports only real single-rank "
                f"EP1/TP1 execution; got {rendered}"
            )

        fixed_values = {
            "model_family": "kimi_k3",
            "model_id": _KIMI_K3_MODEL_ID,
            "model_revision": _KIMI_K3_REVISION,
            "layer_index": 1,
            "hidden_act": "situ",
            "score_function": "sigmoid",
            "selection_method": "noaux_tc",
            "dtype_act": "bf16",
            "dtype_weight": "mxfp4",
            "weight_format": "mxfp4-pack-quantized",
            "weight_group_size": _MXFP4_GROUP_SIZE,
            "weight_source": "synthetic",
            "execution_mode": "eager",
        }
        for name, expected in fixed_values.items():
            if args.get(name) != expected:
                raise UnsupportedOpError(
                    f"Kimi-K3 requires {name}={expected!r}; got {args.get(name)!r}"
                )

        for name in (
            "activation_gate_beta",
            "activation_up_beta",
            "routed_norm_epsilon",
            "routed_output_scale",
        ):
            value = args[name]
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise UnsupportedOpError(f"Kimi-K3 requires positive finite {name}")
        if type(args["normalize_selected_weights"]) is not bool:
            raise UnsupportedOpError("normalize_selected_weights must be boolean")

        try:
            workload = MoeWorkload(
                phase=args["phase"],
                global_num_tokens=args["global_num_tokens"],
                local_num_tokens=args["num_tokens"],
                routing_input_policy=args.get("expert_distribution", "model_native"),
                input_seed=args.get("input_seed", 0),
                source=args["workload_source"],
            )
        except ValueError as error:
            raise UnsupportedOpError(
                f"invalid Kimi-K3 moe_forward workload: {error}"
            ) from error
        if workload.global_num_tokens != workload.local_num_tokens:
            raise UnsupportedOpError(
                "single-rank Kimi-K3 moe_forward requires equal global and local tokens"
            )
        if workload.routing_input_policy != "model_native":
            raise UnsupportedOpError(
                "Kimi-K3 moe_forward requires model_native routing inputs"
            )

        return cls(
            model_id=args["model_id"],
            model_revision=args["model_revision"],
            layer_index=args["layer_index"],
            num_tokens=args["num_tokens"],
            hidden_size=args["hidden"],
            routed_hidden_size=args["routed_hidden"],
            intermediate_size=args["intermediate"],
            num_experts=args["num_experts"],
            top_k=args["top_k"],
            num_shared_experts=args["n_shared_experts"],
            activation=args["hidden_act"],
            activation_gate_beta=float(args["activation_gate_beta"]),
            activation_up_beta=float(args["activation_up_beta"]),
            norm_epsilon=float(args["routed_norm_epsilon"]),
            score_function=args["score_function"],
            selection_method=args["selection_method"],
            normalize_selected_weights=args["normalize_selected_weights"],
            routed_output_scale=float(args["routed_output_scale"]),
            expert_group_count=args["expert_group_count"],
            selected_expert_group_count=args["selected_expert_group_count"],
            activation_dtype=torch.bfloat16,
            activation_dtype_name="bf16",
            weight_format=args["weight_format"],
            weight_group_size=args["weight_group_size"],
            workload=workload,
            weight_source=args["weight_source"],
            execution_mode=args["execution_mode"],
            weight_seed=args.get("weight_seed", 0),
        )

    def quantization_config(self) -> dict[str, Any]:
        """Return K3's routed-only compressed-tensors MXFP4 contract."""

        return {
            "quant_method": "compressed-tensors",
            "format": self.weight_format,
            "quantization_status": "compressed",
            "ignore": [
                "re:.*self_attn.*",
                "re:.*shared_experts.*",
                "re:.*mlp\\.(gate|up|gate_up|down)_proj.*",
                "re:.*lm_head.*",
                "re:.*vision_tower.*",
                "re:.*mm_projector.*",
            ],
            "config_groups": {
                "group_0": {
                    "format": self.weight_format,
                    "targets": ["Linear"],
                    # The checkpoint specifies MXFP4 routed weights but leaves
                    # activation quantization unset. We do not infer MXFP8 from
                    # the paper or MXFP4 from the weight format; instead record the
                    # operand format selected by vLLM at runtime instead.
                    "input_activations": None,
                    "output_activations": None,
                    "weights": {
                        "num_bits": 4,
                        "type": "float",
                        "strategy": "group",
                        "group_size": self.weight_group_size,
                        "symmetric": True,
                        "dynamic": False,
                        "scale_dtype": "torch.uint8",
                    },
                }
            },
        }

    def hf_config_fields(self) -> dict[str, Any]:
        """Return the fields consumed by vLLM's native ``KimiMoE``."""

        return {
            "hidden_size": self.hidden_size,
            "moe_intermediate_size": self.intermediate_size,
            "num_experts": self.num_experts,
            "num_experts_per_token": self.top_k,
            "num_shared_experts": self.num_shared_experts,
            "routed_expert_hidden_size": self.routed_hidden_size,
            "latent_moe_use_norm": True,
            "rms_norm_eps": self.norm_epsilon,
            "hidden_act": self.activation,
            "activation_situ_beta": self.activation_gate_beta,
            "activation_situ_linear_beta": self.activation_up_beta,
            "moe_router_activation_func": self.score_function,
            "topk_method": self.selection_method,
            "moe_renormalize": self.normalize_selected_weights,
            "routed_scaling_factor": self.routed_output_scale,
            "use_grouped_topk": True,
            "num_expert_group": self.expert_group_count,
            "topk_group": self.selected_expert_group_count,
        }

    def vllm_model_config(self) -> SimpleNamespace:
        """Expose only model metadata required by standalone weight processing."""

        return SimpleNamespace(
            model=self.model_id,
            revision=self.model_revision,
            dtype=self.activation_dtype,
            quantization="compressed-tensors",
            is_moe=True,
            word_embeddings_untied_by_checkpoint=False,
            hf_text_config=SimpleNamespace(model_type="kimi_linear"),
        )

    def requested_workload_metadata(self) -> dict[str, Any]:
        """Serialize the framework-neutral request separately from execution."""

        return {
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "layer_index": self.layer_index,
            "boundary": "moe_module",
            "phase": self.workload.phase,
            "global_num_tokens": self.workload.global_num_tokens,
            "local_num_tokens": self.workload.local_num_tokens,
            "input_placement": "single_rank",
            "routing_input_policy": self.workload.routing_input_policy,
            "source": self.workload.source,
            "geometry": {
                "hidden_size": self.hidden_size,
                "routed_expert_hidden_size": self.routed_hidden_size,
                "routed_expert_count": self.num_experts,
                "experts_per_token": self.top_k,
                "routed_expert_intermediate_size": self.intermediate_size,
                "shared_expert_count": self.num_shared_experts,
                "shared_expert_intermediate_size": self.intermediate_size,
                "routed_output_norm_epsilon": self.norm_epsilon,
            },
            "activation": {
                "name": self.activation,
                "gate_beta": self.activation_gate_beta,
                "up_beta": self.activation_up_beta,
            },
            "routing": {
                "score_function": self.score_function,
                "selection_method": self.selection_method,
                "correction_bias_policy": "zero",
                "normalize_selected_weights": self.normalize_selected_weights,
                "routed_output_scale": self.routed_output_scale,
                "expert_group_count": self.expert_group_count,
                "selected_expert_group_count": self.selected_expert_group_count,
            },
            "precision": {
                "module_input_dtype": self.activation_dtype_name,
                "module_output_dtype": self.activation_dtype_name,
                "routed_weight_format": self.weight_format,
                "routed_weight_group_size": self.weight_group_size,
                "shared_weight_dtype": self.activation_dtype_name,
                "activation_operand_format": "framework_selected",
                "weight_source": self.weight_source,
            },
            "topology": {
                "world_size": 1,
                "expert_parallel_size": 1,
                "routed_tensor_parallel_size": 1,
                "shared_tensor_parallel_size": 1,
            },
            "execution_mode": self.execution_mode,
            "synthetic_fixture": {
                "hidden_states": {
                    "distribution": "normal",
                    "mean": 0.0,
                    "standard_deviation": 1.0,
                    "seed": self.workload.input_seed,
                },
                "parameters": {
                    "floating_point_distribution": "normal",
                    "floating_point_mean": 0.0,
                    "floating_point_standard_deviation": 0.02,
                    "normalization_weight_value": 1.0,
                    "router_correction_bias_value": 0.0,
                    "routed_weight_generation": (
                        "per_expert_bf16_to_native_packed_mxfp4"
                    ),
                    "seed": self.weight_seed,
                },
            },
        }


@torch.no_grad()
def _initialize_packed_mxfp4(
    packed_weights: torch.Tensor,
    encoded_scales: torch.Tensor,
    *,
    quantization_args: Any,
    generator: torch.Generator,
    standard_deviation: float,
) -> None:
    """Generate valid packed MXFP4 weights one expert at a time.

    The temporary BF16 source contains only one expert, avoiding a second
    model-sized allocation. Quantization, FP4 packing, and E8M0 scale encoding
    use the same ``compressed-tensors`` functions that produce K3 checkpoints.
    """

    from compressed_tensors.compressors.mx_utils import decompress_mx_scale
    from compressed_tensors.compressors.nvfp4.helpers import pack_fp4_to_uint8
    from compressed_tensors.quantization.lifecycle.forward import quantize
    from compressed_tensors.quantization.utils.mxfp_utils import generate_mx_scales

    group_size = quantization_args.group_size
    if group_size != _MXFP4_GROUP_SIZE:
        raise RuntimeError(
            f"Kimi-K3 requires MXFP4 group size {_MXFP4_GROUP_SIZE}; got {group_size}"
        )
    if packed_weights.ndim != 3 or encoded_scales.ndim != 3:
        raise RuntimeError("Kimi-K3 packed weights and scales must be rank-three")

    rows = packed_weights.shape[1]
    columns = packed_weights.shape[2] * 2
    expected_scale_shape = (
        packed_weights.shape[0],
        rows,
        columns // group_size,
    )
    if tuple(encoded_scales.shape) != expected_scale_shape:
        raise RuntimeError(
            "Kimi-K3 MXFP4 scale shape does not match packed weights: "
            f"expected {expected_scale_shape}, got {tuple(encoded_scales.shape)}"
        )

    for expert_index in range(packed_weights.shape[0]):
        source = torch.randn(
            rows,
            columns,
            dtype=torch.bfloat16,
            device=packed_weights.device,
            generator=generator,
        ).mul_(standard_deviation)
        grouped = source.reshape(rows, columns // group_size, group_size)
        max_abs = grouped.abs().amax(dim=-1).float()
        scale_codes = generate_mx_scales(max_abs, num_bits=4).to(torch.uint8)
        scales = decompress_mx_scale(scale_codes).to(source.dtype)
        zero_points = torch.zeros_like(scales)
        quantized = quantize(
            source,
            scales,
            zero_points,
            quantization_args,
            dtype=source.dtype,
        )
        packed_weights[expert_index].copy_(pack_fp4_to_uint8(quantized))
        encoded_scales[expert_index].copy_(scale_codes)


@torch.no_grad()
def _initialize_synthetic_parameters(
    module: torch.nn.Module,
    *,
    quantization_args: Any,
    seed: int,
) -> None:
    """Initialize K3's ordinary and packed parameters deterministically."""

    device_index = torch.cuda.current_device()
    with torch.random.fork_rng(devices=[device_index]):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        for name, parameter in module.named_parameters():
            if name.endswith(("weight_packed", "weight_scale")):
                continue
            if "correction_bias" in name:
                parameter.zero_()
            elif name.endswith("norm.weight"):
                parameter.fill_(1.0)
            elif parameter.dtype.is_floating_point:
                parameter.normal_(mean=0.0, std=0.02, generator=generator)
            else:
                raise RuntimeError(
                    f"cannot initialize Kimi-K3 parameter {name!r} "
                    f"with dtype {parameter.dtype}"
                )

        routed = module.experts.routed_experts
        _initialize_packed_mxfp4(
            routed.w13_weight_packed,
            routed.w13_weight_scale,
            quantization_args=quantization_args,
            generator=generator,
            standard_deviation=0.02,
        )
        _initialize_packed_mxfp4(
            routed.w2_weight_packed,
            routed.w2_weight_scale,
            quantization_args=quantization_args,
            generator=generator,
            standard_deviation=0.02,
        )


def _quant_descriptor_dtype(descriptor: object | None) -> str | None:
    """Read the resolved operand format from vLLM's pinned quant descriptor."""

    if descriptor is None:
        return None
    value = getattr(descriptor, "dtype", None)
    return None if value is None else str(value)


def _resolved_execution_metadata(
    module: torch.nn.Module,
    spec: KimiK3MoeSpec,
    topk_ids: torch.Tensor,
) -> dict[str, Any]:
    """Describe the concrete K3 runner, quantization, and overlap policy."""

    runner = module.experts
    routed_experts = runner.routed_experts
    quant_method = runner._quant_method
    kernel = quant_method.moe_kernel
    if kernel is None:
        raise RuntimeError("vLLM did not finalize a Kimi-K3 MXFP4 MoE kernel")
    fused_experts = kernel.fused_experts
    quant_config = quant_method.moe_quant_config
    if quant_config is None:
        raise RuntimeError("vLLM did not resolve Kimi-K3 MoE quantization operands")

    return {
        "requested_workload": spec.requested_workload_metadata(),
        "resolved_routing_workload": routing_workload_metadata(
            topk_ids,
            num_experts=spec.num_experts,
            expected_logical_assignments=spec.num_tokens * spec.top_k,
            padding_block_size=None,
        ),
        "resolved_execution": {
            "framework_module": qualified_class_name(module),
            "components": {
                "router_projection": linear_execution_metadata(module.gate),
                "routing": {
                    "router_class": qualified_class_name(runner.router),
                    "selection_method": spec.selection_method,
                    "correction_bias_policy": "zero",
                },
                "routed_input_projection": linear_execution_metadata(
                    module.routed_expert_down_proj
                ),
                "routed_experts": {
                    "runner_class": qualified_class_name(runner),
                    "module_class": qualified_class_name(routed_experts),
                    "quant_method_class": qualified_class_name(quant_method),
                    "kernel_class": qualified_class_name(kernel),
                    "expert_implementation_class": qualified_class_name(fused_experts),
                    # The framework selects this implementation at runtime.
                    # Recording its actual class avoids claiming CUTLASS on a
                    # platform where vLLM resolves a different MXFP4 backend.
                    "kernel_backend": type(fused_experts).__name__,
                    "checkpoint_weight_format": spec.weight_format,
                    "activation_operand_format": _quant_descriptor_dtype(
                        getattr(quant_config, "_a1", None)
                    ),
                    "weight_operand_format": _quant_descriptor_dtype(
                        getattr(quant_config, "_w1", None)
                    ),
                    "weight_group_size": spec.weight_group_size,
                },
                "routed_output_norm": {
                    "module_class": qualified_class_name(module.routed_expert_norm),
                    "epsilon": spec.norm_epsilon,
                },
                "routed_output_projection": linear_execution_metadata(
                    module.routed_expert_up_proj
                ),
                "shared_expert": {
                    "status": "active",
                    "module_class": qualified_class_name(module.shared_experts),
                    "weight_dtype": spec.activation_dtype_name,
                    "gate_up_projection": linear_execution_metadata(
                        module.shared_experts.gate_up_proj
                    ),
                    "down_projection": linear_execution_metadata(
                        module.shared_experts.down_proj
                    ),
                },
                "overlap": {
                    "router_with_routed_input_projection": spec.num_tokens <= 256,
                    "policy": "vllm_native",
                },
                "communication": {"status": "not_active_single_rank"},
            },
            "workspace": {"locked_after_warmup": False},
        },
    }


def prepare(op: Op) -> dict[str, Any]:
    """Construct, initialize, process, and validate native vLLM ``KimiMoE``."""

    spec = KimiK3MoeSpec.from_args(op.args)

    from compressed_tensors.quantization import QuantizationArgs
    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.models.kimi_k3.nvidia.model import KimiMoE
    from vllm.transformers_utils.configs.kimi_linear import KimiLinearConfig

    raw_quant_config = spec.quantization_config()
    quant_config = CompressedTensorsConfig.from_config(copy.deepcopy(raw_quant_config))
    quantization_args = QuantizationArgs.model_validate(
        raw_quant_config["config_groups"]["group_0"]["weights"]
    )
    parallel_config = ParallelConfig(
        tensor_parallel_size=1,
        data_parallel_size=1,
        enable_expert_parallel=False,
        is_moe_model=True,
    )
    vllm_config = VllmConfig(
        parallel_config=parallel_config,
        quant_config=quant_config,
    )
    model_config = spec.vllm_model_config()
    vllm_config.model_config = model_config

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    ensure_single_rank_vllm(vllm_config)

    config = KimiLinearConfig(**spec.hf_config_fields())
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(spec.activation_dtype)
        with torch.device("cuda"), set_current_vllm_config(vllm_config):
            module = KimiMoE(
                config=config,
                vllm_config=vllm_config,
                quant_config=quant_config,
                prefix="operatorx.kimi_k3_moe",
                layer_idx=spec.layer_index,
            )
            _initialize_synthetic_parameters(
                module,
                quantization_args=quantization_args,
                seed=spec.weight_seed,
            )
            process_weights_after_loading(module, model_config, torch.device("cuda"))
    finally:
        torch.set_default_dtype(previous_dtype)

    module.eval()
    input_generator = torch.Generator(device="cuda")
    input_generator.manual_seed(spec.workload.input_seed)
    hidden_states = torch.randn(
        spec.num_tokens,
        spec.hidden_size,
        dtype=spec.activation_dtype,
        device="cuda",
        generator=input_generator,
    )

    with (
        torch.no_grad(),
        capture_routing_ids(module) as captured_routing_ids,
        set_forward_context(None, vllm_config, num_tokens=spec.num_tokens),
    ):
        first_output = module(hidden_states)
    with (
        torch.no_grad(),
        set_forward_context(None, vllm_config, num_tokens=spec.num_tokens),
    ):
        output = module(hidden_states)
    torch.cuda.synchronize()
    correctness = validate_repeated_outputs(first_output, output, hidden_states)
    if len(captured_routing_ids) != 1:
        raise RuntimeError(
            "vLLM Kimi-K3 routing instrumentation expected exactly one router "
            f"invocation; observed {len(captured_routing_ids)}"
        )

    metadata = _resolved_execution_metadata(
        module,
        spec,
        captured_routing_ids[0],
    )
    metadata["correctness"] = correctness
    return {
        "module": module,
        "hidden_states": hidden_states,
        "out": output,
        "vllm_config": vllm_config,
        "metadata": metadata,
    }
