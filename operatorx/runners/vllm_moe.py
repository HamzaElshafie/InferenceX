"""Framework-native DeepSeek MoE module execution through vLLM.

The initial implementation is deliberately single-rank. It constructs vLLM's
``DeepseekV2MoE`` class with synthetic serialized-FP8 weights, executes the
router, routed experts, shared expert, and output combination, and rejects any
request that would imply distributed execution.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from operatorx.core import BackendImpl, Op, UnsupportedOpError

_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16}


@dataclass(frozen=True)
class DeepseekMoeSpec:
    """Validated model semantics needed to construct vLLM's DeepSeek MoE."""

    model_id: str
    num_tokens: int
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    num_shared_experts: int
    activation: str
    score_function: str
    selection_method: str
    normalize_selected_weights: bool
    routed_output_scale: float
    expert_group_count: int
    selected_expert_group_count: int
    activation_dtype: torch.dtype
    weight_block_size: tuple[int, int]
    input_seed: int
    weight_seed: int

    @classmethod
    def from_args(cls, args: Mapping[str, Any]) -> DeepseekMoeSpec:
        """Validate one strict, single-rank DeepSeek module request."""

        required = (
            "model_id",
            "hidden_act",
            "score_function",
            "selection_method",
            "normalize_selected_weights",
            "routed_output_scale",
            "expert_group_count",
            "selected_expert_group_count",
            "weight_block_size_n",
            "weight_block_size_k",
        )
        missing = [name for name in required if args.get(name) is None]
        if missing:
            raise UnsupportedOpError(
                "vLLM moe_forward requires explicit model semantics: "
                + ", ".join(missing)
            )

        integer_fields = (
            "num_tokens",
            "hidden",
            "intermediate",
            "num_experts",
            "top_k",
            "n_shared_experts",
            "expert_group_count",
            "selected_expert_group_count",
            "weight_block_size_n",
            "weight_block_size_k",
        )
        for name in integer_fields:
            value = args[name]
            if type(value) is not int or value <= 0:
                raise UnsupportedOpError(f"vLLM moe_forward requires positive {name}")

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
                "initial vLLM moe_forward supports only real single-rank "
                f"EP1/TP1 execution; got {rendered}"
            )

        if args["dtype_act"] not in _DTYPES:
            raise UnsupportedOpError(
                "vLLM moe_forward requires BF16 or FP16 module input"
            )
        if args["dtype_weight"] != "fp8":
            raise UnsupportedOpError(
                "initial vLLM moe_forward requires serialized FP8 weights"
            )
        if args["hidden_act"] != "silu":
            raise UnsupportedOpError("vLLM DeepSeek MoE supports SiLU activation")
        if args["selection_method"] != "noaux_tc":
            raise UnsupportedOpError(
                "initial vLLM DeepSeek MoE requires noaux_tc routing"
            )
        if type(args["normalize_selected_weights"]) is not bool:
            raise UnsupportedOpError("normalize_selected_weights must be boolean")
        scale = args["routed_output_scale"]
        if type(scale) not in (int, float) or scale <= 0:
            raise UnsupportedOpError("routed_output_scale must be positive")
        for name in ("model_id", "score_function"):
            value = args[name]
            if not isinstance(value, str) or not value.strip():
                raise UnsupportedOpError(f"{name} must be a non-empty string")

        return cls(
            model_id=args["model_id"],
            num_tokens=args["num_tokens"],
            hidden_size=args["hidden"],
            intermediate_size=args["intermediate"],
            num_experts=args["num_experts"],
            top_k=args["top_k"],
            num_shared_experts=args["n_shared_experts"],
            activation=args["hidden_act"],
            score_function=args["score_function"],
            selection_method=args["selection_method"],
            normalize_selected_weights=args["normalize_selected_weights"],
            routed_output_scale=float(scale),
            expert_group_count=args["expert_group_count"],
            selected_expert_group_count=args["selected_expert_group_count"],
            activation_dtype=_DTYPES[args["dtype_act"]],
            weight_block_size=(
                args["weight_block_size_n"],
                args["weight_block_size_k"],
            ),
            input_seed=args.get("input_seed", 0),
            weight_seed=args.get("weight_seed", 0),
        )

    def hf_config(self) -> SimpleNamespace:
        """Build the model fields consumed by vLLM's ``DeepseekV2MoE``."""

        return SimpleNamespace(
            hidden_size=self.hidden_size,
            moe_intermediate_size=self.intermediate_size,
            n_routed_experts=self.num_experts,
            num_experts_per_tok=self.top_k,
            n_shared_experts=self.num_shared_experts,
            hidden_act=self.activation,
            scoring_func=self.score_function,
            topk_method=self.selection_method,
            norm_topk_prob=self.normalize_selected_weights,
            routed_scaling_factor=self.routed_output_scale,
            n_group=self.expert_group_count,
            topk_group=self.selected_expert_group_count,
        )

    def vllm_model_config(self) -> SimpleNamespace:
        """Build the model metadata consumed during this module's setup.

        Constructing vLLM's full ``ModelConfig`` would resolve a remote model
        configuration and tokenizer that this synthetic layer benchmark does
        not use. This view deliberately exposes only the fields read by model-
        parallel initialization and post-load weight processing.
        """

        return SimpleNamespace(
            model=self.model_id,
            dtype=self.activation_dtype,
            quantization="fp8",
            is_moe=True,
        )


def _ensure_single_rank_vllm(vllm_config: Any) -> None:
    """Initialize the one-rank vLLM process groups used by its MoE layer."""

    import torch.distributed as dist
    from vllm.config import set_current_vllm_config
    from vllm.distributed import init_distributed_environment, initialize_model_parallel
    from vllm.distributed.parallel_state import model_parallel_is_initialized
    from vllm.v1.worker.workspace import (
        init_workspace_manager,
        is_workspace_manager_initialized,
        unlock_workspace,
    )

    if dist.is_initialized():
        if dist.get_world_size() != 1 or dist.get_rank() != 0:
            raise UnsupportedOpError(
                "initial vLLM moe_forward requires a one-rank process group"
            )
    else:
        required_env = (
            "MASTER_ADDR",
            "MASTER_PORT",
            "RANK",
            "LOCAL_RANK",
            "WORLD_SIZE",
        )
        missing = [name for name in required_env if not os.environ.get(name)]
        if missing:
            raise RuntimeError(
                "vLLM distributed initialization is missing environment variables: "
                + ", ".join(missing)
            )
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=int(os.environ["LOCAL_RANK"]),
            distributed_init_method="env://",
            backend="nccl",
        )

    if not model_parallel_is_initialized():
        with set_current_vllm_config(vllm_config):
            initialize_model_parallel(tensor_model_parallel_size=1)

    if not is_workspace_manager_initialized():
        init_workspace_manager(torch.device("cuda"))
    else:
        # The manager is process-global. A preceding benchmark case may have
        # locked it after warmup, so reopen growth only during preparation for
        # this case; ``finalize_warmup`` locks it again before measurement.
        unlock_workspace()


@torch.no_grad()
def _initialize_synthetic_parameters(module: torch.nn.Module, seed: int) -> None:
    """Fill module parameters with deterministic, non-degenerate random values."""

    device_index = torch.cuda.current_device()
    with torch.random.fork_rng(devices=[device_index]):
        torch.manual_seed(seed)
        for name, parameter in module.named_parameters():
            if "scale" in name:
                parameter.fill_(1.0)
            elif "correction_bias" in name:
                parameter.zero_()
            elif parameter.dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
                # Generate one expert at a time so initialization does not need
                # a second full-size BF16 copy of the complete expert table.
                for shard in parameter:
                    source = torch.empty_like(shard, dtype=torch.bfloat16)
                    source.normal_(mean=0.0, std=0.02)
                    shard.copy_(source)
            elif parameter.dtype.is_floating_point:
                parameter.normal_(mean=0.0, std=0.02)
            else:
                raise RuntimeError(
                    f"cannot initialize synthetic parameter {name!r} "
                    f"with dtype {parameter.dtype}"
                )


def _resolved_execution_metadata(
    module: torch.nn.Module,
    hidden_states: torch.Tensor,
    spec: DeepseekMoeSpec,
) -> dict[str, Any]:
    """Describe the concrete vLLM MoE backend and its launch configuration."""

    import vllm.envs as vllm_envs
    from vllm.model_executor.layers import fused_moe
    from vllm.model_executor.layers.fused_moe import fused_moe as fused_moe_impl

    experts = module.experts
    quant_method = experts.quant_method
    backend_value = getattr(quant_method.fp8_backend, "value", quant_method.fp8_backend)
    backend = str(backend_value)
    routed_experts: dict[str, Any] = {
        "kernel_backend": backend,
        "tuning_status": "not_applicable",
    }

    # vLLM's JSON tuning tables and launch dictionary configure its Triton MoE
    # backend. Other backends select kernels through different mechanisms, so
    # claiming this lookup describes them would be misleading.
    if "triton" in backend.lower():
        quant_config = quant_method.moe_quant_config
        dtype_name = quant_config.config_name(hidden_states.dtype)
        block_shape = quant_config.block_shape
        block_n, block_k = block_shape or (0, 0)
        file_name = fused_moe_impl.get_config_file_name(
            spec.num_experts,
            spec.intermediate_size,
            dtype_name,
            block_shape,
        )
        override = fused_moe.get_config()
        tuned_configs = (
            None
            if override
            else fused_moe_impl.get_moe_configs(
                spec.num_experts,
                spec.intermediate_size,
                dtype_name,
                block_n,
                block_k,
            )
        )
        resolved_config = fused_moe_impl.try_get_optimal_moe_config(
            tuple(experts.w13_weight.shape),
            tuple(experts.w2_weight.shape),
            spec.top_k,
            dtype_name,
            spec.num_tokens,
            block_shape,
        )

        candidate_paths = []
        if vllm_envs.VLLM_TUNED_CONFIG_FOLDER is not None:
            candidate_paths.append(Path(vllm_envs.VLLM_TUNED_CONFIG_FOLDER) / file_name)
        candidate_paths.append(
            Path(fused_moe_impl.__file__).parent / "configs" / file_name
        )
        tuned_path = next((path for path in candidate_paths if path.exists()), None)

        if override:
            tuning_status = "explicit_override"
            tuned_batch_size = None
        elif tuned_configs:
            tuning_status = "tuned_table"
            tuned_batch_size = min(
                tuned_configs,
                key=lambda batch_size: abs(batch_size - spec.num_tokens),
            )
        else:
            tuning_status = "default_heuristic"
            tuned_batch_size = None

        routed_experts.update(
            {
                "quantization_config": dtype_name,
                "tuning_status": tuning_status,
                "tuning_lookup_key": file_name,
                "tuned_config_file": (
                    str(tuned_path) if tuning_status == "tuned_table" else None
                ),
                "tuned_batch_size": tuned_batch_size,
                "resolved_launch_config": dict(resolved_config),
            }
        )

    module_type = type(module)
    return {
        "resolved_execution": {
            "framework_module": f"{module_type.__module__}.{module_type.__qualname__}",
            "routed_experts": routed_experts,
            "workspace": {"locked_after_warmup": False},
        }
    }


def prepare(op: Op) -> dict[str, Any]:
    """Construct, initialize, process, and validate one native vLLM MoE layer."""

    spec = DeepseekMoeSpec.from_args(op.args)

    from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
    from vllm.forward_context import set_forward_context
    from vllm.model_executor.layers.quantization.fp8 import Fp8Config
    from vllm.model_executor.model_loader.utils import process_weights_after_loading
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2MoE

    quant_config = Fp8Config(
        is_checkpoint_fp8_serialized=True,
        activation_scheme="dynamic",
        weight_block_size=list(spec.weight_block_size),
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
    _ensure_single_rank_vllm(vllm_config)

    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(spec.activation_dtype)
        with torch.device("cuda"), set_current_vllm_config(vllm_config):
            module = DeepseekV2MoE(
                config=spec.hf_config(),
                parallel_config=parallel_config,
                quant_config=quant_config,
                prefix="operatorx.deepseek_moe",
            )
            _initialize_synthetic_parameters(module, spec.weight_seed)
            process_weights_after_loading(module, model_config, torch.device("cuda"))
    finally:
        torch.set_default_dtype(previous_dtype)

    module.eval()
    input_generator = torch.Generator(device="cuda")
    input_generator.manual_seed(spec.input_seed)
    hidden_states = torch.randn(
        spec.num_tokens,
        spec.hidden_size,
        dtype=spec.activation_dtype,
        device="cuda",
        generator=input_generator,
    )

    # This untimed call completes lazy kernel setup and enforces a basic output
    # correctness gate before the common runner starts warmup and measurement.
    with set_forward_context(None, vllm_config, num_tokens=spec.num_tokens):
        output = module(hidden_states)
    torch.cuda.synchronize()
    if output.shape != hidden_states.shape:
        raise RuntimeError(
            f"vLLM MoE returned shape {tuple(output.shape)}, "
            f"expected {tuple(hidden_states.shape)}"
        )
    if output.dtype != spec.activation_dtype:
        raise RuntimeError(
            f"vLLM MoE returned dtype {output.dtype}, expected {spec.activation_dtype}"
        )
    if not torch.isfinite(output).all().item():
        raise RuntimeError("vLLM MoE returned non-finite output")

    metadata = _resolved_execution_metadata(module, hidden_states, spec)

    return {
        "module": module,
        "hidden_states": hidden_states,
        "out": output,
        "vllm_config": vllm_config,
        "metadata": metadata,
    }


@torch.no_grad()
def kernel(context: dict[str, Any]) -> None:
    """Execute the complete framework-native MoE boundary."""

    from vllm.forward_context import set_forward_context

    hidden_states = context["hidden_states"]
    with set_forward_context(
        None,
        context["vllm_config"],
        num_tokens=hidden_states.shape[0],
    ):
        context["out"] = context["module"](hidden_states)


def finalize_warmup(context: dict[str, Any]) -> None:
    """Prevent measured iterations from silently growing vLLM scratch memory."""

    from vllm.v1.worker.workspace import lock_workspace

    lock_workspace()
    context["metadata"]["resolved_execution"]["workspace"]["locked_after_warmup"] = True


def result_metadata(context: dict[str, Any]) -> Mapping[str, Any]:
    """Return resolved execution facts collected outside the timed region."""

    return context["metadata"]


IMPLS = [
    BackendImpl(
        op_type="moe_forward",
        prepare=prepare,
        kernel=kernel,
        finalize_warmup=finalize_warmup,
        result_metadata=result_metadata,
    )
]
