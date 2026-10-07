"""Shared preparation diagnostics for native vLLM MoE module adapters."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch

from operatorx.core import UnsupportedOpError


def ensure_single_rank_vllm(vllm_config: Any) -> None:
    """Initialize the one-rank process groups and vLLM MoE workspace."""

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
        # The manager is process-global; preparation may follow a locked case.
        unlock_workspace()


def qualified_class_name(value: object) -> str:
    """Return a fully qualified class name for execution metadata."""

    value_type = type(value)
    return f"{value_type.__module__}.{value_type.__qualname__}"


def linear_execution_metadata(layer: torch.nn.Module) -> dict[str, Any]:
    """Describe one vLLM linear layer after native weight processing."""

    metadata = {"module_class": qualified_class_name(layer)}
    quant_method = getattr(layer, "quant_method", None)
    if quant_method is None:
        return metadata

    metadata["linear_method_class"] = qualified_class_name(quant_method)
    selected_kernel = getattr(quant_method, "fp8_linear", None)
    if selected_kernel is not None:
        metadata["selected_kernel_class"] = qualified_class_name(selected_kernel)
    return metadata


def routing_workload_metadata(
    topk_ids: torch.Tensor,
    *,
    num_experts: int,
    expected_logical_assignments: int,
    padding_block_size: int | None,
) -> dict[str, Any]:
    """Summarize routing decisions observed during untimed validation.

    Padding is derived only when the selected kernel exposes its token-block
    size. For vLLM's Triton grouped MoE, each active expert bucket is aligned
    to ``BLOCK_SIZE_M`` by its alignment kernel.
    """

    flat_ids = topk_ids.detach().to(device="cpu", dtype=torch.int64).reshape(-1)
    logical_assignments = flat_ids.numel()
    if logical_assignments != expected_logical_assignments:
        raise RuntimeError(
            "vLLM routing returned "
            f"{logical_assignments} assignments, expected "
            f"{expected_logical_assignments}"
        )
    if logical_assignments == 0:
        raise RuntimeError("vLLM routing returned no expert assignments")
    if flat_ids.min().item() < 0 or flat_ids.max().item() >= num_experts:
        raise RuntimeError("vLLM routing returned an expert ID outside model geometry")

    counts = torch.bincount(flat_ids, minlength=num_experts)
    active_counts = counts[counts > 0]
    metadata: dict[str, Any] = {
        "collection_phase": "untimed_validation",
        "logical_assignment_count": logical_assignments,
        "active_expert_count": active_counts.numel(),
        "assignments_per_expert": counts.tolist(),
        "active_expert_load": {
            "min": active_counts.min().item(),
            "mean": active_counts.to(torch.float64).mean().item(),
            "max": active_counts.max().item(),
        },
        "dropped_assignment_count": 0,
    }
    if padding_block_size is not None:
        if padding_block_size <= 0:
            raise RuntimeError("resolved MoE padding block size must be positive")
        padded_counts = (
            torch.div(
                active_counts + padding_block_size - 1,
                padding_block_size,
                rounding_mode="floor",
            )
            * padding_block_size
        )
        post_padding_assignments = padded_counts.sum().item()
        metadata["padding"] = {
            "block_size": padding_block_size,
            "post_padding_assignment_count": post_padding_assignments,
            "padding_assignment_count": post_padding_assignments - logical_assignments,
            "amplification_ratio": post_padding_assignments / logical_assignments,
            "derivation": "sum_of_active_expert_assignments_rounded_up_to_block_size",
        }
    else:
        metadata["padding"] = {
            "status": "unavailable",
            "reason": "selected_backend_does_not_expose_token_block_size",
        }
    return metadata


@contextmanager
def capture_routing_ids(module: torch.nn.Module) -> Iterator[list[torch.Tensor]]:
    """Capture logical IDs from either modular routing or monolithic replay.

    Callbacks observe IDs before optional expert-load-balancing remapping.
    """

    captured: list[torch.Tensor] = []

    def capture(topk_ids: torch.Tensor) -> None:
        if not isinstance(topk_ids, torch.Tensor):
            raise TypeError(
                "vLLM router capture did not provide expert IDs as a tensor"
            )
        captured.append(topk_ids.detach().clone())

    runner = module.experts
    quant_method = getattr(runner, "_quant_method", None)
    if quant_method is not None and quant_method.is_monolithic:
        moe_kernel = getattr(quant_method, "moe_kernel", None)
        implementation = getattr(moe_kernel, "impl", None)
        capture_source = getattr(implementation, "fused_experts", None)
        supports_capture = getattr(
            capture_source, "supports_routing_replay_capture", None
        )
        if supports_capture is None or not supports_capture():
            backend = type(capture_source).__name__
            raise RuntimeError(
                "vLLM monolithic MoE backend does not expose logical routing "
                f"replay: {backend}"
            )
    else:
        capture_source = runner.router

    set_capture_fn = getattr(capture_source, "set_capture_fn", None)
    if set_capture_fn is None:
        raise RuntimeError("vLLM MoE path does not expose logical-ID capture support")

    set_capture_fn(capture)
    try:
        yield captured
    finally:
        set_capture_fn(None)


def validate_repeated_outputs(
    first: torch.Tensor,
    second: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, Any]:
    """Validate the module boundary and identical-input repeatability."""

    expected_shape = tuple(expected.shape)
    expected_dtype = expected.dtype
    for iteration, output in enumerate((first, second), start=1):
        if output.shape != expected_shape:
            raise RuntimeError(
                f"vLLM MoE iteration {iteration} returned shape {tuple(output.shape)}, "
                f"expected {expected_shape}"
            )
        if output.dtype != expected_dtype:
            raise RuntimeError(
                f"vLLM MoE iteration {iteration} returned dtype {output.dtype}, "
                f"expected {expected_dtype}"
            )
        if not torch.isfinite(output).all().item():
            raise RuntimeError(
                f"vLLM MoE iteration {iteration} returned non-finite output"
            )
    if not torch.equal(first, second):
        raise RuntimeError(
            "vLLM MoE produced different outputs for two identical untimed inputs"
        )

    return {
        "status": "passed",
        "output_shape": list(expected_shape),
        "output_dtype": str(expected_dtype),
        "all_values_finite": True,
        "repeatability": {
            "comparison": "exact",
            "untimed_iteration_count": 2,
            "passed": True,
        },
    }
