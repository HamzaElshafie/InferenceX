"""Framework-neutral dimensions of a single MoE layer."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MoeLayerGeometry:
    """Logical expert dimensions, independent of rank placement and kernels.

    The shared-expert intermediate size is the width of *each* shared expert.
    A layer without shared experts sets both shared-expert fields to zero.
    """

    hidden_size: int
    routed_expert_count: int
    experts_per_token: int
    routed_expert_intermediate_size: int
    shared_expert_count: int = 0
    shared_expert_intermediate_size: int = 0

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "routed_expert_count",
            "experts_per_token",
            "routed_expert_intermediate_size",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        if self.experts_per_token > self.routed_expert_count:
            raise ValueError("experts_per_token cannot exceed routed_expert_count")

        for name in ("shared_expert_count", "shared_expert_intermediate_size"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

        if (self.shared_expert_count == 0) != (
            self.shared_expert_intermediate_size == 0
        ):
            raise ValueError(
                "shared_expert_count and shared_expert_intermediate_size "
                "must either both be zero or both be positive"
            )


@dataclass(frozen=True)
class MoeRouting:
    """Model routing rules before a framework chooses its kernels.

    Group fields are paired: both are absent for ungrouped routing, or both
    describe how many expert groups exist and how many may be selected.
    """

    score_function: str
    selection_method: str
    normalize_selected_weights: bool
    routed_output_scale: float
    group_count: int | None = None
    selected_group_count: int | None = None

    def __post_init__(self) -> None:
        for name in ("score_function", "selection_method"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

        if type(self.normalize_selected_weights) is not bool:
            raise ValueError("normalize_selected_weights must be a boolean")

        scale = self.routed_output_scale
        if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
            raise ValueError("routed_output_scale must be a positive finite number")

        if (self.group_count is None) != (self.selected_group_count is None):
            raise ValueError(
                "group_count and selected_group_count must be set together"
            )
        if self.group_count is not None:
            if type(self.group_count) is not int or self.group_count <= 0:
                raise ValueError("group_count must be a positive integer")
            if (
                type(self.selected_group_count) is not int
                or self.selected_group_count <= 0
            ):
                raise ValueError("selected_group_count must be a positive integer")
            if self.selected_group_count > self.group_count:
                raise ValueError("selected_group_count cannot exceed group_count")


@dataclass(frozen=True)
class MoePrecision:
    """Precision declared by a model config, not a resolved kernel dtype.

    ``tensor_dtype`` is the config's ordinary tensor dtype. A framework must
    still verify the actual module input/output and internal operand dtypes.
    """

    tensor_dtype: str
    weight_quant_method: str
    weight_format: str
    activation_scheme: str
    weight_block_size: tuple[int, int]

    def __post_init__(self) -> None:
        for name in (
            "tensor_dtype",
            "weight_quant_method",
            "weight_format",
            "activation_scheme",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")

        if (
            not isinstance(self.weight_block_size, tuple)
            or len(self.weight_block_size) != 2
            or any(
                type(size) is not int or size <= 0 for size in self.weight_block_size
            )
        ):
            raise ValueError("weight_block_size must contain two positive integers")


@dataclass(frozen=True)
class MoeWorkload:
    """Serving-shaped token workload presented to one MoE module invocation.

    Global and local token counts are separate because a future distributed
    topology may shard or replicate the logical workload across ranks. They
    are equal for the current single-rank benchmark.
    """

    phase: str
    global_num_tokens: int
    local_num_tokens: int
    routing_input_policy: str
    input_seed: int
    source: str

    def __post_init__(self) -> None:
        if self.phase not in {"decode", "prefill"}:
            raise ValueError("phase must be 'decode' or 'prefill'")
        for name in ("global_num_tokens", "local_num_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.local_num_tokens > self.global_num_tokens:
            raise ValueError("local_num_tokens cannot exceed global_num_tokens")
        if type(self.input_seed) is not int or self.input_seed < 0:
            raise ValueError("input_seed must be a non-negative integer")
        for name in ("routing_input_policy", "source"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
