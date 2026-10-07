"""Framework-neutral dimensions of a single MoE layer."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MoeLayerGeometry:
    """Logical expert dimensions, independent of rank placement and kernels.

    The shared-expert intermediate size is the width of *each* shared expert.
    A layer without shared experts sets both shared-expert fields to zero.
    A missing routed-expert hidden size means experts consume the module's full
    hidden width. A routed-output norm is applied before the up-projection.
    """

    hidden_size: int
    routed_expert_count: int
    experts_per_token: int
    routed_expert_intermediate_size: int
    shared_expert_count: int = 0
    shared_expert_intermediate_size: int = 0
    routed_expert_hidden_size: int | None = None
    routed_output_norm_eps: float | None = None

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

        if self.routed_expert_hidden_size is not None and (
            type(self.routed_expert_hidden_size) is not int
            or self.routed_expert_hidden_size <= 0
        ):
            raise ValueError("routed_expert_hidden_size must be a positive integer")
        if self.routed_output_norm_eps is not None and (
            type(self.routed_output_norm_eps) not in (int, float)
            or not math.isfinite(self.routed_output_norm_eps)
            or self.routed_output_norm_eps <= 0
        ):
            raise ValueError("routed_output_norm_eps must be positive and finite")


@dataclass(frozen=True)
class MoeActivation:
    """Expert GLU activation, including SiTU's separate gate and up caps."""

    name: str
    gate_beta: float | None = None
    up_beta: float | None = None

    def __post_init__(self) -> None:
        if self.name not in {"silu", "situ"}:
            raise ValueError(f"unsupported MoE activation {self.name!r}")
        if self.name == "situ":
            for field in ("gate_beta", "up_beta"):
                value = getattr(self, field)
                if (
                    type(value) not in (int, float)
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise ValueError(f"{field} must be positive and finite for SiTU")
        elif self.gate_beta is not None or self.up_beta is not None:
            raise ValueError("SiTU beta values are invalid for SiLU")


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
    FP8 block and MXFP4 group sizes are distinct weight granularities; shared
    weights may use the ordinary tensor dtype instead of routed quantization.
    """

    tensor_dtype: str
    weight_quant_method: str
    weight_format: str
    activation_scheme: str
    weight_block_size: tuple[int, int] | None = None
    weight_group_size: int | None = None
    shared_weight_dtype: str | None = None

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

        if (self.weight_block_size is None) == (self.weight_group_size is None):
            raise ValueError("exactly one weight granularity must be specified")
        if self.weight_block_size is not None and (
            not isinstance(self.weight_block_size, tuple)
            or len(self.weight_block_size) != 2
            or any(
                type(size) is not int or size <= 0 for size in self.weight_block_size
            )
        ):
            raise ValueError("weight_block_size must contain two positive integers")
        if self.weight_group_size is not None and (
            type(self.weight_group_size) is not int or self.weight_group_size <= 0
        ):
            raise ValueError("weight_group_size must be a positive integer")
        if self.shared_weight_dtype is not None and (
            not isinstance(self.shared_weight_dtype, str)
            or not self.shared_weight_dtype.strip()
        ):
            raise ValueError("shared_weight_dtype must be a non-empty string")


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
