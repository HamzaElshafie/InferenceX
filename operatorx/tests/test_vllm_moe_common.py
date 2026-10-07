"""Behavioral tests for shared native vLLM MoE validation and diagnostics."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from operatorx.runners.vllm_moe.common import (
    capture_routing_ids,
    routing_workload_metadata,
    validate_repeated_outputs,
)


def test_correctness_gate_reports_repeatable_finite_output() -> None:
    expected = torch.zeros((2, 3), dtype=torch.bfloat16)
    output = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.bfloat16)

    assert validate_repeated_outputs(output, output.clone(), expected) == {
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


def test_routing_metadata_reports_load_and_exact_grouped_gemm_padding() -> None:
    topk_ids = torch.tensor([[0, 1], [0, 2], [0, 2]], dtype=torch.int32)

    assert routing_workload_metadata(
        topk_ids,
        num_experts=4,
        expected_logical_assignments=6,
        padding_block_size=4,
    ) == {
        "collection_phase": "untimed_validation",
        "logical_assignment_count": 6,
        "active_expert_count": 3,
        "assignments_per_expert": [3, 1, 2, 0],
        "active_expert_load": {"min": 1, "mean": 2.0, "max": 3},
        "dropped_assignment_count": 0,
        "padding": {
            "block_size": 4,
            "post_padding_assignment_count": 12,
            "padding_assignment_count": 6,
            "amplification_ratio": 2.0,
            "derivation": ("sum_of_active_expert_assignments_rounded_up_to_block_size"),
        },
    }


def test_router_capture_records_logical_ids_and_unregisters_callback() -> None:
    class Router:
        capture_fn: Any = None

        def set_capture_fn(self, capture_fn: Any) -> None:
            self.capture_fn = capture_fn

        def route(self, topk_ids: torch.Tensor) -> None:
            assert self.capture_fn is not None
            self.capture_fn(topk_ids)

    router = Router()
    module = SimpleNamespace(experts=SimpleNamespace(router=router))
    topk_ids = torch.tensor([[3, 1], [2, 0]], dtype=torch.int32)

    with capture_routing_ids(module) as captured:
        router.route(topk_ids)
        topk_ids.zero_()

    assert len(captured) == 1
    assert torch.equal(captured[0], torch.tensor([[3, 1], [2, 0]], dtype=torch.int32))
    assert router.capture_fn is None


def test_monolithic_kernel_capture_records_replayed_logical_ids() -> None:
    class MonolithicExperts:
        capture_fn: Any = None

        @staticmethod
        def supports_routing_replay_capture() -> bool:
            return True

        def set_capture_fn(self, capture_fn: Any) -> None:
            self.capture_fn = capture_fn

        def replay(self, topk_ids: torch.Tensor) -> None:
            assert self.capture_fn is not None
            self.capture_fn(topk_ids)

    experts = MonolithicExperts()
    quant_method = SimpleNamespace(
        is_monolithic=True,
        moe_kernel=SimpleNamespace(
            impl=SimpleNamespace(fused_experts=experts),
        ),
    )
    module = SimpleNamespace(
        experts=SimpleNamespace(
            _quant_method=quant_method,
            router=SimpleNamespace(set_capture_fn=None),
        )
    )
    topk_ids = torch.tensor([[7, 4], [5, 2]], dtype=torch.int32)

    with capture_routing_ids(module) as captured:
        experts.replay(topk_ids)
        topk_ids.zero_()

    assert len(captured) == 1
    assert torch.equal(captured[0], torch.tensor([[7, 4], [5, 2]], dtype=torch.int32))
    assert experts.capture_fn is None


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
        validate_repeated_outputs(first, second, expected)
