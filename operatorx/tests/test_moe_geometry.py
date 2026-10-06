"""Behavioral checks for framework-neutral MoE layer dimensions."""

import pytest

from operatorx.core.moe import MoeLayerGeometry


def test_routed_only_layer_has_no_shared_experts() -> None:
    geometry = MoeLayerGeometry(
        hidden_size=4096,
        routed_expert_count=64,
        experts_per_token=4,
        routed_expert_intermediate_size=1536,
    )

    assert geometry.shared_expert_count == 0
    assert geometry.shared_expert_intermediate_size == 0


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("hidden_size", 0),
        ("routed_expert_count", -1),
        ("experts_per_token", 0),
        ("routed_expert_intermediate_size", 1.5),
        ("shared_expert_count", -1),
        ("shared_expert_intermediate_size", True),
    ],
)
def test_rejects_invalid_dimensions(field: str, invalid: object) -> None:
    dimensions = {
        "hidden_size": 7168,
        "routed_expert_count": 256,
        "experts_per_token": 8,
        "routed_expert_intermediate_size": 2048,
    }
    dimensions[field] = invalid

    with pytest.raises(ValueError, match=field):
        MoeLayerGeometry(**dimensions)


def test_rejects_more_selected_experts_than_available() -> None:
    with pytest.raises(ValueError, match="experts_per_token cannot exceed"):
        MoeLayerGeometry(
            hidden_size=4096,
            routed_expert_count=4,
            experts_per_token=5,
            routed_expert_intermediate_size=1024,
        )


@pytest.mark.parametrize(
    ("count", "width"),
    [(1, 0), (0, 2048)],
)
def test_rejects_incomplete_shared_expert_geometry(count: int, width: int) -> None:
    with pytest.raises(
        ValueError, match="must either both be zero or both be positive"
    ):
        MoeLayerGeometry(
            hidden_size=7168,
            routed_expert_count=256,
            experts_per_token=8,
            routed_expert_intermediate_size=2048,
            shared_expert_count=count,
            shared_expert_intermediate_size=width,
        )
