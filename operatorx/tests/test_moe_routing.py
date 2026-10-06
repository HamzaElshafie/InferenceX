"""Validation of model routing rules before backend selection."""

import pytest

from operatorx.core.moe import MoeRouting


@pytest.mark.parametrize(
    ("group_count", "selected_group_count", "message"),
    [
        (8, None, "must be set together"),
        (8, 9, "cannot exceed group_count"),
    ],
)
def test_rejects_invalid_expert_group_selection(
    group_count: int, selected_group_count: int | None, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        MoeRouting(
            score_function="sigmoid",
            selection_method="noaux_tc",
            normalize_selected_weights=True,
            routed_output_scale=2.5,
            group_count=group_count,
            selected_group_count=selected_group_count,
        )


def test_rejects_nonfinite_routed_scale() -> None:
    with pytest.raises(ValueError, match="positive finite number"):
        MoeRouting(
            score_function="sigmoid",
            selection_method="noaux_tc",
            normalize_selected_weights=True,
            routed_output_scale=float("nan"),
            group_count=8,
            selected_group_count=4,
        )
