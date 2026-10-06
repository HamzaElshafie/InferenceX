"""Exercise framework-neutral MoE workload validation."""

import pytest
from operatorx.core.moe import MoeWorkload


def test_workload_keeps_global_and_local_token_identity_separate() -> None:
    workload = MoeWorkload(
        phase="decode",
        global_num_tokens=64,
        local_num_tokens=8,
        routing_input_policy="model_native",
        input_seed=17,
        source="controlled_test",
    )

    assert workload.phase == "decode"
    assert workload.global_num_tokens == 64
    assert workload.local_num_tokens == 8


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"phase": "training"}, "phase"),
        ({"global_num_tokens": 0}, "global_num_tokens"),
        ({"local_num_tokens": 65}, "cannot exceed"),
        ({"input_seed": -1}, "input_seed"),
        ({"routing_input_policy": ""}, "routing_input_policy"),
        ({"source": ""}, "source"),
    ],
)
def test_workload_rejects_ambiguous_or_impossible_inputs(
    overrides: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "phase": "prefill",
        "global_num_tokens": 64,
        "local_num_tokens": 64,
        "routing_input_policy": "model_native",
        "input_seed": 17,
        "source": "controlled_test",
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        MoeWorkload(**values)
