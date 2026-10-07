"""Behavioral tests for the vLLM MoE module adapter selection."""

import pytest

from operatorx.core import Op, UnsupportedOpError
from operatorx.runners.vllm_moe.runner import prepare


@pytest.mark.parametrize(
    ("model_id", "family", "message"),
    [
        ("deepseek-ai/DeepSeek-R1-0528", None, "explicit model semantics"),
        ("moonshotai/Kimi-K3", "kimi_k3", "explicit model semantics"),
        ("unknown/model", None, "no native vLLM MoE adapter"),
        ("deepseek-ai/DeepSeek-R1-0528", "kimi_k3", "requires model_family"),
    ],
)
def test_prepare_selects_only_matching_native_adapters(
    model_id: str, family: str | None, message: str
) -> None:
    args = {"model_id": model_id}
    if family is not None:
        args["model_family"] = family

    with pytest.raises(UnsupportedOpError, match=message):
        prepare(Op(type="moe_forward", args=args, backend="vllm"))


def test_prepare_rejects_invalid_model_identity() -> None:
    with pytest.raises(UnsupportedOpError, match="model_id string"):
        prepare(Op(type="moe_forward", args={"model_id": ["Kimi-K3"]}, backend="vllm"))
