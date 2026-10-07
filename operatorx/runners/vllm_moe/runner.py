"""vLLM MoE module entry point and common measured execution lifecycle."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch

from operatorx.core import BackendImpl, Op, UnsupportedOpError
from operatorx.runners.vllm_moe.deepseek import prepare as prepare_deepseek
from operatorx.runners.vllm_moe.kimi_k3 import prepare as prepare_kimi_k3

# Model IDs select adapters by the vLLM module they construct. A family may
# reuse an adapter when its construction and weight processing are equivalent.
_ADAPTERS: Mapping[str, tuple[str, Callable[[Op], dict[str, Any]]]] = {
    "deepseek-ai/DeepSeek-R1-0528": ("deepseek_r1", prepare_deepseek),
    "moonshotai/Kimi-K3": ("kimi_k3", prepare_kimi_k3),
}


def prepare(op: Op) -> dict[str, Any]:
    """Select the native adapter for an explicitly supported model request."""

    model_id = op.args.get("model_id")
    if not isinstance(model_id, str):
        raise UnsupportedOpError("native vLLM MoE requires a model_id string")
    selected = _ADAPTERS.get(model_id)
    if selected is None:
        raise UnsupportedOpError(f"no native vLLM MoE adapter for model {model_id!r}")

    family, adapter = selected
    requested_family = op.args.get("model_family")
    if requested_family is not None and requested_family != family:
        raise UnsupportedOpError(
            f"model {model_id!r} requires model_family={family!r}; "
            f"got {requested_family!r}"
        )
    return adapter(op)


@torch.no_grad()
def kernel(context: dict[str, Any]) -> None:
    """Execute the selected native MoE module within vLLM's forward context."""

    from vllm.forward_context import set_forward_context

    hidden_states = context["hidden_states"]
    with set_forward_context(
        None,
        context["vllm_config"],
        num_tokens=hidden_states.shape[0],
    ):
        context["out"] = context["module"](hidden_states)


def finalize_warmup(context: dict[str, Any]) -> None:
    """Lock vLLM scratch allocation before timed module invocations."""

    from vllm.v1.worker.workspace import lock_workspace

    lock_workspace()
    context["metadata"]["resolved_execution"]["workspace"]["locked_after_warmup"] = True


def result_metadata(context: dict[str, Any]) -> Mapping[str, Any]:
    """Return metadata gathered outside the measured invocation."""

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
