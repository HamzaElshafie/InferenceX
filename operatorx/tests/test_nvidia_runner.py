"""Exercise NVIDIA runner lifecycle hooks without requiring a GPU."""

from operatorx.core import BackendImpl, Op
from operatorx.runners.nvidia import runner


class _FakeEvent:
    """Minimal CUDA event substitute with a deterministic elapsed time."""

    def __init__(self, *, enable_timing):
        assert enable_timing is True

    def record(self):
        return None

    def elapsed_time(self, other):
        assert isinstance(other, _FakeEvent)
        return 1.25


def test_runner_locks_after_warmup_and_emits_backend_metadata(monkeypatch):
    calls = []

    def prepare(op):
        assert op.args == {"m": 2}
        return {"workspace_locked": False}

    def kernel(context):
        phase = "measured" if context["workspace_locked"] else "warmup"
        calls.append(phase)

    def finalize_warmup(context):
        context["workspace_locked"] = True

    def result_metadata(context):
        return {"workspace": {"locked_after_warmup": context["workspace_locked"]}}

    impl = BackendImpl(
        op_type="gemm",
        prepare=prepare,
        kernel=kernel,
        finalize_warmup=finalize_warmup,
        result_metadata=result_metadata,
    )
    monkeypatch.setattr(runner, "_DISPATCH", {("gemm", "fixture"): impl})
    monkeypatch.setattr(runner, "_clear_l2", lambda: None)
    monkeypatch.setattr(runner.torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(runner.torch.cuda, "synchronize", lambda: None)

    op = Op(type="gemm", args={"m": 2}, backend="fixture")
    result = runner.run(op)

    assert calls == ["warmup"] * 5 + ["measured"] * 10
    assert result.metrics == {"latency_us": 1250.0}
    assert result.metadata == {"workspace": {"locked_after_warmup": True}}
