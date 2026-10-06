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
    l2_flushes = []
    monkeypatch.setattr(runner, "_clear_l2", lambda: l2_flushes.append(None))
    monkeypatch.setattr(runner.torch.cuda, "Event", _FakeEvent)
    monkeypatch.setattr(runner.torch.cuda, "synchronize", lambda: None)

    op = Op(type="gemm", args={"m": 2}, backend="fixture")
    result = runner.run(op)

    assert calls == ["warmup"] * 10 + ["measured"] * 101
    assert len(l2_flushes) == 111
    assert result.metrics == {
        "latency_us": 1250.0,
        "latency_min_us": 1250.0,
        "latency_median_us": 1250.0,
        "latency_mean_us": 1250.0,
        "latency_p90_us": 1250.0,
        "latency_max_us": 1250.0,
        "latency_stddev_us": 0.0,
        "latency_cv_percent": 0.0,
    }
    assert result.metadata == {
        "workspace": {"locked_after_warmup": True},
        "timing": {
            "samples_us": [1250.0] * 100,
            "warmup_count": 10,
            "warmup_policy": "same_cache_and_timer_path_as_measurement",
            "measurement_stabilization_count": 1,
            "measurement_stabilization_policy": (
                "one_untimed_cold_l2_invocation_per_buffer_set_after_finalization"
            ),
            "sample_count": 100,
            "timer": "torch.cuda.Event",
            "cache_policy": "cold_l2_before_each_measured_invocation",
            "standard_deviation_method": "sample",
            "stability": {
                "status": "normal",
                "observed_cv_percent": 0.0,
                "threshold_cv_percent": 5.0,
            },
        },
    }
