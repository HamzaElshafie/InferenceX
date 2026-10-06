"""Exercise NVIDIA runner lifecycle hooks without requiring a GPU."""

import json
from contextlib import nullcontext

from operatorx.core import BackendImpl, Op, TraceArtifactTarget
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
        "measurement_pass": "timing",
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


def test_trace_profiles_one_finalized_boundary_and_emits_no_latency(
    tmp_path, monkeypatch
):
    calls = []

    def prepare(op):
        assert op.args == {"m": 2}
        return {"workspace_locked": False}

    def kernel(context):
        calls.append("finalized" if context["workspace_locked"] else "warmup")

    def finalize_warmup(context):
        context["workspace_locked"] = True

    impl = BackendImpl(
        op_type="gemm",
        prepare=prepare,
        kernel=kernel,
        finalize_warmup=finalize_warmup,
        result_metadata=lambda context: {
            "workspace": {"locked_after_warmup": context["workspace_locked"]}
        },
    )
    monkeypatch.setattr(runner, "_DISPATCH", {("gemm", "fixture"): impl})
    l2_flushes = []
    synchronizations = []
    monkeypatch.setattr(runner, "_clear_l2", lambda: l2_flushes.append(None))
    monkeypatch.setattr(
        runner.torch.cuda, "synchronize", lambda: synchronizations.append(None)
    )

    class FakeProfile:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def export_chrome_trace(self, path):
            payload = {
                "traceEvents": [
                    {
                        "ph": "X",
                        "cat": "user_annotation",
                        "name": "operatorx::gemm",
                        "ts": 10,
                        "dur": 20,
                    },
                    {
                        "ph": "X",
                        "cat": "kernel",
                        "name": "fixture_kernel",
                        "ts": 12,
                        "dur": 5,
                        "args": {"stream": 1},
                    },
                ]
            }
            import gzip

            with gzip.open(path, "wt", encoding="utf-8") as trace_file:
                json.dump(payload, trace_file)

    monkeypatch.setattr(
        runner.torch.profiler, "profile", lambda **kwargs: FakeProfile()
    )
    monkeypatch.setattr(
        runner.torch.profiler, "record_function", lambda name: nullcontext()
    )
    artifact = TraceArtifactTarget(
        tmp_path / "profile.json.gz", "run.artifacts/profile.json.gz"
    )

    result = runner.trace(Op(type="gemm", args={"m": 2}, backend="fixture"), artifact)

    assert calls == ["warmup"] * 10 + ["finalized"] * 2
    assert len(l2_flushes) == 12
    assert len(synchronizations) == 4
    assert result.metrics == {}
    assert result.metadata["workspace"] == {"locked_after_warmup": True}
    assert result.metadata["measurement_pass"] == "trace"
    assert result.metadata["trace_capture"] == {
        "tool": "torch.profiler",
        "activities": ["cpu", "cuda"],
        "record_shapes": False,
        "profile_memory": False,
        "with_stack": False,
        "with_flops": False,
        "warmup_count": 10,
        "cache_policy": "cold_l2_before_profiled_boundary",
    }
    assert result.metadata["trace"]["artifact"]["path"] == (
        "run.artifacts/profile.json.gz"
    )
    assert result.metadata["trace"]["kernels"][0]["name"] == "fixture_kernel"
