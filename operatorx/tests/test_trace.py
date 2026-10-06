"""Behavioral tests for normalized profiler trace summaries."""

import gzip
import hashlib
import json

import pytest

from operatorx.core import TraceArtifactTarget
from operatorx.core.trace import summarize_chrome_trace


def test_trace_summary_preserves_order_and_distinguishes_event_types(tmp_path):
    artifact = tmp_path / "profile.json.gz"
    payload = {
        "traceEvents": [
            {
                "ph": "X",
                "cat": "user_annotation",
                "name": "operatorx::moe_forward",
                "ts": 100.0,
                "dur": 80.0,
            },
            {
                "ph": "X",
                "cat": "cuda_runtime",
                "name": "cudaLaunchKernel",
                "ts": 105.0,
                "dur": 2.0,
                "args": {"correlation": 17},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "fused_moe_kernel",
                "ts": 112.0,
                "dur": 15.0,
                "args": {"device": 0, "stream": 7, "correlation": 17},
            },
            {
                "ph": "X",
                "cat": "gpu_memcpy",
                "name": "Memcpy DtoD",
                "ts": 128.0,
                "dur": 3.0,
                "args": {"device": 0, "stream": 7},
            },
            {
                "ph": "X",
                "cat": "kernel",
                "name": "ncclDevKernel_AllReduce",
                "ts": 135.0,
                "dur": 11.0,
                "args": {"Device Id": 0, "stream_id": 9},
            },
        ]
    }
    with gzip.open(artifact, "wt", encoding="utf-8") as trace_file:
        json.dump(payload, trace_file)

    summary = summarize_chrome_trace(
        TraceArtifactTarget(artifact, "run.artifacts/profile.json.gz"),
        boundary_name="operatorx::moe_forward",
    )

    assert summary["artifact"] == {
        "path": "run.artifacts/profile.json.gz",
        "format": "pytorch_chrome_trace",
        "compression": "gzip",
        "size_bytes": artifact.stat().st_size,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    assert summary["authoritative_for_latency"] is False
    assert summary["duration_accounting"] == (
        "summed event durations include overlap and are not wall latency"
    )
    assert summary["profiled_boundary_cpu_duration_us"] == 80.0
    assert summary["kernel_count"] == 2
    assert summary["summed_kernel_duration_us"] == 26.0
    assert summary["streams"] == ["7", "9"]
    assert summary["kernels"] == [
        {
            "sequence": 0,
            "name": "fused_moe_kernel",
            "start_offset_us": 12.0,
            "duration_us": 15.0,
            "device": 0,
            "stream": 7,
            "correlation_id": 17,
        },
        {
            "sequence": 1,
            "name": "ncclDevKernel_AllReduce",
            "start_offset_us": 35.0,
            "duration_us": 11.0,
            "device": 0,
            "stream": 9,
        },
    ]
    assert summary["memory_copy_count"] == 1
    assert summary["cuda_runtime_call_count"] == 1
    assert summary["communication"] == {
        "detection": "kernel_name_heuristic",
        "kernel_count": 1,
        "kernels": [summary["kernels"][1]],
    }


@pytest.mark.parametrize(
    ("events", "message"),
    [
        ([], "boundary range"),
        (
            [
                {
                    "ph": "X",
                    "cat": "user_annotation",
                    "name": "operatorx::gemm",
                    "ts": 1,
                    "dur": 2,
                }
            ],
            "CUDA kernel",
        ),
    ],
)
def test_trace_summary_rejects_incomplete_capture(tmp_path, events, message):
    artifact = tmp_path / "profile.json"
    artifact.write_text(json.dumps({"traceEvents": events}))

    with pytest.raises(ValueError, match=message):
        summarize_chrome_trace(
            TraceArtifactTarget(artifact, "run.artifacts/profile.json"),
            boundary_name="operatorx::gemm",
        )


@pytest.mark.parametrize("reference", ["", "/absolute/trace.json", "../trace.json"])
def test_trace_artifact_reference_must_be_portable(reference, tmp_path):
    with pytest.raises(ValueError):
        TraceArtifactTarget(tmp_path / "trace.json", reference)
