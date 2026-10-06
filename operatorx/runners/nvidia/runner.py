from __future__ import annotations

from importlib import import_module

import torch

from operatorx.core import (
    BackendImpl,
    Op,
    Result,
    TraceArtifactTarget,
    UnsupportedOpError,
)
from operatorx.core.timing import summarize_latencies
from operatorx.core.trace import summarize_chrome_trace

_BACKENDS = [
    "torch",
    "deepgemm",
    "flashinfer",
    "deepep",
    "sglang",
    "flashinfer_comm",
    "sglang_comm",
    "vllm",
]
_DISPATCH: dict[tuple[str, str], BackendImpl] = {}
_L2_BUF: dict[int, torch.Tensor] = {}


def _load() -> None:
    if _DISPATCH:
        return
    for name in _BACKENDS:
        try:
            mod = import_module(f"operatorx.runners.nvidia.backends.{name}")
        except ImportError:
            continue
        for impl in getattr(mod, "IMPLS", []):
            _DISPATCH[(impl.op_type, name)] = impl


def _clear_l2() -> None:
    """Flush L2 by writing zeros to a buffer sized to the device's L2 cache."""
    dev = torch.cuda.current_device()
    buf = _L2_BUF.get(dev)
    if buf is None:
        l2 = torch.cuda.get_device_properties(dev).L2_cache_size
        buf = torch.empty(l2, dtype=torch.int8, device=dev)
        _L2_BUF[dev] = buf
    buf.zero_()


_WARMUP = 10
_ITERS = 100
_NUM_BUFFER_SETS = 1


def _stabilize_measurement_state(impl: BackendImpl, contexts: list[object]) -> None:
    """Establish active steady state after the required warmup synchronization.

    A synchronization boundary can leave a floating-clock GPU briefly idle. The
    first subsequent invocation may then run at a lower clock than later queued
    work, even though synchronization itself is outside the CUDA-event interval.
    Enqueueing one untimed, cold-L2 invocation per buffer set without another
    synchronization makes sample zero represent the requested warm-runtime state.
    This policy does not model post-idle latency, which requires a separate case.

    References:
      NVIDIA TensorRT, GPU Clock Locking and Floating Clock:
      https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/benchmarking.html#gpu-clock-locking-and-floating-clock
      NVIDIA Nsight Compute, Clock Control:
      https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#clock-control
    """

    for context in contexts:
        _clear_l2()
        impl.kernel(context)


def run(op: Op) -> Result:
    _load()
    impl = _DISPATCH.get((op.type, op.backend))
    if impl is None:
        raise UnsupportedOpError(
            f"nvidia/{op.backend} has no impl for op_type={op.type!r}"
        )

    ctxs = [impl.prepare(op) for _ in range(_NUM_BUFFER_SETS)]

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(_WARMUP + _ITERS)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(_WARMUP + _ITERS)]

    for i in range(_WARMUP):
        _clear_l2()
        starts[i].record()
        impl.kernel(ctxs[i % _NUM_BUFFER_SETS])
        ends[i].record()
    torch.cuda.synchronize()
    if impl.finalize_warmup is not None:
        for context in ctxs:
            impl.finalize_warmup(context)

    _stabilize_measurement_state(impl, ctxs)

    for i in range(_ITERS):
        event_index = _WARMUP + i
        _clear_l2()
        starts[event_index].record()
        impl.kernel(ctxs[i % _NUM_BUFFER_SETS])
        ends[event_index].record()
    torch.cuda.synchronize()

    times_us = [
        starts[_WARMUP + i].elapsed_time(ends[_WARMUP + i]) * 1000.0
        for i in range(_ITERS)
    ]
    timing = summarize_latencies(times_us)
    metadata = (
        dict(impl.result_metadata(ctxs[0])) if impl.result_metadata is not None else {}
    )
    metadata["measurement_pass"] = "timing"
    metadata["timing"] = timing.metadata(
        warmup_count=_WARMUP,
        warmup_policy="same_cache_and_timer_path_as_measurement",
        measurement_stabilization_count=len(ctxs),
        measurement_stabilization_policy=(
            "one_untimed_cold_l2_invocation_per_buffer_set_after_finalization"
        ),
        timer="torch.cuda.Event",
        cache_policy="cold_l2_before_each_measured_invocation",
    )
    return Result(op=op, metrics=timing.metrics(), metadata=metadata)


def trace(op: Op, artifact: TraceArtifactTarget) -> Result:
    """Capture one warmed boundary invocation in a non-authoritative GPU trace.

    Preparation, warmup, workspace finalization, and L2 flushing happen before
    profiler capture. Only the requested operation boundary is traced. Profiling
    changes execution, so this pass intentionally emits no latency metric; clean
    latency remains the responsibility of :func:`run`.
    """

    _load()
    impl = _DISPATCH.get((op.type, op.backend))
    if impl is None:
        raise UnsupportedOpError(
            f"nvidia/{op.backend} has no impl for op_type={op.type!r}"
        )

    context = impl.prepare(op)
    for _ in range(_WARMUP):
        _clear_l2()
        impl.kernel(context)
    torch.cuda.synchronize()
    if impl.finalize_warmup is not None:
        impl.finalize_warmup(context)

    _stabilize_measurement_state(impl, [context])
    torch.cuda.synchronize()
    _clear_l2()
    torch.cuda.synchronize()

    artifact.path.parent.mkdir(parents=True, exist_ok=True)
    boundary_name = f"operatorx::{op.type}"
    with (
        torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
        ) as profile,
        torch.profiler.record_function(boundary_name),
    ):
        impl.kernel(context)
        torch.cuda.synchronize()
    profile.export_chrome_trace(str(artifact.path))

    metadata = (
        dict(impl.result_metadata(context)) if impl.result_metadata is not None else {}
    )
    metadata["measurement_pass"] = "trace"
    metadata["trace_capture"] = {
        "tool": "torch.profiler",
        "activities": ["cpu", "cuda"],
        "record_shapes": False,
        "profile_memory": False,
        "with_stack": False,
        "with_flops": False,
        "warmup_count": _WARMUP,
        "cache_policy": "cold_l2_before_profiled_boundary",
    }
    metadata["trace"] = summarize_chrome_trace(artifact, boundary_name=boundary_name)
    return Result(op=op, metadata=metadata)
