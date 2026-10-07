"""Trace artifact contracts and normalized Chrome-trace summaries."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TraceArtifactTarget:
    """Filesystem destination and portable result reference for one raw trace."""

    path: Path
    reference: str

    def __post_init__(self) -> None:
        if not self.reference:
            raise ValueError("trace artifact reference must not be empty")
        reference = Path(self.reference)
        if reference.is_absolute() or ".." in reference.parts:
            raise ValueError("trace artifact reference must be a safe relative path")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as artifact_file:
        for chunk in iter(lambda: artifact_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_trace(path: Path) -> Mapping[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as trace_file:
            payload = json.load(trace_file)
    else:
        with path.open(encoding="utf-8") as trace_file:
            payload = json.load(trace_file)
    if not isinstance(payload, Mapping):
        raise TypeError("trace artifact must contain a JSON object")
    return payload


def _duration_events(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_events = payload.get("traceEvents")
    if not isinstance(raw_events, list):
        raise TypeError("trace artifact is missing a traceEvents list")
    return [
        event
        for event in raw_events
        if isinstance(event, Mapping)
        and event.get("ph") == "X"
        and isinstance(event.get("dur"), int | float)
        and isinstance(event.get("ts"), int | float)
    ]


def _category(event: Mapping[str, Any]) -> str:
    return str(event.get("cat", "")).casefold()


def _argument(event: Mapping[str, Any], *names: str) -> Any:
    arguments = event.get("args")
    if not isinstance(arguments, Mapping):
        return None
    normalized = {
        str(key).casefold().replace("_", "").replace(" ", ""): value
        for key, value in arguments.items()
    }
    for name in names:
        value = normalized.get(name.casefold().replace("_", "").replace(" ", ""))
        if value is not None:
            return value
    return None


def _normalize_event(
    event: Mapping[str, Any], *, sequence: int, origin_us: float
) -> dict[str, Any]:
    normalized: dict[str, Any] = {
        "sequence": sequence,
        "name": str(event.get("name", "")),
        "start_offset_us": float(event["ts"]) - origin_us,
        "duration_us": float(event["dur"]),
    }
    for output_name, aliases in {
        "device": ("device", "device_id"),
        "stream": ("stream", "stream_id"),
        "correlation_id": ("correlation", "correlation_id", "external_id"),
    }.items():
        value = _argument(event, *aliases)
        if value is not None:
            normalized[output_name] = value
    return normalized


def _is_communication_kernel(name: str) -> bool:
    lowered = name.casefold()
    return any(token in lowered for token in ("nccl", "rccl", "deepep"))


def summarize_chrome_trace(
    target: TraceArtifactTarget, *, boundary_name: str
) -> dict[str, Any]:
    """Normalize one PyTorch Chrome trace without treating it as timing data.

    Kernel duration sums are useful accounting values but are not wall latency:
    kernels on different streams may overlap. The raw trace remains the source of
    truth for timeline analysis and for information not represented here.
    """

    events = _duration_events(_read_trace(target.path))
    boundary_events = [event for event in events if event.get("name") == boundary_name]
    if not boundary_events:
        raise ValueError(f"trace does not contain boundary range {boundary_name!r}")
    boundary = min(boundary_events, key=lambda event: float(event["ts"]))
    origin_us = float(boundary["ts"])

    kernel_events = [event for event in events if "kernel" in _category(event)]
    if not kernel_events:
        raise ValueError("trace does not contain any CUDA kernel events")
    kernel_events.sort(key=lambda event: float(event["ts"]))
    kernels = [
        _normalize_event(event, sequence=index, origin_us=origin_us)
        for index, event in enumerate(kernel_events)
    ]

    # Kineto emits the host-side CUDA API submission and the resulting device
    # activity as separate events. Classify copies by their GPU activity category,
    # not by names such as ``cudaMemcpyAsync`` on ``cuda_runtime`` events; those
    # runtime calls remain available below and share a correlation ID with the
    # physical transfer they initiated.
    copy_events = [event for event in events if "memcpy" in _category(event)]
    copy_events.sort(key=lambda event: float(event["ts"]))
    memory_copies = [
        _normalize_event(event, sequence=index, origin_us=origin_us)
        for index, event in enumerate(copy_events)
    ]

    runtime_events = [event for event in events if "cuda_runtime" in _category(event)]
    runtime_events.sort(key=lambda event: float(event["ts"]))
    runtime_calls = [
        _normalize_event(event, sequence=index, origin_us=origin_us)
        for index, event in enumerate(runtime_events)
    ]

    communication = [
        kernel for kernel in kernels if _is_communication_kernel(str(kernel["name"]))
    ]
    streams = list(
        dict.fromkeys(str(kernel["stream"]) for kernel in kernels if "stream" in kernel)
    )
    return {
        "artifact": {
            "path": target.reference,
            "format": "pytorch_chrome_trace",
            "compression": "gzip" if target.path.suffix == ".gz" else "none",
            "size_bytes": target.path.stat().st_size,
            "sha256": _sha256(target.path),
        },
        "authoritative_for_latency": False,
        "duration_accounting": (
            "summed event durations include overlap and are not wall latency"
        ),
        "profiled_iteration_count": 1,
        "profiled_boundary_cpu_duration_us": float(boundary["dur"]),
        "kernel_count": len(kernels),
        "summed_kernel_duration_us": sum(
            float(kernel["duration_us"]) for kernel in kernels
        ),
        "kernels": kernels,
        "streams": streams,
        "memory_copy_count": len(memory_copies),
        "summed_memory_copy_duration_us": sum(
            float(copy["duration_us"]) for copy in memory_copies
        ),
        "memory_copies": memory_copies,
        "cuda_runtime_call_count": len(runtime_calls),
        "summed_cuda_runtime_duration_us": sum(
            float(call["duration_us"]) for call in runtime_calls
        ),
        "cuda_runtime_calls": runtime_calls,
        "communication": {
            "detection": "kernel_name_heuristic",
            "kernel_count": len(communication),
            "kernels": communication,
        },
    }
