from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from dataclasses import dataclass

HIGH_VARIANCE_CV_PERCENT = 5.0


@dataclass(frozen=True)
class LatencyStatistics:
    """Descriptive statistics for latency samples recorded in execution order."""

    samples_us: tuple[float, ...]
    minimum_us: float
    median_us: float
    mean_us: float
    p90_us: float
    maximum_us: float
    stddev_us: float
    cv_percent: float

    @property
    def high_variance(self) -> bool:
        """Return whether relative latency dispersion exceeds the reporting limit."""

        return self.cv_percent > HIGH_VARIANCE_CV_PERCENT

    def metrics(self) -> dict[str, float]:
        """Return flat metrics with a compatibility alias for median latency."""

        return {
            # Existing result consumers interpret latency_us as the median.
            "latency_us": self.median_us,
            "latency_min_us": self.minimum_us,
            "latency_median_us": self.median_us,
            "latency_mean_us": self.mean_us,
            "latency_p90_us": self.p90_us,
            "latency_max_us": self.maximum_us,
            "latency_stddev_us": self.stddev_us,
            "latency_cv_percent": self.cv_percent,
        }

    def metadata(
        self,
        *,
        warmup_count: int,
        warmup_policy: str,
        measurement_stabilization_count: int,
        measurement_stabilization_policy: str,
        timer: str,
        cache_policy: str,
    ) -> dict:
        """Return the samples, measurement policy, and stability assessment."""

        stability = {
            "status": "high_variance" if self.high_variance else "normal",
            "observed_cv_percent": self.cv_percent,
            "threshold_cv_percent": HIGH_VARIANCE_CV_PERCENT,
        }
        if self.high_variance:
            stability["note"] = (
                f"Observed latency CV of {self.cv_percent:.2f}% exceeded the configured "
                f"{HIGH_VARIANCE_CV_PERCENT:.2f}% stability threshold across "
                f"{len(self.samples_us)} measured iterations. The result was retained, "
                "but performance comparisons should be interpreted cautiously. "
                "Possible contributors include GPU clock or thermal variation, "
                "contention from concurrent workloads, unexpected allocator or "
                "compilation activity, and synchronization jitter."
            )

        return {
            "samples_us": list(self.samples_us),
            "warmup_count": warmup_count,
            "warmup_policy": warmup_policy,
            "measurement_stabilization_count": measurement_stabilization_count,
            "measurement_stabilization_policy": measurement_stabilization_policy,
            "sample_count": len(self.samples_us),
            "timer": timer,
            "cache_policy": cache_policy,
            "standard_deviation_method": "sample",
            "stability": stability,
        }


def summarize_latencies(samples_us: Iterable[float]) -> LatencyStatistics:
    """Calculate latency statistics without changing the samples' execution order."""

    samples = tuple(float(sample) for sample in samples_us)
    if len(samples) < 2:
        raise ValueError("at least two latency samples are required")
    if any(not math.isfinite(sample) or sample < 0.0 for sample in samples):
        raise ValueError("latency samples must be finite and non-negative")

    mean_us = statistics.fmean(samples)
    if mean_us == 0.0:
        raise ValueError("mean latency must be positive")
    # These iterations sample the benchmark's broader runtime behavior rather
    # than exhaust every future execution, so use Bessel-corrected sample SD.
    stddev_us = statistics.stdev(samples, xbar=mean_us)
    cv_percent = stddev_us / mean_us * 100.0

    return LatencyStatistics(
        samples_us=samples,
        minimum_us=min(samples),
        median_us=statistics.median(samples),
        mean_us=mean_us,
        p90_us=statistics.quantiles(samples, n=10, method="inclusive")[8],
        maximum_us=max(samples),
        stddev_us=stddev_us,
        cv_percent=cv_percent,
    )
