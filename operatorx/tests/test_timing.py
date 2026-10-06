import math

import pytest

from operatorx.core.timing import summarize_latencies


def test_summarize_latencies_preserves_samples_and_flags_high_variance():
    statistics = summarize_latencies([10.0, 20.0, 30.0, 40.0, 50.0])

    assert statistics.metrics() == pytest.approx(
        {
            "latency_us": 30.0,
            "latency_min_us": 10.0,
            "latency_median_us": 30.0,
            "latency_mean_us": 30.0,
            "latency_p90_us": 46.0,
            "latency_max_us": 50.0,
            "latency_stddev_us": math.sqrt(250.0),
            "latency_cv_percent": math.sqrt(250.0) / 30.0 * 100.0,
        }
    )
    assert statistics.metadata(
        warmup_count=10,
        warmup_policy="test_warmup_policy",
        measurement_stabilization_count=1,
        measurement_stabilization_policy="test_stabilization_policy",
        timer="test_timer",
        cache_policy="test_cache_policy",
    ) == {
        "samples_us": [10.0, 20.0, 30.0, 40.0, 50.0],
        "warmup_count": 10,
        "warmup_policy": "test_warmup_policy",
        "measurement_stabilization_count": 1,
        "measurement_stabilization_policy": "test_stabilization_policy",
        "sample_count": 5,
        "timer": "test_timer",
        "cache_policy": "test_cache_policy",
        "standard_deviation_method": "sample",
        "stability": {
            "status": "high_variance",
            "observed_cv_percent": math.sqrt(250.0) / 30.0 * 100.0,
            "threshold_cv_percent": 5.0,
            "note": (
                "Observed latency CV of 52.70% exceeded the configured 5.00% stability "
                "threshold across 5 measured iterations. The result was retained, but "
                "performance comparisons should be interpreted cautiously. Possible "
                "contributors include GPU clock or thermal variation, contention from "
                "concurrent workloads, unexpected allocator or compilation activity, "
                "and synchronization jitter."
            ),
        },
    }


@pytest.mark.parametrize(
    "samples",
    [[], [1.0], [-1.0, 1.0], [1.0, math.nan], [0.0, 0.0]],
)
def test_summarize_latencies_rejects_invalid_samples(samples):
    with pytest.raises(ValueError):
        summarize_latencies(samples)
