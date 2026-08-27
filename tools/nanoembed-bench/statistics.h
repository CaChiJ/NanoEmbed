// Pure descriptive statistics used by nanoembed-bench.
//
// Percentiles use the "lower" order statistic: after sorting N samples, the
// zero-based index is floor(q * (N - 1)).  This deliberately preserves the
// percentile semantics used by the M3/M3.5 harness while making them explicit
// and testable.

#pragma once

#include <cstddef>
#include <vector>

namespace nanoembed::bench {

struct DistributionStats {
    bool   available = false;
    size_t count     = 0;
    double min       = 0.0;
    double max       = 0.0;
    double mean      = 0.0;
    double p50       = 0.0;
    double p90       = 0.0;
    double p95       = 0.0;
    double p99       = 0.0;
    // Population standard deviation: the samples are the complete measured
    // window, not an estimator for independently repeated benchmark runs.
    double stddev    = 0.0;
    // Median absolute deviation from p50, using the same lower percentile.
    double mad       = 0.0;
};

// Returns unavailable for an empty input or for non-finite samples.  Rejecting
// the whole set avoids silently hiding a broken timer behind plausible stats.
DistributionStats describe_samples(const std::vector<double> & samples);

// Derive descriptive throughput observations from contiguous, non-overlapping
// windows of exactly window_size_items per-request latency samples.  A trailing
// partial window is intentionally omitted.  The benchmark's authoritative
// throughput remains total_items / measured wall time; these observations only
// describe within-run variation.
std::vector<double> fixed_item_window_throughputs(
        const std::vector<double> & latency_ms,
        size_t                      window_size_items);

} // namespace nanoembed::bench
