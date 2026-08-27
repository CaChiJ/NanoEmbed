#include "statistics.h"

#include <algorithm>
#include <cmath>

namespace nanoembed::bench {

namespace {

double lower_percentile(const std::vector<double> & sorted, double q) {
    const size_t index = static_cast<size_t>(
        q * static_cast<double>(sorted.size() - 1));
    return sorted[index];
}

} // namespace

DistributionStats describe_samples(const std::vector<double> & samples) {
    DistributionStats out;
    if (samples.empty()) return out;
    for (double sample : samples) {
        if (!std::isfinite(sample)) return out;
    }

    std::vector<double> sorted = samples;
    std::sort(sorted.begin(), sorted.end());

    long double sum = 0.0;
    for (double sample : samples) sum += static_cast<long double>(sample);
    const double mean = static_cast<double>(sum / samples.size());

    long double squared_deviation_sum = 0.0;
    for (double sample : samples) {
        const long double d = static_cast<long double>(sample) - mean;
        squared_deviation_sum += d * d;
    }

    const double median = lower_percentile(sorted, 0.50);
    std::vector<double> absolute_deviations;
    absolute_deviations.reserve(samples.size());
    for (double sample : samples) {
        absolute_deviations.push_back(std::fabs(sample - median));
    }
    std::sort(absolute_deviations.begin(), absolute_deviations.end());

    out.available = true;
    out.count     = samples.size();
    out.min       = sorted.front();
    out.max       = sorted.back();
    out.mean      = mean;
    out.p50       = median;
    out.p90       = lower_percentile(sorted, 0.90);
    out.p95       = lower_percentile(sorted, 0.95);
    out.p99       = lower_percentile(sorted, 0.99);
    out.stddev    = std::sqrt(static_cast<double>(
        squared_deviation_sum / samples.size()));
    out.mad       = lower_percentile(absolute_deviations, 0.50);
    return out;
}

std::vector<double> fixed_item_window_throughputs(
        const std::vector<double> & latency_ms,
        size_t                      window_size_items) {
    std::vector<double> out;
    if (window_size_items == 0) return out;

    const size_t complete_windows = latency_ms.size() / window_size_items;
    out.reserve(complete_windows);
    for (size_t window = 0; window < complete_windows; ++window) {
        long double duration_ms = 0.0;
        for (size_t item = 0; item < window_size_items; ++item) {
            const double sample = latency_ms[window * window_size_items + item];
            if (!std::isfinite(sample) || sample < 0.0) return {};
            duration_ms += static_cast<long double>(sample);
        }
        if (duration_ms <= 0.0) return {};
        out.push_back(static_cast<double>(
            1000.0L * window_size_items / duration_ms));
    }
    return out;
}

} // namespace nanoembed::bench
