#include "statistics.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

int g_failures = 0;

#define EXPECT_TRUE(cond)                                                          \
    do {                                                                           \
        if (!(cond)) {                                                             \
            std::fprintf(stderr, "FAIL: %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
            ++g_failures;                                                          \
        }                                                                          \
    } while (0)

#define EXPECT_NEAR(a, b, tolerance)                                               \
    do {                                                                           \
        const double actual_  = static_cast<double>(a);                            \
        const double expected_ = static_cast<double>(b);                           \
        if (std::fabs(actual_ - expected_) > (tolerance)) {                        \
            std::fprintf(stderr,                                                   \
                "FAIL: %s ~= %s (got %.12g vs %.12g) (%s:%d)\n",                 \
                #a, #b, actual_, expected_, __FILE__, __LINE__);                   \
            ++g_failures;                                                          \
        }                                                                          \
    } while (0)

void test_empty_and_invalid_are_unavailable() {
    EXPECT_TRUE(!nanoembed::bench::describe_samples({}).available);
    EXPECT_TRUE(!nanoembed::bench::describe_samples(
        {1.0, std::numeric_limits<double>::quiet_NaN()}).available);
}

void test_descriptive_statistics_and_lower_percentiles() {
    // Unsorted input also guards that caller-owned samples are not mutated.
    const std::vector<double> samples = {10, 1, 9, 2, 8, 3, 7, 4, 6, 5};
    const auto stats = nanoembed::bench::describe_samples(samples);

    EXPECT_TRUE(stats.available);
    EXPECT_TRUE(stats.count == 10);
    EXPECT_NEAR(stats.min, 1.0, 1e-12);
    EXPECT_NEAR(stats.max, 10.0, 1e-12);
    EXPECT_NEAR(stats.mean, 5.5, 1e-12);
    // floor(q * (N - 1)): indices 4, 8, 8, and 8 respectively.
    EXPECT_NEAR(stats.p50, 5.0, 1e-12);
    EXPECT_NEAR(stats.p90, 9.0, 1e-12);
    EXPECT_NEAR(stats.p95, 9.0, 1e-12);
    EXPECT_NEAR(stats.p99, 9.0, 1e-12);
    EXPECT_NEAR(stats.stddev, std::sqrt(8.25), 1e-12);
    EXPECT_NEAR(stats.mad, 2.0, 1e-12);
    EXPECT_NEAR(samples.front(), 10.0, 1e-12);
}

void test_single_sample_has_zero_spread() {
    const auto stats = nanoembed::bench::describe_samples({42.0});
    EXPECT_TRUE(stats.available);
    EXPECT_TRUE(stats.count == 1);
    EXPECT_NEAR(stats.mean, 42.0, 1e-12);
    EXPECT_NEAR(stats.p99, 42.0, 1e-12);
    EXPECT_NEAR(stats.stddev, 0.0, 1e-12);
    EXPECT_NEAR(stats.mad, 0.0, 1e-12);
}

void test_fixed_item_windows_are_contiguous_and_drop_tail() {
    const std::vector<double> latency_ms = {
        100, 100, 100, 100,
        250, 250, 250, 250,
        999, // partial tail is not a four-item window
    };
    const auto rates = nanoembed::bench::fixed_item_window_throughputs(
        latency_ms, 4);
    EXPECT_TRUE(rates.size() == 2);
    EXPECT_NEAR(rates[0], 10.0, 1e-12);
    EXPECT_NEAR(rates[1], 4.0, 1e-12);

    EXPECT_TRUE(nanoembed::bench::fixed_item_window_throughputs(
        latency_ms, 0).empty());
    EXPECT_TRUE(nanoembed::bench::fixed_item_window_throughputs(
        {0.0, 0.0}, 2).empty());
}

} // namespace

int main() {
    test_empty_and_invalid_are_unavailable();
    test_descriptive_statistics_and_lower_percentiles();
    test_single_sample_has_zero_spread();
    test_fixed_item_windows_are_contiguous_and_drop_tail();

    if (g_failures != 0) {
        std::fprintf(stderr, "%d bench statistics test(s) failed\n", g_failures);
        return 1;
    }
    std::printf("bench statistics tests: OK\n");
    return 0;
}
