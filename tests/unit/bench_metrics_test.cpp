#include "metrics.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace nanoembed::bench;

namespace {

int failures = 0;

void check(bool ok, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

bool near(double actual, double expected) {
    return std::fabs(actual - expected) < 0.001;
}

} // namespace

int main() {
    const char * rollup =
        "00400000-00452000 r--p 00000000 00:00 0 [rollup]\n"
        "Rss:                 100 kB\n"
        "Pss:                  80 kB\n"
        "Pss_Anon:             50 kB\n"
        "Pss_File:             30 kB\n"
        "Shared_Clean:         11 kB\n"
        "Shared_Dirty:         12 kB\n"
        "Private_Clean:        13 kB\n"
        "Private_Dirty:        14 kB\n"
        "Anonymous:            55 kB\n";

    const MemSample parsed = parse_smaps_rollup(rollup);
    check(parsed.valid, "rollup with RSS is valid");
    check(parsed.rss_bytes == 100u * 1024u, "RSS parsed");
    check(parsed.pss_bytes == 80u * 1024u, "PSS parsed");
    check(parsed.uss_bytes == 27u * 1024u, "USS is private clean + dirty");
    check(parsed.has_pss_anon && parsed.pss_anon_bytes == 50u * 1024u,
          "Pss_Anon availability and value parsed");
    check(parsed.has_pss_file && parsed.pss_file_bytes == 30u * 1024u,
          "Pss_File availability and value parsed");
    check(parsed.has_anonymous && parsed.anonymous_bytes == 55u * 1024u,
          "Anonymous availability and value parsed");
    check(parsed.has_private_clean && parsed.has_private_dirty,
          "private breakdown availability parsed");
    check(parsed.has_shared_clean && parsed.has_shared_dirty,
          "shared breakdown availability parsed");

    const MemSample old_kernel = parse_smaps_rollup(
        "Rss: 10 kB\nPss: 9 kB\nPrivate_Clean: 0 kB\nPrivate_Dirty: 8 kB\n");
    check(old_kernel.valid, "minimal rollup is valid");
    check(!old_kernel.has_pss_anon && !old_kernel.has_pss_file,
          "absent optional PSS breakdown stays unavailable");
    check(old_kernel.has_private_clean && old_kernel.private_clean_bytes == 0,
          "present zero is distinct from an absent field");

    std::vector<MemSample> samples;
    for (size_t multiplier = 1; multiplier <= 5; ++multiplier) {
        MemSample sample = parsed;
        sample.rss_bytes = multiplier * 100u * 1024u;
        sample.pss_bytes = multiplier * 80u * 1024u;
        sample.uss_bytes = multiplier * 27u * 1024u;
        samples.push_back(sample);
    }
    const MemSampleSummary summary = summarize_mem_samples(samples);
    check(summary.total_samples == 5, "all supplied samples counted");
    check(summary.rss.valid_samples == 5, "RSS valid sample count");
    check(near(summary.rss.average_bytes, 300.0 * 1024.0), "RSS average");
    check(summary.rss.peak_bytes == 500u * 1024u,
          "last/final sample participates in the peak");
    check(summary.rss.p50_bytes == 300u * 1024u, "RSS lower p50");
    check(summary.rss.p75_bytes == 400u * 1024u, "RSS lower p75");
    check(summary.rss.p90_bytes == 400u * 1024u, "RSS lower p90");
    check(summary.rss.p95_bytes == 400u * 1024u, "RSS lower p95");
    check(summary.rss.p99_bytes == 400u * 1024u, "RSS lower p99");
    check(near(summary.pss.average_bytes, 240.0 * 1024.0), "PSS average");
    check(summary.uss.peak_bytes == 135u * 1024u, "USS peak");


    std::printf("bench metrics tests: %s\n", failures == 0 ? "OK" : "FAILED");
    return failures == 0 ? 0 : 1;
}
