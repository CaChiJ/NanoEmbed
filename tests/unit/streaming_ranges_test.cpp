#include "streaming_execution.h"

#include <atomic>
#include <cstdio>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

int failures = 0;

#define EXPECT_TRUE(condition) do {                                             \
    if (!(condition)) {                                                        \
        std::fprintf(stderr, "FAIL: %s (%s:%d)\n", #condition, __FILE__, __LINE__); \
        ++failures;                                                            \
    }                                                                          \
} while (0)

template <typename F>
void expect_throw(F && fn, const char * needle) {
    try {
        fn();
        std::fprintf(stderr, "FAIL: expected exception containing %s\n", needle);
        ++failures;
    } catch (const std::exception & error) {
        if (std::string(error.what()).find(needle) == std::string::npos) {
            std::fprintf(stderr, "FAIL: '%s' lacks '%s'\n", error.what(), needle);
            ++failures;
        }
    }
}

void test_page_ranges() {
    using nanoembed::StreamingByteRange;
    using nanoembed::streaming_detail::page_align_and_coalesce;

    // Unaligned ranges sharing/adjacent on pages coalesce.
    auto got = page_align_and_coalesce({{3, 7}, {4090, 20}, {8192, 1}}, 16384, 4096);
    EXPECT_TRUE(got == std::vector<StreamingByteRange>({{0, 12288}}));

    // A real gap is not turned into a min/max span.
    got = page_align_and_coalesce({{1, 2}, {12289, 2}}, 20000, 4096);
    EXPECT_TRUE(got == std::vector<StreamingByteRange>({{0, 4096}, {12288, 4096}}));

    // Mapping edge is clipped rather than advising beyond the mapping.
    got = page_align_and_coalesce({{9990, 10}}, 10000, 4096);
    EXPECT_TRUE(got == std::vector<StreamingByteRange>({{8192, 1808}}));
    expect_throw([&] { (void) page_align_and_coalesce({{9999, 2}}, 10000, 4096); },
                 "exceeds mapped file");
    expect_throw([&] {
        (void) page_align_and_coalesce(
            {{std::numeric_limits<size_t>::max() - 1, 4}},
            std::numeric_limits<size_t>::max(), 4096);
    }, "overflow");
}

void test_token_rows() {
    using nanoembed::StreamingByteRange;
    using nanoembed::streaming_detail::token_row_ranges;

    // Duplicate IDs disappear. Rows 1 and 2 share a page; row 7 is disjoint.
    auto got = token_row_ranges(4096, 8 * 600, 600, 8,
                                {2, 1, 2, 7}, 16384, 4096);
    EXPECT_TRUE(got == std::vector<StreamingByteRange>({{4096, 8192}}));
    expect_throw([&] {
        (void) token_row_ranges(0, 1024, 128, 8, {-1}, 4096, 4096);
    }, "outside validated");
    expect_throw([&] {
        (void) token_row_ranges(0, 1024, 128, 8, {8}, 4096, 4096);
    }, "outside validated");

    // Quantized row strides are bytes, not element counts. Exercise Q8-like
    // (34 bytes/block) and mixed K-like (144 bytes/block) packed rows.
    got = token_row_ranges(128, 4 * 1088, 1088, 4, {0, 3}, 8192, 4096);
    EXPECT_TRUE(got == std::vector<StreamingByteRange>({{0, 8192}}));
    got = token_row_ranges(8192, 4 * 2304, 2304, 4, {0, 3}, 20000, 4096);
    EXPECT_TRUE(got == std::vector<StreamingByteRange>({{8192, 11808}}));
}

void test_leases_and_failures() {
    using namespace nanoembed::streaming_detail;
    struct Event { size_t offset; AdviceKind kind; };
    std::mutex events_mutex;
    std::vector<Event> events;
    AdviceFunction advice = [&](size_t offset, size_t, AdviceKind kind, std::string &) {
        std::lock_guard<std::mutex> lock(events_mutex);
        events.push_back({offset, kind});
        return 0;
    };
    ResidencyCoordinator coordinator(65536, 4096, advice);
    coordinator.set_classification({{0, 4096}}, {{4096, 4096}}, {{{8192, 4096}}});
    coordinator.retain_common({{0, 4096}});

    auto first = coordinator.acquire("layer:0", {{8192, 4096}}, false);
    auto second = coordinator.acquire("layer:0", {{8192, 4096}}, false);
    first.mark_compute_complete();
    first.release();
    {
        std::lock_guard<std::mutex> lock(events_mutex);
        size_t drops = 0;
        for (const auto & event : events) if (event.kind == AdviceKind::DontNeed) ++drops;
        EXPECT_TRUE(drops == 0); // another context still owns the layer
    }
    second.mark_compute_complete();
    second.release();
    const auto diag = coordinator.diagnostics();
    EXPECT_TRUE(diag.leases_acquired == 2);
    EXPECT_TRUE(diag.leases_released == 2);
    EXPECT_TRUE(diag.lease_high_water == 2);
    EXPECT_TRUE(diag.compute_completions == 2);
    EXPECT_TRUE(diag.dontneed_calls == 1);
    EXPECT_TRUE(diag.common_class_bytes == 4096 && diag.common_class_pages == 1);
    EXPECT_TRUE(diag.token_class_bytes == 4096 && diag.token_class_pages == 1);
    EXPECT_TRUE(diag.layer_class_bytes == 4096 && diag.layer_class_pages == 1);
    EXPECT_TRUE(diag.resolved_mode == "internal-streaming-linux");
    EXPECT_TRUE(!diag.poisoned);

    // A shared page protected by common lifetime must never be dropped.
    auto shared = coordinator.acquire("token", {{0, 4096}}, true);
    shared.mark_compute_complete();
    shared.release();
    EXPECT_TRUE(coordinator.diagnostics().dontneed_calls == 1);

    // Postcompute DONTNEED failure is loud and permanently poisons the model.
    ResidencyCoordinator bad(65536, 4096,
        [](size_t, size_t, AdviceKind kind, std::string & error) {
            if (kind == AdviceKind::DontNeed) { error = "injected"; return -1; }
            return 0;
        });
    auto lease = bad.acquire("layer:0", {{8192, 4096}}, false);
    lease.mark_compute_complete();
    expect_throw([&] { lease.release(); }, "MADV_DONTNEED failed");
    EXPECT_TRUE(bad.diagnostics().poisoned);
    expect_throw([&] { (void) bad.acquire("layer:0", {{8192, 4096}}, false); },
                 "poisoned");

    // Releasing before the graph reports synchronous completion is observable.
    ResidencyCoordinator early(65536, 4096, advice);
    auto premature = early.acquire("layer:0", {{8192, 4096}}, false);
    expect_throw([&] { premature.release(); }, "before synchronous compute");
    EXPECT_TRUE(early.diagnostics().premature_release_attempts == 1);

    // Precompute WILLNEED failure fails the acquisition without inventing an
    // active lease or poisoning otherwise valid coordinator state.
    bool fail_once = true;
    ResidencyCoordinator precompute(65536, 4096,
        [&](size_t, size_t, AdviceKind kind, std::string & error) {
            if (kind == AdviceKind::WillNeed && fail_once) {
                fail_once = false; error = "injected"; return -1;
            }
            return 0;
        });
    expect_throw([&] { (void) precompute.acquire("layer:0", {{8192, 4096}}, false); },
                 "MADV_WILLNEED failed");
    EXPECT_TRUE(precompute.diagnostics().active_leases == 0);
    EXPECT_TRUE(!precompute.diagnostics().poisoned);
    auto retry = precompute.acquire("layer:0", {{8192, 4096}}, false);
    retry.mark_compute_complete();
    retry.release();
}

void test_concurrent_same_region() {
    using namespace nanoembed::streaming_detail;
    std::atomic<int> willneed{0};
    std::atomic<int> dontneed{0};
    ResidencyCoordinator coordinator(65536, 4096,
        [&](size_t, size_t, AdviceKind kind, std::string &) {
            if (kind == AdviceKind::WillNeed) ++willneed; else ++dontneed;
            return 0;
        });
    std::atomic<int> ready{0};
    std::atomic<bool> go{false};
    auto run = [&] {
        auto lease = coordinator.acquire("layer:7", {{16384, 4096}}, false);
        ++ready;
        while (!go.load()) std::this_thread::yield();
        lease.mark_compute_complete();
        lease.release();
    };
    std::thread a(run), b(run);
    while (ready.load() != 2) std::this_thread::yield();
    EXPECT_TRUE(coordinator.diagnostics().lease_high_water == 2);
    EXPECT_TRUE(dontneed.load() == 0);
    go = true;
    a.join(); b.join();
    EXPECT_TRUE(dontneed.load() == 1);
}

} // namespace

int main() {
    test_page_ranges();
    test_token_rows();
    test_leases_and_failures();
    test_concurrent_same_region();
#if !defined(__linux__)
    expect_throw([] { nanoembed::InternalStreamingContext context; }, "unsupported outside Linux");
    expect_throw([] { nanoembed::InternalStreamingModel model("does-not-matter"); },
                 "unsupported outside Linux");
#endif
    std::printf("streaming_ranges_test: %s\n", failures == 0 ? "ok" : "FAIL");
    return failures == 0 ? 0 : 1;
}
