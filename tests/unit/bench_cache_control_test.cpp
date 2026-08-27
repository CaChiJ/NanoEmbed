#include "cache_control.h"

#include <cmath>
#include <cstdio>
#include <vector>

#if defined(__linux__)
#include <cstdlib>
#include <unistd.h>
#endif

using namespace nanoembed::bench;

namespace {

int failures = 0;

void check(bool ok, const char * what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

} // namespace

int main() {
    const FileResidency summary = summarize_residency(
        5 * 4096 - 7, 4096, {1, 0, 3, 0, 1});
    check(summary.platform_supported, "pure residency summary is portable");
    check(summary.collected, "valid residency vector is collected");
    check(summary.total_pages == 5, "partial final page is counted");
    check(summary.resident_pages == 3, "mincore low bit means resident");
    check(std::fabs(summary.resident_percent - 60.0) < 1e-12,
          "resident percentage uses all file pages");

    const FileResidency invalid = summarize_residency(8192, 4096, {1});
    check(!invalid.collected, "dimension mismatch is explicit");
    check(invalid.status == "residency_vector_size_mismatch",
          "dimension mismatch has stable status");

    const FileResidency missing = query_file_residency(
        "/definitely/not/a/nanoembed/model.gguf");
#if defined(__linux__)
    check(missing.platform_supported, "Linux reports cache control support");
    check(!missing.collected && missing.error_number != 0,
          "missing Linux file reports an observation error");

    char temp_path[] = "/tmp/nanoembed-cache-control-XXXXXX";
    const int temp_fd = ::mkstemp(temp_path);
    check(temp_fd >= 0, "Linux cache test file created");
    if (temp_fd >= 0) {
        const long live_page_size = ::sysconf(_SC_PAGESIZE);
        check(live_page_size > 0, "Linux page size is available");
        check(::ftruncate(temp_fd, 2 * live_page_size) == 0,
              "Linux cache test file sized");
        ::close(temp_fd);
        const FileResidency live = query_file_residency(temp_path);
        check(live.collected && live.total_pages == 2,
              "Linux mincore residency query succeeds");
        const CacheEvictionResult eviction = evict_file_pages(temp_path);
        check(eviction.requested && eviction.platform_supported,
              "Linux eviction request is supported");
        check(eviction.eviction_call_succeeded,
              "Linux posix_fadvise request succeeds");
        check(eviction.before_eviction.collected &&
                  eviction.after_eviction_before_worker.collected,
              "Linux eviction records before and after residency");
        ::unlink(temp_path);
    }
#else
    check(!cache_control_platform_supported(),
          "non-Linux reports cache control unsupported");
    check(!missing.platform_supported && !missing.collected,
          "unsupported residency is machine-readable");
    const CacheEvictionResult eviction = evict_file_pages("unused");
    check(eviction.requested && !eviction.platform_supported,
          "unsupported eviction preserves requested intent");
    check(!eviction.cold_cache_verified,
          "unsupported eviction is never verified cold");
#endif

    std::printf("bench cache-control tests: %s\n",
                failures == 0 ? "OK" : "FAILED");
    return failures == 0 ? 0 : 1;
}
