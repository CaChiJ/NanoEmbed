// OS-split resource metrics for nanoembed-bench.
//
// Conventions:
//   - "current" = state at time of call.
//   - "peak"    = high-water-mark since process start (OS-tracked).
//   - 0 means "unsupported on this OS" rather than "actually zero" for the
//     I/O field (page faults / CPU are universally available via getrusage).

#pragma once

#include <cstddef>
#include <cstdint>

namespace nanoembed::bench {

struct Snapshot {
    double  cpu_user_sec        = 0.0;
    double  cpu_sys_sec         = 0.0;
    long    page_faults_major   = 0;     // ru_majflt
    long    page_faults_minor   = 0;     // ru_minflt
    size_t  io_read_bytes       = 0;     // disk read bytes (0 if unsupported)
    size_t  rss_current_bytes   = 0;
};

// Capture user/sys CPU + page faults + IO + current RSS in one call.
Snapshot snapshot();

// Process peak RSS since start.
size_t peak_rss_bytes();

// Process current RSS (separate so the sampler thread can call it cheaply).
size_t current_rss_bytes();

// OS label, e.g. "macos" / "linux" / "unknown".
const char * os_label();

} // namespace nanoembed::bench
