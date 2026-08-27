// File page-cache observation and best-effort eviction for cold benchmarks.
//
// The public structs and pure summary helper are portable so their semantics
// can be unit-tested on every development platform. Live residency/eviction is
// deliberately Linux-only; unsupported platforms return an explicit status.

#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace nanoembed::bench {

struct FileResidency {
    bool        platform_supported = false;
    bool        collected          = false;
    size_t      file_size_bytes    = 0;
    size_t      page_size_bytes    = 0;
    size_t      total_pages        = 0;
    size_t      resident_pages     = 0;
    double      resident_percent   = 0.0;
    int         error_number       = 0;
    std::string status;
};

struct CacheEvictionResult {
    bool          requested                = false;
    bool          platform_supported       = false;
    bool          eviction_call_succeeded  = false;
    bool          cold_cache_verified      = false;
    int           eviction_error_number    = 0;
    std::string   status;
    FileResidency before_eviction;
    FileResidency after_eviction_before_worker;
    FileResidency after_worker;
};

// Pure conversion from mincore's per-page residency vector. A non-zero low bit
// means resident. Invalid dimensions produce collected=false.
FileResidency summarize_residency(size_t                            file_size_bytes,
                                  size_t                            page_size_bytes,
                                  const std::vector<unsigned char> & page_state);

bool cache_control_platform_supported();

// Read-only: maps without touching contents and asks mincore which file pages
// are resident. The mapping is removed before return.
FileResidency query_file_residency(const std::string & path);

// Linux: observe, unmap, request POSIX_FADV_DONTNEED, then observe again. A
// successful advisory call is not sufficient verification: verified means the
// second mincore observation saw zero resident pages.
CacheEvictionResult evict_file_pages(const std::string & path);

} // namespace nanoembed::bench
