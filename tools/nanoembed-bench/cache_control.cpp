#include "cache_control.h"

#include <cerrno>
#include <cstring>

#if defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace nanoembed::bench {

namespace {

#if defined(__linux__)
std::string errno_status(const char * operation, int error_number) {
    return std::string(operation) + ": " + std::strerror(error_number);
}
#endif

} // namespace

FileResidency summarize_residency(
        size_t                            file_size_bytes,
        size_t                            page_size_bytes,
        const std::vector<unsigned char> & page_state) {
    FileResidency out;
    out.platform_supported = true;
    out.file_size_bytes     = file_size_bytes;
    out.page_size_bytes     = page_size_bytes;
    if (file_size_bytes == 0 || page_size_bytes == 0) {
        out.status = "invalid_file_or_page_size";
        return out;
    }

    out.total_pages = (file_size_bytes + page_size_bytes - 1) / page_size_bytes;
    if (page_state.size() != out.total_pages) {
        out.status = "residency_vector_size_mismatch";
        return out;
    }

    for (unsigned char page : page_state) {
        if ((page & 1u) != 0) ++out.resident_pages;
    }
    out.resident_percent =
        100.0 * static_cast<double>(out.resident_pages) /
        static_cast<double>(out.total_pages);
    out.collected = true;
    out.status    = "collected";
    return out;
}

bool cache_control_platform_supported() {
#if defined(__linux__)
    return true;
#else
    return false;
#endif
}

FileResidency query_file_residency(const std::string & path) {
#if defined(__linux__)
    FileResidency out;
    out.platform_supported = true;

    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        out.error_number = errno;
        out.status = errno_status("open", out.error_number);
        return out;
    }

    struct stat st {};
    if (::fstat(fd, &st) != 0 || st.st_size <= 0) {
        out.error_number = errno;
        out.status = st.st_size == 0
            ? "model_file_is_empty"
            : errno_status("fstat", out.error_number);
        ::close(fd);
        return out;
    }

    const long raw_page_size = ::sysconf(_SC_PAGESIZE);
    if (raw_page_size <= 0) {
        out.status = "sysconf_page_size_failed";
        ::close(fd);
        return out;
    }
    const size_t file_size = static_cast<size_t>(st.st_size);
    const size_t page_size = static_cast<size_t>(raw_page_size);
    const size_t page_count = (file_size + page_size - 1) / page_size;

    void * mapping = ::mmap(nullptr, file_size, PROT_NONE, MAP_SHARED, fd, 0);
    if (mapping == MAP_FAILED) {
        out.error_number = errno;
        out.status = errno_status("mmap", out.error_number);
        ::close(fd);
        return out;
    }

    std::vector<unsigned char> page_state(page_count, 0);
    if (::mincore(mapping, file_size, page_state.data()) != 0) {
        out.error_number = errno;
        out.status = errno_status("mincore", out.error_number);
        ::munmap(mapping, file_size);
        ::close(fd);
        return out;
    }

    ::munmap(mapping, file_size);
    ::close(fd);
    return summarize_residency(file_size, page_size, page_state);
#else
    (void) path;
    FileResidency out;
    out.status = "unsupported_platform_linux_required";
    return out;
#endif
}

CacheEvictionResult evict_file_pages(const std::string & path) {
    CacheEvictionResult out;
    out.requested          = true;
    out.platform_supported = cache_control_platform_supported();
    if (!out.platform_supported) {
        (void) path;
        out.status = "unsupported_platform_linux_required";
        return out;
    }

#if defined(__linux__)
    // query_file_residency unmaps before returning. Keeping no model mapping
    // alive while issuing DONTNEED avoids making the advisory call fail solely
    // because this observer pinned a mapping itself.
    out.before_eviction = query_file_residency(path);

    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        out.eviction_error_number = errno;
        out.status = errno_status("open_for_posix_fadvise",
                                  out.eviction_error_number);
        out.after_eviction_before_worker = query_file_residency(path);
        return out;
    }

    const int advice_error = ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    ::close(fd);
    out.eviction_call_succeeded = advice_error == 0;
    out.eviction_error_number   = advice_error;
    out.after_eviction_before_worker = query_file_residency(path);

    out.cold_cache_verified =
        out.eviction_call_succeeded &&
        out.before_eviction.collected &&
        out.after_eviction_before_worker.collected &&
        out.after_eviction_before_worker.resident_pages == 0;

    if (!out.eviction_call_succeeded) {
        out.status = errno_status("posix_fadvise", advice_error);
    } else if (!out.before_eviction.collected) {
        out.status = "eviction_requested_but_pre_eviction_residency_unavailable";
    } else if (!out.after_eviction_before_worker.collected) {
        out.status = "eviction_requested_but_post_eviction_residency_unavailable";
    } else if (!out.cold_cache_verified) {
        out.status = "eviction_requested_but_pages_remain_resident";
    } else {
        out.status = "verified_zero_resident_pages_before_worker";
    }
#endif
    return out;
}

} // namespace nanoembed::bench
