#include "metrics.h"

#include <sys/resource.h>

#if defined(__APPLE__)
#  include <mach/mach.h>
#  include <libproc.h>
#  include <sys/proc_info.h>
#  include <unistd.h>
#elif defined(__linux__)
#  include <fstream>
#  include <string>
#  include <unistd.h>
#endif

namespace nanoembed::bench {

namespace {

double tv_to_sec(const timeval & tv) {
    return static_cast<double>(tv.tv_sec) +
           static_cast<double>(tv.tv_usec) * 1e-6;
}

#if defined(__APPLE__)

size_t macos_current_rss() {
    mach_task_basic_info_data_t info{};
    mach_msg_type_number_t      count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) {
        return 0;
    }
    return static_cast<size_t>(info.resident_size);
}

size_t macos_peak_rss() {
    mach_task_basic_info_data_t info{};
    mach_msg_type_number_t      count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                  reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) {
        return 0;
    }
    return static_cast<size_t>(info.resident_size_max);
}

size_t macos_io_read_bytes() {
    // proc_pid_rusage with RUSAGE_INFO_CURRENT exposes ri_diskio_bytesread
    // (cumulative bytes read from disk). Available since macOS 10.9.
    rusage_info_current ri{};
    if (proc_pid_rusage(getpid(), RUSAGE_INFO_CURRENT,
                         reinterpret_cast<rusage_info_t *>(&ri)) != 0) {
        return 0;
    }
    return static_cast<size_t>(ri.ri_diskio_bytesread);
}

#elif defined(__linux__)

size_t parse_kb_field(const std::string & status, const std::string & key) {
    const auto pos = status.find(key);
    if (pos == std::string::npos) return 0;
    const auto eol = status.find('\n', pos);
    const std::string line = status.substr(pos, eol - pos);
    // Format: "VmRSS:\t  12345 kB"
    size_t i = 0;
    while (i < line.size() && (line[i] < '0' || line[i] > '9')) ++i;
    size_t kb = 0;
    while (i < line.size() && line[i] >= '0' && line[i] <= '9') {
        kb = kb * 10 + static_cast<size_t>(line[i] - '0');
        ++i;
    }
    return kb * 1024;
}

std::string slurp(const char * path) {
    std::ifstream f(path);
    if (!f) return {};
    std::string out((std::istreambuf_iterator<char>(f)),
                     std::istreambuf_iterator<char>());
    return out;
}

size_t linux_current_rss() {
    return parse_kb_field(slurp("/proc/self/status"), "VmRSS:");
}

size_t linux_peak_rss() {
    return parse_kb_field(slurp("/proc/self/status"), "VmHWM:");
}

size_t linux_io_read_bytes() {
    const auto io = slurp("/proc/self/io");
    if (io.empty()) return 0;
    const std::string key = "read_bytes:";
    const auto pos = io.find(key);
    if (pos == std::string::npos) return 0;
    size_t i = pos + key.size();
    while (i < io.size() && (io[i] == ' ' || io[i] == '\t')) ++i;
    size_t v = 0;
    while (i < io.size() && io[i] >= '0' && io[i] <= '9') {
        v = v * 10 + static_cast<size_t>(io[i] - '0');
        ++i;
    }
    return v;
}

#endif

} // namespace

size_t current_rss_bytes() {
#if defined(__APPLE__)
    return macos_current_rss();
#elif defined(__linux__)
    return linux_current_rss();
#else
    return 0;
#endif
}

size_t peak_rss_bytes() {
#if defined(__APPLE__)
    return macos_peak_rss();
#elif defined(__linux__)
    return linux_peak_rss();
#else
    return 0;
#endif
}

Snapshot snapshot() {
    Snapshot s;

    rusage ru{};
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        s.cpu_user_sec      = tv_to_sec(ru.ru_utime);
        s.cpu_sys_sec       = tv_to_sec(ru.ru_stime);
        s.page_faults_major = ru.ru_majflt;
        s.page_faults_minor = ru.ru_minflt;
    }

    s.rss_current_bytes = current_rss_bytes();

#if defined(__APPLE__)
    s.io_read_bytes = macos_io_read_bytes();
#elif defined(__linux__)
    s.io_read_bytes = linux_io_read_bytes();
#endif

    return s;
}

const char * os_label() {
#if defined(__APPLE__)
    return "macos";
#elif defined(__linux__)
    return "linux";
#else
    return "unknown";
#endif
}

} // namespace nanoembed::bench
