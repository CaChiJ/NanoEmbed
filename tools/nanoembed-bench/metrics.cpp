#include "metrics.h"

#include <sys/resource.h>
#include <sys/utsname.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <unistd.h>

namespace nanoembed::bench {

namespace {

double tv_to_sec(const timeval & tv) {
    return static_cast<double>(tv.tv_sec) +
           static_cast<double>(tv.tv_usec) * 1e-6;
}

std::string slurp(const std::string & path) {
    std::ifstream f(path);
    if (!f) return {};
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string proc_path(int pid, const char * name) {
    return "/proc/" + std::to_string(pid) + "/" + name;
}

// Advance past a line; returns npos-safe next line start.
size_t next_line(std::string_view text, size_t pos) {
    const size_t nl = text.find('\n', pos);
    return (nl == std::string_view::npos) ? text.size() : nl + 1;
}

size_t parse_leading_number(std::string_view s) {
    size_t i = 0;
    while (i < s.size() && (s[i] == ' ' || s[i] == '\t')) ++i;
    size_t v = 0;
    while (i < s.size() && s[i] >= '0' && s[i] <= '9') {
        v = v * 10 + static_cast<size_t>(s[i] - '0');
        ++i;
    }
    return v;
}

} // namespace

namespace {

// Value following "Key:" on a line that starts with Key, unscaled.
struct ParsedValue {
    size_t value = 0;
    bool found = false;
};

ParsedValue find_value_line(std::string_view text, std::string_view key) {
    for (size_t pos = 0; pos < text.size(); pos = next_line(text, pos)) {
        if (text.compare(pos, key.size(), key) != 0) continue;
        const size_t eol  = text.find('\n', pos);
        const auto   line = text.substr(pos + key.size(),
                                        (eol == std::string_view::npos)
                                            ? std::string_view::npos
                                            : eol - pos - key.size());
        return {parse_leading_number(line), true};
    }
    return {};
}

size_t parse_value_line(std::string_view text, std::string_view key) {
    return find_value_line(text, key).value;
}

ParsedValue find_kb_line(std::string_view text, std::string_view key) {
    ParsedValue result = find_value_line(text, key);
    result.value *= 1024;
    return result;
}

} // namespace

size_t parse_kb_line(std::string_view text, std::string_view key) {
    return parse_value_line(text, key) * 1024;
}

MemSample parse_smaps_rollup(std::string_view rollup) {
    MemSample s;
    if (rollup.empty()) return s;

    const ParsedValue rss           = find_kb_line(rollup, "Rss:");
    const ParsedValue pss           = find_kb_line(rollup, "Pss:");
    const ParsedValue pss_anon      = find_kb_line(rollup, "Pss_Anon:");
    const ParsedValue pss_file      = find_kb_line(rollup, "Pss_File:");
    const ParsedValue anonymous     = find_kb_line(rollup, "Anonymous:");
    const ParsedValue private_clean = find_kb_line(rollup, "Private_Clean:");
    const ParsedValue private_dirty = find_kb_line(rollup, "Private_Dirty:");
    const ParsedValue shared_clean  = find_kb_line(rollup, "Shared_Clean:");
    const ParsedValue shared_dirty  = find_kb_line(rollup, "Shared_Dirty:");

    s.rss_bytes           = rss.value;
    s.pss_bytes           = pss.value;
    s.pss_anon_bytes      = pss_anon.value;
    s.pss_file_bytes      = pss_file.value;
    s.anonymous_bytes     = anonymous.value;
    s.private_clean_bytes = private_clean.value;
    s.private_dirty_bytes = private_dirty.value;
    s.shared_clean_bytes  = shared_clean.value;
    s.shared_dirty_bytes  = shared_dirty.value;
    s.uss_bytes           = private_clean.value + private_dirty.value;
    s.has_rss           = rss.found;
    s.has_pss           = pss.found;
    s.has_pss_anon      = pss_anon.found;
    s.has_pss_file      = pss_file.found;
    s.has_anonymous     = anonymous.found;
    s.has_private_clean = private_clean.found;
    s.has_private_dirty = private_dirty.found;
    s.has_shared_clean  = shared_clean.found;
    s.has_shared_dirty  = shared_dirty.found;
    s.has_uss           = private_clean.found && private_dirty.found;
    // Rss is the only field guaranteed non-zero for a live process; treat its
    // absence as "the file was not in the shape we expect".
    s.valid = s.has_rss && s.rss_bytes > 0;
    return s;
}

MemSampleSummary summarize_mem_samples(const std::vector<MemSample> & samples) {
    MemSampleSummary out;
    out.total_samples = samples.size();
    std::vector<size_t> rss, pss, uss, pss_anon, pss_file, anonymous;
    std::vector<size_t> private_clean, private_dirty, shared_clean, shared_dirty;

    auto add = [](SampledSizeSummary & dst, std::vector<size_t> & values,
                  size_t bytes, bool available) {
        if (!available) return;
        ++dst.valid_samples;
        dst.average_bytes += static_cast<double>(bytes);
        dst.peak_bytes = std::max(dst.peak_bytes, bytes);
        values.push_back(bytes);
    };
    for (const MemSample & sample : samples) {
        if (!sample.valid) continue;
        add(out.rss, rss, sample.rss_bytes, sample.has_rss);
        add(out.pss, pss, sample.pss_bytes, sample.has_pss);
        add(out.uss, uss, sample.uss_bytes, sample.has_uss);
        add(out.pss_anon, pss_anon, sample.pss_anon_bytes, sample.has_pss_anon);
        add(out.pss_file, pss_file, sample.pss_file_bytes, sample.has_pss_file);
        add(out.anonymous, anonymous, sample.anonymous_bytes, sample.has_anonymous);
        add(out.private_clean, private_clean, sample.private_clean_bytes,
            sample.has_private_clean);
        add(out.private_dirty, private_dirty, sample.private_dirty_bytes,
            sample.has_private_dirty);
        add(out.shared_clean, shared_clean, sample.shared_clean_bytes,
            sample.has_shared_clean);
        add(out.shared_dirty, shared_dirty, sample.shared_dirty_bytes,
            sample.has_shared_dirty);
    }
    auto finish = [](SampledSizeSummary & summary, std::vector<size_t> & values) {
        if (values.empty()) return;
        summary.average_bytes /= static_cast<double>(values.size());
        std::sort(values.begin(), values.end());
        const auto lower = [&](double q) {
            return values[static_cast<size_t>(q * static_cast<double>(values.size() - 1))];
        };
        summary.p50_bytes = lower(0.50);
        summary.p75_bytes = lower(0.75);
        summary.p90_bytes = lower(0.90);
        summary.p95_bytes = lower(0.95);
        summary.p99_bytes = lower(0.99);
    };
    finish(out.rss, rss); finish(out.pss, pss); finish(out.uss, uss);
    finish(out.pss_anon, pss_anon); finish(out.pss_file, pss_file);
    finish(out.anonymous, anonymous); finish(out.private_clean, private_clean);
    finish(out.private_dirty, private_dirty); finish(out.shared_clean, shared_clean);
    finish(out.shared_dirty, shared_dirty);
    return out;
}

std::string read_proc_file(int pid, const char * name) {
    return slurp(proc_path(pid, name));
}

size_t read_peak_rss(int pid) {
    return parse_kb_line(read_proc_file(pid, "status"), "VmHWM:");
}

size_t read_current_rss(int pid) {
    // statm is a single short line of page counts; field 2 is resident. Far
    // cheaper than status (no per-field formatting) and far cheaper than
    // smaps_rollup (no page-table walk).
    const std::string statm = read_proc_file(pid, "statm");
    if (statm.empty()) return 0;

    const size_t sp = statm.find(' ');
    if (sp == std::string::npos) return 0;

    const size_t resident_pages = parse_leading_number(
        std::string_view(statm).substr(sp + 1));
    const long page_size = sysconf(_SC_PAGESIZE);
    return resident_pages * static_cast<size_t>(page_size > 0 ? page_size : 4096);
}

MemSample read_mem_sample(int pid) {
    return parse_smaps_rollup(read_proc_file(pid, "smaps_rollup"));
}

bool reset_peak_rss(int pid) {
    const std::string path = proc_path(pid, "clear_refs");
    FILE * f = std::fopen(path.c_str(), "w");
    if (!f) return false;
    const bool ok = std::fputs("5\n", f) >= 0;
    return (std::fclose(f) == 0) && ok;
}

ResourceCounters read_self_counters() {
    ResourceCounters c;

    rusage ru{};
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        c.cpu_user_sec      = tv_to_sec(ru.ru_utime);
        c.cpu_sys_sec       = tv_to_sec(ru.ru_stime);
        c.page_faults_major = ru.ru_majflt;
        c.page_faults_minor = ru.ru_minflt;
    }

    // /proc/self/io is in raw bytes, not kB. read_bytes counts what actually
    // hit the block layer, so page-cache hits read as 0 — that is the intent:
    // it measures real disk I/O, not mmap traffic.
    c.io_read_bytes = parse_value_line(slurp("/proc/self/io"), "read_bytes:");

    return c;
}

Environment read_environment() {
    Environment e;

    utsname u{};
    if (uname(&u) == 0) e.kernel = u.release;

    const std::string cpuinfo = slurp("/proc/cpuinfo");
    const size_t      pos     = cpuinfo.find("model name");
    if (pos != std::string::npos) {
        const size_t colon = cpuinfo.find(':', pos);
        const size_t eol   = cpuinfo.find('\n', pos);
        if (colon != std::string::npos && eol != std::string::npos && colon < eol) {
            size_t b = colon + 1;
            while (b < eol && (cpuinfo[b] == ' ' || cpuinfo[b] == '\t')) ++b;
            e.cpu_model = cpuinfo.substr(b, eol - b);
        }
    }

    e.nproc     = static_cast<int>(sysconf(_SC_NPROCESSORS_ONLN));
    e.page_size = sysconf(_SC_PAGESIZE);
    return e;
}

} // namespace nanoembed::bench
