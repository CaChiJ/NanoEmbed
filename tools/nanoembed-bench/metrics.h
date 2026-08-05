// Linux-only process metrics for nanoembed-bench.
//
// The bench harness is a two-process design: an orchestrator (parent) measures
// a worker (child) that was fork+exec'd, so nothing the harness allocates ever
// lands in the numbers we report. That split is why every reader here takes an
// explicit pid — the parent reads /proc/<worker>/..., never /proc/self/...
//
// The one exception is read_self_counters(): CPU time and page faults come from
// getrusage(), which only reports on the calling process. The worker samples
// those itself at the window boundaries and ships the delta over a pipe.
//
// Conventions:
//   - All sizes are bytes. /proc reports kB; conversion happens here.
//   - A failed read yields 0 / valid=false rather than throwing.

#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace nanoembed::bench {

// One memory observation of a process, from /proc/<pid>/smaps_rollup.
//
//   rss = every resident page, shared libraries included -> what the device
//         must actually have free. This is the gating metric.
//   pss = shared pages divided by their mapcount -> sums correctly across
//         several workers mapping the same model.
//   uss = private pages only -> what is reclaimed if this process dies.
//
// On M3 the three nearly coincide (the model is read into anonymous memory).
// They diverge once M4 mmaps the GGUF.
struct MemSample {
    size_t rss_bytes = 0;
    size_t pss_bytes = 0;
    size_t uss_bytes = 0;   // Private_Clean + Private_Dirty
    bool   valid     = false;
};

// Counters a process can only observe about itself (getrusage + /proc/self/io).
struct ResourceCounters {
    double cpu_user_sec      = 0.0;
    double cpu_sys_sec       = 0.0;
    long   page_faults_major = 0;
    long   page_faults_minor = 0;
    size_t io_read_bytes     = 0;
};

// Identifies the machine a run happened on, so a baseline recorded elsewhere
// is detectable rather than silently compared against.
struct Environment {
    std::string kernel;      // uname -r
    std::string cpu_model;   // /proc/cpuinfo "model name"
    int         nproc     = 0;
    long        page_size = 0;
};

// ---- Pure parsers (no I/O; unit-testable) ----------------------------------

// Value of a "Key:  1234 kB" line in a /proc status-style file, in bytes.
// Matches on line starts only, so "Rss:" cannot be found inside "Pss_Anon:".
// Returns 0 when the key is absent.
size_t parse_kb_line(std::string_view text, std::string_view key);

MemSample parse_smaps_rollup(std::string_view rollup);

// ---- /proc readers ---------------------------------------------------------

// Contents of /proc/<pid>/<name>, or "" if unreadable.
std::string read_proc_file(int pid, const char * name);

// Peak RSS since the last reset (VmHWM). Exact — the kernel tracks it, so
// unlike sampling it cannot miss a spike between two observations.
size_t read_peak_rss(int pid);

// Current RSS via /proc/<pid>/statm. ~0.004 ms — cheap enough to poll at 50 ms.
size_t read_current_rss(int pid);

// Full RSS/PSS/USS via /proc/<pid>/smaps_rollup. ~6 ms on a 300 MB process:
// it walks the page table under the target's mmap_lock, so polling this at the
// RSS cadence would perturb the very process being measured. Poll it slowly.
MemSample read_mem_sample(int pid);

// Reset VmHWM to the current RSS by writing "5" to /proc/<pid>/clear_refs
// (Linux >= 4.0). This is what makes peak RSS scoped to the measurement window
// instead of the whole process lifetime. Returns false if the write failed.
bool reset_peak_rss(int pid);

ResourceCounters read_self_counters();

Environment read_environment();

} // namespace nanoembed::bench
