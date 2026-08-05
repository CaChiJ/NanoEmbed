// nanoembed-bench: run one scenario, write JSON metrics to stdout / --out.
//
// Linux only. The measurement design depends on /proc/<pid>/{status,statm,
// smaps_rollup,clear_refs}, which have no portable equivalent.
//
// Usage:
//   nanoembed-bench --model PATH --inputs FILE
//                   [--scenario NAME] [--warmup N] [--iter N]
//                   [--cls] [--no-normalize] [--threads N]
//                   [--max-seq-len N] [--out PATH]
//                   [--rss-interval-ms N] [--rollup-interval-ms N]
//   nanoembed-bench --selftest [--selftest-alloc-mb N]
//
// ---- Why two processes -----------------------------------------------------
//
// The orchestrator (parent) never loads a model. It fork+exec's itself with
// --worker, and the worker does all the embedding. The parent then measures
// the worker from the outside via /proc.
//
// exec, not just fork: a plain fork leaves the child sharing the parent's
// dirtied pages copy-on-write, and those pages count in the child's RSS. exec
// replaces the address space outright, so the worker starts from a clean ~1 MB
// floor no matter what the parent touched. Measured, parent holding 8 MB:
//
//     fork only    RSS=8552 kB at start -> 70192 kB after a 60 MB workload
//     fork + exec  RSS=1152 kB at start -> 62796 kB after the same workload
//
// That is what makes RSS — not just USS — a trustworthy number here, which
// matters because the M4 memory gate is stated in terms of RSS.
//
// ---- Why the window is reset ----------------------------------------------
//
// VmHWM is a high-water mark that never decreases on its own, so reading it at
// the end would report a peak covering model load and warmup — i.e. a different
// window than the CPU and page-fault deltas, which are scoped to the
// measurement loop. Writing "5" to clear_refs resets VmHWM to the current RSS,
// so we can report both windows separately and honestly:
//
//     rss_peak_lifetime_mb  whole worker lifetime, load spike included
//     rss_peak_window_mb    steady-state embedding only
//
// PSS and USS have no kernel high-water mark, so their peaks are sampled and
// are named _sampled to say so.

#include "nanoembed/nanoembed.h"

#include "metrics.h"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace nanoembed::bench;
using Clock = std::chrono::steady_clock;

// Fds the worker inherits across exec. Fixed by convention rather than passed
// in argv: fewer moving parts, and exec preserves them as long as they are not
// O_CLOEXEC (dup2 targets never are).
constexpr int kCtrlFd = 3;   // parent -> worker
constexpr int kRepFd  = 4;   // worker -> parent

constexpr char kMsgReady = 'R';
constexpr char kMsgGo    = 'G';
constexpr char kMsgDone  = 'D';
constexpr char kMsgExit  = 'X';

// What the worker ships back once the window closes. Both sides are the same
// binary, so the layout is identical by construction.
struct WorkerReport {
    double             cpu_user_sec      = 0.0;
    double             cpu_sys_sec       = 0.0;
    long long          page_faults_major = 0;
    long long          page_faults_minor = 0;
    unsigned long long io_read_bytes     = 0;
    unsigned long long total_items       = 0;
    double             wall_sec          = 0.0;
    unsigned long long n_latencies       = 0;
};

// ---- Args ------------------------------------------------------------------

struct Args {
    std::string model_path;
    std::string inputs_path;
    std::string scenario = "single_short_f16";
    std::string out_path;              // empty = stdout
    int  warmup             = 5;
    int  iter               = 50;
    int  threads            = 0;       // 0 = auto
    int  max_seq_len        = 0;       // 0 = model default
    bool use_cls            = false;
    bool normalize          = true;
    int  rss_interval_ms    = 50;
    int  rollup_interval_ms = 500;
    int  timeout_sec        = 1800;

    bool is_worker          = false;
    bool selftest           = false;
    int  selftest_alloc_mb  = 0;       // >0 = worker allocates instead of embedding
};

void print_usage(const char * prog) {
    std::fprintf(stderr,
        "usage: %s --model PATH --inputs FILE [--scenario NAME] [--warmup N]\n"
        "       [--iter N] [--cls] [--no-normalize] [--threads N]\n"
        "       [--max-seq-len N] [--out PATH]\n"
        "       [--rss-interval-ms N] [--rollup-interval-ms N] [--timeout-sec N]\n"
        "       %s --selftest [--selftest-alloc-mb N]\n",
        prog, prog);
}

bool parse_args(int argc, char ** argv, Args & a) {
    bool bad = false;

    auto need_val = [&](int & i) -> const char * {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "error: %s expects a value\n", argv[i]);
            bad = true;
            return nullptr;
        }
        return argv[++i];
    };

    for (int i = 1; i < argc && !bad; ++i) {
        const char * t = argv[i];
        auto str_opt = [&](const char * name, std::string & dst) {
            if (std::strcmp(t, name) != 0) return false;
            const char * v = need_val(i);
            if (v) dst = v;
            return true;
        };
        auto int_opt = [&](const char * name, int & dst) {
            if (std::strcmp(t, name) != 0) return false;
            const char * v = need_val(i);
            if (v) dst = std::atoi(v);
            return true;
        };

        if      (str_opt("--model",    a.model_path))  { }
        else if (str_opt("--inputs",   a.inputs_path)) { }
        else if (str_opt("--scenario", a.scenario))    { }
        else if (str_opt("--out",      a.out_path))    { }
        else if (int_opt("--warmup",             a.warmup))             { }
        else if (int_opt("--iter",               a.iter))               { }
        else if (int_opt("--threads",            a.threads))            { }
        else if (int_opt("--max-seq-len",        a.max_seq_len))        { }
        else if (int_opt("--rss-interval-ms",    a.rss_interval_ms))    { }
        else if (int_opt("--rollup-interval-ms", a.rollup_interval_ms)) { }
        else if (int_opt("--timeout-sec",        a.timeout_sec))        { }
        else if (int_opt("--selftest-alloc-mb",  a.selftest_alloc_mb))  { }
        else if (std::strcmp(t, "--cls")          == 0) { a.use_cls   = true;  }
        else if (std::strcmp(t, "--no-normalize") == 0) { a.normalize = false; }
        else if (std::strcmp(t, "--worker")       == 0) { a.is_worker = true;  }
        else if (std::strcmp(t, "--selftest")     == 0) { a.selftest  = true;  }
        else if (std::strcmp(t, "-h") == 0 || std::strcmp(t, "--help") == 0) {
            print_usage(argv[0]);
            return false;
        } else {
            std::fprintf(stderr, "error: unknown arg: %s\n", t);
            print_usage(argv[0]);
            return false;
        }
    }

    if (bad) return false;

    if (a.rss_interval_ms <= 0 || a.rollup_interval_ms <= 0 || a.timeout_sec <= 0) {
        std::fprintf(stderr, "error: intervals and timeout must be positive\n");
        return false;
    }

    if (a.selftest) {
        if (a.selftest_alloc_mb <= 0) a.selftest_alloc_mb = 64;
        return true;
    }
    if (a.selftest_alloc_mb > 0) return true;   // worker in selftest mode

    if (a.model_path.empty() || a.inputs_path.empty()) {
        print_usage(argv[0]);
        return false;
    }
    return true;
}

// ---- Pipe I/O --------------------------------------------------------------

bool write_all(int fd, const void * buf, size_t n) {
    const auto * p = static_cast<const unsigned char *>(buf);
    while (n > 0) {
        const ssize_t w = ::write(fd, p, n);
        if (w < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        p += w;
        n -= static_cast<size_t>(w);
    }
    return true;
}

bool read_all(int fd, void * buf, size_t n) {
    auto * p = static_cast<unsigned char *>(buf);
    while (n > 0) {
        const ssize_t r = ::read(fd, p, n);
        if (r < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (r == 0) return false;      // worker died
        p += r;
        n -= static_cast<size_t>(r);
    }
    return true;
}

// Block until fd has data or the timeout expires. Guards against a wedged
// worker hanging the harness forever.
bool wait_readable(int fd, int timeout_sec) {
    pollfd pfd{};
    pfd.fd     = fd;
    pfd.events = POLLIN;
    for (;;) {
        const int rc = ::poll(&pfd, 1, timeout_sec * 1000);
        if (rc < 0 && errno == EINTR) continue;
        return rc > 0;
    }
}

// ---- Worker ----------------------------------------------------------------

std::vector<std::string> read_lines(const std::string & path) {
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "worker: cannot open inputs: %s\n", path.c_str());
        return {};
    }
    std::vector<std::string> out;
    std::string              line;
    while (std::getline(f, line)) {
        if (!line.empty()) out.push_back(line);
    }
    return out;
}

int run_worker(const Args & a) {
    std::vector<double> latencies;
    std::vector<std::string> inputs;

    nanoembed_model *   model = nullptr;
    nanoembed_context * ctx   = nullptr;

    const bool synthetic = a.selftest_alloc_mb > 0;

    if (!synthetic) {
        inputs = read_lines(a.inputs_path);
        if (inputs.empty()) {
            std::fprintf(stderr, "worker: empty inputs file\n");
            return 1;
        }

        model = nanoembed_load_model(a.model_path.c_str());
        if (!model) {
            std::fprintf(stderr, "worker: load_model failed: %s\n", nanoembed_last_error());
            return 1;
        }

        nanoembed_context_params p = nanoembed_context_default_params();
        p.pooling   = a.use_cls ? NANOEMBED_POOL_CLS : NANOEMBED_POOL_MEAN;
        p.normalize = a.normalize ? 1 : 0;
        if (a.threads     > 0) p.n_threads   = a.threads;
        if (a.max_seq_len > 0) p.max_seq_len = a.max_seq_len;

        ctx = nanoembed_new_context(model, p);
        if (!ctx) {
            std::fprintf(stderr, "worker: new_context failed: %s\n", nanoembed_last_error());
            nanoembed_free_model(model);
            return 1;
        }
    }

    const size_t n_expected = synthetic
        ? 1u
        : static_cast<size_t>(a.iter) * inputs.size();

    // Allocated AND touched before the window opens, so clear_refs folds it
    // into the baseline instead of letting it show up as window growth.
    latencies.assign(n_expected, 0.0);

    const int          H = synthetic ? 1 : nanoembed_n_embed(model);
    std::vector<float> out_buf(static_cast<size_t>(H));

    if (!synthetic) {
        for (int w = 0; w < a.warmup; ++w) {
            for (const auto & t : inputs) {
                if (nanoembed_embed(ctx, t.c_str(), out_buf.data()) != NANOEMBED_OK) {
                    std::fprintf(stderr, "worker: warmup embed failed: %s\n",
                                 nanoembed_last_error());
                    nanoembed_free_context(ctx);
                    nanoembed_free_model(model);
                    return 1;
                }
            }
        }
    }

    // Ready: everything that should sit in the baseline is now resident.
    if (!write_all(kRepFd, &kMsgReady, 1)) return 1;

    char go = 0;
    if (!read_all(kCtrlFd, &go, 1) || go != kMsgGo) return 1;

    // ---- Measurement window ----
    const ResourceCounters c0 = read_self_counters();
    const auto             t_start = Clock::now();

    size_t n_lat = 0;
    unsigned long long total_items = 0;

    if (synthetic) {
        // Stand-in workload with a known memory footprint, so the harness can
        // be verified without a GGUF: touch N MB, hold it long enough for the
        // slow smaps_rollup sampler to see it, then release.
        const size_t n = static_cast<size_t>(a.selftest_alloc_mb) << 20;
        std::vector<unsigned char> block(n, 1u);
        std::this_thread::sleep_for(std::chrono::milliseconds(2000));
        // Defeat any chance the optimizer elides the allocation.
        if (block[n / 2] != 1u) std::fprintf(stderr, "unreachable\n");
        latencies[n_lat++] = 0.0;
        total_items = 1;
    } else {
        for (int it = 0; it < a.iter; ++it) {
            for (const auto & t : inputs) {
                const auto t0 = Clock::now();
                const int  rc = nanoembed_embed(ctx, t.c_str(), out_buf.data());
                const auto t1 = Clock::now();
                if (rc != NANOEMBED_OK) {
                    std::fprintf(stderr, "worker: embed failed: %s\n",
                                 nanoembed_last_error());
                    nanoembed_free_context(ctx);
                    nanoembed_free_model(model);
                    return 1;
                }
                latencies[n_lat++] =
                    std::chrono::duration<double, std::milli>(t1 - t0).count();
                ++total_items;
            }
        }
    }

    const auto             t_end = Clock::now();
    const ResourceCounters c1    = read_self_counters();
    // ---- Window closed ----

    WorkerReport rep;
    rep.cpu_user_sec      = c1.cpu_user_sec - c0.cpu_user_sec;
    rep.cpu_sys_sec       = c1.cpu_sys_sec  - c0.cpu_sys_sec;
    rep.page_faults_major = c1.page_faults_major - c0.page_faults_major;
    rep.page_faults_minor = c1.page_faults_minor - c0.page_faults_minor;
    rep.io_read_bytes     = c1.io_read_bytes - c0.io_read_bytes;
    rep.total_items       = total_items;
    rep.wall_sec          = std::chrono::duration<double>(t_end - t_start).count();
    rep.n_latencies       = n_lat;

    bool ok = write_all(kRepFd, &kMsgDone, 1) &&
              write_all(kRepFd, &rep, sizeof(rep)) &&
              write_all(kRepFd, latencies.data(), n_lat * sizeof(double));

    // Stay alive until the parent has finished reading /proc/<us>/... — those
    // entries vanish the moment we exit.
    char bye = 0;
    ok = ok && read_all(kCtrlFd, &bye, 1) && bye == kMsgExit;

    if (ctx)   nanoembed_free_context(ctx);
    if (model) nanoembed_free_model(model);
    return ok ? 0 : 1;
}

// ---- Parent ----------------------------------------------------------------

struct Measurement {
    size_t    peak_rss_lifetime = 0;
    size_t    peak_rss_window   = 0;
    MemSample baseline;
    MemSample final_sample;

    double rss_avg = 0.0, rss_max = 0.0;
    double pss_avg = 0.0, pss_max = 0.0;
    double uss_avg = 0.0, uss_max = 0.0;

    size_t rss_samples    = 0;
    size_t rollup_samples = 0;
    bool   hwm_reset      = false;

    WorkerReport        report;
    std::vector<double> latencies;
};

// Build the worker's argv. Everything the worker needs travels here, since
// exec means it inherits no memory from us.
std::vector<std::string> worker_argv(const Args & a) {
    std::vector<std::string> v = {"nanoembed-bench", "--worker"};
    auto add = [&](const char * k, const std::string & val) {
        v.push_back(k);
        v.push_back(val);
    };
    if (a.selftest_alloc_mb > 0) {
        add("--selftest-alloc-mb", std::to_string(a.selftest_alloc_mb));
        return v;
    }
    add("--model",       a.model_path);
    add("--inputs",      a.inputs_path);
    add("--warmup",      std::to_string(a.warmup));
    add("--iter",        std::to_string(a.iter));
    add("--threads",     std::to_string(a.threads));
    if (a.max_seq_len > 0) add("--max-seq-len", std::to_string(a.max_seq_len));
    if (a.use_cls)    v.push_back("--cls");
    if (!a.normalize) v.push_back("--no-normalize");
    return v;
}

[[noreturn]] void exec_worker(const Args & a, int ctrl_read, int rep_write) {
    // Move both fds out of the 3/4 range first, so assigning one cannot clobber
    // the other regardless of what numbers pipe() handed out.
    const int c = ::fcntl(ctrl_read, F_DUPFD, 10);
    const int r = ::fcntl(rep_write, F_DUPFD, 10);
    if (c < 0 || r < 0) _exit(127);
    if (::dup2(c, kCtrlFd) < 0 || ::dup2(r, kRepFd) < 0) _exit(127);

    // Everything above kRepFd is the parent's business.
    for (int fd = kRepFd + 1; fd < 256; ++fd) ::close(fd);

    const std::vector<std::string> argv_s = worker_argv(a);
    std::vector<char *>            argv_c;
    argv_c.reserve(argv_s.size() + 1);
    for (const auto & s : argv_s) argv_c.push_back(const_cast<char *>(s.c_str()));
    argv_c.push_back(nullptr);

    // /proc/self/exe rather than argv[0]: immune to PATH and to symlinks.
    ::execv("/proc/self/exe", argv_c.data());
    std::fprintf(stderr, "parent: execv failed: %s\n", std::strerror(errno));
    _exit(127);
}

bool run_parent(const Args & a, Measurement & m) {
    int ctrl[2], rep[2];
    if (::pipe(ctrl) != 0 || ::pipe(rep) != 0) {
        std::fprintf(stderr, "parent: pipe failed: %s\n", std::strerror(errno));
        return false;
    }

    const pid_t pid = ::fork();
    if (pid < 0) {
        std::fprintf(stderr, "parent: fork failed: %s\n", std::strerror(errno));
        return false;
    }
    if (pid == 0) {
        ::close(ctrl[1]);
        ::close(rep[0]);
        exec_worker(a, ctrl[0], rep[1]);
    }

    ::close(ctrl[0]);
    ::close(rep[1]);

    auto fail = [&](const char * why) {
        std::fprintf(stderr, "parent: %s\n", why);
        ::kill(pid, SIGKILL);
        int st = 0;
        ::waitpid(pid, &st, 0);
        ::close(ctrl[1]);
        ::close(rep[0]);
        return false;
    };

    // --- wait for the worker to finish loading + warming up ---
    if (!wait_readable(rep[0], a.timeout_sec)) return fail("timed out waiting for worker READY");
    char msg = 0;
    if (!read_all(rep[0], &msg, 1) || msg != kMsgReady) return fail("worker died before READY");

    // Peak so far covers exec -> end of warmup. Must be read before the reset
    // destroys it.
    const size_t peak_through_warmup = read_peak_rss(pid);

    m.hwm_reset = reset_peak_rss(pid);
    if (!m.hwm_reset) {
        std::fprintf(stderr,
                     "parent: warning: could not reset VmHWM (clear_refs); "
                     "rss_peak_window_mb will include load and warmup\n");
    }
    m.baseline = read_mem_sample(pid);

    // --- sampler: cheap RSS often, expensive smaps_rollup rarely ---
    std::atomic<bool>      stop{false};
    std::vector<size_t>    rss_samples;
    std::vector<MemSample> rollup_samples;
    rss_samples.reserve(4096);
    rollup_samples.reserve(512);

    std::thread sampler([&] {
        auto next_rollup = Clock::now();
        while (!stop.load(std::memory_order_relaxed)) {
            const size_t rss = read_current_rss(pid);
            if (rss > 0) rss_samples.push_back(rss);

            const auto now = Clock::now();
            if (now >= next_rollup) {
                const MemSample s = read_mem_sample(pid);
                if (s.valid) rollup_samples.push_back(s);
                next_rollup = now + std::chrono::milliseconds(a.rollup_interval_ms);
            }
            std::this_thread::sleep_for(
                std::chrono::milliseconds(a.rss_interval_ms));
        }
    });

    auto stop_sampler = [&] {
        stop.store(true, std::memory_order_relaxed);
        sampler.join();
    };

    if (!write_all(ctrl[1], &kMsgGo, 1)) { stop_sampler(); return fail("could not signal GO"); }

    // --- wait for the window to close ---
    if (!wait_readable(rep[0], a.timeout_sec)) { stop_sampler(); return fail("timed out waiting for worker DONE"); }
    if (!read_all(rep[0], &msg, 1) || msg != kMsgDone) { stop_sampler(); return fail("worker died during measurement"); }
    if (!read_all(rep[0], &m.report, sizeof(m.report))) { stop_sampler(); return fail("truncated worker report"); }

    m.latencies.resize(static_cast<size_t>(m.report.n_latencies));
    if (!m.latencies.empty() &&
        !read_all(rep[0], m.latencies.data(), m.latencies.size() * sizeof(double))) {
        stop_sampler();
        return fail("truncated latency payload");
    }

    stop_sampler();

    // Worker is still alive (blocked on EXIT), so /proc/<pid> is still valid.
    m.peak_rss_window   = read_peak_rss(pid);
    m.final_sample      = read_mem_sample(pid);
    m.peak_rss_lifetime = std::max(peak_through_warmup, m.peak_rss_window);

    if (!write_all(ctrl[1], &kMsgExit, 1)) return fail("could not signal EXIT");

    int status = 0;
    if (::waitpid(pid, &status, 0) < 0) return fail("waitpid failed");
    ::close(ctrl[1]);
    ::close(rep[0]);

    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        std::fprintf(stderr, "parent: worker exited abnormally (status=%d)\n", status);
        return false;
    }

    // --- aggregate samples ---
    const double mb = 1024.0 * 1024.0;
    m.rss_samples    = rss_samples.size();
    m.rollup_samples = rollup_samples.size();

    if (!rss_samples.empty()) {
        double sum = 0.0;
        for (size_t v : rss_samples) {
            sum += static_cast<double>(v);
            m.rss_max = std::max(m.rss_max, static_cast<double>(v) / mb);
        }
        m.rss_avg = sum / static_cast<double>(rss_samples.size()) / mb;
    }
    if (!rollup_samples.empty()) {
        double ps = 0.0, us = 0.0;
        for (const auto & s : rollup_samples) {
            ps += static_cast<double>(s.pss_bytes);
            us += static_cast<double>(s.uss_bytes);
            m.pss_max = std::max(m.pss_max, static_cast<double>(s.pss_bytes) / mb);
            m.uss_max = std::max(m.uss_max, static_cast<double>(s.uss_bytes) / mb);
        }
        const auto n = static_cast<double>(rollup_samples.size());
        m.pss_avg = ps / n / mb;
        m.uss_avg = us / n / mb;
    }
    return true;
}

// ---- JSON ------------------------------------------------------------------

double pct(std::vector<double> & v, double q) {
    if (v.empty()) return 0.0;
    const size_t n = v.size();
    size_t i = static_cast<size_t>(q * static_cast<double>(n - 1));
    if (i >= n) i = n - 1;
    std::nth_element(v.begin(), v.begin() + static_cast<std::ptrdiff_t>(i), v.end());
    return v[i];
}

void escape_json_string(const std::string & s, std::string & out) {
    out.push_back('"');
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(c);
                }
        }
    }
    out.push_back('"');
}

struct JsonWriter {
    std::string j;
    std::string indent = "  ";

    void str(const char * k, const std::string & v, bool comma) {
        j += indent; j += '"'; j += k; j += "\": ";
        std::string esc;
        escape_json_string(v, esc);
        j += esc;
        j += comma ? ",\n" : "\n";
    }
    void num_i(const char * k, long long v, bool comma) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%lld", v);
        j += indent; j += '"'; j += k; j += "\": "; j += buf;
        j += comma ? ",\n" : "\n";
    }
    void num_f(const char * k, double v, bool comma) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.6g", v);
        j += indent; j += '"'; j += k; j += "\": "; j += buf;
        j += comma ? ",\n" : "\n";
    }
    void num_b(const char * k, bool v, bool comma) {
        j += indent; j += '"'; j += k; j += "\": "; j += (v ? "true" : "false");
        j += comma ? ",\n" : "\n";
    }
    void open(const char * k) {
        j += indent; j += '"'; j += k; j += "\": {\n";
        indent += "  ";
    }
    void close(bool comma) {
        indent.resize(indent.size() - 2);
        j += indent; j += "}";
        j += comma ? ",\n" : "\n";
    }
};

std::string build_json(const Args & a, Measurement & m, const Environment & env) {
    const double mb = 1024.0 * 1024.0;

    const double p50 = pct(m.latencies, 0.50);
    const double p90 = pct(m.latencies, 0.90);
    const double p99 = pct(m.latencies, 0.99);
    const double throughput = m.report.wall_sec > 0.0
        ? static_cast<double>(m.report.total_items) / m.report.wall_sec
        : 0.0;

    JsonWriter w;
    w.j += "{\n";

    w.str  ("scenario",    a.scenario,   true);
    w.str  ("os",          "linux",      true);
    w.str  ("model",       a.model_path, true);
    w.str  ("inputs",      a.inputs_path, true);
    w.num_i("warmup",      a.warmup,     true);
    w.num_i("iter",        a.iter,       true);
    w.num_i("total_items", static_cast<long long>(m.report.total_items), true);
    w.num_i("threads",     a.threads,    true);
    w.str  ("pooling",     a.use_cls ? "cls" : "mean", true);
    w.num_i("normalize",   a.normalize ? 1 : 0, true);

    // Fingerprint of the machine. compare.py refuses to compare across
    // mismatched environments, which is what a bare number could never express.
    w.open("environment");
    w.str  ("kernel",          env.kernel,    true);
    w.str  ("cpu_model",       env.cpu_model, true);
    w.num_i("nproc",           env.nproc,     true);
    w.num_i("page_size_bytes", env.page_size, false);
    w.close(true);

    w.open("measurement");
    w.str  ("mode",               "fork-exec-isolated",  true);
    w.num_b("hwm_reset",          m.hwm_reset,           true);
    w.num_i("rss_interval_ms",    a.rss_interval_ms,     true);
    w.num_i("rollup_interval_ms", a.rollup_interval_ms,  true);
    w.num_i("rss_samples",        static_cast<long long>(m.rss_samples),    true);
    w.num_i("rollup_samples",     static_cast<long long>(m.rollup_samples), false);
    w.close(true);

    w.open("metrics");
    // Exact, kernel-tracked (VmHWM). Lifetime is the number the M4 gate is
    // stated against; window isolates steady-state embedding.
    w.num_f("rss_peak_lifetime_mb", static_cast<double>(m.peak_rss_lifetime) / mb, true);
    w.num_f("rss_peak_window_mb",   static_cast<double>(m.peak_rss_window)   / mb, true);
    w.num_f("rss_baseline_mb",      static_cast<double>(m.baseline.rss_bytes) / mb, true);
    w.num_f("rss_avg_mb",           m.rss_avg, true);
    w.num_f("rss_max_sampled_mb",   m.rss_max, true);
    // No kernel high-water mark exists for PSS/USS, so these are sample maxima.
    w.num_f("pss_baseline_mb",      static_cast<double>(m.baseline.pss_bytes) / mb, true);
    w.num_f("pss_avg_mb",           m.pss_avg, true);
    w.num_f("pss_peak_sampled_mb",  m.pss_max, true);
    w.num_f("uss_baseline_mb",      static_cast<double>(m.baseline.uss_bytes) / mb, true);
    w.num_f("uss_avg_mb",           m.uss_avg, true);
    w.num_f("uss_peak_sampled_mb",  m.uss_max, true);

    w.num_f("cpu_user_sec",         m.report.cpu_user_sec, true);
    w.num_f("cpu_sys_sec",          m.report.cpu_sys_sec,  true);
    w.num_i("page_faults_major",    m.report.page_faults_major, true);
    w.num_i("page_faults_minor",    m.report.page_faults_minor, true);
    w.num_i("io_read_bytes",        static_cast<long long>(m.report.io_read_bytes), true);
    w.num_f("wall_sec",             m.report.wall_sec, true);
    w.num_f("latency_p50_ms",       p50, true);
    w.num_f("latency_p90_ms",       p90, true);
    w.num_f("latency_p99_ms",       p99, true);
    w.num_f("throughput_items_per_sec", throughput, false);
    w.close(false);

    w.j += "}\n";
    return w.j;
}

// ---- Selftest --------------------------------------------------------------

// Verifies the harness itself, with no GGUF required — the only bench check
// that can run in CI. Dirties memory in the parent first, so a regression that
// reintroduced plain fork (or dropped exec) would show up as an inflated
// worker baseline.
int run_selftest(Args a) {
    constexpr int kParentDirtyMb = 64;

    std::vector<unsigned char> dirt(static_cast<size_t>(kParentDirtyMb) << 20, 7u);
    // Keep it resident and un-elided for the duration of the child's run.
    volatile unsigned char sink = dirt[dirt.size() / 2];
    (void) sink;

    a.scenario  = "selftest";
    a.warmup    = 0;
    a.iter      = 1;
    a.timeout_sec = std::min(a.timeout_sec, 120);

    Measurement m;
    if (!run_parent(a, m)) {
        std::fprintf(stderr, "selftest: FAIL harness could not run a worker\n");
        return 1;
    }

    const double mb       = 1024.0 * 1024.0;
    const double alloc_mb = static_cast<double>(a.selftest_alloc_mb);
    const double baseline = static_cast<double>(m.baseline.rss_bytes) / mb;
    const double window   = static_cast<double>(m.peak_rss_window) / mb;
    const double growth   = window - baseline;
    const double uss_growth =
        m.uss_max - static_cast<double>(m.baseline.uss_bytes) / mb;

    std::printf("selftest: parent dirtied %d MB, worker allocated %d MB\n",
                kParentDirtyMb, a.selftest_alloc_mb);
    std::printf("  rss_baseline      = %8.2f MB\n", baseline);
    std::printf("  rss_peak_window   = %8.2f MB  (growth %.2f MB)\n", window, growth);
    std::printf("  uss_peak_sampled  = %8.2f MB  (growth %.2f MB)\n", m.uss_max, uss_growth);
    std::printf("  hwm_reset=%s rss_samples=%zu rollup_samples=%zu\n",
                m.hwm_reset ? "true" : "false", m.rss_samples, m.rollup_samples);

    int failures = 0;
    auto check = [&](bool ok, const char * what) {
        std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
        if (!ok) ++failures;
    };

    // The load-bearing assertion: exec gave the worker a fresh address space,
    // so the parent's 64 MB is nowhere in its baseline.
    check(baseline < 32.0,
          "worker baseline is unaffected by parent-dirtied memory (< 32 MB)");
    check(m.hwm_reset, "VmHWM reset via clear_refs succeeded");
    check(growth > alloc_mb * 0.90 && growth < alloc_mb * 1.15,
          "windowed RSS growth matches the worker allocation (+/-15%)");
    check(m.rollup_samples >= 2, "smaps_rollup sampled at least twice");
    check(uss_growth > alloc_mb * 0.85,
          "sampled USS growth matches the worker allocation (>85%)");

    std::printf("selftest: %s\n", failures == 0 ? "OK" : "FAILED");
    return failures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char ** argv) {
    Args a;
    if (!parse_args(argc, argv, a)) return 2;

    if (a.is_worker)  return run_worker(a);
    if (a.selftest)   return run_selftest(a);

    Measurement m;
    if (!run_parent(a, m)) return 1;

    const std::string j = build_json(a, m, read_environment());
    if (a.out_path.empty()) {
        std::fputs(j.c_str(), stdout);
    } else {
        std::ofstream f(a.out_path);
        if (!f) {
            std::fprintf(stderr, "error: cannot write %s\n", a.out_path.c_str());
            return 1;
        }
        f << j;
    }
    return 0;
}
