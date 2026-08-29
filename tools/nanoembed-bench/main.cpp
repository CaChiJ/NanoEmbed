// nanoembed-bench: run one scenario, write JSON metrics to stdout / --out.
//
// Linux only. The measurement design depends on /proc/<pid>/{status,statm,
// smaps_rollup,clear_refs}, which have no portable equivalent.
//
// Usage:
//   nanoembed-bench --model PATH --inputs FILE
//                   [--scenario NAME] [--warmup N] [--iter N]
//                   [--cls] [--no-normalize] [--threads N]
//                   [--streaming] [--partition PRESET]
//                   [--max-seq-len N] [--out PATH]
//                   [--cache-state cold|warm] [--strict-cold]
//                   [--memory-profile]
//                   [--memory-profile-interval-ms N]
//                   [--raw-samples-out PATH]
//   nanoembed-bench --selftest [--selftest-alloc-mb N]
//                   [--memory-profile]
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

#include "cache_control.h"
#include "metrics.h"
#include "statistics.h"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
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

// Version 3 separates API-batch latency/throughput from per-item equivalents.
constexpr int    kResultSchemaVersion          = 3;
constexpr size_t kThroughputWindowSizeItems    = 10;
constexpr size_t kMinThroughputWindowsForStats = 2;

// What the worker ships back once the window closes. Both sides are the same
// binary, so the layout is identical by construction.
struct WorkerReport {
    double             cpu_user_sec      = 0.0;
    double             cpu_sys_sec       = 0.0;
    long long          page_faults_major = 0;
    long long          page_faults_minor = 0;
    unsigned long long io_read_bytes     = 0;
    unsigned long long total_items       = 0;
    unsigned long long total_batches     = 0;
    double             wall_sec          = 0.0;
    unsigned long long n_latencies       = 0;
    int                resolved_pooling  = NANOEMBED_POOL_MODEL_DEFAULT;
    int                resolved_threads  = -1; // -1 = public API cannot expose auto result
    int                resolved_max_seq_len = 0;
    int                resolved_max_batch = 0;
    int                resolved_normalize   = 0;
    int                requested_execution_mode = -1; // 0=eager, 1=streaming
    int                resolved_execution_mode  = -1;
    int                strict_mode_context_creation_succeeded = 0;
    int                execution_mode_contract_version = 1;
    unsigned long long warmup_items         = 0;
    // Cold-only wall-clock phases. Negative means not applicable/not measured.
    double             model_load_ms                = -1.0;
    double             context_create_ms            = -1.0;
    double             first_request_latency_ms     = -1.0;
    double             startup_to_first_result_ms   = -1.0;
};

// ---- Args ------------------------------------------------------------------

// Names both requested and worker-resolved pooling values.
const char * pool_name(nanoembed_pool_type p) {
    switch (p) {
        case NANOEMBED_POOL_MEAN: return "mean";
        case NANOEMBED_POOL_CLS:  return "cls";
        case NANOEMBED_POOL_LAST: return "last";
        default:                  return "model-default";
    }
}

enum class CacheState {
    Warm,
    Cold,
};

const char * cache_state_name(CacheState state) {
    return state == CacheState::Cold ? "cold" : "warm";
}

const char * execution_mode_name(bool streaming) {
    return streaming ? "streaming" : "eager";
}

struct Args {
    std::string model_path;
    std::string inputs_path;
    std::string scenario = "single_short_f16";
    std::string out_path;              // empty = stdout
    std::string raw_samples_out_path;  // empty = do not persist raw latencies
    int  warmup             = 5;
    int  iter               = 50;
    int  threads            = 0;       // 0 = auto
    int  max_seq_len        = 0;       // 0 = no CLI override (library default)
    int  batch_size         = 1;       // items per measured API batch
    int  max_batch          = 64;      // context subdivision limit
    bool batch_control      = false;   // M4-style sequential calls, one timed batch window
    // Unset means the model's own pooling, matching the library default.
    // Naming one here would mean-pool a last-token model and time a graph the
    // library never builds.
    nanoembed_pool_type pooling = NANOEMBED_POOL_MODEL_DEFAULT;
    bool normalize          = true;
    bool streaming          = false;
    // Which streaming partition preset to select. Empty leaves the library's
    // own default. Not a context parameter -- the library reads it from the
    // environment -- so the worker exports it before loading the model.
    std::string partition;
    bool memory_profile     = false;
    int  memory_profile_interval_ms = 25;
    int  timeout_sec        = 1800;
    CacheState cache_state  = CacheState::Warm;
    bool strict_cold        = false;

    bool is_worker          = false;
    bool selftest           = false;
    int  selftest_alloc_mb  = 0;       // >0 = worker allocates instead of embedding
};

void print_usage(const char * prog) {
    std::fprintf(stderr,
        "usage: %s --model PATH --inputs FILE [--scenario NAME] [--warmup N]\n"
        "       [--iter N] [--cls] [--no-normalize] [--threads N]\n"
        "       [--streaming] [--partition PRESET]\n"
        "       [--batch-size N] [--max-batch N] [--batch-control]\n"
        "       [--max-seq-len N] [--out PATH]\n"
        "       [--cache-state cold|warm] [--strict-cold]\n"
        "       [--memory-profile] [--memory-profile-interval-ms N]\n"
        "       [--raw-samples-out PATH]\n"
        "       [--timeout-sec N]\n"
        "       %s --selftest [--selftest-alloc-mb N] [--memory-profile]\n",
        prog, prog);
}

bool parse_args(int argc, char ** argv, Args & a) {
    bool bad = false;
    bool warmup_explicit = false;
    bool iter_explicit   = false;

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
        else if (str_opt("--raw-samples-out", a.raw_samples_out_path)) { }
        else if (std::strcmp(t, "--warmup") == 0) {
            const char * v = need_val(i);
            if (v) a.warmup = std::atoi(v);
            warmup_explicit = true;
        }
        else if (std::strcmp(t, "--iter") == 0) {
            const char * v = need_val(i);
            if (v) a.iter = std::atoi(v);
            iter_explicit = true;
        }
        else if (int_opt("--threads",            a.threads))            { }
        else if (int_opt("--max-seq-len",        a.max_seq_len))        { }
        else if (int_opt("--batch-size",         a.batch_size))         { }
        else if (int_opt("--max-batch",          a.max_batch))          { }
        else if (int_opt("--memory-profile-interval-ms",
                         a.memory_profile_interval_ms))                  { }
        // Backward-compatible alias for pre-A3 invocations. It configures the
        // cadence but deliberately does not turn profiling on.
        else if (int_opt("--rollup-interval-ms",
                         a.memory_profile_interval_ms))                  { }
        else if (int_opt("--timeout-sec",        a.timeout_sec))        { }
        else if (int_opt("--selftest-alloc-mb",  a.selftest_alloc_mb))  { }
        else if (std::strcmp(t, "--mean")         == 0) { a.pooling = NANOEMBED_POOL_MEAN; }
        else if (std::strcmp(t, "--cls")          == 0) { a.pooling = NANOEMBED_POOL_CLS;  }
        else if (std::strcmp(t, "--last")         == 0) { a.pooling = NANOEMBED_POOL_LAST; }
        else if (std::strcmp(t, "--no-normalize") == 0) { a.normalize = false; }
        else if (std::strcmp(t, "--streaming")      == 0) { a.streaming = true; }
        else if (std::strcmp(t, "--batch-control")  == 0) { a.batch_control = true; }
        else if (std::strcmp(t, "--partition")      == 0 && i + 1 < argc) { a.partition = argv[++i]; }
        else if (std::strcmp(t, "--memory-profile") == 0) { a.memory_profile = true; }
        else if (std::strcmp(t, "--strict-cold")    == 0) { a.strict_cold = true; }
        else if (std::strcmp(t, "--cache-state") == 0) {
            const char * v = need_val(i);
            if (v && std::strcmp(v, "cold") == 0) {
                a.cache_state = CacheState::Cold;
            } else if (v && std::strcmp(v, "warm") == 0) {
                a.cache_state = CacheState::Warm;
            } else if (v) {
                std::fprintf(stderr,
                             "error: --cache-state must be cold or warm\n");
                bad = true;
            }
        }
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

    if (a.cache_state == CacheState::Cold) {
        if (!warmup_explicit) a.warmup = 0;
        if (!iter_explicit)   a.iter   = 1;
    }

    if (a.memory_profile_interval_ms <= 0 || a.timeout_sec <= 0 ||
        a.warmup < 0 || a.iter <= 0 || a.batch_size <= 0 || a.max_batch <= 0) {
        std::fprintf(stderr,
                     "error: interval, timeout and iter must be positive; "
                     "warmup must be non-negative\n");
        return false;
    }
    if (a.cache_state == CacheState::Cold &&
        (a.warmup != 0 || a.iter != 1)) {
        std::fprintf(stderr,
                     "error: cold execution requires --warmup 0 --iter 1; "
                     "the runner creates one fresh worker per selected input\n");
        return false;
    }
    // Cold execution measures first use. One fresh worker still performs
    // exactly one timed request, but that request may be a batch: the runner
    // hands each worker one sub-batch worth of inputs. Later items in the
    // batch legitimately reuse pages the first item faulted in -- that reuse
    // is the effect layer batching exists to produce, not measurement bleed.
    if (a.strict_cold && a.cache_state != CacheState::Cold) {
        std::fprintf(stderr,
                     "error: --strict-cold requires --cache-state cold\n");
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
    if (a.cache_state == CacheState::Warm && a.warmup == 0) {
        std::fprintf(stderr,
                     "error: warm execution requires at least one warmup pass; "
                     "use --cache-state cold for first-use measurement\n");
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
    std::vector<uint32_t> batch_item_counts;
    std::vector<std::string> inputs;
    std::vector<float> out_buf(a.selftest_alloc_mb > 0 ? 1u : 0u);

    nanoembed_model *   model = nullptr;
    nanoembed_context * ctx   = nullptr;

    nanoembed_pool_type resolved_pooling = NANOEMBED_POOL_MODEL_DEFAULT;
    int resolved_threads    = -1;
    int resolved_max_seq_len = 0;
    int resolved_max_batch = 0;
    int resolved_normalize   = 0;

    const bool synthetic = a.selftest_alloc_mb > 0;
    const bool cold = !synthetic && a.cache_state == CacheState::Cold;

    if (!synthetic) {
        inputs = read_lines(a.inputs_path);
        if (inputs.empty()) {
            std::fprintf(stderr, "worker: empty inputs file\n");
            return 1;
        }
        if (cold && inputs.size() > static_cast<size_t>(a.batch_size)) {
            std::fprintf(stderr,
                         "worker: cold execution runs exactly one batch; got %zu "
                         "inputs for batch size %d\n",
                         inputs.size(), a.batch_size);
            return 1;
        }
    }

    double model_load_ms     = -1.0;
    double context_create_ms = -1.0;
    auto load_model_and_context = [&]() -> bool {
        // Set before the model is loaded: the partition is resolved once, at
        // model initialization, and never re-read.
        if (!a.partition.empty()) {
            setenv("NANOEMBED_STREAMING_PARTITION", a.partition.c_str(), 1);
        }
        const auto load_start = Clock::now();
        model = nanoembed_load_model(a.model_path.c_str());
        const auto load_end = Clock::now();
        model_load_ms =
            std::chrono::duration<double, std::milli>(load_end - load_start).count();
        if (!model) {
            std::fprintf(stderr, "worker: load_model failed: %s\n",
                         nanoembed_last_error());
            return false;
        }

        nanoembed_context_params p = nanoembed_context_default_params();
        p.pooling   = a.pooling;
        p.normalize = a.normalize ? 1 : 0;
        p.use_streaming = a.streaming ? 1 : 0;
        p.max_batch = a.max_batch;
        if (a.threads     > 0) p.n_threads   = a.threads;
        if (a.max_seq_len > 0) p.max_seq_len = a.max_seq_len;

        const auto context_start = Clock::now();
        ctx = nanoembed_new_context(model, p);
        const auto context_end = Clock::now();
        context_create_ms = std::chrono::duration<double, std::milli>(
            context_end - context_start).count();
        if (!ctx) {
            std::fprintf(stderr, "worker: new_context failed: %s\n",
                         nanoembed_last_error());
            nanoembed_free_model(model);
            model = nullptr;
            return false;
        }

        // Resolve only facts available through the public ABI. In particular,
        // auto thread selection is internal to Embedder and must remain null
        // instead of being guessed here.
        resolved_pooling =
            p.pooling == NANOEMBED_POOL_MODEL_DEFAULT
                ? nanoembed_model_default_pooling(model)
                : p.pooling;
        const int model_max_seq_len = nanoembed_model_max_seq_len(model);
        if (model_max_seq_len <= 0) {
            std::fprintf(stderr, "worker: model max_seq_len query failed: %s\n",
                         nanoembed_last_error());
            nanoembed_free_context(ctx);
            nanoembed_free_model(model);
            ctx = nullptr;
            model = nullptr;
            return false;
        }

        resolved_threads     = p.n_threads > 0 ? p.n_threads : -1;
        resolved_max_seq_len = std::min(p.max_seq_len, model_max_seq_len);
        resolved_max_batch   = p.max_batch;
        resolved_normalize   = p.normalize != 0 ? 1 : 0;
        return true;
    };

    const size_t batches_per_iteration = synthetic || cold
        ? 1u
        : 1u + (inputs.size() - 1u) / static_cast<size_t>(a.batch_size);
    const size_t n_expected = synthetic || cold
        ? 1u
        : static_cast<size_t>(a.iter) * batches_per_iteration;
    latencies.assign(n_expected, 0.0);
    batch_item_counts.assign(n_expected, 0);
    unsigned long long warmup_items = 0;

    std::vector<const char *> batch_ptrs;
    batch_ptrs.reserve(static_cast<size_t>(a.batch_size));
    auto prepare_output = [&] {
        const size_t H = static_cast<size_t>(nanoembed_n_embed(model));
        if (H == 0 || static_cast<size_t>(a.batch_size) >
                          std::numeric_limits<size_t>::max() / H) {
            throw std::runtime_error("benchmark batch output size overflow");
        }
        out_buf.resize(H * static_cast<size_t>(a.batch_size));
    };
    auto execute_batch = [&](size_t begin, size_t end) -> int {
        batch_ptrs.clear();
        for (size_t i = begin; i < end; ++i) batch_ptrs.push_back(inputs[i].c_str());
        if (!a.batch_control) {
            return nanoembed_embed_batch(
                ctx, batch_ptrs.data(), static_cast<int>(batch_ptrs.size()), out_buf.data());
        }
        const size_t H = static_cast<size_t>(nanoembed_n_embed(model));
        for (size_t i = 0; i < batch_ptrs.size(); ++i) {
            const int rc = nanoembed_embed(ctx, batch_ptrs[i], out_buf.data() + i * H);
            if (rc != NANOEMBED_OK) return rc;
        }
        return NANOEMBED_OK;
    };

    // Warm mode deliberately loads, creates the context, and runs every
    // selected warmup input before READY. Cold mode sends READY first: GO then
    // scopes counters, memory profiling and VmHWM across load through result.
    if (!synthetic && !cold) {
        if (!load_model_and_context()) return 1;
        prepare_output();
        for (int w = 0; w < a.warmup; ++w) {
            for (size_t begin = 0; begin < inputs.size();
                 begin += static_cast<size_t>(a.batch_size)) {
                const size_t end = std::min(
                    begin + static_cast<size_t>(a.batch_size), inputs.size());
                if (execute_batch(begin, end) != NANOEMBED_OK) {
                    std::fprintf(stderr, "worker: warmup embed failed: %s\n",
                                 nanoembed_last_error());
                    nanoembed_free_context(ctx);
                    nanoembed_free_model(model);
                    return 1;
                }
                warmup_items += end - begin;
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
    unsigned long long total_batches = 0;

    double first_request_latency_ms = -1.0;
    if (synthetic) {
        // Stand-in workload with a known memory footprint, so the harness can
        // be verified without a GGUF: touch N MB, hold it long enough for the
        // detailed sampler to see it, then release. 200 ms is deliberately
        // short-lived relative to the old 500 ms cadence, but long enough for
        // a requested 10-25 ms profile interval to observe reliably.
        const size_t n = static_cast<size_t>(a.selftest_alloc_mb) << 20;
        std::vector<unsigned char> block(n, 1u);
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        // Defeat any chance the optimizer elides the allocation.
        if (block[n / 2] != 1u) std::fprintf(stderr, "unreachable\n");
        latencies[n_lat++] = 0.0;
        batch_item_counts[0] = 1;
        total_items = 1;
        total_batches = 1;
    } else if (cold) {
        if (!load_model_and_context()) return 1;
        prepare_output();
        // One timed request over every input this worker was given. At batch
        // size 1 that is the historical single-item cold measurement.
        const auto t0 = Clock::now();
        const int rc = execute_batch(0, inputs.size());
        const auto t1 = Clock::now();
        if (rc != NANOEMBED_OK) {
            std::fprintf(stderr, "worker: first cold embed failed: %s\n",
                         nanoembed_last_error());
            nanoembed_free_context(ctx);
            nanoembed_free_model(model);
            return 1;
        }
        first_request_latency_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        latencies[n_lat++] = first_request_latency_ms;
        batch_item_counts[0] = static_cast<uint32_t>(inputs.size());
        total_items = inputs.size();
        total_batches = 1;
    } else {
        for (int it = 0; it < a.iter; ++it) {
            for (size_t begin = 0; begin < inputs.size();
                 begin += static_cast<size_t>(a.batch_size)) {
                const size_t end = std::min(
                    begin + static_cast<size_t>(a.batch_size), inputs.size());
                const auto t0 = Clock::now();
                const int  rc = execute_batch(begin, end);
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
                batch_item_counts[n_lat - 1] = static_cast<uint32_t>(end - begin);
                total_items += end - begin;
                ++total_batches;
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
    rep.total_batches     = total_batches;
    rep.wall_sec          = std::chrono::duration<double>(t_end - t_start).count();
    rep.n_latencies       = n_lat;
    rep.resolved_pooling  = resolved_pooling;
    rep.resolved_threads  = resolved_threads;
    rep.resolved_max_seq_len = resolved_max_seq_len;
    rep.resolved_max_batch = resolved_max_batch;
    rep.resolved_normalize   = resolved_normalize;
    if (!synthetic) {
        rep.requested_execution_mode = a.streaming ? 1 : 0;
        // nanoembed_new_context has a strict no-fallback contract: successful
        // creation resolves exactly the requested selector or fails.
        rep.resolved_execution_mode = a.streaming ? 1 : 0;
        rep.strict_mode_context_creation_succeeded = ctx != nullptr ? 1 : 0;
    }
    rep.warmup_items         = warmup_items;
    if (cold) {
        rep.model_load_ms              = model_load_ms;
        rep.context_create_ms          = context_create_ms;
        rep.first_request_latency_ms   = first_request_latency_ms;
        // Starts immediately after GO and ends when the first embed returns.
        // It excludes input-file parsing, pipe synchronization and cache
        // eviction, but includes model load, context construction and all
        // worker-side setup on the critical path to the result.
        rep.startup_to_first_result_ms = rep.wall_sec * 1000.0;
    }

    bool ok = write_all(kRepFd, &kMsgDone, 1) &&
              write_all(kRepFd, &rep, sizeof(rep)) &&
              write_all(kRepFd, latencies.data(), n_lat * sizeof(double)) &&
              write_all(kRepFd, batch_item_counts.data(), n_lat * sizeof(uint32_t));

    // Stay alive until the parent has finished reading /proc/<us>/... — those
    // entries vanish the moment we exit.
    char bye = 0;
    ok = ok && read_all(kCtrlFd, &bye, 1);
    if (ok) ok = bye == kMsgExit;

    if (ctx)   nanoembed_free_context(ctx);
    if (model) nanoembed_free_model(model);
    return ok ? 0 : 1;
}

// ---- Parent ----------------------------------------------------------------

struct TimedMemSample {
    std::string sample_role;
    MemSample sample;
    long long monotonic_timestamp_ns = 0;
    double read_duration_ms = 0.0;
};

long long monotonic_timestamp_ns(Clock::time_point value) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        value.time_since_epoch()).count();
}

struct Measurement {
    size_t    peak_rss_lifetime = 0;
    size_t    peak_rss_window   = 0;
    MemSample baseline;
    MemSample final_sample;
    MemSampleSummary sampled_memory;

    size_t memory_profile_samples_requested = 0;
    size_t memory_profile_samples_attempted = 0;
    size_t memory_profile_samples_effective = 0;
    bool   memory_profile_final_collected   = false;
    bool   hwm_reset                        = false;

    CacheEvictionResult cache_eviction;

    std::vector<double> memory_profile_read_durations_ms;
    std::vector<TimedMemSample> memory_profile_observations;

    long long go_monotonic_timestamp_ns          = 0;
    long long done_marker_monotonic_timestamp_ns = 0;
    bool      has_go_timestamp                   = false;
    bool      has_done_marker_timestamp          = false;

    WorkerReport        report;
    std::vector<double> latencies;
    std::vector<uint32_t> batch_item_counts;
};

MemSample read_current_rss_sample(int pid) {
    MemSample sample;
    sample.rss_bytes = read_current_rss(pid);
    sample.has_rss   = sample.rss_bytes > 0;
    sample.valid     = sample.has_rss;
    return sample;
}

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
    add("--batch-size",  std::to_string(a.batch_size));
    add("--max-batch",   std::to_string(a.max_batch));
    add("--cache-state", cache_state_name(a.cache_state));
    if (a.max_seq_len > 0) add("--max-seq-len", std::to_string(a.max_seq_len));
    switch (a.pooling) {
        case NANOEMBED_POOL_MEAN: v.push_back("--mean"); break;
        case NANOEMBED_POOL_CLS:  v.push_back("--cls");  break;
        case NANOEMBED_POOL_LAST: v.push_back("--last"); break;
        default: break;   // MODEL_DEFAULT: say nothing, let the worker decide
    }
    if (!a.normalize) v.push_back("--no-normalize");
    if (a.streaming) v.push_back("--streaming");
    if (a.batch_control) v.push_back("--batch-control");
    // The worker is exec'd and inherits no memory, so a knob missing here is
    // silently the default rather than an error.
    if (!a.partition.empty()) add("--partition", a.partition);
    return v;
}

[[noreturn]] void exec_worker(const Args & a, int ctrl_read, int rep_write) {
    // Move both fds out of the 3/4 range first, so assigning one cannot clobber
    // the other regardless of what numbers pipe() handed out.
    const int c = ::fcntl(ctrl_read, F_DUPFD, 10);
    const int r = ::fcntl(rep_write, F_DUPFD, 10);
    if (c < 0 || r < 0) _exit(127);
    if (::dup2(c, kCtrlFd) < 0 || ::dup2(r, kRepFd) < 0) _exit(127);

    // Everything above the report fd is the parent's business.
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
    if (a.cache_state == CacheState::Cold && a.selftest_alloc_mb <= 0) {
        // This is intentionally the last model-file operation before fork.
        // Verification describes the instant after fadvise and before the
        // worker exists; model loading may (and in M3 does) repopulate pages.
        m.cache_eviction = evict_file_pages(a.model_path);
        if (a.strict_cold && !m.cache_eviction.cold_cache_verified) {
            std::fprintf(stderr,
                         "parent: strict cold-cache verification failed: %s\n",
                         m.cache_eviction.status.c_str());
            return false;
        }
    }

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

    // Warm READY follows model load/context/warmup. Cold READY precedes all
    // model work, so GO opens the startup-to-first-result measurement window.
    if (!wait_readable(rep[0], a.timeout_sec)) return fail("timed out waiting for worker READY");
    char msg = 0;
    if (!read_all(rep[0], &msg, 1) || msg != kMsgReady) return fail("worker died before READY");

    // In warm mode this covers exec -> end of warmup. In cold mode it is the
    // pre-model baseline. It must be read before the reset destroys it.
    const size_t peak_through_warmup = read_peak_rss(pid);

    m.hwm_reset = reset_peak_rss(pid);
    if (!m.hwm_reset) {
        std::fprintf(stderr,
                     "parent: warning: could not reset VmHWM (clear_refs); "
                     "rss_peak_window_mb will include load and warmup\n");
    }
    // Detailed rollups are opt-in because reading smaps_rollup walks the page
    // table and can contend with mmap/page-fault work. The normal path only
    // uses the cheap statm boundary read here and never creates a sampler
    // thread.
    if (a.memory_profile) {
        const auto t0 = Clock::now();
        m.baseline = read_mem_sample(pid);
        const auto t1 = Clock::now();
        m.memory_profile_observations.push_back({
            "baseline", m.baseline, monotonic_timestamp_ns(t1),
            std::chrono::duration<double, std::milli>(t1 - t0).count(),
        });
    } else {
        m.baseline = read_current_rss_sample(pid);
    }

    std::atomic<bool>      stop_sampling{false};
    std::atomic<bool>      start_sampling{false};
    std::mutex             sampler_mutex;
    std::condition_variable sampler_cv;
    std::vector<MemSample> profile_samples;
    std::thread            sampler;

    profile_samples.reserve(512);
    m.memory_profile_read_durations_ms.reserve(512);

    if (a.memory_profile) {
        sampler = std::thread([&] {
            std::unique_lock<std::mutex> lock(sampler_mutex);
            sampler_cv.wait(lock, [&] {
                return start_sampling.load(std::memory_order_relaxed) ||
                       stop_sampling.load(std::memory_order_relaxed);
            });

            const auto interval =
                std::chrono::milliseconds(a.memory_profile_interval_ms);
            while (!stop_sampling.load(std::memory_order_relaxed)) {
                lock.unlock();
                const auto t0 = Clock::now();
                const MemSample sample = read_mem_sample(pid);
                const auto t1 = Clock::now();
                const double read_duration_ms =
                    std::chrono::duration<double, std::milli>(t1 - t0).count();
                m.memory_profile_read_durations_ms.push_back(read_duration_ms);
                m.memory_profile_observations.push_back({
                    "periodic", sample, monotonic_timestamp_ns(t1),
                    read_duration_ms,
                });
                if (sample.valid) profile_samples.push_back(sample);
                lock.lock();

                // Wake promptly on stop instead of making the parent wait a
                // full (possibly user-increased) sampling interval.
                if (sampler_cv.wait_for(lock, interval, [&] {
                        return stop_sampling.load(std::memory_order_relaxed);
                    })) {
                    break;
                }
            }
        });
    }

    auto stop_sampler = [&] {
        if (!sampler.joinable()) return;
        stop_sampling.store(true, std::memory_order_relaxed);
        sampler_cv.notify_all();
        sampler.join();
    };

    if (a.memory_profile) {
        m.go_monotonic_timestamp_ns = monotonic_timestamp_ns(Clock::now());
        m.has_go_timestamp = true;
    }
    if (!write_all(ctrl[1], &kMsgGo, 1)) {
        stop_sampler();
        return fail("could not signal GO");
    }
    if (a.memory_profile) {
        start_sampling.store(true, std::memory_order_relaxed);
        sampler_cv.notify_all();
    }

    // --- wait for the window to close ---
    if (!wait_readable(rep[0], a.timeout_sec)) { stop_sampler(); return fail("timed out waiting for worker DONE"); }
    if (!read_all(rep[0], &msg, 1) || msg != kMsgDone) { stop_sampler(); return fail("worker died during measurement"); }
    if (a.memory_profile) {
        m.done_marker_monotonic_timestamp_ns = monotonic_timestamp_ns(Clock::now());
        m.has_done_marker_timestamp = true;
    }
    if (!read_all(rep[0], &m.report, sizeof(m.report))) { stop_sampler(); return fail("truncated worker report"); }

    if (a.selftest_alloc_mb <= 0) {
        const int requested = a.streaming ? 1 : 0;
        if (m.report.execution_mode_contract_version != 1 ||
            m.report.strict_mode_context_creation_succeeded != 1 ||
            m.report.requested_execution_mode != requested ||
            m.report.resolved_execution_mode != requested) {
            stop_sampler();
            return fail("worker execution-mode resolution claim mismatch");
        }
    }

    m.latencies.resize(static_cast<size_t>(m.report.n_latencies));
    if (!m.latencies.empty() &&
        !read_all(rep[0], m.latencies.data(), m.latencies.size() * sizeof(double))) {
        stop_sampler();
        return fail("truncated latency payload");
    }
    m.batch_item_counts.resize(static_cast<size_t>(m.report.n_latencies));
    if (!m.batch_item_counts.empty() &&
        !read_all(rep[0], m.batch_item_counts.data(),
                  m.batch_item_counts.size() * sizeof(uint32_t))) {
        stop_sampler();
        return fail("truncated batch-item-count payload");
    }

    stop_sampler();

    // Worker is still alive (blocked on EXIT), so /proc/<pid> is still valid.
    m.peak_rss_window = read_peak_rss(pid);
    if (a.memory_profile) {
        const auto t0 = Clock::now();
        m.final_sample = read_mem_sample(pid);
        const auto t1 = Clock::now();
        m.memory_profile_read_durations_ms.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
        m.memory_profile_observations.push_back({
            "final", m.final_sample, monotonic_timestamp_ns(t1),
            std::chrono::duration<double, std::milli>(t1 - t0).count(),
        });
        m.memory_profile_final_collected = m.final_sample.valid;
        if (m.final_sample.valid) profile_samples.push_back(m.final_sample);
    } else {
        // Boundary-only statm keeps final RSS observable without ever walking
        // smaps_rollup in an authoritative performance run.
        m.final_sample = read_current_rss_sample(pid);
    }
    m.peak_rss_lifetime = std::max(peak_through_warmup, m.peak_rss_window);

    if (a.cache_state == CacheState::Cold && a.selftest_alloc_mb <= 0) {
        // This observation is descriptive only. In-memory M3 loading reads the
        // full GGUF, so a verified-cold pre-worker file will normally be warm
        // again here; it does not retroactively invalidate the start state.
        m.cache_eviction.after_worker = query_file_residency(a.model_path);
    }

    if (!write_all(ctrl[1], &kMsgExit, 1)) return fail("could not signal EXIT");

    int status = 0;
    if (::waitpid(pid, &status, 0) < 0) return fail("waitpid failed");
    ::close(ctrl[1]);
    ::close(rep[0]);

    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        std::fprintf(stderr, "parent: worker exited abnormally (status=%d)\n", status);
        return false;
    }

    if (a.memory_profile) {
        // One periodic observation is requested immediately and then once per
        // interval, plus the mandatory final observation. Actual attempts can
        // be lower when the OS deschedules the sampler or a rollup read itself
        // exceeds the requested cadence.
        const double wall_ms = m.report.wall_sec * 1000.0;
        const size_t periodic_requested = wall_ms > 0.0
            ? static_cast<size_t>(std::floor(
                  wall_ms / static_cast<double>(a.memory_profile_interval_ms))) + 1
            : 1;
        m.memory_profile_samples_requested = periodic_requested + 1;
        m.memory_profile_samples_attempted =
            m.memory_profile_read_durations_ms.size();
        m.memory_profile_samples_effective = profile_samples.size();
        // profile_samples already contains final_sample when it was valid.
        // Keeping aggregation pure makes this inclusion regression-testable.
        m.sampled_memory = summarize_mem_samples(profile_samples);
    }
    return true;
}

// ---- JSON ------------------------------------------------------------------

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
    void num_b_or_null(const char * k, bool available, bool v, bool comma) {
        if (available) num_b(k, v, comma);
        else           null(k, comma);
    }
    void null(const char * k, bool comma) {
        j += indent; j += '"'; j += k; j += "\": null";
        j += comma ? ",\n" : "\n";
    }
    void num_i_or_null(const char * k, bool available, long long v, bool comma) {
        if (available) num_i(k, v, comma);
        else           null(k, comma);
    }
    void num_f_or_null(const char * k, bool available, double v, bool comma) {
        if (available) num_f(k, v, comma);
        else           null(k, comma);
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

std::string build_json(const Args & a, const Measurement & m, const Environment & env) {
    const double mb = 1024.0 * 1024.0;
    const bool cold_requested = a.cache_state == CacheState::Cold;
    const char * effective_cache_regime = !cold_requested
        ? "warm"
        : (m.cache_eviction.cold_cache_verified ? "cold" : "cold-unverified");
    const bool execution_mode_applicable = a.selftest_alloc_mb <= 0;
    const char * requested_execution_mode = execution_mode_applicable
        ? execution_mode_name(a.streaming) : "not-applicable";
    const char * resolved_execution_mode = execution_mode_applicable
        ? execution_mode_name(m.report.resolved_execution_mode == 1)
        : "not-applicable";

    const DistributionStats latency = describe_samples(m.latencies);
    std::vector<double> item_latencies;
    if (m.batch_item_counts.size() == m.latencies.size()) {
        item_latencies.reserve(m.latencies.size());
        for (size_t i = 0; i < m.latencies.size(); ++i) {
            if (m.batch_item_counts[i] != 0) {
                item_latencies.push_back(
                    m.latencies[i] / static_cast<double>(m.batch_item_counts[i]));
            }
        }
    }
    const DistributionStats item_latency = describe_samples(item_latencies);
    const bool throughput_available =
        m.report.wall_sec > 0.0 && m.report.total_items > 0;
    const double throughput = throughput_available
        ? static_cast<double>(m.report.total_items) / m.report.wall_sec
        : 0.0;
    const bool batch_throughput_available =
        m.report.wall_sec > 0.0 && m.report.total_batches > 0;
    const double batch_throughput = batch_throughput_available
        ? static_cast<double>(m.report.total_batches) / m.report.wall_sec
        : 0.0;
    const std::vector<double> window_rates = fixed_item_window_throughputs(
        m.latencies, kThroughputWindowSizeItems);
    const DistributionStats window_throughput = describe_samples(window_rates);
    const size_t complete_window_count =
        m.latencies.size() / kThroughputWindowSizeItems;
    const bool window_input_valid =
        a.batch_size == 1 && latency.available &&
        window_rates.size() == complete_window_count;
    const bool window_stats_available =
        window_input_valid &&
        window_rates.size() >= kMinThroughputWindowsForStats;
    const char * window_status = !window_input_valid
        ? "unavailable"
        : (window_stats_available ? "collected" : "insufficient_samples");

    const bool peak_lifetime_available = m.peak_rss_lifetime > 0;
    const bool peak_window_available   = m.peak_rss_window > 0;
    const bool baseline_available      = m.baseline.valid && m.baseline.has_rss;
    const bool final_rss_available     = m.final_sample.valid && m.final_sample.has_rss;
    const bool rss_samples_available   =
        a.memory_profile && m.sampled_memory.rss.available();
    const bool pss_baseline_available  =
        a.memory_profile && m.baseline.valid && m.baseline.has_pss;
    const bool pss_final_available     =
        a.memory_profile && m.final_sample.valid && m.final_sample.has_pss;
    const bool uss_baseline_available  =
        a.memory_profile && m.baseline.valid && m.baseline.has_uss;
    const bool uss_final_available     =
        a.memory_profile && m.final_sample.valid && m.final_sample.has_uss;
    const bool pss_samples_available   =
        a.memory_profile && m.sampled_memory.pss.available();
    const bool uss_samples_available   =
        a.memory_profile && m.sampled_memory.uss.available();
    const DistributionStats sampler_read_duration =
        describe_samples(m.memory_profile_read_durations_ms);
    const bool profile_ratio_available =
        a.memory_profile && m.memory_profile_samples_attempted > 0;
    const double profile_valid_ratio = profile_ratio_available
        ? static_cast<double>(m.memory_profile_samples_effective) /
              static_cast<double>(m.memory_profile_samples_attempted)
        : 0.0;
    auto detailed_status = [&](bool available) {
        return !a.memory_profile
            ? "not_collected"
            : (available ? "collected" : "unavailable");
    };

    JsonWriter w;
    w.j += "{\n";

    auto write_distribution = [&](const char * name,
                                  const DistributionStats & stats,
                                  bool comma) {
        w.open(name);
        w.num_i("count", static_cast<long long>(stats.count), true);
        w.num_f_or_null("min_ms", stats.available, stats.min, true);
        w.num_f_or_null("max_ms", stats.available, stats.max, true);
        w.num_f_or_null("mean_ms", stats.available, stats.mean, true);
        w.num_f_or_null("p50_ms", stats.available, stats.p50, true);
        w.num_f_or_null("p90_ms", stats.available, stats.p90, true);
        w.num_f_or_null("p95_ms", stats.available, stats.p95, true);
        w.num_f_or_null("p99_ms", stats.available, stats.p99, true);
        w.num_f_or_null("stddev_ms", stats.available, stats.stddev, true);
        w.num_f_or_null("mad_ms", stats.available, stats.mad, false);
        w.close(comma);
    };

    w.num_i("schema_version", kResultSchemaVersion, true);
    w.str  ("scenario",    a.scenario,   true);
    w.str  ("os",          "linux",      true);
    w.str  ("model",       a.model_path, true);
    w.str  ("inputs",      a.inputs_path, true);
    w.num_i("warmup",      a.warmup,     true);
    w.num_i("iter",        a.iter,       true);
    w.num_i("total_items", static_cast<long long>(m.report.total_items), true);
    w.num_i("total_batches", static_cast<long long>(m.report.total_batches), true);
    w.num_i("batch_size", a.batch_size, true);
    w.num_i("max_batch", a.max_batch, true);
    w.num_i("threads",     a.threads,    true);
    w.str  ("pooling",     pool_name(a.pooling), true);
    w.num_i("normalize",   a.normalize ? 1 : 0, true);
    w.str  ("requested_execution_mode", requested_execution_mode, true);
    w.str  ("resolved_execution_mode", resolved_execution_mode, true);

    // The legacy flat fields above remain during the schema-v2 transition.
    // New consumers should use this explicit intent/effective-value split.
    w.open("settings");
    w.open("requested");
    w.num_i("warmup",    a.warmup, true);
    w.num_i("iter",      a.iter, true);
    w.num_i("threads",   a.threads, true);
    w.num_i("batch_size", a.batch_size, true);
    w.num_i("max_batch", a.max_batch, true);
    w.num_b("batch_control", a.batch_control, true);
    w.str  ("pooling",   pool_name(a.pooling), true);
    w.num_b("normalize", a.normalize, true);
    w.str  ("execution_mode", requested_execution_mode, true);
    // Empty means the library's own default. compare.py diffs the whole
    // requested-settings object, so a preset change shows up as an explained
    // identity difference instead of an unexplained metric shift.
    w.str  ("streaming_partition", a.partition.empty() ? "default" : a.partition, true);
    w.str  ("cache_state", cache_state_name(a.cache_state), true);
    w.num_b("strict_cold", a.strict_cold, true);
    w.num_b("memory_profile", a.memory_profile, true);
    w.num_i("memory_profile_interval_ms",
            a.memory_profile_interval_ms, true);
    w.num_i("timeout_sec", a.timeout_sec, true);
    w.num_b("raw_samples_requested", !a.raw_samples_out_path.empty(), true);
    w.num_i_or_null("max_seq_len", a.max_seq_len > 0, a.max_seq_len, true);
    w.str("max_seq_len_source", a.max_seq_len > 0 ? "cli" : "library-default", false);
    w.close(true);
    w.open("resolved");
    w.num_i_or_null("threads", m.report.resolved_threads > 0,
                    m.report.resolved_threads, true);
    w.num_i("max_batch", m.report.resolved_max_batch, true);
    w.str("threads_collection_status",
          m.report.resolved_threads > 0 ? "collected" : "unavailable", true);
    w.str("pooling", pool_name(static_cast<nanoembed_pool_type>(
              m.report.resolved_pooling)), true);
    w.num_b("normalize", m.report.resolved_normalize != 0, true);
    w.str("execution_mode", resolved_execution_mode, true);
    w.str("cache_regime", effective_cache_regime, true);
    w.num_i_or_null("max_seq_len", m.report.resolved_max_seq_len > 0,
                    m.report.resolved_max_seq_len, false);
    w.close(false);
    w.close(true);

    w.open("execution_mode_resolution");
    w.num_i("contract_version", m.report.execution_mode_contract_version, true);
    w.str("requested_execution_mode", requested_execution_mode, true);
    w.str("resolved_execution_mode", resolved_execution_mode, true);
    w.num_b("strict_no_fallback", execution_mode_applicable, true);
    w.num_b("context_creation_succeeded_with_exact_request",
            m.report.strict_mode_context_creation_succeeded == 1, true);
    w.str("evidence",
          "frozen nanoembed_context_params.use_streaming request plus successful strict context creation",
          false);
    w.close(true);

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
    w.str  ("cache_regime_requested", cache_state_name(a.cache_state), true);
    w.str  ("cache_regime",       effective_cache_regime, true);
    w.str  ("execution_shape",
            a.cache_state == CacheState::Cold
                ? "one-fresh-worker-one-first-inference"
                : "one-worker-warmup-then-timed-repetitions",
            true);
    w.num_i("warmup_items_executed",
            static_cast<long long>(m.report.warmup_items), true);
    w.str("latency_scope",
          a.cache_state == CacheState::Cold
              ? "first-inference-only-after-model-and-context-creation"
              : "timed-inference-window-warmup-excluded",
          true);
    w.str("throughput_scope",
          a.cache_state == CacheState::Cold
              ? "worker-startup-through-first-result-excludes-eviction-and-process-launch"
              : "timed-inference-window",
          true);
    w.str("resource_counter_scope",
          a.cache_state == CacheState::Cold
              ? "model-load-through-first-result"
              : "timed-inference-window",
          true);
    w.num_b("memory_profile_enabled", a.memory_profile, true);
    w.num_i("memory_profile_interval_requested_ms",
            a.memory_profile_interval_ms, true);
    w.num_i("memory_profile_samples_requested",
            static_cast<long long>(m.memory_profile_samples_requested), true);
    w.num_i("memory_profile_samples_attempted",
            static_cast<long long>(m.memory_profile_samples_attempted), true);
    w.num_i("memory_profile_samples_effective",
            static_cast<long long>(m.memory_profile_samples_effective), true);
    w.num_f_or_null("memory_profile_valid_sample_ratio",
                    profile_ratio_available, profile_valid_ratio, true);
    w.num_b("memory_profile_final_sample_collected",
            m.memory_profile_final_collected, true);
    w.num_b("memory_profile_final_in_aggregates",
            a.memory_profile && m.memory_profile_final_collected, true);
    w.str("latency_result_role",
          a.memory_profile ? "diagnostic" : "authoritative", true);
    w.num_b("memory_profile_may_perturb_memory_lifecycle",
            a.memory_profile, true);
    w.str("memory_profile_observer_effect",
          a.memory_profile
              ? "smaps_rollup page-table walks may perturb timing and page residency"
              : "not_applicable_detailed_sampling_disabled",
          true);
    w.num_f_or_null("memory_profile_read_duration_min_ms",
                    sampler_read_duration.available,
                    sampler_read_duration.min, true);
    w.num_f_or_null("memory_profile_read_duration_mean_ms",
                    sampler_read_duration.available,
                    sampler_read_duration.mean, true);
    w.num_f_or_null("memory_profile_read_duration_max_ms",
                    sampler_read_duration.available,
                    sampler_read_duration.max, true);
    w.str  ("percentile_method", "lower: floor(q * (count - 1))", true);
    w.str  ("stddev_kind", "population", true);
    w.num_b("hwm_reset",          m.hwm_reset,           true);
    // Legacy schema-v1 names remain readable, but periodic RSS now comes from
    // the same opt-in rollup observation rather than a separate statm poller.
    w.num_i_or_null("rss_interval_ms", a.memory_profile,
                    a.memory_profile_interval_ms, true);
    w.num_i_or_null("rollup_interval_ms", a.memory_profile,
                    a.memory_profile_interval_ms, true);
    w.num_i("rss_samples", static_cast<long long>(
                m.sampled_memory.rss.valid_samples), true);
    w.num_i("rollup_samples", static_cast<long long>(
                m.memory_profile_samples_effective), true);

    const CacheEvictionResult & cache = m.cache_eviction;
    auto write_residency = [&](const char * name,
                               const FileResidency & residency,
                               bool comma) {
        w.open(name);
        w.str("collection_status",
              !cold_requested
                  ? "not_requested"
                  : (residency.collected ? "collected" : "unavailable"),
              true);
        w.num_i_or_null("file_size_bytes", residency.collected,
                        static_cast<long long>(residency.file_size_bytes), true);
        w.num_i_or_null("page_size_bytes", residency.collected,
                        static_cast<long long>(residency.page_size_bytes), true);
        w.num_i_or_null("total_pages", residency.collected,
                        static_cast<long long>(residency.total_pages), true);
        w.num_i_or_null("resident_pages", residency.collected,
                        static_cast<long long>(residency.resident_pages), true);
        w.num_f_or_null("resident_percent", residency.collected,
                        residency.resident_percent, true);
        w.num_i_or_null("error_number",
                        cold_requested && residency.error_number != 0,
                        residency.error_number, true);
        w.str("status", cold_requested ? residency.status : "not_requested", false);
        w.close(comma);
    };

    w.open("cache_control");
    w.num_b("cold_cache_requested", cold_requested, true);
    w.num_b("strict_cold", a.strict_cold, true);
    w.num_b_or_null("platform_supported", cold_requested,
                    cache.platform_supported, true);
    w.num_b_or_null("eviction_call_succeeded", cold_requested,
                    cache.eviction_call_succeeded, true);
    w.num_i_or_null("eviction_error_number",
                    cold_requested && cache.eviction_error_number != 0,
                    cache.eviction_error_number, true);
    w.num_b_or_null("cold_cache_verified", cold_requested,
                    cache.cold_cache_verified, true);
    w.str("verification_definition",
          cold_requested
              ? "mincore observed zero resident model-file pages after fadvise and before worker fork"
              : "not_requested",
          true);
    w.str("verification_status", cold_requested ? cache.status : "not_requested", true);
    write_residency("before_eviction", cache.before_eviction, true);
    write_residency("after_eviction_before_worker",
                    cache.after_eviction_before_worker, true);
    write_residency("after_worker_load_and_first_result",
                    cache.after_worker, false);
    w.close(false);
    w.close(true);

    w.open("metrics");
    w.open("collection_status");
    w.str("rss_peak_lifetime", peak_lifetime_available ? "collected" : "unavailable", true);
    w.str("rss_peak_window",   peak_window_available   ? "collected" : "unavailable", true);
    w.str("rss_baseline",      baseline_available      ? "collected" : "unavailable", true);
    w.str("rss_final",         final_rss_available     ? "collected" : "unavailable", true);
    w.str("rss_sampled",       detailed_status(rss_samples_available), true);
    w.str("pss_baseline",      detailed_status(pss_baseline_available), true);
    w.str("pss_final",         detailed_status(pss_final_available), true);
    w.str("pss_sampled",       detailed_status(pss_samples_available), true);
    w.str("uss_baseline",      detailed_status(uss_baseline_available), true);
    w.str("uss_final",         detailed_status(uss_final_available), true);
    w.str("uss_sampled",       detailed_status(uss_samples_available), true);
    w.str("latency",           latency.available ? "collected" : "unavailable", true);
    w.str("single_request_items_per_sec",
          throughput_available ? "collected" : "unavailable", true);
    w.str("fixed_item_window_throughput", window_status, false);
    w.close(true);

    // Exact, kernel-tracked (VmHWM). Lifetime is the number the M4 gate is
    // stated against; window isolates steady-state embedding.
    w.num_f_or_null("rss_peak_lifetime_mb", peak_lifetime_available,
                    static_cast<double>(m.peak_rss_lifetime) / mb, true);
    w.num_f_or_null("rss_peak_window_mb", peak_window_available,
                    static_cast<double>(m.peak_rss_window) / mb, true);
    w.num_f_or_null("rss_baseline_mb", baseline_available,
                    static_cast<double>(m.baseline.rss_bytes) / mb, true);
    w.num_f_or_null("rss_final_mb", final_rss_available,
                    static_cast<double>(m.final_sample.rss_bytes) / mb, true);
    w.num_f_or_null("rss_avg_mb", rss_samples_available,
                    m.sampled_memory.rss.average_bytes / mb, true);
    w.num_f_or_null("rss_max_sampled_mb", rss_samples_available,
                    static_cast<double>(m.sampled_memory.rss.peak_bytes) / mb, true);
    w.num_f_or_null("rss_sampled_p50_mb", rss_samples_available,
                    static_cast<double>(m.sampled_memory.rss.p50_bytes) / mb, true);
    w.num_f_or_null("rss_sampled_p75_mb", rss_samples_available,
                    static_cast<double>(m.sampled_memory.rss.p75_bytes) / mb, true);
    w.num_f_or_null("rss_sampled_p90_mb", rss_samples_available,
                    static_cast<double>(m.sampled_memory.rss.p90_bytes) / mb, true);
    w.num_f_or_null("rss_sampled_p95_mb", rss_samples_available,
                    static_cast<double>(m.sampled_memory.rss.p95_bytes) / mb, true);
    w.num_f_or_null("rss_sampled_p99_mb", rss_samples_available,
                    static_cast<double>(m.sampled_memory.rss.p99_bytes) / mb, true);
    // No kernel high-water mark exists for PSS/USS, so these are sample maxima.
    w.num_f_or_null("pss_baseline_mb", pss_baseline_available,
                    static_cast<double>(m.baseline.pss_bytes) / mb, true);
    w.num_f_or_null("pss_final_mb", pss_final_available,
                    static_cast<double>(m.final_sample.pss_bytes) / mb, true);
    w.num_f_or_null("pss_avg_mb", pss_samples_available,
                    m.sampled_memory.pss.average_bytes / mb, true);
    w.num_f_or_null("pss_peak_sampled_mb", pss_samples_available,
                    static_cast<double>(m.sampled_memory.pss.peak_bytes) / mb, true);
    w.num_f_or_null("pss_sampled_p50_mb", pss_samples_available,
                    static_cast<double>(m.sampled_memory.pss.p50_bytes) / mb, true);
    w.num_f_or_null("pss_sampled_p75_mb", pss_samples_available,
                    static_cast<double>(m.sampled_memory.pss.p75_bytes) / mb, true);
    w.num_f_or_null("pss_sampled_p90_mb", pss_samples_available,
                    static_cast<double>(m.sampled_memory.pss.p90_bytes) / mb, true);
    w.num_f_or_null("pss_sampled_p95_mb", pss_samples_available,
                    static_cast<double>(m.sampled_memory.pss.p95_bytes) / mb, true);
    w.num_f_or_null("pss_sampled_p99_mb", pss_samples_available,
                    static_cast<double>(m.sampled_memory.pss.p99_bytes) / mb, true);
    w.num_f_or_null("uss_baseline_mb", uss_baseline_available,
                    static_cast<double>(m.baseline.uss_bytes) / mb, true);
    w.num_f_or_null("uss_final_mb", uss_final_available,
                    static_cast<double>(m.final_sample.uss_bytes) / mb, true);
    w.num_f_or_null("uss_avg_mb", uss_samples_available,
                    m.sampled_memory.uss.average_bytes / mb, true);
    w.num_f_or_null("uss_peak_sampled_mb", uss_samples_available,
                    static_cast<double>(m.sampled_memory.uss.peak_bytes) / mb, true);
    w.num_f_or_null("uss_sampled_p50_mb", uss_samples_available,
                    static_cast<double>(m.sampled_memory.uss.p50_bytes) / mb, true);
    w.num_f_or_null("uss_sampled_p75_mb", uss_samples_available,
                    static_cast<double>(m.sampled_memory.uss.p75_bytes) / mb, true);
    w.num_f_or_null("uss_sampled_p90_mb", uss_samples_available,
                    static_cast<double>(m.sampled_memory.uss.p90_bytes) / mb, true);
    w.num_f_or_null("uss_sampled_p95_mb", uss_samples_available,
                    static_cast<double>(m.sampled_memory.uss.p95_bytes) / mb, true);
    w.num_f_or_null("uss_sampled_p99_mb", uss_samples_available,
                    static_cast<double>(m.sampled_memory.uss.p99_bytes) / mb, true);

    auto write_breakdown = [&](const char * name,
                               bool baseline_field_available,
                               size_t baseline_bytes,
                               const SampledSizeSummary & sampled,
                               bool final_field_available,
                               size_t final_bytes,
                               bool comma) {
        w.open(name);
        w.str("collection_status", detailed_status(
                  baseline_field_available || sampled.available() ||
                  final_field_available), true);
        w.num_f_or_null("baseline_mb",
                        a.memory_profile && baseline_field_available,
                        static_cast<double>(baseline_bytes) / mb, true);
        w.num_f_or_null("average_mb",
                        a.memory_profile && sampled.available(),
                        sampled.average_bytes / mb, true);
        w.num_f_or_null("peak_sampled_mb",
                        a.memory_profile && sampled.available(),
                        static_cast<double>(sampled.peak_bytes) / mb, true);
        w.num_f_or_null("final_mb",
                        a.memory_profile && final_field_available,
                        static_cast<double>(final_bytes) / mb, false);
        w.close(comma);
    };

    w.open("memory_breakdown");
    write_breakdown("pss_anon", m.baseline.has_pss_anon,
                    m.baseline.pss_anon_bytes, m.sampled_memory.pss_anon,
                    m.final_sample.has_pss_anon, m.final_sample.pss_anon_bytes, true);
    write_breakdown("pss_file", m.baseline.has_pss_file,
                    m.baseline.pss_file_bytes, m.sampled_memory.pss_file,
                    m.final_sample.has_pss_file, m.final_sample.pss_file_bytes, true);
    write_breakdown("anonymous", m.baseline.has_anonymous,
                    m.baseline.anonymous_bytes, m.sampled_memory.anonymous,
                    m.final_sample.has_anonymous, m.final_sample.anonymous_bytes, true);
    write_breakdown("private_clean", m.baseline.has_private_clean,
                    m.baseline.private_clean_bytes, m.sampled_memory.private_clean,
                    m.final_sample.has_private_clean,
                    m.final_sample.private_clean_bytes, true);
    write_breakdown("private_dirty", m.baseline.has_private_dirty,
                    m.baseline.private_dirty_bytes, m.sampled_memory.private_dirty,
                    m.final_sample.has_private_dirty,
                    m.final_sample.private_dirty_bytes, true);
    write_breakdown("shared_clean", m.baseline.has_shared_clean,
                    m.baseline.shared_clean_bytes, m.sampled_memory.shared_clean,
                    m.final_sample.has_shared_clean,
                    m.final_sample.shared_clean_bytes, true);
    write_breakdown("shared_dirty", m.baseline.has_shared_dirty,
                    m.baseline.shared_dirty_bytes, m.sampled_memory.shared_dirty,
                    m.final_sample.has_shared_dirty,
                    m.final_sample.shared_dirty_bytes, false);
    w.close(true);

    w.num_f("cpu_user_sec",         m.report.cpu_user_sec, true);
    w.num_f("cpu_sys_sec",          m.report.cpu_sys_sec,  true);
    w.num_i("page_faults_major",    m.report.page_faults_major, true);
    w.num_i("page_faults_minor",    m.report.page_faults_minor, true);
    w.num_i("io_read_bytes",        static_cast<long long>(m.report.io_read_bytes), true);
    w.num_f("wall_sec",             m.report.wall_sec, true);
    w.num_i("latency_count",         static_cast<long long>(latency.count), true);
    w.num_f_or_null("latency_min_ms",    latency.available, latency.min, true);
    w.num_f_or_null("latency_max_ms",    latency.available, latency.max, true);
    w.num_f_or_null("latency_mean_ms",   latency.available, latency.mean, true);
    w.num_f_or_null("latency_p50_ms",    latency.available, latency.p50, true);
    w.num_f_or_null("latency_p90_ms",    latency.available, latency.p90, true);
    w.num_f_or_null("latency_p95_ms",    latency.available, latency.p95, true);
    w.num_f_or_null("latency_p99_ms",    latency.available, latency.p99, true);
    w.num_f_or_null("latency_stddev_ms", latency.available, latency.stddev, true);
    w.num_f_or_null("latency_mad_ms",    latency.available, latency.mad, true);
    w.num_f_or_null("batch_latency_p50_ms", latency.available, latency.p50, true);
    w.num_f_or_null("batch_latency_p90_ms", latency.available, latency.p90, true);
    w.num_f_or_null("batch_latency_p95_ms", latency.available, latency.p95, true);
    w.num_f_or_null("batch_latency_p99_ms", latency.available, latency.p99, true);
    w.num_f_or_null("item_latency_p50_ms", item_latency.available, item_latency.p50, true);
    w.num_f_or_null("item_latency_p90_ms", item_latency.available, item_latency.p90, true);
    w.num_f_or_null("item_latency_p95_ms", item_latency.available, item_latency.p95, true);
    w.num_f_or_null("item_latency_p99_ms", item_latency.available, item_latency.p99, true);
    w.num_f_or_null("batches_per_sec", batch_throughput_available,
                    batch_throughput, true);
    w.num_f_or_null("items_per_sec", throughput_available, throughput, true);
    w.num_f_or_null("single_request_items_per_sec",
                    throughput_available && a.batch_size == 1, throughput, true);
    w.num_f_or_null("page_faults_major_per_item", throughput_available,
                    static_cast<double>(m.report.page_faults_major) /
                        static_cast<double>(m.report.total_items), true);
    w.num_f_or_null("page_faults_minor_per_item", throughput_available,
                    static_cast<double>(m.report.page_faults_minor) /
                        static_cast<double>(m.report.total_items), true);
    w.num_f_or_null("io_read_bytes_per_item", throughput_available,
                    static_cast<double>(m.report.io_read_bytes) /
                        static_cast<double>(m.report.total_items), true);

    const bool cold_phases_available =
        a.cache_state == CacheState::Cold &&
        m.report.model_load_ms >= 0.0 &&
        m.report.context_create_ms >= 0.0 &&
        m.report.first_request_latency_ms >= 0.0 &&
        m.report.startup_to_first_result_ms >= 0.0;
    auto one_phase = [&](double value) {
        return describe_samples(cold_phases_available
            ? std::vector<double>{value}
            : std::vector<double>{});
    };
    w.open("cold_phase_timings");
    w.str("collection_status",
          a.cache_state != CacheState::Cold
              ? "not_applicable_warm"
              : (cold_phases_available ? "collected" : "unavailable"),
          true);
    w.str("startup_boundary",
          "worker GO through first embed result; excludes cache eviction, process launch, input parsing and pipe synchronization",
          true);
    w.num_i("fresh_worker_count", cold_phases_available ? 1 : 0, true);
    write_distribution("model_load_ms",
                       one_phase(m.report.model_load_ms), true);
    write_distribution("context_create_ms",
                       one_phase(m.report.context_create_ms), true);
    write_distribution("first_request_latency_ms",
                       one_phase(m.report.first_request_latency_ms), true);
    write_distribution("startup_to_first_result_ms",
                       one_phase(m.report.startup_to_first_result_ms), false);
    w.close(true);

    w.open("fixed_item_window_throughput");
    w.str("collection_status", window_status, true);
    w.num_b("canonical", false, true);
    w.str("source", "sum-of-per-request-latencies", true);
    w.num_i("window_size_items",
            static_cast<long long>(kThroughputWindowSizeItems), true);
    w.num_i("minimum_windows_for_statistics",
            static_cast<long long>(kMinThroughputWindowsForStats), true);
    w.num_i("complete_windows", static_cast<long long>(complete_window_count), true);
    w.num_i("dropped_tail_items", static_cast<long long>(
                m.latencies.size() % kThroughputWindowSizeItems), true);
    w.num_i("count", static_cast<long long>(window_rates.size()), true);
    w.num_f_or_null("min_items_per_sec", window_stats_available,
                    window_throughput.min, true);
    w.num_f_or_null("max_items_per_sec", window_stats_available,
                    window_throughput.max, true);
    w.num_f_or_null("mean_items_per_sec", window_stats_available,
                    window_throughput.mean, true);
    w.num_f_or_null("p50_items_per_sec", window_stats_available,
                    window_throughput.p50, true);
    w.num_f_or_null("p90_items_per_sec", window_stats_available,
                    window_throughput.p90, true);
    w.num_f_or_null("p95_items_per_sec", window_stats_available,
                    window_throughput.p95, true);
    w.num_f_or_null("p99_items_per_sec", window_stats_available,
                    window_throughput.p99, true);
    w.num_f_or_null("stddev_items_per_sec", window_stats_available,
                    window_throughput.stddev, true);
    w.num_f_or_null("mad_items_per_sec", window_stats_available,
                    window_throughput.mad, false);
    w.close(false);
    w.close(false);

    w.j += "}\n";
    return w.j;
}

std::string build_raw_samples_json(const Args & a, const Measurement & m) {
    std::ostringstream f;
    std::string escaped_scenario;
    escape_json_string(a.scenario, escaped_scenario);
    f << "{\n"
      << "  \"schema_version\": 2,\n"
      << "  \"scenario\": " << escaped_scenario << ",\n"
      << "  \"latency_unit\": \"ms\",\n"
      << "  \"latency_scope\": \"one requested API batch\",\n"
      << "  \"latency_ms\": [";
    f << std::setprecision(17);
    for (size_t i = 0; i < m.latencies.size(); ++i) {
        if (i != 0) f << ", ";
        f << m.latencies[i];
    }
    f << "],\n"
      << "  \"batch_item_counts\": [";
    for (size_t i = 0; i < m.batch_item_counts.size(); ++i) {
        if (i != 0) f << ", ";
        f << m.batch_item_counts[i];
    }
    f << "],\n"
      << "  \"memory_profile\": {\n"
      << "    \"schema_version\": 1,\n"
      << "    \"collection_status\": \""
      << (a.memory_profile ? "collected" : "disabled") << "\",\n"
      << "    \"clock_source\": \"std::chrono::steady_clock\",\n"
      << "    \"clock_scope\": \"process-local monotonic epoch; compare only within this native invocation\",\n"
      << "    \"timestamp_unit\": \"ns\",\n"
      << "    \"timestamp_semantics\": \"smaps_rollup read completion; subtract read_duration_ms for approximate read start\",\n"
      << "    \"done_marker_timestamp_semantics\": \"parent steady_clock immediately after the DONE byte is received\",\n"
      << "    \"go_timestamp_semantics\": \"parent steady_clock immediately before the GO byte is written\",\n"
      << "    \"elapsed_unit\": \"ms\",\n"
      << "    \"sample_values_unit\": \"bytes\",\n";

    auto nullable_integer = [&](bool available, long long value) {
        if (available) f << value;
        else           f << "null";
    };
    auto nullable_size = [&](bool available, size_t value) {
        if (available) f << value;
        else           f << "null";
    };

    f << "    \"go_monotonic_timestamp_ns\": ";
    nullable_integer(a.memory_profile && m.has_go_timestamp,
                     m.go_monotonic_timestamp_ns);
    f << ",\n    \"done_marker_monotonic_timestamp_ns\": ";
    nullable_integer(a.memory_profile && m.has_done_marker_timestamp,
                     m.done_marker_monotonic_timestamp_ns);
    f << ",\n    \"done_marker_elapsed_ms_from_go\": ";
    if (a.memory_profile && m.has_go_timestamp &&
        m.has_done_marker_timestamp) {
        f << static_cast<double>(m.done_marker_monotonic_timestamp_ns -
                                 m.go_monotonic_timestamp_ns) / 1e6;
    } else {
        f << "null";
    }
    f << ",\n"
      << "    \"sample_count\": "
      << (a.memory_profile ? m.memory_profile_observations.size() : 0)
      << ",\n"
      << "    \"aggregation_contract\": {\n"
      << "      \"included_sample_roles\": "
      << (a.memory_profile ? "[\"periodic\", \"final\"]" : "null")
      << ",\n"
      << "      \"baseline_excluded\": "
      << (a.memory_profile ? "true" : "null") << ",\n"
      << "      \"valid_samples_only\": "
      << (a.memory_profile ? "true" : "null") << ",\n"
      << "      \"final_included\": "
      << (a.memory_profile ? "true" : "null") << ",\n"
      << "      \"percentile_method\": "
      << (a.memory_profile
              ? "\"lower: floor(q * (count - 1))\""
              : "null") << "\n"
      << "    },\n"
      << "    \"samples\": ";

    if (!a.memory_profile) {
        f << "null\n";
    } else {
        std::vector<TimedMemSample> observations =
            m.memory_profile_observations;
        std::stable_sort(observations.begin(), observations.end(),
                         [](const TimedMemSample & lhs,
                            const TimedMemSample & rhs) {
                             return lhs.monotonic_timestamp_ns <
                                    rhs.monotonic_timestamp_ns;
                         });
        f << "[\n";
        for (size_t i = 0; i < observations.size(); ++i) {
            const TimedMemSample & observation = observations[i];
            const MemSample & sample = observation.sample;
            std::string escaped_role;
            escape_json_string(observation.sample_role, escaped_role);
            f << "      {\n"
              << "        \"sample_role\": " << escaped_role << ",\n"
              << "        \"monotonic_timestamp_ns\": "
              << observation.monotonic_timestamp_ns << ",\n"
              << "        \"elapsed_ms_from_go\": "
              << static_cast<double>(observation.monotonic_timestamp_ns -
                                     m.go_monotonic_timestamp_ns) / 1e6
              << ",\n"
              << "        \"read_duration_ms\": "
              << observation.read_duration_ms << ",\n"
              << "        \"valid\": "
              << (sample.valid ? "true" : "false") << ",\n"
              << "        \"rss_bytes\": ";
            nullable_size(sample.has_rss, sample.rss_bytes);
            f << ",\n        \"pss_bytes\": ";
            nullable_size(sample.has_pss, sample.pss_bytes);
            f << ",\n        \"uss_bytes\": ";
            nullable_size(sample.has_uss, sample.uss_bytes);
            f << ",\n        \"breakdown_bytes\": {\n"
              << "          \"pss_anon_bytes\": ";
            nullable_size(sample.has_pss_anon, sample.pss_anon_bytes);
            f << ",\n          \"pss_file_bytes\": ";
            nullable_size(sample.has_pss_file, sample.pss_file_bytes);
            f << ",\n          \"anonymous_bytes\": ";
            nullable_size(sample.has_anonymous, sample.anonymous_bytes);
            f << ",\n          \"private_clean_bytes\": ";
            nullable_size(sample.has_private_clean,
                          sample.private_clean_bytes);
            f << ",\n          \"private_dirty_bytes\": ";
            nullable_size(sample.has_private_dirty,
                          sample.private_dirty_bytes);
            f << ",\n          \"shared_clean_bytes\": ";
            nullable_size(sample.has_shared_clean, sample.shared_clean_bytes);
            f << ",\n          \"shared_dirty_bytes\": ";
            nullable_size(sample.has_shared_dirty, sample.shared_dirty_bytes);
            f << "\n        }\n"
              << "      }" << (i + 1 == observations.size() ? "\n" : ",\n");
        }
        f << "    ]\n";
    }
    f << "  }\n}\n";
    return f.str();
}

bool write_raw_samples(const Args & a, const Measurement & m) {
    if (a.raw_samples_out_path.empty()) return true;

    std::ofstream f(a.raw_samples_out_path);
    if (!f) {
        std::fprintf(stderr, "error: cannot write raw samples to %s\n",
                     a.raw_samples_out_path.c_str());
        return false;
    }

    f << build_raw_samples_json(a, m);
    if (!f) {
        std::fprintf(stderr, "error: failed while writing raw samples to %s\n",
                     a.raw_samples_out_path.c_str());
        return false;
    }
    return true;
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
    a.cache_state = CacheState::Warm;
    a.strict_cold = false;
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
    const bool uss_available =
        a.memory_profile && m.baseline.has_uss &&
        m.sampled_memory.uss.available();
    const double uss_peak = uss_available
        ? static_cast<double>(m.sampled_memory.uss.peak_bytes) / mb
        : 0.0;
    const double uss_growth = uss_available
        ? uss_peak - static_cast<double>(m.baseline.uss_bytes) / mb
        : 0.0;

    std::printf("selftest: parent dirtied %d MB, worker allocated %d MB\n",
                kParentDirtyMb, a.selftest_alloc_mb);
    std::printf("  rss_baseline      = %8.2f MB\n", baseline);
    std::printf("  rss_peak_window   = %8.2f MB  (growth %.2f MB)\n", window, growth);
    if (a.memory_profile) {
        std::printf("  uss_peak_sampled  = %8.2f MB  (growth %.2f MB)\n",
                    uss_peak, uss_growth);
    } else {
        std::printf("  uss_peak_sampled  = not collected\n");
    }
    std::printf("  hwm_reset=%s memory_profile=%s attempted=%zu effective=%zu\n",
                m.hwm_reset ? "true" : "false",
                a.memory_profile ? "true" : "false",
                m.memory_profile_samples_attempted,
                m.memory_profile_samples_effective);

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

    const std::string selftest_json = build_json(a, m, read_environment());
    const std::string selftest_raw_json = build_raw_samples_json(a, m);
    if (a.memory_profile) {
        check(m.memory_profile_samples_attempted >= 2,
              "short-lived spike triggered at least two smaps_rollup attempts");
        check(m.memory_profile_samples_effective >= 2,
              "short-lived spike produced at least two valid samples");
        check(m.memory_profile_final_collected,
              "final smaps_rollup sample was collected");
        check(m.sampled_memory.total_samples ==
                  m.memory_profile_samples_effective,
              "final sample participates in memory aggregation");
        check(m.memory_profile_observations.size() ==
                  m.memory_profile_samples_attempted + 1,
              "raw memory timeline preserves baseline plus every attempt");
        check(!m.memory_profile_observations.empty() &&
                  m.memory_profile_observations.front().sample_role == "baseline" &&
                  m.memory_profile_observations.back().sample_role == "final" &&
                  m.memory_profile_observations.front().monotonic_timestamp_ns <
                      m.go_monotonic_timestamp_ns &&
                  m.memory_profile_observations.back().monotonic_timestamp_ns >=
                      m.done_marker_monotonic_timestamp_ns,
              "raw memory timeline brackets GO and DONE with baseline/final");
        check(uss_growth > alloc_mb * 0.85,
              "sampled USS growth captured the short-lived allocation (>85%)");
        check(selftest_json.find("\"memory_profile_enabled\": true") !=
                  std::string::npos &&
                  selftest_json.find("\"latency_result_role\": \"diagnostic\"") !=
                  std::string::npos,
              "profile-on JSON labels latency diagnostic");
        check(selftest_json.find("\"rss_sampled_p75_mb\": null") ==
                  std::string::npos &&
                  selftest_json.find("\"pss_sampled_p99_mb\": null") ==
                  std::string::npos &&
                  selftest_json.find("\"uss_sampled_p50_mb\": null") ==
                  std::string::npos,
              "profile-on JSON publishes sampled memory percentiles");
        check(selftest_raw_json.find("\"sample_role\": \"baseline\"") !=
                  std::string::npos &&
                  selftest_raw_json.find("\"sample_role\": \"periodic\"") !=
                  std::string::npos &&
                  selftest_raw_json.find("\"sample_role\": \"final\"") !=
                  std::string::npos &&
                  selftest_raw_json.find(
                      "\"done_marker_monotonic_timestamp_ns\": null") ==
                  std::string::npos,
              "profile-on raw JSON preserves roles and DONE marker");
    } else {
        check(m.memory_profile_samples_attempted == 0 &&
                  m.memory_profile_samples_effective == 0,
              "profile-off creates no smaps_rollup samples");
        check(!m.baseline.has_pss && !m.final_sample.has_pss,
              "profile-off boundaries use statm, not smaps_rollup");
        check(m.final_sample.valid && m.final_sample.has_rss,
              "profile-off still records final RSS");
        check(selftest_json.find("\"memory_profile_enabled\": false") !=
                  std::string::npos &&
                  selftest_json.find("\"pss_baseline_mb\": null") !=
                  std::string::npos &&
                  selftest_json.find("\"uss_peak_sampled_mb\": null") !=
                  std::string::npos,
              "profile-off JSON reports PSS/USS as null");
        check(selftest_json.find("\"rss_sampled_p50_mb\": null") !=
                  std::string::npos &&
                  selftest_json.find("\"pss_sampled_p75_mb\": null") !=
                  std::string::npos &&
                  selftest_json.find("\"uss_sampled_p99_mb\": null") !=
                  std::string::npos,
              "profile-off JSON disables sampled memory percentiles");
        check(m.memory_profile_observations.empty() &&
                  selftest_raw_json.find(
                      "\"collection_status\": \"disabled\"") !=
                  std::string::npos &&
                  selftest_raw_json.find("\"samples\": null") !=
                  std::string::npos &&
                  selftest_raw_json.find(
                      "\"done_marker_monotonic_timestamp_ns\": null") !=
                  std::string::npos,
              "profile-off raw JSON contains no memory observations");
    }

    if (!write_raw_samples(a, m)) ++failures;

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

    // Raw samples are serialized only after the authoritative worker window
    // has closed. runner.py normally points this at a temporary file and then
    // builds one optional, hashed sidecar for the aggregated result.
    if (!write_raw_samples(a, m)) return 1;

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
