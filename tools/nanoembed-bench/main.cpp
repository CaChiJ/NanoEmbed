// nanoembed-bench: run one scenario, write JSON metrics to stdout / --out.
//
// Usage:
//   nanoembed-bench --model PATH --inputs FILE
//                   [--scenario NAME] [--warmup N] [--iter N]
//                   [--cls] [--no-normalize] [--threads N]
//                   [--max-seq-len N] [--out PATH]
//
// Metrics (see PLAN.md §벤치마크):
//   - peak / mean RSS (50 ms background sampling thread)
//   - CPU user + sys (delta across measurement window)
//   - major / minor page faults (delta)
//   - disk read bytes (delta; 0 if unsupported)
//   - latency p50 / p90 / p99 across all single-input embed calls
//   - throughput items/sec (total items / wall time)

#include "nanoembed/nanoembed.h"

#include "metrics.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace nanoembed::bench;

// ---- Args ------------------------------------------------------------------

struct Args {
    std::string model_path;
    std::string inputs_path;
    std::string scenario   = "single_short_f16";
    std::string out_path;          // empty = stdout
    int  warmup       = 5;         // warmup *iterations*, each touches every input
    int  iter         = 50;        // measurement iterations
    int  threads      = 0;         // 0 = auto
    int  max_seq_len  = 0;         // 0 = use model default
    bool use_cls      = false;
    bool normalize    = true;
};

void print_usage(const char * prog) {
    std::fprintf(stderr,
        "usage: %s --model PATH --inputs FILE [--scenario NAME] [--warmup N]\n"
        "       [--iter N] [--cls] [--no-normalize] [--threads N]\n"
        "       [--max-seq-len N] [--out PATH]\n",
        prog);
}

bool parse_args(int argc, char ** argv, Args & a) {
    auto need_val = [&](int & i) -> const char * {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "error: %s expects a value\n", argv[i]);
            return nullptr;
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        const char * t = argv[i];
        if (std::strcmp(t, "--model")        == 0) { auto v=need_val(i); if(!v) return false; a.model_path = v; }
        else if (std::strcmp(t, "--inputs")  == 0) { auto v=need_val(i); if(!v) return false; a.inputs_path = v; }
        else if (std::strcmp(t, "--scenario")== 0) { auto v=need_val(i); if(!v) return false; a.scenario = v; }
        else if (std::strcmp(t, "--out")     == 0) { auto v=need_val(i); if(!v) return false; a.out_path = v; }
        else if (std::strcmp(t, "--warmup")  == 0) { auto v=need_val(i); if(!v) return false; a.warmup = std::atoi(v); }
        else if (std::strcmp(t, "--iter")    == 0) { auto v=need_val(i); if(!v) return false; a.iter = std::atoi(v); }
        else if (std::strcmp(t, "--threads") == 0) { auto v=need_val(i); if(!v) return false; a.threads = std::atoi(v); }
        else if (std::strcmp(t, "--max-seq-len")==0){ auto v=need_val(i); if(!v) return false; a.max_seq_len = std::atoi(v); }
        else if (std::strcmp(t, "--cls")          == 0) { a.use_cls = true; }
        else if (std::strcmp(t, "--no-normalize") == 0) { a.normalize = false; }
        else if (std::strcmp(t, "-h") == 0 || std::strcmp(t, "--help") == 0) {
            print_usage(argv[0]); return false;
        } else {
            std::fprintf(stderr, "error: unknown arg: %s\n", t);
            print_usage(argv[0]);
            return false;
        }
    }
    if (a.model_path.empty() || a.inputs_path.empty()) {
        print_usage(argv[0]);
        return false;
    }
    return true;
}

// ---- Helpers ---------------------------------------------------------------

std::vector<std::string> read_lines(const std::string & path) {
    std::ifstream f(path);
    if (!f) {
        std::fprintf(stderr, "error: cannot open inputs: %s\n", path.c_str());
        return {};
    }
    std::vector<std::string> out;
    std::string              line;
    while (std::getline(f, line)) {
        if (!line.empty()) out.push_back(line);
    }
    return out;
}

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

// ---- Main ------------------------------------------------------------------

int run(const Args & a) {
    auto inputs = read_lines(a.inputs_path);
    if (inputs.empty()) {
        std::fprintf(stderr, "error: empty inputs file\n");
        return 1;
    }

    nanoembed_model * model = nanoembed_load_model(a.model_path.c_str());
    if (!model) {
        std::fprintf(stderr, "load_model failed: %s\n", nanoembed_last_error());
        return 1;
    }

    nanoembed_context_params p = nanoembed_context_default_params();
    p.pooling   = a.use_cls ? NANOEMBED_POOL_CLS : NANOEMBED_POOL_MEAN;
    p.normalize = a.normalize ? 1 : 0;
    if (a.threads     > 0) p.n_threads   = a.threads;
    if (a.max_seq_len > 0) p.max_seq_len = a.max_seq_len;

    nanoembed_context * ctx = nanoembed_new_context(model, p);
    if (!ctx) {
        std::fprintf(stderr, "new_context failed: %s\n", nanoembed_last_error());
        nanoembed_free_model(model);
        return 1;
    }

    const int H = nanoembed_n_embed(model);
    std::vector<float> out_buf(static_cast<size_t>(H));

    // Warmup.
    for (int w = 0; w < a.warmup; ++w) {
        for (const auto & t : inputs) {
            nanoembed_embed(ctx, t.c_str(), out_buf.data());
        }
    }

    // Background RSS sampler (50 ms cadence).
    std::atomic<bool>     stop{false};
    std::vector<size_t>   rss_samples;
    rss_samples.reserve(static_cast<size_t>(a.iter) * 4);
    std::thread sampler([&]() {
        while (!stop.load(std::memory_order_relaxed)) {
            rss_samples.push_back(current_rss_bytes());
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    });

    const Snapshot rs_start = snapshot();
    const auto     t_start  = std::chrono::steady_clock::now();

    std::vector<double> latencies_ms;
    latencies_ms.reserve(static_cast<size_t>(a.iter) * inputs.size());

    int total_items = 0;
    for (int it = 0; it < a.iter; ++it) {
        for (const auto & t : inputs) {
            const auto t0 = std::chrono::steady_clock::now();
            const int rc  = nanoembed_embed(ctx, t.c_str(), out_buf.data());
            const auto t1 = std::chrono::steady_clock::now();
            if (rc != NANOEMBED_OK) {
                stop.store(true, std::memory_order_relaxed);
                sampler.join();
                std::fprintf(stderr, "embed failed: %s\n", nanoembed_last_error());
                nanoembed_free_context(ctx);
                nanoembed_free_model(model);
                return 1;
            }
            latencies_ms.push_back(
                std::chrono::duration<double, std::milli>(t1 - t0).count());
            ++total_items;
        }
    }

    const auto     t_end  = std::chrono::steady_clock::now();
    const Snapshot rs_end = snapshot();

    stop.store(true, std::memory_order_relaxed);
    sampler.join();

    const size_t peak_rss     = peak_rss_bytes();
    size_t       rss_sum      = 0;
    for (size_t v : rss_samples) rss_sum += v;
    const double rss_avg = rss_samples.empty()
        ? 0.0
        : static_cast<double>(rss_sum) / static_cast<double>(rss_samples.size());

    const double wall_sec  = std::chrono::duration<double>(t_end - t_start).count();
    const double throughput = wall_sec > 0.0 ? total_items / wall_sec : 0.0;

    const double p50 = pct(latencies_ms, 0.50);
    const double p90 = pct(latencies_ms, 0.90);
    const double p99 = pct(latencies_ms, 0.99);

    nanoembed_free_context(ctx);
    nanoembed_free_model(model);

    // ---- JSON output -------------------------------------------------------

    std::string j;
    j.reserve(2048);
    j += "{\n";

    auto add_str = [&](const char * key, const std::string & v, bool comma) {
        j += "  \""; j += key; j += "\": ";
        std::string esc;
        escape_json_string(v, esc);
        j += esc;
        j += comma ? ",\n" : "\n";
    };
    auto add_i = [&](const char * key, long long v, bool comma) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%lld", v);
        j += "  \""; j += key; j += "\": "; j += buf;
        j += comma ? ",\n" : "\n";
    };
    auto add_f = [&](const char * key, double v, int prec, bool comma) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.*g", prec, v);
        j += "  \""; j += key; j += "\": "; j += buf;
        j += comma ? ",\n" : "\n";
    };

    add_str("scenario",       a.scenario,                        true);
    add_str("os",             os_label(),                        true);
    add_str("model",          a.model_path,                      true);
    add_str("inputs",         a.inputs_path,                     true);
    add_i  ("n_inputs",       static_cast<long long>(inputs.size()), true);
    add_i  ("warmup",         a.warmup,                          true);
    add_i  ("iter",           a.iter,                            true);
    add_i  ("total_items",    total_items,                       true);
    add_i  ("threads",        a.threads,                         true);
    add_str("pooling",        a.use_cls ? "cls" : "mean",        true);
    add_i  ("normalize",      a.normalize ? 1 : 0,               true);

    j += "  \"metrics\": {\n";
    auto mline_f = [&](const char * key, double v, int prec, bool comma) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.*g", prec, v);
        j += "    \""; j += key; j += "\": "; j += buf;
        j += comma ? ",\n" : "\n";
    };
    auto mline_i = [&](const char * key, long long v, bool comma) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%lld", v);
        j += "    \""; j += key; j += "\": "; j += buf;
        j += comma ? ",\n" : "\n";
    };

    mline_f("rss_peak_mb",            static_cast<double>(peak_rss) / (1024.0 * 1024.0), 6, true);
    mline_f("rss_avg_mb",             rss_avg / (1024.0 * 1024.0),                        6, true);
    mline_f("cpu_user_sec",           rs_end.cpu_user_sec - rs_start.cpu_user_sec,        6, true);
    mline_f("cpu_sys_sec",            rs_end.cpu_sys_sec  - rs_start.cpu_sys_sec,         6, true);
    mline_i("page_faults_major",      static_cast<long long>(rs_end.page_faults_major - rs_start.page_faults_major), true);
    mline_i("page_faults_minor",      static_cast<long long>(rs_end.page_faults_minor - rs_start.page_faults_minor), true);
    mline_i("io_read_bytes",          static_cast<long long>(rs_end.io_read_bytes - rs_start.io_read_bytes),         true);
    mline_f("wall_sec",               wall_sec,                                            6, true);
    mline_f("latency_p50_ms",         p50,                                                 6, true);
    mline_f("latency_p90_ms",         p90,                                                 6, true);
    mline_f("latency_p99_ms",         p99,                                                 6, true);
    mline_f("throughput_items_per_sec", throughput,                                        6, false);
    j += "  }\n";

    j += "}\n";

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

} // namespace

int main(int argc, char ** argv) {
    Args a;
    if (!parse_args(argc, argv, a)) return 2;
    return run(a);
}
