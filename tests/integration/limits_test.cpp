// Sequence-length limit behaviour.
//
// The golden corpus is all short sentences, so it never exercised the top of
// the length range. This covers the boundary:
//   1. an input longer than the model's context embeds without aborting
//      (it used to overflow the graph memory pool and kill the process),
//   2. a max_seq_len above the model's context is clamped rather than
//      indexing the positional embedding table out of bounds,
//   3. use_streaming is rejected until M4 implements it.
//
// Skipped if NANOEMBED_TEST_MODEL is unset.

#include "nanoembed/nanoembed.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool cond, const char * what) {
    if (!cond) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_failures;
    }
}

// Comfortably more WordPiece pieces than bge-small's 512-token context.
std::string long_text() {
    std::string s;
    for (int i = 0; i < 400; ++i) {
        s += "embedding tokenization stress ";
    }
    return s;
}

bool all_finite(const std::vector<float> & v) {
    for (float x : v) {
        if (!std::isfinite(x)) return false;
    }
    return true;
}

double l2_norm(const std::vector<float> & v) {
    double sum = 0.0;
    for (float x : v) sum += static_cast<double>(x) * x;
    return std::sqrt(sum);
}

} // namespace

int main() {
    const char * model_path = std::getenv("NANOEMBED_TEST_MODEL");
    if (!model_path) {
        std::fprintf(stderr, "[limits_test] skip: NANOEMBED_TEST_MODEL not set\n");
        return 0;
    }

    nanoembed_model * model = nanoembed_load_model(model_path);
    if (!model) {
        std::fprintf(stderr, "load_model failed: %s\n", nanoembed_last_error());
        return 1;
    }

    const int H = nanoembed_n_embed(model);
    const std::string text = long_text();

    // ---- 1. Over-length input at the model's own limit -------------------
    std::vector<float> at_limit(static_cast<size_t>(H));
    {
        nanoembed_context_params p = nanoembed_context_default_params();
        nanoembed_context * ctx = nanoembed_new_context(model, p);
        check(ctx != nullptr, "new_context with default params");
        if (ctx) {
            const int rc = nanoembed_embed(ctx, text.c_str(), at_limit.data());
            check(rc == NANOEMBED_OK, "embed of over-length input returns OK");
            check(all_finite(at_limit), "over-length embedding is finite");
            check(std::abs(l2_norm(at_limit) - 1.0) < 1e-3,
                  "over-length embedding is L2-normalized");
            nanoembed_free_context(ctx);
        }
    }

    // ---- 2. max_seq_len beyond the model's context is clamped ------------
    {
        nanoembed_context_params p = nanoembed_context_default_params();
        p.max_seq_len = 100000;
        nanoembed_context * ctx = nanoembed_new_context(model, p);
        check(ctx != nullptr, "new_context with oversized max_seq_len");
        if (ctx) {
            std::vector<float> got(static_cast<size_t>(H));
            const int rc = nanoembed_embed(ctx, text.c_str(), got.data());
            check(rc == NANOEMBED_OK, "embed with oversized max_seq_len returns OK");
            check(all_finite(got), "clamped embedding is finite");

            // Clamping means this must be the same computation as case 1.
            bool same = true;
            for (int i = 0; i < H; ++i) {
                if (std::abs(got[static_cast<size_t>(i)] -
                             at_limit[static_cast<size_t>(i)]) > 1e-6f) {
                    same = false;
                    break;
                }
            }
            check(same, "oversized max_seq_len matches the clamped result");
            nanoembed_free_context(ctx);
        }
    }

    // ---- 3. Streaming is not silently ignored ----------------------------
    {
        nanoembed_context_params p = nanoembed_context_default_params();
        p.use_streaming = 1;
        nanoembed_context * ctx = nanoembed_new_context(model, p);
        check(ctx == nullptr, "use_streaming is rejected before M4");
        if (ctx) nanoembed_free_context(ctx);
    }

    nanoembed_free_model(model);

    std::printf("limits_test: %s\n", g_failures == 0 ? "ok" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
