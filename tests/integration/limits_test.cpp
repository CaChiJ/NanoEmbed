// Sequence-length limit behaviour.
//
// The golden corpus is all short sentences, so it never exercises the top of
// the length range. This covers the boundary:
//   1. an input longer than the model's context embeds without aborting
//      (it used to overflow the graph memory pool and kill the process),
//   2. a max_seq_len above the model's context is clamped rather than
//      indexing the positional embedding table out of bounds,
//   3. use_streaming is rejected until M4 implements it.
//
// Runs for every configured family; each is skipped when its model env var is
// unset.

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

// Comfortably past a 512-token context for any tokenizer. A repeated *phrase*
// rather than a repeated word on purpose: BPE vocabularies carry merges for
// long runs of one repeated token, so "hello hello hello ..." can collapse to
// far fewer tokens than words and quietly stop being a long input.
std::string long_text() {
    std::string s;
    for (int i = 0; i < 150; ++i) {
        s += "the quick brown fox jumps over the lazy dog. ";
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

// Reserving a context of S costs O(S^2) in attention scores alone. Short-context
// encoders can afford their whole window; a 32k-context decoder cannot — its
// full reservation runs to tens of gigabytes — so the clamp case only runs
// where the clamped-to value is actually affordable.
constexpr int kAffordableContext = 2048;

struct ModelUnderTest {
    const char * label;
    const char * model_env;
};

void run_model(const ModelUnderTest & m) {
    const char * model_path = std::getenv(m.model_env);
    if (!model_path) {
        std::fprintf(stderr, "[limits_test] skip %s: %s not set\n", m.label, m.model_env);
        return;
    }

    nanoembed_model * model = nanoembed_load_model(model_path);
    if (!model) {
        std::fprintf(stderr, "load_model failed (%s): %s\n", m.label, nanoembed_last_error());
        ++g_failures;
        return;
    }

    const int         H          = nanoembed_n_embed(model);
    const int         model_ctx  = nanoembed_model_max_seq_len(model);
    const std::string text       = long_text();

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
    if (model_ctx > 0 && model_ctx <= kAffordableContext) {
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
    } else {
        std::fprintf(stderr,
            "[limits_test] %s: skipping the clamp case, context %d would reserve "
            "O(n^2) activations\n", m.label, model_ctx);
    }

    // ---- 3. Streaming is not silently ignored ----------------------------
    {
        nanoembed_context_params p = nanoembed_context_default_params();
        p.use_streaming = 1;
        nanoembed_context * ctx = nanoembed_new_context(model, p);
        check(ctx == nullptr, "use_streaming is rejected before M4");
        if (ctx) nanoembed_free_context(ctx);
    }

    std::fprintf(stderr, "[limits_test] %s: n_embed=%d model_ctx=%d\n",
                 m.label, H, model_ctx);
    nanoembed_free_model(model);
}

const ModelUnderTest kModels[] = {
    {"bert",   "NANOEMBED_TEST_MODEL"},
    {"gemma3", "NANOEMBED_TEST_MODEL_GEMMA3"},
};

} // namespace

int main() {
    for (const ModelUnderTest & m : kModels) {
        run_model(m);
    }
    std::printf("limits_test: %s\n", g_failures == 0 ? "ok" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
