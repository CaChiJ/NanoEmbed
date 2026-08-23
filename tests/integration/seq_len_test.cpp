// Sequence-length and buffer-reuse regression test (M3.5).
//
// Before M3.5, embed() carved every tensor out of a fixed 256 MiB bump arena
// with no reuse. That arena overflowed at S≈270 and aborted inside ggml, even
// though the model advertises (and the default context params request) a
// 512-token context. These cases pin down that the graph allocator fixed it:
//
//   1. a full-length input at the advertised max_seq_len returns OK;
//   2. an over-length input is truncated rather than failing;
//   3. the reused activation buffer survives many calls and keeps returning
//      the same embedding for the same text (no cross-call corruption).
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

// `reps` copies of a short phrase, roughly 10 tokens each for any tokenizer.
// A phrase rather than a single repeated word on purpose: BPE vocabularies
// carry merges for long runs of one repeated token, so "hello hello hello ..."
// can collapse to far fewer tokens than words and stop being a long input
// without the test noticing.
std::string repeated_text(int reps) {
    std::string s;
    s.reserve(static_cast<size_t>(reps) * 45);
    for (int i = 0; i < reps; ++i) {
        s += "the quick brown fox jumps over the lazy dog. ";
    }
    return s;
}

float cosine(const std::vector<float> & a, const std::vector<float> & b) {
    double aa = 0.0, bb = 0.0, ab = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        aa += static_cast<double>(a[i]) * a[i];
        bb += static_cast<double>(b[i]) * b[i];
        ab += static_cast<double>(a[i]) * b[i];
    }
    if (aa <= 0.0 || bb <= 0.0) return 0.0f;
    return static_cast<float>(ab / (std::sqrt(aa) * std::sqrt(bb)));
}

int failures = 0;

void check(bool cond, const char * what) {
    if (!cond) {
        std::fprintf(stderr, "[seq_len_test] FAIL: %s\n", what);
        ++failures;
    }
}

struct ModelUnderTest {
    const char * label;
    const char * model_env;
};

void run_model(const ModelUnderTest & m) {
    const char * model_path = std::getenv(m.model_env);
    if (!model_path) {
        std::fprintf(stderr, "[seq_len_test] skip %s: %s not set\n", m.label, m.model_env);
        return;
    }

    nanoembed_model * model = nanoembed_load_model(model_path);
    if (!model) {
        std::fprintf(stderr, "load_model failed (%s): %s\n", m.label, nanoembed_last_error());
        ++failures;
        return;
    }

    // Deliberately the defaults: the pre-M3.5 abort was reachable without the
    // caller opting into anything, because max_seq_len defaults to 512.
    const nanoembed_context_params params = nanoembed_context_default_params();

    nanoembed_context * ctx = nanoembed_new_context(model, params);
    if (!ctx) {
        std::fprintf(stderr, "new_context failed (%s): %s\n", m.label, nanoembed_last_error());
        nanoembed_free_model(model);
        ++failures;
        return;
    }

    const int H = nanoembed_n_embed(model);
    std::vector<float> out(static_cast<size_t>(H));

    // 1. Full-length input. 60 phrases run well past the 512-token cap, so
    //    after truncation this is the worst case the buffer was reserved for.
    check(nanoembed_embed(ctx, repeated_text(60).c_str(), out.data()) == NANOEMBED_OK,
          "embed at max sequence length should succeed");

    // 2. Far past the cap — truncation, not failure.
    check(nanoembed_embed(ctx, repeated_text(400).c_str(), out.data()) == NANOEMBED_OK,
          "over-length input should be truncated, not rejected");

    // 3. Buffer reuse must be stable. Interleave a long input between repeats
    //    of a short one so each short call reuses memory the long call wrote.
    const std::string  probe = "the quick brown fox jumps over the lazy dog";
    std::vector<float> first(static_cast<size_t>(H));
    check(nanoembed_embed(ctx, probe.c_str(), first.data()) == NANOEMBED_OK,
          "probe embed should succeed");

    const std::string  long_text = repeated_text(60);
    std::vector<float> again(static_cast<size_t>(H));
    bool  all_ok    = true;
    float worst_cos = 1.0f;
    // A few rounds is enough: corruption from a shared buffer shows up on the
    // first reuse, and each round costs a full-context forward pass — about a
    // second on the 270M model.
    for (int i = 0; i < 4; ++i) {
        if (nanoembed_embed(ctx, long_text.c_str(), out.data()) != NANOEMBED_OK ||
            nanoembed_embed(ctx, probe.c_str(), again.data())   != NANOEMBED_OK) {
            all_ok = false;
            break;
        }
        worst_cos = std::fmin(worst_cos, cosine(first, again));
    }
    check(all_ok, "repeated alternating embeds should all succeed");
    check(worst_cos >= 0.99999f,
          "reused buffer must not perturb repeated embeddings");

    std::fprintf(stderr, "[seq_len_test] %s: worst repeat cosine=%.6f\n",
                 m.label, static_cast<double>(worst_cos));

    nanoembed_free_context(ctx);
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
    std::printf("seq_len_test: %s\n", failures == 0 ? "ok" : "FAIL");
    return failures == 0 ? 0 : 1;
}
