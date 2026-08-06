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
// Skipped if NANOEMBED_TEST_MODEL is unset.

#include "nanoembed/nanoembed.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

// `words` whitespace-separated copies of a token that WordPiece keeps whole,
// so token count tracks word count closely and we comfortably reach the cap.
std::string repeated_text(int words) {
    std::string s;
    s.reserve(static_cast<size_t>(words) * 5);
    for (int i = 0; i < words; ++i) {
        if (i) s += ' ';
        s += "hello";
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

} // namespace

int main() {
    const char * model_path = std::getenv("NANOEMBED_TEST_MODEL");
    if (!model_path) {
        std::fprintf(stderr, "[seq_len_test] skip: NANOEMBED_TEST_MODEL not set\n");
        return 0;
    }

    nanoembed_model * model = nanoembed_load_model(model_path);
    if (!model) {
        std::fprintf(stderr, "load_model failed: %s\n", nanoembed_last_error());
        return 1;
    }

    // Deliberately the defaults: the pre-M3.5 abort was reachable without the
    // caller opting into anything, because max_seq_len defaults to 512.
    const nanoembed_context_params params = nanoembed_context_default_params();

    nanoembed_context * ctx = nanoembed_new_context(model, params);
    if (!ctx) {
        std::fprintf(stderr, "new_context failed: %s\n", nanoembed_last_error());
        nanoembed_free_model(model);
        return 1;
    }

    const int H = nanoembed_n_embed(model);
    std::vector<float> out(static_cast<size_t>(H));

    // 1. Full-length input. 600 words saturates the 512-token cap (incl. the
    //    [CLS]/[SEP] pair), so this is the worst case the handle reserved for.
    check(nanoembed_embed(ctx, repeated_text(600).c_str(), out.data()) == NANOEMBED_OK,
          "embed at max sequence length should succeed");

    // 2. Far past the cap — truncation, not failure.
    check(nanoembed_embed(ctx, repeated_text(4000).c_str(), out.data()) == NANOEMBED_OK,
          "over-length input should be truncated, not rejected");

    // 3. Buffer reuse must be stable. Interleave a long input between repeats
    //    of a short one so each short call reuses memory the long call wrote.
    const std::string  probe = "the quick brown fox jumps over the lazy dog";
    std::vector<float> first(static_cast<size_t>(H));
    check(nanoembed_embed(ctx, probe.c_str(), first.data()) == NANOEMBED_OK,
          "probe embed should succeed");

    const std::string  long_text = repeated_text(600);
    std::vector<float> again(static_cast<size_t>(H));
    bool  all_ok    = true;
    float worst_cos = 1.0f;
    // A handful of rounds is enough: corruption from a shared buffer shows up
    // on the first reuse, and each round costs a full-context forward pass.
    for (int i = 0; i < 8; ++i) {
        if (nanoembed_embed(ctx, long_text.c_str(), out.data()) != NANOEMBED_OK ||
            nanoembed_embed(ctx, probe.c_str(), again.data())   != NANOEMBED_OK) {
            all_ok = false;
            break;
        }
        worst_cos = std::fmin(worst_cos, cosine(first, again));
    }
    check(all_ok, "repeated alternating embeds should all succeed");
    check(worst_cos >= 0.99999f, "reused buffer must not perturb repeated embeddings");

    nanoembed_free_context(ctx);
    nanoembed_free_model(model);

    if (failures == 0) {
        std::fprintf(stderr, "[seq_len_test] ok (worst repeat cosine %.6f)\n",
                     static_cast<double>(worst_cos));
    }
    return failures == 0 ? 0 : 1;
}
