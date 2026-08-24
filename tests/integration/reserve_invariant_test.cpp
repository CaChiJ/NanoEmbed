// Context creation reserves the graph the context will actually execute: no
// first-call resize, because an unexpected grow moves OOM from context
// construction back into embed().
//
// Scope, measured rather than assumed: the reservation is dominated by the
// attention activations, so the buffer is byte-identical across every pooling
// and normalization combination (bge-small 16521216, harrier 15208448). This
// therefore pins the sequence-length axis only -- it would still pass if
// reserve() went back to ignoring the context's pooling and normalize flags.
// The alternate-pooling and normalize=false configs are run so the assertion
// keeps holding if a future pooling op ever does change the graph's size.

#include "embedder.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool cond, const char * what) {
    if (!cond) {
        std::fprintf(stderr, "[reserve_invariant_test] FAIL: %s\n", what);
        ++g_failures;
    }
}

std::string long_text() {
    std::string text;
    for (int i = 0; i < 400; ++i) {
        text += "the quick brown fox jumps over the lazy dog. ";
    }
    return text;
}

void run_config(nanoembed::Embedder & e,
                nanoembed::PoolType  pooling,
                bool                 normalize,
                const char *         label) {
    nanoembed::ComputeScratch scratch;
    e.reserve(scratch, 512, pooling, normalize);
    check(e.reserved_seq_len(scratch) == std::min(512, e.max_seq_len()),
          "reserve records the effective maximum sequence length");
    const size_t before = e.graph_buffer_size(scratch);

    nanoembed::EmbedderConfig cfg;
    cfg.n_threads   = 1;
    cfg.max_seq_len = 512;
    cfg.pooling     = pooling;
    cfg.normalize   = normalize;

    std::vector<float> out(static_cast<size_t>(e.n_embed()));
    e.embed(scratch, long_text(), cfg, out.data());

    const size_t after = e.graph_buffer_size(scratch);
    check(before > 0, "reserve creates an activation buffer");
    check(before == after, label);
    for (float v : out) {
        if (!std::isfinite(v)) {
            check(false, "reserved-graph embedding is finite");
            break;
        }
    }
}

struct ModelUnderTest {
    const char * label;
    const char * env;
};

void run_model(const ModelUnderTest & m) {
    const char * path = std::getenv(m.env);
    if (!path) {
        std::fprintf(stderr, "[reserve_invariant_test] skip %s: %s not set\n",
                     m.label, m.env);
        return;
    }

    nanoembed::Embedder e(path);
    const nanoembed::PoolType model_pool = e.default_pooling();
    const nanoembed::PoolType other_pool =
        model_pool == nanoembed::PoolType::Mean
            ? nanoembed::PoolType::Cls
            : nanoembed::PoolType::Mean;

    run_config(e, model_pool, true,
               "model-default normalized graph does not resize on first embed");
    run_config(e, other_pool, true,
               "alternate-pooling graph does not resize on first embed");
    run_config(e, model_pool, false,
               "non-normalized graph does not resize on first embed");

    std::fprintf(stderr, "[reserve_invariant_test] %s: reservation stable\n", m.label);
}

const ModelUnderTest kModels[] = {
    {"bert",   "NANOEMBED_TEST_MODEL"},
    {"gemma3", "NANOEMBED_TEST_MODEL_GEMMA3"},
};

} // namespace

int main() {
    for (const ModelUnderTest & m : kModels) run_model(m);
    std::printf("reserve_invariant_test: %s\n", g_failures == 0 ? "ok" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
