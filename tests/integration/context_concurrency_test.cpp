// Distinct contexts must be safe to use concurrently against one shared model.
// The public C ABI promises this, and 503fa2a (pre-M3.6) moved every mutable
// compute resource from the model into nanoembed_context to make it true. The
// promise had no test until now.

#include "nanoembed/nanoembed.h"

#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <thread>
#include <vector>

namespace {

int g_failures = 0;

void check(bool cond, const char * what) {
    if (!cond) {
        std::fprintf(stderr, "[context_concurrency_test] FAIL: %s\n", what);
        ++g_failures;
    }
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

struct ModelUnderTest {
    const char * label;
    const char * env;
};

void run_model(const ModelUnderTest & m) {
    const char * path = std::getenv(m.env);
    if (!path) {
        std::fprintf(stderr, "[context_concurrency_test] skip %s: %s not set\n",
                     m.label, m.env);
        return;
    }

    nanoembed_model * model = nanoembed_load_model(path);
    check(model != nullptr, "shared model loads");
    if (!model) return;

    nanoembed_context_params p = nanoembed_context_default_params();
    p.n_threads   = 1;
    p.max_batch   = 3;
    p.max_seq_len = 128;

    nanoembed_context * a = nanoembed_new_context(model, p);
    nanoembed_context * b = nanoembed_new_context(model, p);
    check(a != nullptr && b != nullptr, "two contexts can be created on one model");
    if (!a || !b) {
        nanoembed_free_context(a);
        nanoembed_free_context(b);
        nanoembed_free_model(model);
        return;
    }

    const int H = nanoembed_n_embed(model);
    const char * text_a = "the quick brown fox jumps over the lazy dog";
    const char * text_b = "edge embeddings should remain deterministic under concurrency";
    std::vector<float> ref_a(static_cast<size_t>(H));
    std::vector<float> ref_b(static_cast<size_t>(H));
    check(nanoembed_embed(a, text_a, ref_a.data()) == NANOEMBED_OK,
          "first context reference embed succeeds");
    check(nanoembed_embed(b, text_b, ref_b.data()) == NANOEMBED_OK,
          "second context reference embed succeeds");

    std::atomic<bool> start{false};
    std::atomic<int>  worker_failures{0};
    auto worker = [&](nanoembed_context * ctx,
                      const char *        text,
                      const std::vector<float> & ref) {
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        std::vector<float> got(static_cast<size_t>(H));
        for (int i = 0; i < 4; ++i) {
            const int   rc = nanoembed_embed(ctx, text, got.data());
            const float c  = cosine(got, ref);
            if (rc != NANOEMBED_OK || !std::isfinite(c) || c < 0.99999f) {
                worker_failures.fetch_add(1, std::memory_order_relaxed);
                return;
            }
        }
    };

    std::thread ta(worker, a, text_a, std::cref(ref_a));
    std::thread tb(worker, b, text_b, std::cref(ref_b));
    start.store(true, std::memory_order_release);
    ta.join();
    tb.join();

    check(worker_failures.load(std::memory_order_relaxed) == 0,
          "parallel contexts match their sequential references");

    const char * batch_a[] = {
        "short", "a longer first-context batch sentence", "중복 batch"
    };
    const char * batch_b[] = {
        "another", "第二の context", "edge embeddings under true batching"
    };
    std::vector<float> batch_ref_a(static_cast<size_t>(H) * 3);
    std::vector<float> batch_ref_b(static_cast<size_t>(H) * 3);
    check(nanoembed_embed_batch(a, batch_a, 3, batch_ref_a.data()) == NANOEMBED_OK,
          "first context batch reference succeeds");
    check(nanoembed_embed_batch(b, batch_b, 3, batch_ref_b.data()) == NANOEMBED_OK,
          "second context batch reference succeeds");
    start.store(false, std::memory_order_release);
    worker_failures.store(0, std::memory_order_release);
    auto batch_worker = [&](nanoembed_context * ctx,
                            const char * const * texts,
                            const std::vector<float> & ref) {
        while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
        std::vector<float> got(static_cast<size_t>(H) * 3);
        for (int i = 0; i < 3; ++i) {
            const int rc = nanoembed_embed_batch(ctx, texts, 3, got.data());
            for (int item = 0; item < 3; ++item) {
                const size_t offset = static_cast<size_t>(item) * H;
                std::vector<float> expected(ref.begin() + static_cast<std::ptrdiff_t>(offset),
                                            ref.begin() + static_cast<std::ptrdiff_t>(offset + H));
                std::vector<float> actual(got.begin() + static_cast<std::ptrdiff_t>(offset),
                                          got.begin() + static_cast<std::ptrdiff_t>(offset + H));
                if (rc != NANOEMBED_OK || cosine(actual, expected) < 0.99999f) {
                    worker_failures.fetch_add(1, std::memory_order_relaxed);
                    return;
                }
            }
        }
    };
    std::thread batch_ta(batch_worker, a, batch_a, std::cref(batch_ref_a));
    std::thread batch_tb(batch_worker, b, batch_b, std::cref(batch_ref_b));
    start.store(true, std::memory_order_release);
    batch_ta.join();
    batch_tb.join();
    check(worker_failures.load(std::memory_order_relaxed) == 0,
          "parallel true batches match sequential batch references");
    std::fprintf(stderr, "[context_concurrency_test] %s: parallel contexts stable\n",
                 m.label);

    nanoembed_free_context(a);
    nanoembed_free_context(b);
    nanoembed_free_model(model);
}

const ModelUnderTest kModels[] = {
    {"bert",   "NANOEMBED_TEST_MODEL"},
    {"gemma3", "NANOEMBED_TEST_MODEL_GEMMA3"},
};

} // namespace

int main() {
    for (const ModelUnderTest & m : kModels) run_model(m);
    std::printf("context_concurrency_test: %s\n", g_failures == 0 ? "ok" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
