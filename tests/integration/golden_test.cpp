// End-to-end golden parity vs sentence-transformers.
//
// Loads the NEGD fixture (text + reference embedding) produced by
// tools/dump_golden.py and asserts cosine similarity ≥ 0.9999 per sample
// and ≥ 0.99999 mean.
//
// bge-small-en-v1.5 uses CLS pooling + L2 normalize by default in
// sentence-transformers; we mirror that here.
//
// Skipped if NANOEMBED_TEST_MODEL or NANOEMBED_GOLDEN_FIXTURE is unset.

#include "nanoembed/nanoembed.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct GoldenSample {
    std::string         text;
    std::vector<float>  embedding;
};

std::vector<GoldenSample> load_negd(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open NEGD: " + path);

    auto must_read = [&](void * dst, std::streamsize n) {
        f.read(static_cast<char *>(dst), n);
        if (f.gcount() != n) {
            throw std::runtime_error("NEGD truncated or unreadable: " + path);
        }
    };

    char magic[4];
    must_read(magic, 4);
    if (std::string(magic, 4) != "NEGD") throw std::runtime_error("bad NEGD magic");

    uint32_t version, n_samples, n_embed;
    must_read(&version,   4);
    must_read(&n_samples, 4);
    must_read(&n_embed,   4);
    if (version != 1) throw std::runtime_error("unsupported NEGD version");

    std::vector<GoldenSample> samples;
    samples.reserve(n_samples);
    for (uint32_t i = 0; i < n_samples; ++i) {
        GoldenSample s;
        uint32_t text_len;
        must_read(&text_len, 4);
        s.text.resize(text_len);
        must_read(s.text.data(), text_len);
        s.embedding.resize(n_embed);
        must_read(s.embedding.data(),
                  static_cast<std::streamsize>(n_embed) * static_cast<std::streamsize>(sizeof(float)));
        samples.push_back(std::move(s));
    }
    return samples;
}

float cosine(const float * a, const float * b, int n) {
    double aa = 0.0, bb = 0.0, ab = 0.0;
    for (int i = 0; i < n; ++i) {
        aa += static_cast<double>(a[i]) * a[i];
        bb += static_cast<double>(b[i]) * b[i];
        ab += static_cast<double>(a[i]) * b[i];
    }
    if (aa <= 0.0 || bb <= 0.0) return 0.0f;
    return static_cast<float>(ab / (std::sqrt(aa) * std::sqrt(bb)));
}

} // namespace

int main() {
    const char * model_path  = std::getenv("NANOEMBED_TEST_MODEL");
    const char * golden_path = std::getenv("NANOEMBED_GOLDEN_FIXTURE");
    if (!model_path || !golden_path) {
        std::fprintf(stderr, "[golden_test] skip: env not set\n");
        return 0;
    }

    std::vector<GoldenSample> samples;
    try {
        samples = load_negd(golden_path);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[golden_test] %s\n", e.what());
        return 1;
    }
    if (samples.empty()) {
        std::fprintf(stderr, "[golden_test] empty fixture\n");
        return 1;
    }

    nanoembed_model * model = nanoembed_load_model(model_path);
    if (!model) {
        std::fprintf(stderr, "load_model failed: %s\n", nanoembed_last_error());
        return 1;
    }

    // The fixture comes from sentence-transformers, which uses whatever the
    // model's own 1_Pooling config says — CLS for bge-small, last-token for
    // harrier. Naming one here would make this test model-specific for no
    // reason, and would silently compare the wrong pooling for the other.
    nanoembed_context_params params = nanoembed_context_default_params();
    params.pooling   = NANOEMBED_POOL_MODEL_DEFAULT;
    params.normalize = 1;

    nanoembed_context * ctx = nanoembed_new_context(model, params);
    if (!ctx) {
        std::fprintf(stderr, "new_context failed: %s\n", nanoembed_last_error());
        nanoembed_free_model(model);
        return 1;
    }

    const int H = nanoembed_n_embed(model);
    if (H <= 0 || static_cast<int>(samples[0].embedding.size()) != H) {
        std::fprintf(stderr, "[golden_test] dim mismatch: model=%d fixture=%zu\n",
                     H, samples[0].embedding.size());
        nanoembed_free_context(ctx);
        nanoembed_free_model(model);
        return 1;
    }

    std::vector<float> got(static_cast<size_t>(H));
    int   n_fail  = 0;
    float min_cos = 1.0f;
    double sum_cos = 0.0;
    constexpr float PER_SAMPLE_TOL = 0.9999f;
    constexpr float MEAN_TOL       = 0.99999f;

    for (size_t i = 0; i < samples.size(); ++i) {
        const auto & s = samples[i];
        const int rc = nanoembed_embed(ctx, s.text.c_str(), got.data());
        if (rc != NANOEMBED_OK) {
            std::fprintf(stderr, "embed failed sample[%zu]: %s\n", i, nanoembed_last_error());
            nanoembed_free_context(ctx);
            nanoembed_free_model(model);
            return 1;
        }
        const float c = cosine(got.data(), s.embedding.data(), H);
        if (c < min_cos) min_cos = c;
        sum_cos += c;
        if (c < PER_SAMPLE_TOL) {
            ++n_fail;
            if (n_fail <= 3) {
                std::fprintf(stderr,
                    "FAIL sample[%zu] cosine=%.6f (tol=%.4f) text=\"%.80s\"\n",
                    i, static_cast<double>(c), static_cast<double>(PER_SAMPLE_TOL),
                    s.text.c_str());
            }
        }
    }

    const size_t n_total = samples.size();
    const float  mean_cos = static_cast<float>(sum_cos / static_cast<double>(n_total));
    std::printf("[golden_test] %zu/%zu samples vs sentence-transformers: "
                "min cosine=%.6f mean cosine=%.6f (per-sample tol %.4f, mean tol %.5f)\n",
                n_total - static_cast<size_t>(n_fail), n_total,
                static_cast<double>(min_cos),
                static_cast<double>(mean_cos),
                static_cast<double>(PER_SAMPLE_TOL),
                static_cast<double>(MEAN_TOL));

    nanoembed_free_context(ctx);
    nanoembed_free_model(model);

    const bool ok = (n_fail == 0) && (mean_cos >= MEAN_TOL);
    std::printf("golden_test: %s\n", ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}
