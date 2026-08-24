// End-to-end golden parity vs sentence-transformers.
//
// Loads the NEGD fixture (text + reference embedding) produced by
// tools/dump_golden.py and asserts cosine similarity per sample and on the
// mean. This is the whole pipeline at once — tokenizer, graph, pooling,
// normalization — so it is the test that says the model is actually correct
// rather than merely self-consistent.
//
// Runs for every configured family:
//
//   bge-small-en-v1.5    CLS pooling         NANOEMBED_TEST_MODEL
//   harrier-oss-v1-270m  last-token pooling  NANOEMBED_TEST_MODEL_GEMMA3
//
// Pooling is left at MODEL_DEFAULT rather than named here: the fixture follows
// each model's own 1_Pooling config, so naming one mode would compare the
// wrong thing for the other model.
//
// Each model is skipped when its env vars are unset.

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

struct ModelUnderTest {
    const char * label;
    const char * model_env;
    const char * fixture_env;
    float        per_sample_tol;
    float        mean_tol;
};

// Returns 0 on success or skip, 1 on failure.
int run_model(const ModelUnderTest & m) {
    const char * model_path  = std::getenv(m.model_env);
    const char * golden_path = std::getenv(m.fixture_env);
    if (!model_path || !golden_path) {
        std::fprintf(stderr, "[golden_test] skip %s: %s/%s not set\n",
                     m.label, m.model_env, m.fixture_env);
        return 0;
    }

    std::vector<GoldenSample> samples;
    try {
        samples = load_negd(golden_path);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "[golden_test] %s: %s\n", m.label, e.what());
        return 1;
    }
    if (samples.empty()) {
        std::fprintf(stderr, "[golden_test] %s: empty fixture\n", m.label);
        return 1;
    }

    nanoembed_model * model = nanoembed_load_model(model_path);
    if (!model) {
        std::fprintf(stderr, "load_model failed (%s): %s\n", m.label, nanoembed_last_error());
        return 1;
    }

    nanoembed_context_params params = nanoembed_context_default_params();
    params.pooling   = NANOEMBED_POOL_MODEL_DEFAULT;
    params.normalize = 1;

    nanoembed_context * ctx = nanoembed_new_context(model, params);
    if (!ctx) {
        std::fprintf(stderr, "new_context failed (%s): %s\n", m.label, nanoembed_last_error());
        nanoembed_free_model(model);
        return 1;
    }

    const int H = nanoembed_n_embed(model);
    if (H <= 0 || static_cast<int>(samples[0].embedding.size()) != H) {
        std::fprintf(stderr, "[golden_test] %s dim mismatch: model=%d fixture=%zu\n",
                     m.label, H, samples[0].embedding.size());
        nanoembed_free_context(ctx);
        nanoembed_free_model(model);
        return 1;
    }

    std::vector<float> got(static_cast<size_t>(H));
    int    n_fail  = 0;
    float  min_cos = 1.0f;
    double sum_cos = 0.0;

    for (size_t i = 0; i < samples.size(); ++i) {
        const auto & s = samples[i];
        const int rc = nanoembed_embed(ctx, s.text.c_str(), got.data());
        if (rc != NANOEMBED_OK) {
            std::fprintf(stderr, "embed failed (%s) sample[%zu]: %s\n",
                         m.label, i, nanoembed_last_error());
            nanoembed_free_context(ctx);
            nanoembed_free_model(model);
            return 1;
        }
        const float c = cosine(got.data(), s.embedding.data(), H);
        if (c < min_cos) min_cos = c;
        sum_cos += c;
        if (c < m.per_sample_tol) {
            ++n_fail;
            if (n_fail <= 3) {
                std::fprintf(stderr,
                    "FAIL %s sample[%zu] cosine=%.6f (tol=%.4f) text=\"%.80s\"\n",
                    m.label, i, static_cast<double>(c),
                    static_cast<double>(m.per_sample_tol), s.text.c_str());
            }
        }
    }

    const size_t n_total  = samples.size();
    const float  mean_cos = static_cast<float>(sum_cos / static_cast<double>(n_total));
    std::printf("[golden_test] %s: %zu/%zu samples vs sentence-transformers, "
                "min cosine=%.6f mean cosine=%.6f\n",
                m.label, n_total - static_cast<size_t>(n_fail), n_total,
                static_cast<double>(min_cos), static_cast<double>(mean_cos));

    nanoembed_free_context(ctx);
    nanoembed_free_model(model);

    return (n_fail == 0 && mean_cos >= m.mean_tol) ? 0 : 1;
}

const ModelUnderTest kModels[] = {
    // Same-precision paths: only implementation noise is allowed.
    {"bert",       "NANOEMBED_TEST_MODEL",           "NANOEMBED_GOLDEN_FIXTURE",
                   0.9999f, 0.99999f},
    {"gemma3-f32", "NANOEMBED_TEST_MODEL_GEMMA3",    "NANOEMBED_GOLDEN_FIXTURE_GEMMA3",
                   0.9999f, 0.99999f},

    // The measured q8_0 result is min=0.999117, mean=0.999754 against the same
    // F32 sentence-transformers oracle. These gates leave platform margin while
    // still catching a material quantized-kernel or model-selection regression.
    {"gemma3-q8",  "NANOEMBED_TEST_MODEL_GEMMA3_Q8", "NANOEMBED_GOLDEN_FIXTURE_GEMMA3",
                   0.9985f, 0.9995f},
};

int main() {
    int rc = 0;
    for (const ModelUnderTest & m : kModels) {
        if (run_model(m) != 0) rc = 1;
    }
    std::printf("golden_test: %s\n", rc == 0 ? "ok" : "FAIL");
    return rc;
}
