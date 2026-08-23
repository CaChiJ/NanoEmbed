// Unit tests for src/gguf_scanner.cpp.
//
// Covers:
//   - Happy path on bge-small-en-v1.5 (skipped if NANOEMBED_TEST_MODEL is unset).
//   - Fail-fast on missing file.
//   - Fail-fast on non-BERT architecture (synthetic minimal GGUF).
//   - Fail-fast on missing required metadata (synthetic minimal GGUF).
//
// And for src/arch/gemma3_arch.cpp:
//   - Happy path on harrier-oss-v1-270m (skipped if NANOEMBED_TEST_MODEL_GEMMA3
//     is unset). This is the file whose head geometry breaks BERT's
//     head_dim == n_embed / n_head assumption, so the shape assertions here
//     are the regression guard for that.

#include "arch/gemma3_arch.h"
#include "gguf_scanner.h"

#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <unistd.h>

namespace {

int g_failures = 0;

#define EXPECT_TRUE(cond)                                                              \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::fprintf(stderr, "FAIL: %s (%s:%d)\n", #cond, __FILE__, __LINE__);     \
            ++g_failures;                                                              \
        }                                                                              \
    } while (0)

#define EXPECT_EQ_INT(a, b)                                                            \
    do {                                                                               \
        long long _a = static_cast<long long>(a);                                      \
        long long _b = static_cast<long long>(b);                                      \
        if (_a != _b) {                                                                \
            std::fprintf(stderr,                                                       \
                "FAIL: %s == %s (got %lld vs %lld) (%s:%d)\n",                         \
                #a, #b, _a, _b, __FILE__, __LINE__);                                   \
            ++g_failures;                                                              \
        }                                                                              \
    } while (0)

#define EXPECT_NEAR_F(a, b, tol)                                                       \
    do {                                                                               \
        double _a = static_cast<double>(a);                                            \
        double _b = static_cast<double>(b);                                            \
        if (std::fabs(_a - _b) > (tol)) {                                              \
            std::fprintf(stderr,                                                       \
                "FAIL: %s ~= %s (got %g vs %g) (%s:%d)\n",                             \
                #a, #b, _a, _b, __FILE__, __LINE__);                                   \
            ++g_failures;                                                              \
        }                                                                              \
    } while (0)

// ---- Test cases --------------------------------------------------------

bool test_bge_small_happy_path() {
    const char * path = std::getenv("NANOEMBED_TEST_MODEL");
    if (path == nullptr) {
        std::fprintf(stderr, "[scanner_test] skip happy_path: set NANOEMBED_TEST_MODEL to enable\n");
        return true;
    }

    nanoembed::ScanResult r = nanoembed::scan_gguf(path);
    const auto & m = r.manifest();

    // Hyperparameters match bge-small-en-v1.5.
    EXPECT_EQ_INT(m.arch.n_layer,     12);
    EXPECT_EQ_INT(m.arch.n_embed,     384);
    EXPECT_EQ_INT(m.arch.n_head,      12);
    EXPECT_EQ_INT(m.arch.n_ff,        1536);
    EXPECT_EQ_INT(m.arch.n_vocab,     30522);
    EXPECT_EQ_INT(m.arch.max_seq_len, 512);

    // Read from the file, not assumed. bge-small states 2 (CLS), but BERT
    // embedding models disagree — all-MiniLM and the e5 family are mean — and
    // since pooling now defaults to the model's own, hardcoding one here would
    // hand those models the wrong pooling with nothing to signal it.
    EXPECT_EQ_INT(m.arch.pooling_type, 2);

    // 12 layer slots, all 16 tensors per slot populated.
    EXPECT_EQ_INT(m.layers.size(), 12);
    for (size_t i = 0; i < m.layers.size(); ++i) {
        const auto & s = m.layers[i];
        EXPECT_TRUE(s.attn_q_w.valid());    EXPECT_TRUE(s.attn_q_b.valid());
        EXPECT_TRUE(s.attn_k_w.valid());    EXPECT_TRUE(s.attn_k_b.valid());
        EXPECT_TRUE(s.attn_v_w.valid());    EXPECT_TRUE(s.attn_v_b.valid());
        EXPECT_TRUE(s.attn_o_w.valid());    EXPECT_TRUE(s.attn_o_b.valid());
        EXPECT_TRUE(s.attn_norm_w.valid()); EXPECT_TRUE(s.attn_norm_b.valid());
        EXPECT_TRUE(s.ffn_up_w.valid());    EXPECT_TRUE(s.ffn_up_b.valid());
        EXPECT_TRUE(s.ffn_down_w.valid());  EXPECT_TRUE(s.ffn_down_b.valid());
        EXPECT_TRUE(s.ffn_norm_w.valid());  EXPECT_TRUE(s.ffn_norm_b.valid());
    }

    // Embedding tensors.
    EXPECT_TRUE(m.tok_embed_w.valid());
    EXPECT_TRUE(m.pos_embed_w.valid());
    EXPECT_TRUE(m.type_embed_w.valid());
    EXPECT_TRUE(m.embed_norm_w.valid());
    EXPECT_TRUE(m.embed_norm_b.valid());

    // Spot-check tensor shapes.
    EXPECT_EQ_INT(m.tok_embed_w.ne[0], m.arch.n_embed);
    EXPECT_EQ_INT(m.tok_embed_w.ne[1], m.arch.n_vocab);
    EXPECT_EQ_INT(m.layers[0].attn_q_w.ne[0], m.arch.n_embed);
    EXPECT_EQ_INT(m.layers[0].attn_q_w.ne[1], m.arch.n_embed);
    EXPECT_EQ_INT(m.layers[0].ffn_up_w.ne[1], m.arch.n_ff);

    return g_failures == 0;
}

bool test_missing_file() {
    bool threw = false;
    std::string msg;
    try {
        nanoembed::scan_gguf("/nonexistent/path/that/should/not/exist.gguf");
    } catch (const nanoembed::ScanError & e) {
        threw = true;
        msg = e.what();
    }
    EXPECT_TRUE(threw);
    EXPECT_TRUE(msg.find("failed to open") != std::string::npos);
    return g_failures == 0;
}

// Helper: write a minimal synthetic GGUF (metadata-only) to a tmp path.
// gguf_write_to_file does not care about the extension.
std::string write_temp_gguf(void (*populate)(gguf_context *)) {
    char path_template[] = "/tmp/nanoembed_test_XXXXXX";
    int fd = mkstemp(path_template);
    if (fd < 0) {
        std::fprintf(stderr, "mkstemp failed\n");
        return {};
    }
    close(fd);

    gguf_context * g = gguf_init_empty();
    populate(g);
    bool ok = gguf_write_to_file(g, path_template, /*only_meta=*/true);
    gguf_free(g);
    if (!ok) {
        std::remove(path_template);
        return {};
    }
    return std::string(path_template);
}

bool test_non_bert_architecture() {
    auto populate = [](gguf_context * g) {
        gguf_set_val_str(g, "general.architecture", "llama");
    };
    std::string path = write_temp_gguf(populate);
    EXPECT_TRUE(!path.empty());
    if (path.empty()) return false;

    bool threw = false;
    std::string msg;
    try {
        nanoembed::scan_gguf(path);
    } catch (const nanoembed::ScanError & e) {
        threw = true;
        msg = e.what();
    }
    std::remove(path.c_str());

    EXPECT_TRUE(threw);
    EXPECT_TRUE(msg.find("bert") != std::string::npos);
    return g_failures == 0;
}

bool test_bert_missing_hyperparameters() {
    // Marked "bert" but missing bert.block_count etc. — scanner must fail fast.
    auto populate = [](gguf_context * g) {
        gguf_set_val_str(g, "general.architecture", "bert");
        // Intentionally do not set bert.block_count etc.
    };
    std::string path = write_temp_gguf(populate);
    EXPECT_TRUE(!path.empty());
    if (path.empty()) return false;

    bool threw = false;
    std::string msg;
    try {
        nanoembed::scan_gguf(path);
    } catch (const nanoembed::ScanError & e) {
        threw = true;
        msg = e.what();
    }
    std::remove(path.c_str());

    EXPECT_TRUE(threw);
    EXPECT_TRUE(msg.find("missing required metadata") != std::string::npos);
    return g_failures == 0;
}

bool test_harrier_happy_path() {
    const char * path = std::getenv("NANOEMBED_TEST_MODEL_GEMMA3");
    if (path == nullptr) {
        std::fprintf(stderr,
            "[scanner_test] skip gemma3 happy_path: set NANOEMBED_TEST_MODEL_GEMMA3 to enable\n");
        return true;
    }

    const nanoembed::Gemma3Manifest m = nanoembed::scan_gemma3(path);
    const nanoembed::ArchParams &   a = m.params;

    EXPECT_TRUE(a.name == "gemma3");
    EXPECT_EQ_INT(a.n_layer,     18);
    EXPECT_EQ_INT(a.n_embed,     640);
    EXPECT_EQ_INT(a.n_ff,        2048);
    EXPECT_EQ_INT(a.n_vocab,     262144);
    EXPECT_EQ_INT(a.max_seq_len, 32768);

    // The reason this family needs its own scanner: 4 query heads of width
    // 256 over a 640-wide residual stream, sharing one KV head. Deriving
    // head_dim as n_embed / n_head would give 160 and corrupt every reshape.
    EXPECT_EQ_INT(a.n_head,    4);
    EXPECT_EQ_INT(a.n_head_kv, 1);
    EXPECT_EQ_INT(a.head_dim,  256);
    EXPECT_TRUE(a.head_dim * a.n_head != a.n_embed);

    EXPECT_TRUE(a.causal);
    EXPECT_NEAR_F(a.rope_freq_base, 1e6f, 1.0);
    EXPECT_NEAR_F(a.norm_eps, 1e-6f, 1e-9);

    EXPECT_TRUE(m.pooling == nanoembed::PoolType::Last);
    EXPECT_NEAR_F(m.embed_scale, std::sqrt(640.0), 1e-4);   // not folded into token_embd
    EXPECT_NEAR_F(m.attn_scale,  1.0 / 16.0,       1e-6);   // 1/sqrt(query_pre_attn_scalar)

    // 18 blocks, every one of the 13 slots filled.
    EXPECT_EQ_INT(m.layers.size(), 18);
    for (size_t i = 0; i < m.layers.size(); ++i) {
        const auto & L = m.layers[i];
        EXPECT_TRUE(L.attn_norm.valid());
        EXPECT_TRUE(L.attn_q.valid());       EXPECT_TRUE(L.attn_k.valid());
        EXPECT_TRUE(L.attn_v.valid());       EXPECT_TRUE(L.attn_output.valid());
        EXPECT_TRUE(L.attn_q_norm.valid());  EXPECT_TRUE(L.attn_k_norm.valid());
        EXPECT_TRUE(L.post_attention_norm.valid());
        EXPECT_TRUE(L.ffn_norm.valid());     EXPECT_TRUE(L.post_ffw_norm.valid());
        EXPECT_TRUE(L.ffn_gate.valid());     EXPECT_TRUE(L.ffn_up.valid());
        EXPECT_TRUE(L.ffn_down.valid());
    }

    EXPECT_TRUE(m.tok_embed.valid());
    EXPECT_TRUE(m.output_norm.valid());
    EXPECT_EQ_INT(m.tok_embed.ne[0], a.n_embed);
    EXPECT_EQ_INT(m.tok_embed.ne[1], a.n_vocab);

    // Asymmetric projections: Q is 4x wider than K/V on the output side.
    EXPECT_EQ_INT(m.layers[0].attn_q.ne[0], a.n_embed);
    EXPECT_EQ_INT(m.layers[0].attn_q.ne[1], a.n_head * a.head_dim);
    EXPECT_EQ_INT(m.layers[0].attn_k.ne[1], a.n_head_kv * a.head_dim);
    EXPECT_EQ_INT(m.layers[0].attn_output.ne[0], a.n_head * a.head_dim);
    EXPECT_EQ_INT(m.layers[0].attn_output.ne[1], a.n_embed);
    EXPECT_EQ_INT(m.layers[0].attn_q_norm.ne[0], a.head_dim);

    return g_failures == 0;
}

bool test_gemma3_missing_hyperparameters() {
    auto populate = [](gguf_context * g) {
        gguf_set_val_str(g, "general.architecture", "gemma3");
        // Intentionally do not set gemma3.block_count etc.
    };
    std::string path = write_temp_gguf(populate);
    EXPECT_TRUE(!path.empty());
    if (path.empty()) return false;

    bool threw = false;
    std::string msg;
    try {
        nanoembed::scan_gemma3(path);
    } catch (const nanoembed::ScanError & e) {
        threw = true;
        msg = e.what();
    }
    std::remove(path.c_str());

    EXPECT_TRUE(threw);
    EXPECT_TRUE(msg.find("missing required metadata") != std::string::npos);
    return g_failures == 0;
}

bool test_gemma3_scanner_rejects_bert() {
    // Each family's scanner must refuse the other's file rather than reading
    // absent keys as zeros.
    const char * path = std::getenv("NANOEMBED_TEST_MODEL");
    if (path == nullptr) {
        std::fprintf(stderr,
            "[scanner_test] skip gemma3_rejects_bert: set NANOEMBED_TEST_MODEL to enable\n");
        return true;
    }

    bool threw = false;
    std::string msg;
    try {
        nanoembed::scan_gemma3(path);
    } catch (const nanoembed::ScanError & e) {
        threw = true;
        msg = e.what();
    }
    EXPECT_TRUE(threw);
    EXPECT_TRUE(msg.find("gemma3") != std::string::npos);
    return g_failures == 0;
}

} // namespace

int main() {
    int rc = 0;
    g_failures = 0; if (!test_bge_small_happy_path())            rc = 1;
    g_failures = 0; if (!test_missing_file())                    rc = 1;
    g_failures = 0; if (!test_non_bert_architecture())           rc = 1;
    g_failures = 0; if (!test_bert_missing_hyperparameters())    rc = 1;
    g_failures = 0; if (!test_harrier_happy_path())              rc = 1;
    g_failures = 0; if (!test_gemma3_missing_hyperparameters())  rc = 1;
    g_failures = 0; if (!test_gemma3_scanner_rejects_bert())     rc = 1;

    std::printf("scanner_test: %s\n", rc == 0 ? "ok" : "FAIL");
    return rc;
}
