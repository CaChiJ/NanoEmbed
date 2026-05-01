// Unit tests for src/gguf_scanner.cpp.
//
// Covers:
//   - Happy path on bge-small-en-v1.5 (skipped if NANOEMBED_TEST_MODEL is unset).
//   - Fail-fast on missing file.
//   - Fail-fast on non-BERT architecture (synthetic minimal GGUF).
//   - Fail-fast on missing required metadata (synthetic minimal GGUF).

#include "gguf_scanner.h"

#include "gguf.h"

#include <cstdio>
#include <cstdlib>
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

} // namespace

int main() {
    int rc = 0;
    g_failures = 0; if (!test_bge_small_happy_path())            rc = 1;
    g_failures = 0; if (!test_missing_file())                    rc = 1;
    g_failures = 0; if (!test_non_bert_architecture())           rc = 1;
    g_failures = 0; if (!test_bert_missing_hyperparameters())    rc = 1;

    std::printf("scanner_test: %s\n", rc == 0 ? "ok" : "FAIL");
    return rc;
}
