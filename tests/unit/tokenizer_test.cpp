// Unit tests for src/tokenizer/wordpiece.cpp.
//
// Loads the HuggingFace-derived TSV fixture and asserts our WordPiece
// tokenizer produces byte-equal token IDs for every sentence.
//
// Skipped if either NANOEMBED_TEST_MODEL or NANOEMBED_TOKENIZER_FIXTURE
// is unset (CMake sets both when models/bge-small-en-v1.5-f16.gguf
// exists locally).

#include "tokenizer/wordpiece.h"

#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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

struct FixtureSample {
    std::string      text;
    std::vector<int> ids;
};

std::vector<FixtureSample> load_tsv(const std::string & path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("cannot open fixture: " + path);
    }
    std::vector<FixtureSample> out;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        const auto tab = line.find('\t');
        if (tab == std::string::npos) {
            throw std::runtime_error("malformed fixture line: " + line);
        }
        FixtureSample s;
        s.text = line.substr(0, tab);

        std::stringstream ss(line.substr(tab + 1));
        std::string       num;
        while (std::getline(ss, num, ',')) {
            s.ids.push_back(std::stoi(num));
        }
        out.push_back(std::move(s));
    }
    return out;
}

bool test_special_tokens_match_metadata() {
    const char * model_path = std::getenv("NANOEMBED_TEST_MODEL");
    if (!model_path) {
        std::fprintf(stderr, "[tokenizer_test] skip metadata: NANOEMBED_TEST_MODEL not set\n");
        return true;
    }

    gguf_init_params params;
    params.no_alloc = true;
    params.ctx      = nullptr;
    gguf_context * ctx = gguf_init_from_file(model_path, params);
    EXPECT_TRUE(ctx != nullptr);
    if (!ctx) return false;

    nanoembed::WordPieceTokenizer tok = nanoembed::WordPieceTokenizer::from_gguf(ctx);
    gguf_free(ctx);

    EXPECT_EQ_INT(tok.cls_id(),     101);
    EXPECT_EQ_INT(tok.sep_id(),     102);
    EXPECT_EQ_INT(tok.pad_id(),     0);
    EXPECT_EQ_INT(tok.unk_id(),     100);
    EXPECT_EQ_INT(tok.vocab_size(), 30522);

    return g_failures == 0;
}

bool test_hf_parity() {
    const char * model_path   = std::getenv("NANOEMBED_TEST_MODEL");
    const char * fixture_path = std::getenv("NANOEMBED_TOKENIZER_FIXTURE");
    if (!model_path || !fixture_path) {
        std::fprintf(stderr, "[tokenizer_test] skip parity: env not set\n");
        return true;
    }

    gguf_init_params params;
    params.no_alloc = true;
    params.ctx      = nullptr;
    gguf_context * ctx = gguf_init_from_file(model_path, params);
    EXPECT_TRUE(ctx != nullptr);
    if (!ctx) return false;

    nanoembed::WordPieceTokenizer tok = nanoembed::WordPieceTokenizer::from_gguf(ctx);
    gguf_free(ctx);

    const auto samples = load_tsv(fixture_path);
    EXPECT_TRUE(!samples.empty());

    int n_pass = 0;
    int n_fail = 0;
    for (size_t i = 0; i < samples.size(); ++i) {
        const auto got = tok.encode(samples[i].text);
        if (got == samples[i].ids) {
            ++n_pass;
            continue;
        }
        ++n_fail;
        if (n_fail <= 3) {
            std::fprintf(stderr, "FAIL sample[%zu]: text=\"%s\"\n  exp:",
                         i, samples[i].text.c_str());
            for (int x : samples[i].ids) std::fprintf(stderr, " %d", x);
            std::fprintf(stderr, "\n  got:");
            for (int x : got) std::fprintf(stderr, " %d", x);
            std::fprintf(stderr, "\n");
        }
    }

    std::printf("tokenizer: %d/%zu samples match HF (n_fail=%d)\n",
                n_pass, samples.size(), n_fail);
    EXPECT_EQ_INT(n_fail, 0);

    return g_failures == 0;
}

} // namespace

int main() {
    int rc = 0;
    g_failures = 0; if (!test_special_tokens_match_metadata()) rc = 1;
    g_failures = 0; if (!test_hf_parity())                     rc = 1;
    std::printf("tokenizer_test: %s\n", rc == 0 ? "ok" : "FAIL");
    return rc;
}
