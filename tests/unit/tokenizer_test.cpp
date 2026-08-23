// Tokenizer parity tests.
//
// Loads a HuggingFace-derived TSV fixture per model and asserts our tokenizer
// produces byte-equal token IDs for every sentence. This is the oracle for
// "we tokenize like HuggingFace"; everything downstream is meaningless if it
// fails, so it runs for every supported family:
//
//   bge-small-en-v1.5   WordPiece            NANOEMBED_TEST_MODEL
//   harrier-oss-v1-270m SentencePiece BPE    NANOEMBED_TEST_MODEL_GEMMA3
//
// Models go through create_tokenizer() rather than a concrete class, so the
// GGUF's own tokenizer.ggml.model tag decides the implementation — the same
// dispatch the library uses. Expected constants come from the fixture header
// instead of being hardcoded, which is what lets one binary cover both.
//
// Each model is skipped when its env vars are unset.

#include "tokenizer/spm_bpe.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/wordpiece.h"

#include "gguf.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
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

struct Fixture {
    std::map<std::string, std::string> header;   // "# k=v k=v" line
    std::vector<FixtureSample>         samples;

    // -1 is the fixture's encoding for "this family has no such token", which
    // is also a safe default for a key an older fixture did not write.
    int id(const char * key) const {
        const auto it = header.find(key);
        return it == header.end() ? -1 : std::stoi(it->second);
    }
};

Fixture load_tsv(const std::string & path) {
    std::ifstream f(path);
    if (!f) {
        throw std::runtime_error("cannot open fixture: " + path);
    }
    Fixture     fx;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;

        if (line[0] == '#') {
            std::stringstream ss(line.substr(1));
            std::string       kv;
            while (ss >> kv) {
                const auto eq = kv.find('=');
                if (eq != std::string::npos) {
                    fx.header[kv.substr(0, eq)] = kv.substr(eq + 1);
                }
            }
            continue;
        }

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
        fx.samples.push_back(std::move(s));
    }
    return fx;
}

std::unique_ptr<nanoembed::Tokenizer> open_tokenizer(const char * model_path) {
    gguf_init_params params;
    params.no_alloc = true;
    params.ctx      = nullptr;
    gguf_context * ctx = gguf_init_from_file(model_path, params);
    if (ctx == nullptr) {
        throw std::runtime_error(std::string("cannot open GGUF: ") + model_path);
    }
    auto tok = nanoembed::create_tokenizer(ctx);
    gguf_free(ctx);
    return tok;
}

// ---- Per-model cases ---------------------------------------------------

struct ModelUnderTest {
    const char * label;
    const char * model_env;
    const char * fixture_env;
};

bool test_metadata(const ModelUnderTest & m) {
    const char * model_path   = std::getenv(m.model_env);
    const char * fixture_path = std::getenv(m.fixture_env);
    if (!model_path || !fixture_path) {
        std::fprintf(stderr, "[tokenizer_test] skip %s metadata: %s/%s not set\n",
                     m.label, m.model_env, m.fixture_env);
        return true;
    }

    const auto tok = open_tokenizer(model_path);
    const auto fx  = load_tsv(fixture_path);

    EXPECT_EQ_INT(tok->vocab_size(), fx.id("vocab_size"));

    // Special-token IDs are family-specific, so they are checked against the
    // concrete type the registry chose. Getting the wrong implementation for a
    // file would show up here before the parity loop.
    if (const auto * wp = dynamic_cast<const nanoembed::WordPieceTokenizer *>(tok.get())) {
        EXPECT_EQ_INT(wp->cls_id(), fx.id("cls"));
        EXPECT_EQ_INT(wp->sep_id(), fx.id("sep"));
        EXPECT_EQ_INT(wp->pad_id(), fx.id("pad"));
        EXPECT_EQ_INT(wp->unk_id(), fx.id("unk"));
    } else if (const auto * bp = dynamic_cast<const nanoembed::SpmBpeTokenizer *>(tok.get())) {
        EXPECT_EQ_INT(bp->bos_id(), fx.id("bos"));
        EXPECT_EQ_INT(bp->eos_id(), fx.id("eos"));
        EXPECT_EQ_INT(bp->pad_id(), fx.id("pad"));

        // harrier's GGUF carries no tokenizer.ggml.unknown_token_id, and with
        // a complete <0x00>..<0xFF> fallback set it never needs one: no
        // character is unrepresentable. Reporting -1 is the honest reading of
        // a file that does not state it, so this is checked only when it does.
        // from_gguf() already refuses a file that has neither.
        if (bp->unk_id() >= 0) {
            EXPECT_EQ_INT(bp->unk_id(), fx.id("unk"));
        }

        // The wrapper this model actually applies. The GGUF's add_eos_token
        // says false while HuggingFace appends <eos> anyway, so this asserts
        // we resolved that conflict in HuggingFace's favour. A last-token
        // pooling model reads its embedding off this token.
        EXPECT_EQ_INT(bp->prefix_id(), fx.id("bos"));
        EXPECT_EQ_INT(bp->suffix_id(), fx.id("eos"));
    } else {
        std::fprintf(stderr, "FAIL: %s got an unrecognized tokenizer type\n", m.label);
        ++g_failures;
    }

    return g_failures == 0;
}

bool test_hf_parity(const ModelUnderTest & m) {
    const char * model_path   = std::getenv(m.model_env);
    const char * fixture_path = std::getenv(m.fixture_env);
    if (!model_path || !fixture_path) {
        std::fprintf(stderr, "[tokenizer_test] skip %s parity: %s/%s not set\n",
                     m.label, m.model_env, m.fixture_env);
        return true;
    }

    const auto tok = open_tokenizer(model_path);
    const auto fx  = load_tsv(fixture_path);
    EXPECT_TRUE(!fx.samples.empty());

    int n_pass = 0;
    int n_fail = 0;
    for (size_t i = 0; i < fx.samples.size(); ++i) {
        const auto got = tok->encode(fx.samples[i].text);
        if (got == fx.samples[i].ids) {
            ++n_pass;
            continue;
        }
        ++n_fail;
        if (n_fail <= 3) {
            std::fprintf(stderr, "FAIL %s sample[%zu]: text=\"%s\"\n  exp:",
                         m.label, i, fx.samples[i].text.c_str());
            for (int x : fx.samples[i].ids) std::fprintf(stderr, " %d", x);
            std::fprintf(stderr, "\n  got:");
            for (int x : got) std::fprintf(stderr, " %d", x);
            std::fprintf(stderr, "\n");
        }
    }

    std::printf("tokenizer[%s]: %d/%zu samples match HF (n_fail=%d)\n",
                m.label, n_pass, fx.samples.size(), n_fail);
    EXPECT_EQ_INT(n_fail, 0);

    return g_failures == 0;
}

// Truncation must keep the wrapper tokens. For a last-token-pooling model the
// trailing marker *is* the embedding position, so dropping it to make room for
// content would silently change what gets pooled.
bool test_truncation(const ModelUnderTest & m) {
    const char * model_path   = std::getenv(m.model_env);
    const char * fixture_path = std::getenv(m.fixture_env);
    if (!model_path || !fixture_path) {
        std::fprintf(stderr, "[tokenizer_test] skip %s truncation: env not set\n", m.label);
        return true;
    }

    const auto tok = open_tokenizer(model_path);
    const auto fx  = load_tsv(fixture_path);
    EXPECT_TRUE(!fx.samples.empty());
    if (fx.samples.empty()) return false;

    // Taken from HuggingFace's own output rather than named per family.
    const int prefix = fx.samples[0].ids.front();
    const int suffix = fx.samples[0].ids.back();

    std::string long_text;
    for (int i = 0; i < 400; ++i) long_text += "embedding tokenization stress ";

    for (const int limit : {8, 32, 128}) {
        const auto ids = tok->encode(long_text, limit);
        EXPECT_EQ_INT(static_cast<int>(ids.size()), limit);
        if (ids.empty()) continue;
        EXPECT_EQ_INT(ids.front(), prefix);
        EXPECT_EQ_INT(ids.back(),  suffix);
    }

    // Untruncated, the same text must exceed every limit above — otherwise the
    // assertions are vacuous.
    EXPECT_TRUE(tok->encode(long_text, 4096).size() > 128);

    return g_failures == 0;
}

const ModelUnderTest kModels[] = {
    {"bert",   "NANOEMBED_TEST_MODEL",        "NANOEMBED_TOKENIZER_FIXTURE"},
    {"gemma3", "NANOEMBED_TEST_MODEL_GEMMA3", "NANOEMBED_TOKENIZER_FIXTURE_GEMMA3"},
};

} // namespace

int main() {
    int rc = 0;
    for (const ModelUnderTest & m : kModels) {
        g_failures = 0; if (!test_metadata(m))  rc = 1;
        g_failures = 0; if (!test_hf_parity(m)) rc = 1;
        g_failures = 0; if (!test_truncation(m)) rc = 1;
    }
    std::printf("tokenizer_test: %s\n", rc == 0 ? "ok" : "FAIL");
    return rc;
}
