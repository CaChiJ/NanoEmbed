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
#include "tokenizer/disk_merge_index.h"
#include "tokenizer/sha256.h"
#include "tokenizer/tokenizer.h"
#include "tokenizer/wordpiece.h"

#include "gguf.h"

#include <cstdio>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {

int g_failures = 0;

void set_cache_environment(const std::filesystem::path & value) {
#ifdef _WIN32
    if (_wputenv_s(L"NANOEMBED_CACHE_DIR", value.c_str()) != 0) {
        throw std::runtime_error("cannot set NANOEMBED_CACHE_DIR");
    }
#else
    if (setenv("NANOEMBED_CACHE_DIR", value.c_str(), 1) != 0) {
        throw std::runtime_error("cannot set NANOEMBED_CACHE_DIR");
    }
#endif
}

class TemporaryCacheDirectory {
public:
    TemporaryCacheDirectory() {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path() /
                ("nanoembed-bpe-test-" + std::to_string(stamp));
        if (!std::filesystem::create_directory(path_)) {
            throw std::runtime_error("cannot create tokenizer test cache directory");
        }
#ifdef _WIN32
        const wchar_t * old = _wgetenv(L"NANOEMBED_CACHE_DIR");
#else
        const char * old = std::getenv("NANOEMBED_CACHE_DIR");
#endif
        if (old != nullptr) {
            had_old_ = true;
            old_ = old;
        }
        set_cache_environment(path_);
    }

    ~TemporaryCacheDirectory() {
        try {
            if (had_old_) set_cache_environment(old_);
#ifdef _WIN32
            else _wputenv_s(L"NANOEMBED_CACHE_DIR", L"");
#else
            else unsetenv("NANOEMBED_CACHE_DIR");
#endif
        } catch (...) {
        }
        std::error_code ignored;
        std::filesystem::remove_all(path_, ignored);
    }

    const std::filesystem::path & path() const { return path_; }

private:
    std::filesystem::path path_;
    std::filesystem::path old_;
    bool had_old_ = false;
};

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

    // Both supported families wrap content in two required special tokens.
    // Returning those two under a declared cap of one would violate the
    // tokenizer contract and make the graph allocator grow past its context.
    bool rejected_too_small = false;
    try {
        (void) tok->encode(long_text, 1);
    } catch (const nanoembed::TokenizerError &) {
        rejected_too_small = true;
    }
    EXPECT_TRUE(rejected_too_small);

    return g_failures == 0;
}

bool test_wordpiece_rejects_out_of_vocab_special_id() {
    gguf_context * g = gguf_init_empty();
    const char * tokens[] = {"[PAD]", "[UNK]", "[CLS]", "[SEP]"};
    gguf_set_arr_str(g, "tokenizer.ggml.tokens", tokens, 4);
    gguf_set_val_u32(g, "tokenizer.ggml.cls_token_id", 99);
    gguf_set_val_u32(g, "tokenizer.ggml.seperator_token_id", 3);
    gguf_set_val_u32(g, "tokenizer.ggml.padding_token_id", 0);
    gguf_set_val_u32(g, "tokenizer.ggml.unknown_token_id", 1);

    bool threw = false;
    try {
        (void) nanoembed::WordPieceTokenizer::from_gguf(g);
    } catch (const nanoembed::TokenizerError & e) {
        threw = std::string(e.what()).find("outside the tokenizer vocabulary") !=
                std::string::npos;
    }
    gguf_free(g);
    EXPECT_TRUE(threw);
    return g_failures == 0;
}

bool test_bpe_rejects_wrong_optional_type() {
    gguf_context * g = gguf_init_empty();
    const char * tokens[] = {"<s>", "</s>", "<unk>", "a", "b", "ab"};
    const char * merges[] = {"a b"};
    gguf_set_arr_str(g, "tokenizer.ggml.tokens", tokens, 6);
    gguf_set_arr_str(g, "tokenizer.ggml.merges", merges, 1);
    gguf_set_val_bool(g, "tokenizer.ggml.bos_token_id", true);
    gguf_set_val_u32(g, "tokenizer.ggml.unknown_token_id", 2);

    bool threw = false;
    try {
        (void) nanoembed::SpmBpeTokenizer::from_gguf(g);
    } catch (const nanoembed::TokenizerError & e) {
        const std::string msg = e.what();
        threw = msg.find("bos_token_id") != std::string::npos &&
                msg.find("wrong type") != std::string::npos;
    }
    gguf_free(g);
    EXPECT_TRUE(threw);
    return g_failures == 0;
}

gguf_context * make_synthetic_bpe(std::vector<const char *> tokens,
                                  std::vector<const char *> merges) {
    gguf_context * g = gguf_init_empty();
    gguf_set_arr_str(g, "tokenizer.ggml.tokens", tokens.data(), tokens.size());
    gguf_set_arr_str(g, "tokenizer.ggml.merges", merges.data(), merges.size());
    gguf_set_val_u32(g, "tokenizer.ggml.unknown_token_id", 0);
    return g;
}

bool test_sha256_vectors() {
    const auto empty = nanoembed::detail::sha256("", 0);
    const auto abc = nanoembed::detail::sha256("abc", 3);
    EXPECT_TRUE(nanoembed::detail::sha256_hex(empty) ==
                "e3b0c44298fc1c149afbf4c8996fb924"
                "27ae41e4649b934ca495991b7852b855");
    EXPECT_TRUE(nanoembed::detail::sha256_hex(abc) ==
                "ba7816bf8f01cfea414140de5dae2223"
                "b00361a396177a9cb410ff61f20015ad");
    return g_failures == 0;
}

void flip_file_byte(const std::filesystem::path & path, std::streamoff offset) {
    std::fstream file(path, std::ios::in | std::ios::out | std::ios::binary);
    if (!file) throw std::runtime_error("cannot open cache for corruption test");
    file.seekg(offset);
    char byte = 0;
    file.read(&byte, 1);
    if (!file) throw std::runtime_error("cannot read cache byte for corruption test");
    byte ^= static_cast<char>(0x5a);
    file.seekp(offset);
    file.write(&byte, 1);
    if (!file) throw std::runtime_error("cannot write cache byte for corruption test");
}

bool test_bpe_disk_cache_lifecycle() {
    const std::vector<const char *> tokens = {"<unk>", "a", "b", "c", "ab", "abc", "bc"};
    const std::vector<const char *> merges = {"a b", "ab c", "a b"};
    gguf_context * g = make_synthetic_bpe(tokens, merges);

    auto cold = nanoembed::SpmBpeTokenizer::from_gguf(g);
    EXPECT_TRUE(!cold.merge_cache_hit());
    EXPECT_TRUE(cold.merge_record_count() == 2);
    EXPECT_TRUE(cold.merge_fence_bytes() <= 64 * 1024);
    EXPECT_TRUE(cold.encode("abc") == std::vector<int>({5}));
    const std::filesystem::path path = cold.merge_cache_path();
    EXPECT_TRUE(std::filesystem::file_size(path) == 3 * 4096);
    nanoembed::DiskMergeIndex direct = nanoembed::DiskMergeIndex::from_gguf(
        g, gguf_find_key(g, "tokenizer.ggml.tokens"),
        gguf_find_key(g, "tokenizer.ggml.merges"));
    nanoembed::DiskMergeIndex::LookupScratch scratch;
    nanoembed::DiskMergeIndex::Rule duplicate_rule;
    EXPECT_TRUE(direct.find((uint64_t{1} << 32) | 2, scratch, duplicate_rule));
    EXPECT_EQ_INT(duplicate_rule.rank, 0);
    EXPECT_EQ_INT(duplicate_rule.merged, 4);
    nanoembed::DiskMergeIndex::Rule missing_rule;
    EXPECT_TRUE(!direct.find((uint64_t{3} << 32) | 1, scratch, missing_rule));

    auto warm = nanoembed::SpmBpeTokenizer::from_gguf(g);
    EXPECT_TRUE(warm.merge_cache_hit());
    EXPECT_TRUE(warm.encode("abc") == std::vector<int>({5}));

    // Both header and payload damage must be noticed before metadata can be
    // discarded, rebuilt atomically, and then produce the same IDs.
    flip_file_byte(path, 8);
    auto header_rebuilt = nanoembed::SpmBpeTokenizer::from_gguf(g);
    EXPECT_TRUE(!header_rebuilt.merge_cache_hit());
    EXPECT_TRUE(header_rebuilt.encode("abc") == std::vector<int>({5}));

    flip_file_byte(path, 12);
    auto endian_rebuilt = nanoembed::SpmBpeTokenizer::from_gguf(g);
    EXPECT_TRUE(!endian_rebuilt.merge_cache_hit());
    EXPECT_TRUE(endian_rebuilt.encode("abc") == std::vector<int>({5}));

    flip_file_byte(path, 72);
    auto digest_rebuilt = nanoembed::SpmBpeTokenizer::from_gguf(g);
    EXPECT_TRUE(!digest_rebuilt.merge_cache_hit());
    EXPECT_TRUE(digest_rebuilt.encode("abc") == std::vector<int>({5}));

    flip_file_byte(path, 4096);
    auto payload_rebuilt = nanoembed::SpmBpeTokenizer::from_gguf(g);
    EXPECT_TRUE(!payload_rebuilt.merge_cache_hit());
    EXPECT_TRUE(payload_rebuilt.encode("abc") == std::vector<int>({5}));

    // A record edited in place after load is NOT detected, and this asserts
    // that deliberately. The per-page CRC used to run on every page read and
    // would have caught it, but it measured 96% of encode time, so encode now
    // performs structural checks only. Editing the first record's key hides
    // the "a b" merge, and encode silently returns the unmerged pieces instead
    // of the merged ID -- a plausible wrong answer, which is the risk this
    // trade accepts. Load-time SHA-256 still rejects the same damage, so the
    // window is only between one load and the next.
    flip_file_byte(path, 2 * 4096 + 16);
    EXPECT_TRUE(payload_rebuilt.encode("abc") == std::vector<int>({1, 2, 3}));
    flip_file_byte(path, 2 * 4096 + 16);

    // A merged ID is a row index into the token embedding table, and ggml
    // aborts the process on an out-of-range row rather than raising anything a
    // caller could catch. Corrupting the first record's merged field must
    // therefore surface as an ordinary TokenizerError, not a crash.
    flip_file_byte(path, 2 * 4096 + 16 + 12);
    bool out_of_vocab_rejected = false;
    try {
        (void) payload_rebuilt.encode("abc");
    } catch (const nanoembed::TokenizerError &) {
        out_of_vocab_rejected = true;
    }
    EXPECT_TRUE(out_of_vocab_rejected);
    flip_file_byte(path, 2 * 4096 + 16 + 12);

    flip_file_byte(path, 2 * 4096 + 16);
    auto runtime_rebuilt = nanoembed::SpmBpeTokenizer::from_gguf(g);
    EXPECT_TRUE(!runtime_rebuilt.merge_cache_hit());
    EXPECT_TRUE(runtime_rebuilt.encode("abc") == std::vector<int>({5}));

    std::filesystem::resize_file(path, 2 * 4096);
    bool short_read_rejected = false;
    try {
        (void) runtime_rebuilt.encode("abc");
    } catch (const nanoembed::TokenizerError &) {
        short_read_rejected = true;
    }
    EXPECT_TRUE(short_read_rejected);
    auto truncation_rebuilt = nanoembed::SpmBpeTokenizer::from_gguf(g);
    EXPECT_TRUE(!truncation_rebuilt.merge_cache_hit());
    EXPECT_TRUE(truncation_rebuilt.encode("abc") == std::vector<int>({5}));

    // Changing either source array produces a distinct content-addressed
    // cache rather than accepting an index for another tokenizer.
    const std::vector<const char *> other_merges = {"b c"};
    gguf_context * other = make_synthetic_bpe(tokens, other_merges);
    auto distinct = nanoembed::SpmBpeTokenizer::from_gguf(other);
    EXPECT_TRUE(distinct.merge_cache_path() != path);

    const auto valid_root = path.parent_path();
    const auto invalid_root = valid_root / "not-a-directory";
    { std::ofstream marker(invalid_root); marker << "file"; }
    set_cache_environment(invalid_root);
    bool unwritable_rejected = false;
    try {
        (void) nanoembed::SpmBpeTokenizer::from_gguf(g);
    } catch (const nanoembed::TokenizerError &) {
        unwritable_rejected = true;
    }
    set_cache_environment(valid_root);
    EXPECT_TRUE(unwritable_rejected);
    gguf_free(other);
    gguf_free(g);
    return g_failures == 0;
}

bool test_bpe_concurrent_cache_creation() {
    const std::vector<const char *> tokens = {"<unk>", "a", "b", "ab"};
    const std::vector<const char *> merges = {"a b"};
    gguf_context * seed = make_synthetic_bpe(tokens, merges);
    auto existing = nanoembed::SpmBpeTokenizer::from_gguf(seed);
    const auto path = existing.merge_cache_path();
    gguf_free(seed);
    std::filesystem::remove(path);

    std::atomic<bool> start{false};
    std::atomic<int> ready{0};
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&] {
            ++ready;
            while (!start.load(std::memory_order_acquire)) std::this_thread::yield();
            try {
                gguf_context * g = make_synthetic_bpe(tokens, merges);
                auto tokenizer = nanoembed::SpmBpeTokenizer::from_gguf(g);
                if (tokenizer.encode("ab") != std::vector<int>({3})) ++failures;
                gguf_free(g);
            } catch (...) {
                ++failures;
            }
        });
    }
    while (ready.load() != 4) std::this_thread::yield();
    start.store(true, std::memory_order_release);
    for (auto & thread : threads) thread.join();
    EXPECT_EQ_INT(failures.load(), 0);

    gguf_context * verify_g = make_synthetic_bpe(tokens, merges);
    auto verify = nanoembed::SpmBpeTokenizer::from_gguf(verify_g);
    EXPECT_TRUE(verify.merge_cache_hit());
    EXPECT_TRUE(verify.encode("ab") == std::vector<int>({3}));
    gguf_free(verify_g);

#ifndef _WIN32
    // A rename that is safe only between threads can still expose a partial
    // file between processes. Re-run the same race with distinct PIDs so temp
    // naming and atomic replacement are covered too.
    std::filesystem::remove(path);
    std::vector<pid_t> children;
    for (int i = 0; i < 4; ++i) {
        const pid_t pid = fork();
        if (pid == 0) {
            try {
                gguf_context * child_g = make_synthetic_bpe(tokens, merges);
                auto child_tokenizer = nanoembed::SpmBpeTokenizer::from_gguf(child_g);
                const bool ok = child_tokenizer.encode("ab") == std::vector<int>({3});
                gguf_free(child_g);
                _exit(ok ? 0 : 1);
            } catch (...) {
                _exit(1);
            }
        }
        if (pid < 0) {
            ++failures;
            break;
        }
        children.push_back(pid);
    }
    for (const pid_t child : children) {
        int child_status = 0;
        if (waitpid(child, &child_status, 0) != child ||
            !WIFEXITED(child_status) || WEXITSTATUS(child_status) != 0) {
            ++failures;
        }
    }
    EXPECT_EQ_INT(failures.load(), 0);
    gguf_context * process_verify_g = make_synthetic_bpe(tokens, merges);
    auto process_verify = nanoembed::SpmBpeTokenizer::from_gguf(process_verify_g);
    EXPECT_TRUE(process_verify.merge_cache_hit());
    EXPECT_TRUE(process_verify.encode("ab") == std::vector<int>({3}));
    gguf_free(process_verify_g);
#endif
    for (const auto & entry : std::filesystem::directory_iterator(path.parent_path())) {
        const std::string name = entry.path().filename().string();
        EXPECT_TRUE(name.find(path.filename().string() + ".tmp.") != 0);
    }
    return g_failures == 0;
}

bool test_discard_consumed_tokenizer_metadata() {
    gguf_context * g = gguf_init_empty();
    const char * tokens[] = {"<unk>", "a", "b", "ab"};
    const char * merges[] = {"a b"};
    const float scores[] = {0.0f, 1.0f, 2.0f, 3.0f};
    const int32_t token_types[] = {0, 1, 1, 1};

    gguf_set_val_str(g, "tokenizer.ggml.model", "llama");
    gguf_set_arr_str(g, "tokenizer.ggml.tokens", tokens, 4);
    gguf_set_arr_str(g, "tokenizer.ggml.merges", merges, 1);
    gguf_set_arr_data(g, "tokenizer.ggml.scores", GGUF_TYPE_FLOAT32,
                      scores, 4);
    gguf_set_arr_data(g, "tokenizer.ggml.token_type", GGUF_TYPE_INT32,
                      token_types, 4);
    gguf_set_val_u32(g, "tokenizer.ggml.unknown_token_id", 0);

    auto tok = nanoembed::create_tokenizer(g);
    const std::vector<int> before = tok->encode("ab");

    nanoembed::discard_consumed_tokenizer_metadata(g);

    EXPECT_TRUE(gguf_find_key(g, "tokenizer.ggml.tokens") < 0);
    EXPECT_TRUE(gguf_find_key(g, "tokenizer.ggml.merges") < 0);
    EXPECT_TRUE(gguf_find_key(g, "tokenizer.ggml.scores") < 0);
    EXPECT_TRUE(gguf_find_key(g, "tokenizer.ggml.token_type") < 0);
    EXPECT_TRUE(gguf_find_key(g, "tokenizer.ggml.model") >= 0);
    EXPECT_TRUE(gguf_find_key(g, "tokenizer.ggml.unknown_token_id") >= 0);
    EXPECT_TRUE(tok->encode("ab") == before);

    gguf_free(g);
    return g_failures == 0;
}

const ModelUnderTest kModels[] = {
    {"bert",   "NANOEMBED_TEST_MODEL",        "NANOEMBED_TOKENIZER_FIXTURE"},
    {"gemma3", "NANOEMBED_TEST_MODEL_GEMMA3", "NANOEMBED_TOKENIZER_FIXTURE_GEMMA3"},
};

} // namespace

int main() {
    TemporaryCacheDirectory cache;
    int rc = 0;
    g_failures = 0;
    if (!test_sha256_vectors()) rc = 1;
    g_failures = 0;
    if (!test_bpe_disk_cache_lifecycle()) rc = 1;
    g_failures = 0;
    if (!test_bpe_concurrent_cache_creation()) rc = 1;
    g_failures = 0;
    if (!test_wordpiece_rejects_out_of_vocab_special_id()) rc = 1;
    g_failures = 0;
    if (!test_bpe_rejects_wrong_optional_type()) rc = 1;
    g_failures = 0;
    if (!test_discard_consumed_tokenizer_metadata()) rc = 1;
    for (const ModelUnderTest & m : kModels) {
        g_failures = 0; if (!test_metadata(m))  rc = 1;
        g_failures = 0; if (!test_hf_parity(m)) rc = 1;
        g_failures = 0; if (!test_truncation(m)) rc = 1;
    }
    std::printf("tokenizer_test: %s\n", rc == 0 ? "ok" : "FAIL");
    return rc;
}
