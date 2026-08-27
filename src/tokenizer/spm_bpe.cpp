#include "spm_bpe.h"

#include "gguf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <queue>
#include <string>
#include <vector>

namespace nanoembed {

namespace {

// U+2581 LOWER ONE EIGHTH BLOCK — SentencePiece's space marker.
constexpr char kSpaceMarker[] = "\xE2\x96\x81";

// Byte length of the UTF-8 sequence starting with `c`. An invalid lead byte
// yields 1 so the scan always advances and the byte-fallback path handles it.
size_t utf8_len(unsigned char c) {
    if (c < 0x80)          return 1;
    if ((c >> 5) == 0x06)  return 2;
    if ((c >> 4) == 0x0E)  return 3;
    if ((c >> 3) == 0x1E)  return 4;
    return 1;
}

bool is_single_codepoint(const std::string & s) {
    return !s.empty() && utf8_len(static_cast<unsigned char>(s[0])) == s.size();
}

// "<0xNN>" -> byte value, or -1.
int parse_byte_token(const std::string & s) {
    if (s.size() != 6 || s[0] != '<' || s[1] != '0' || s[2] != 'x' || s[5] != '>') {
        return -1;
    }
    int v = 0;
    for (int i = 3; i <= 4; ++i) {
        const char c = s[static_cast<size_t>(i)];
        int d;
        if (c >= '0' && c <= '9')      d = c - '0';
        else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
        else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
        else return -1;
        v = v * 16 + d;
    }
    return v;
}

uint64_t pair_key(int left, int right) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(left)) << 32) |
            static_cast<uint32_t>(right);
}

int optional_id(gguf_context * ctx, const char * key) {
    const int64_t k = gguf_find_key(ctx, key);
    if (k < 0) return -1;
    switch (gguf_get_kv_type(ctx, k)) {
        case GGUF_TYPE_UINT32: {
            const uint32_t value = gguf_get_val_u32(ctx, k);
            if (value > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
                throw TokenizerError(std::string(key) + " is out of int range");
            }
            return static_cast<int>(value);
        }
        case GGUF_TYPE_INT32:  return gguf_get_val_i32(ctx, k);
        default:
            throw TokenizerError(std::string(key) + " has the wrong type");
    }
}

bool optional_flag(gguf_context * ctx, const char * key, bool fallback) {
    const int64_t k = gguf_find_key(ctx, key);
    if (k < 0) return fallback;
    if (gguf_get_kv_type(ctx, k) != GGUF_TYPE_BOOL) {
        throw TokenizerError(std::string(key) + " has the wrong type");
    }
    return gguf_get_val_bool(ctx, k);
}

int64_t require_str_array(gguf_context * ctx, const char * key) {
    const int64_t k = gguf_find_key(ctx, key);
    if (k < 0 || gguf_get_kv_type(ctx, k) != GGUF_TYPE_ARRAY) {
        throw TokenizerError(std::string(key) + " missing or not an array");
    }
    if (gguf_get_arr_type(ctx, k) != GGUF_TYPE_STRING) {
        throw TokenizerError(std::string(key) + " is not an array of strings");
    }
    return k;
}

} // namespace

// ---- Construction ----------------------------------------------------------

SpmBpeTokenizer SpmBpeTokenizer::from_gguf(gguf_context * ctx) {
    if (ctx == nullptr) {
        throw TokenizerError("from_gguf: null gguf context");
    }

    SpmBpeTokenizer t;
    std::fill(std::begin(t.byte_id_), std::end(t.byte_id_), -1);

    // ---- Vocab ----
    const int64_t tk = require_str_array(ctx, "tokenizer.ggml.tokens");
    const size_t  n  = gguf_get_arr_n(ctx, tk);
    if (n == 0 || n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw TokenizerError("tokenizer.ggml.tokens has an unsupported size");
    }

    t.vocab_size_ = static_cast<int>(n);
    for (size_t i = 0; i < n; ++i) {
        std::string piece = gguf_get_arr_str(ctx, tk, i);
        const int   id    = static_cast<int>(i);

        const int b = parse_byte_token(piece);
        if (b >= 0) t.byte_id_[b] = id;
        if (is_single_codepoint(piece)) t.char_ix_.emplace(piece, id);
    }

    // ---- Merges ----
    const int64_t mk = require_str_array(ctx, "tokenizer.ggml.merges");
    const size_t  nm = gguf_get_arr_n(ctx, mk);
    if (nm > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw TokenizerError("tokenizer.ggml.merges is too large");
    }
    t.merge_index_ = DiskMergeIndex::from_gguf(ctx, tk, mk);

    // ---- Special tokens ----
    t.bos_id_ = optional_id(ctx, "tokenizer.ggml.bos_token_id");
    t.eos_id_ = optional_id(ctx, "tokenizer.ggml.eos_token_id");
    t.pad_id_ = optional_id(ctx, "tokenizer.ggml.padding_token_id");
    t.unk_id_ = optional_id(ctx, "tokenizer.ggml.unknown_token_id");

    auto validate_optional_id = [&](int id, const char * key) {
        if (id < -1 || id >= t.vocab_size_) {
            throw TokenizerError(std::string(key) + " is outside the tokenizer vocabulary");
        }
    };
    validate_optional_id(t.bos_id_, "tokenizer.ggml.bos_token_id");
    validate_optional_id(t.eos_id_, "tokenizer.ggml.eos_token_id");
    validate_optional_id(t.pad_id_, "tokenizer.ggml.padding_token_id");
    validate_optional_id(t.unk_id_, "tokenizer.ggml.unknown_token_id");

    if (optional_flag(ctx, "tokenizer.ggml.add_bos_token", true) && t.bos_id_ >= 0) {
        t.prefix_id_ = t.bos_id_;
    }

    // What gets appended is stated twice and the two disagree for
    // harrier-oss-v1-270m: add_eos_token is false, yet HuggingFace appends
    // <eos> because its post-processor template says "<bos> A <eos>" —
    // independent of that flag. The converter recorded the real answer in
    // suffix_token_id, so that key wins when present.
    //
    // This is not cosmetic. The model pools the LAST token, so getting this
    // wrong means pooling the final content token instead of <eos> and
    // producing a completely different embedding, with nothing in the output
    // to suggest a tokenizer bug.
    const int suffix = optional_id(ctx, "tokenizer.ggml.suffix_token_id");
    validate_optional_id(suffix, "tokenizer.ggml.suffix_token_id");
    if (suffix >= 0) {
        t.suffix_id_ = suffix;
    } else if (optional_flag(ctx, "tokenizer.ggml.add_eos_token", false) && t.eos_id_ >= 0) {
        t.suffix_id_ = t.eos_id_;
    }

    // Every character has to be representable: either all 256 byte-fallback
    // tokens exist, or there is an unknown token to fall back to. Checked here
    // so encode() cannot fail partway through a sequence.
    const bool all_bytes =
        std::find(std::begin(t.byte_id_), std::end(t.byte_id_), -1) == std::end(t.byte_id_);
    if (!all_bytes && t.unk_id_ < 0) {
        throw TokenizerError(
            "vocab has neither a complete <0x00>..<0xFF> byte-fallback set nor "
            "an unknown-token id; out-of-vocabulary characters would be undecodable");
    }

    return t;
}

// ---- BPE -------------------------------------------------------------------

uint64_t SpmBpeTokenizer::bpe(const std::string & s,
                              int budget,
                              std::vector<int> & out) const {
    if (budget <= 0 || s.empty()) return 0;

    // Doubly-linked symbol list. `id < 0` marks a symbol absorbed by a merge.
    struct Symbol {
        int id;
        int prev;
        int next;
    };
    std::vector<Symbol> syms;
    syms.reserve(s.size());

    for (size_t i = 0; i < s.size();) {
        const size_t len = std::min(utf8_len(static_cast<unsigned char>(s[i])), s.size() - i);
        const std::string ch = s.substr(i, len);

        const auto it = char_ix_.find(ch);
        if (it != char_ix_.end()) {
            syms.push_back({it->second, 0, 0});
        } else {
            for (size_t b = 0; b < len; ++b) {
                const int id = byte_id_[static_cast<unsigned char>(s[i + b])];
                syms.push_back({id >= 0 ? id : unk_id_, 0, 0});
            }
        }
        i += len;
    }
    if (syms.empty()) return 0;

    const int n = static_cast<int>(syms.size());
    for (int i = 0; i < n; ++i) {
        syms[static_cast<size_t>(i)].prev = i - 1;
        syms[static_cast<size_t>(i)].next = (i + 1 < n) ? i + 1 : -1;
    }

    // Candidate merge. `lid`/`rid` record the operand IDs at push time so a
    // candidate invalidated by an earlier merge is recognised and dropped.
    struct Candidate {
        int rank;
        int left;
        int right;
        int lid;
        int rid;
        int merged;
    };
    // Lowest rank first; ties broken leftmost, which is what applying the
    // merge list in order does when the same pair occurs more than once.
    const auto worse = [](const Candidate & a, const Candidate & b) {
        if (a.rank != b.rank) return a.rank > b.rank;
        return a.left > b.left;
    };
    std::priority_queue<Candidate, std::vector<Candidate>, decltype(worse)> queue(worse);
    DiskMergeIndex::LookupScratch page_cache;

    const auto offer = [&](int l, int r) {
        if (l < 0 || r < 0) return;
        DiskMergeIndex::Rule rule;
        if (!merge_index_.find(pair_key(syms[static_cast<size_t>(l)].id,
                                        syms[static_cast<size_t>(r)].id),
                               page_cache, rule)) return;
        queue.push(Candidate{rule.rank, l, r,
                             syms[static_cast<size_t>(l)].id,
                             syms[static_cast<size_t>(r)].id,
                             rule.merged});
    };

    for (int i = 0; i + 1 < n; ++i) offer(i, i + 1);

    while (!queue.empty()) {
        const Candidate c = queue.top();
        queue.pop();

        Symbol & left = syms[static_cast<size_t>(c.left)];
        if (left.id != c.lid || left.next != c.right) continue;
        Symbol & right = syms[static_cast<size_t>(c.right)];
        if (right.id != c.rid) continue;

        left.id   = c.merged;
        left.next = right.next;
        if (right.next >= 0) syms[static_cast<size_t>(right.next)].prev = c.left;
        right.id = -1;

        offer(left.prev, c.left);
        offer(c.left, left.next);
    }

    int produced = 0;
    for (int i = 0; i >= 0 && produced < budget; i = syms[static_cast<size_t>(i)].next) {
        const int id = syms[static_cast<size_t>(i)].id;
        if (id < 0) continue;   // defensive; merged-away symbols are unlinked
        out.push_back(id);
        ++produced;
    }
    return page_cache.page_reads;
}

// ---- encode ----------------------------------------------------------------

std::vector<int> SpmBpeTokenizer::encode(const std::string & text,
                                         int                 max_seq_len_override) const {
    return encode_impl(text, max_seq_len_override).ids;
}

SpmBpeTokenizer::DiagnosticEncoding SpmBpeTokenizer::encode_with_diagnostics(
    const std::string & text, int max_seq_len_override) const {
    return encode_impl(text, max_seq_len_override);
}

SpmBpeTokenizer::DiagnosticEncoding SpmBpeTokenizer::encode_impl(
    const std::string & text, int max_seq_len_override) const {
    const int limit = max_seq_len_override > 0 ? max_seq_len_override : max_seq_len_;

    const int reserved = (prefix_id_ >= 0 ? 1 : 0) + (suffix_id_ >= 0 ? 1 : 0);
    if (limit < reserved) {
        throw TokenizerError("BPE max sequence length is smaller than its special-token wrapper");
    }
    const int budget = std::max(limit - reserved, 0);

    // Normalize: every ASCII space becomes the SentencePiece marker. Done as a
    // whole-string pass rather than fused into the scan because the marker is
    // an ordinary vocab character that takes part in merges like any other.
    std::string normalized;
    normalized.reserve(text.size());
    for (const char c : text) {
        if (c == ' ') normalized += kSpaceMarker;
        else          normalized += c;
    }

    std::vector<int> ids;
    ids.reserve(static_cast<size_t>(std::min(budget, 1024)) + 2);

    if (prefix_id_ >= 0) ids.push_back(prefix_id_);
    const uint64_t page_reads = bpe(normalized, budget, ids);
    if (suffix_id_ >= 0) ids.push_back(suffix_id_);

    return DiagnosticEncoding{std::move(ids), page_reads};
}

} // namespace nanoembed
