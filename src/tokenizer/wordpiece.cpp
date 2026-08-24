#include "wordpiece.h"

#include "gguf.h"

#include <cstdint>
#include <cstring>
#include <limits>

namespace nanoembed {

namespace {

// HuggingFace BertTokenizer drops words longer than this.
constexpr size_t kMaxInputCharsPerWord = 100;

bool is_ascii_whitespace(unsigned char c) {
    // Matches BERT BasicTokenizer._is_whitespace for the ASCII range.
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

bool is_ascii_control(unsigned char c) {
    // \t \n \r are whitespace, not control, in BERT.
    if (c == '\t' || c == '\n' || c == '\r') return false;
    return c < 0x20 || c == 0x7f;
}

bool is_ascii_punct(unsigned char c) {
    // Mirrors HF: ASCII 33-47, 58-64, 91-96, 123-126.
    return (c >= 33 && c <= 47) ||
           (c >= 58 && c <= 64) ||
           (c >= 91 && c <= 96) ||
           (c >= 123 && c <= 126);
}

unsigned u32_from_kv(gguf_context * ctx, const char * key) {
    int64_t k = gguf_find_key(ctx, key);
    if (k < 0) {
        throw TokenizerError(std::string("missing tokenizer metadata key: ") + key);
    }
    if (gguf_get_kv_type(ctx, k) != GGUF_TYPE_UINT32) {
        throw TokenizerError(std::string("wrong type for ") + key);
    }
    const uint32_t value = gguf_get_val_u32(ctx, k);
    if (value > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
        throw TokenizerError(std::string("token id is out of int range: ") + key);
    }
    return value;
}

// llama.cpp's BERT GGUF converter stores vocab in SentencePiece-style notation:
//   - begin-of-word piece = "▁X"  (U+2581 LOWER ONE EIGHTH BLOCK prefix)
//   - sub-word piece      = "X"   (no prefix)
// HuggingFace BertTokenizer uses WordPiece-style notation:
//   - begin-of-word piece = "X"   (no prefix)
//   - sub-word piece      = "##X"
// IDs line up between the two; only the string representation differs.
// Normalize at load time so our greedy WordPiece matcher can run on
// the familiar HF strings.
std::string normalize_piece(const char * raw) {
    static const char kBow[3] = {'\xE2', '\x96', '\x81'};  // U+2581 in UTF-8
    constexpr size_t kBowSize = sizeof(kBow);

    const size_t len = std::strlen(raw);

    // Special tokens like [CLS], [PAD], [UNK], [unusedN] — keep as-is.
    if (len > 0 && raw[0] == '[') {
        return std::string(raw, len);
    }
    if (len >= kBowSize && std::memcmp(raw, kBow, kBowSize) == 0) {
        return std::string(raw + kBowSize, len - kBowSize);
    }
    return std::string("##") + raw;
}

} // namespace

// ---- Construction ----------------------------------------------------------

WordPieceTokenizer WordPieceTokenizer::from_gguf(gguf_context * ctx) {
    if (ctx == nullptr) {
        throw TokenizerError("from_gguf: null gguf context");
    }

    WordPieceTokenizer t;

    // Special token IDs. Note: GGUF spells the SEP key as "seperator" (sic).
    t.cfg_.cls_id = static_cast<int>(u32_from_kv(ctx, "tokenizer.ggml.cls_token_id"));
    t.cfg_.sep_id = static_cast<int>(u32_from_kv(ctx, "tokenizer.ggml.seperator_token_id"));
    t.cfg_.pad_id = static_cast<int>(u32_from_kv(ctx, "tokenizer.ggml.padding_token_id"));
    t.cfg_.unk_id = static_cast<int>(u32_from_kv(ctx, "tokenizer.ggml.unknown_token_id"));

    // Vocab.
    int64_t tk = gguf_find_key(ctx, "tokenizer.ggml.tokens");
    if (tk < 0 || gguf_get_kv_type(ctx, tk) != GGUF_TYPE_ARRAY) {
        throw TokenizerError("tokenizer.ggml.tokens missing or not an array");
    }
    if (gguf_get_arr_type(ctx, tk) != GGUF_TYPE_STRING) {
        throw TokenizerError("tokenizer.ggml.tokens is not an array of strings");
    }

    const size_t n = gguf_get_arr_n(ctx, tk);
    if (n == 0 || n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw TokenizerError("tokenizer.ggml.tokens has an unsupported size");
    }
    t.vocab_.reserve(n);
    t.ix_.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        std::string piece = normalize_piece(gguf_get_arr_str(ctx, tk, i));
        t.vocab_.emplace_back(piece);
        t.ix_.emplace(std::move(piece), static_cast<int>(i));
    }

    auto validate_special = [&](int id, const char * key) {
        if (id < 0 || id >= static_cast<int>(n)) {
            throw TokenizerError(std::string(key) + " is outside the tokenizer vocabulary");
        }
    };
    validate_special(t.cfg_.cls_id, "tokenizer.ggml.cls_token_id");
    validate_special(t.cfg_.sep_id, "tokenizer.ggml.seperator_token_id");
    validate_special(t.cfg_.pad_id, "tokenizer.ggml.padding_token_id");
    validate_special(t.cfg_.unk_id, "tokenizer.ggml.unknown_token_id");

    return t;
}

// ---- BasicTokenizer --------------------------------------------------------

std::vector<std::string> WordPieceTokenizer::basic_tokenize(const std::string & text) const {
    std::vector<std::string> out;
    std::string cur;
    cur.reserve(32);

    auto flush = [&]() {
        if (!cur.empty()) {
            out.push_back(cur);
            cur.clear();
        }
    };

    for (size_t i = 0; i < text.size(); ++i) {
        unsigned char c = static_cast<unsigned char>(text[i]);

        // Strip ASCII control characters by replacing with whitespace.
        if (is_ascii_control(c)) {
            flush();
            continue;
        }
        if (is_ascii_whitespace(c)) {
            flush();
            continue;
        }
        if (is_ascii_punct(c)) {
            flush();
            out.emplace_back(1, static_cast<char>(c));
            continue;
        }

        // Lowercase ASCII A-Z. (Non-ASCII bytes are passed through verbatim;
        // see the file-level note about the accuracy oracle.)
        if (cfg_.do_lower_case && c >= 'A' && c <= 'Z') {
            c = static_cast<unsigned char>(c - 'A' + 'a');
        }

        cur.push_back(static_cast<char>(c));
    }
    flush();

    return out;
}

// ---- WordPiece (greedy longest-prefix) -------------------------------------

std::vector<int> WordPieceTokenizer::wordpiece(const std::string & word) const {
    std::vector<int> out;

    if (word.size() > kMaxInputCharsPerWord) {
        out.push_back(cfg_.unk_id);
        return out;
    }

    size_t start = 0;
    bool   ok    = true;

    while (start < word.size()) {
        size_t end      = word.size();
        int    piece_id = -1;

        while (start < end) {
            std::string substr = word.substr(start, end - start);
            if (start > 0) {
                substr.insert(0, "##");
            }
            auto it = ix_.find(substr);
            if (it != ix_.end()) {
                piece_id = it->second;
                break;
            }
            --end;
        }

        if (piece_id < 0) {
            ok = false;
            break;
        }
        out.push_back(piece_id);
        start = end;
    }

    if (!ok || out.empty()) {
        out.clear();
        out.push_back(cfg_.unk_id);
    }
    return out;
}

// ---- encode ----------------------------------------------------------------

std::vector<int> WordPieceTokenizer::encode(const std::string & text,
                                             int                 max_seq_len_override) const {
    const int limit = (max_seq_len_override > 0) ? max_seq_len_override : cfg_.max_seq_len;

    std::vector<int> ids;
    if (limit < 2) {
        throw TokenizerError("WordPiece max sequence length must be at least 2");
    }

    ids.reserve(static_cast<size_t>(limit));
    ids.push_back(cfg_.cls_id);

    const std::vector<std::string> words = basic_tokenize(text);

    bool truncated = false;
    for (const auto & w : words) {
        const std::vector<int> pieces = wordpiece(w);
        for (int id : pieces) {
            if (static_cast<int>(ids.size()) >= limit - 1) {
                truncated = true;
                break;
            }
            ids.push_back(id);
        }
        if (truncated) break;
    }

    ids.push_back(cfg_.sep_id);
    return ids;
}

} // namespace nanoembed
