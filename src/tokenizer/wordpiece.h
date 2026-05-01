// BERT WordPiece tokenizer.
//
// Pulls vocab + special tokens from a GGUF context (the same tokenizer.ggml.*
// metadata that llama.cpp's BERT converter emits) and produces token IDs
// matching HuggingFace's BertTokenizer for the bge-small-en-v1.5 model.
//
// Scope (M3a): English BERT (do_lower_case=true). ASCII whitespace +
// punctuation handling. Non-ASCII text passes through untouched (no
// Unicode NFD / accent stripping yet — accuracy oracle is the HF token-ID
// fixture, so any divergence shows up there and we patch in response).

#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct gguf_context;

namespace nanoembed {

struct TokenizerConfig {
    int  cls_id        = 101;
    int  sep_id        = 102;
    int  pad_id        = 0;
    int  unk_id        = 100;
    int  max_seq_len   = 512;
    bool do_lower_case = true;
};

class TokenizerError : public std::runtime_error {
public:
    explicit TokenizerError(const std::string & w) : std::runtime_error(w) {}
};

class WordPieceTokenizer {
public:
    // Build by reading vocab + special token IDs from GGUF metadata.
    // Throws TokenizerError on missing keys.
    static WordPieceTokenizer from_gguf(gguf_context * ctx);

    // Tokenize. Returns token IDs INCLUDING leading [CLS] and trailing [SEP],
    // truncated so the result fits in max_seq_len.
    std::vector<int> encode(const std::string & text) const;

    int cls_id()      const noexcept { return cfg_.cls_id; }
    int sep_id()      const noexcept { return cfg_.sep_id; }
    int pad_id()      const noexcept { return cfg_.pad_id; }
    int unk_id()      const noexcept { return cfg_.unk_id; }
    int max_seq_len() const noexcept { return cfg_.max_seq_len; }
    int vocab_size()  const noexcept { return static_cast<int>(vocab_.size()); }

    // Read-only access to the loaded vocab (id -> piece string).
    const std::vector<std::string> & vocab() const noexcept { return vocab_; }

    // Override max sequence length (defaults to GGUF / config value, 512).
    void set_max_seq_len(int n) { cfg_.max_seq_len = n; }

private:
    TokenizerConfig                      cfg_;
    std::vector<std::string>             vocab_;  // id -> piece string
    std::unordered_map<std::string, int> ix_;     // piece string -> id

    std::vector<std::string> basic_tokenize(const std::string & text) const;
    std::vector<int>         wordpiece    (const std::string & word) const;
};

} // namespace nanoembed
