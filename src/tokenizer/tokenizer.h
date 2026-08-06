// Tokenizer interface + registry.
//
// A GGUF states its tokenizer family in `tokenizer.ggml.model`, independently
// of `general.architecture`: bge-small says "bert" (WordPiece), jina-v5-nano
// says "gpt2" (byte-level BPE). Model swapping therefore needs this seam as
// well as the architecture one — the two vary separately.
//
// Adding a family means implementing Tokenizer and adding one line to
// create_tokenizer(); nothing else in the tree changes.

#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct gguf_context;

namespace nanoembed {

class TokenizerError : public std::runtime_error {
public:
    explicit TokenizerError(const std::string & w) : std::runtime_error(w) {}
};

class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    // Token IDs including whatever special tokens the family prepends or
    // appends, truncated to fit the effective limit.
    //
    // max_seq_len_override > 0 overrides the configured limit for this call;
    // 0 means "use the configured value".
    virtual std::vector<int> encode(const std::string & text,
                                    int                 max_seq_len_override = 0) const = 0;

    virtual int  max_seq_len() const noexcept = 0;
    virtual int  vocab_size()  const noexcept = 0;
    virtual void set_max_seq_len(int n)       = 0;
};

// Build the tokenizer a GGUF declares. Throws TokenizerError if the family is
// unknown or its metadata is missing.
std::unique_ptr<Tokenizer> create_tokenizer(gguf_context * ctx);

} // namespace nanoembed
