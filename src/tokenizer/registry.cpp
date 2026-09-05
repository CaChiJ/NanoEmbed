#include "tokenizer.h"

#include "spm_bpe.h"
#include "wordpiece.h"

#include "gguf.h"

namespace nanoembed {

namespace {

std::string read_tokenizer_family(gguf_context * ctx) {
    const int64_t k = gguf_find_key(ctx, "tokenizer.ggml.model");
    if (k < 0) {
        throw TokenizerError("GGUF has no tokenizer.ggml.model key");
    }
    if (gguf_get_kv_type(ctx, k) != GGUF_TYPE_STRING) {
        throw TokenizerError("tokenizer.ggml.model is not a string");
    }
    return gguf_get_val_str(ctx, k);
}

} // namespace

std::unique_ptr<Tokenizer> create_tokenizer(gguf_context * ctx) {
    const std::string family = read_tokenizer_family(ctx);

    if (family == "bert") {
        return std::make_unique<WordPieceTokenizer>(WordPieceTokenizer::from_gguf(ctx));
    }

    // Gemma 3 (and Llama) vocabs declare themselves as "llama". The tag names
    // the vocab flavour, not the algorithm — SpmBpeTokenizer requires a merge
    // table and says so if the file is really a unigram SentencePiece model.
    if (family == "llama") {
        return std::make_unique<SpmBpeTokenizer>(SpmBpeTokenizer::from_gguf(ctx));
    }

    // "gpt2" is byte-level BPE, a different pre-tokenization and byte mapping
    // from the SentencePiece flavour above. Named explicitly so the error is
    // actionable rather than a generic "unsupported".
    if (family == "gpt2") {
        throw TokenizerError(
            "tokenizer family 'gpt2' (byte-level BPE) is recognized but not "
            "implemented");
    }

    throw TokenizerError("unsupported tokenizer family: '" + family + "'");
}

void discard_consumed_tokenizer_metadata(gguf_context * ctx) noexcept {
    if (ctx == nullptr) return;

    // These arrays dominate the GGUF metadata heap. Both supported tokenizer
    // implementations copy/compile everything they need from them during
    // create_tokenizer(); retaining the GGUF copies afterwards is redundant.
    // Keep scalar IDs, flags, and tokenizer.ggml.model so diagnostics can
    // still describe the configured tokenizer family and wrapper.
    constexpr const char * kConsumedArrays[] = {
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.merges",
        "tokenizer.ggml.scores",
        "tokenizer.ggml.token_type",
    };
    for (const char * key : kConsumedArrays) {
        (void) gguf_remove_key(ctx, key);
    }
}

} // namespace nanoembed
