#include "tokenizer.h"

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

    // "gpt2" (byte-level BPE) is what jina-embeddings-v5-text-nano ships; it
    // lands with the eurobert milestone. Naming it explicitly keeps the error
    // actionable instead of a generic "unsupported".
    if (family == "gpt2") {
        throw TokenizerError(
            "tokenizer family 'gpt2' (byte-level BPE) is not implemented yet — "
            "planned with eurobert support (PLAN.md M3.6)");
    }

    throw TokenizerError("unsupported tokenizer family: '" + family + "'");
}

} // namespace nanoembed
