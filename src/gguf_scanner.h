// GGUF scanner: turn ggml's flat tensor list into a typed BERT manifest.
//
// Scope:
//   - Open a GGUF file (header only, no_alloc=true).
//   - Validate general.architecture == "bert".
//   - Read BERT hyperparameters from metadata KV.
//   - Map every required tensor name to a TensorRef slot.
//   - Fail fast at scan time if anything is missing or shaped wrong.
//
// Everything here is BERT-specific by design — it is this family's private
// detail, not a shared contract. The family-agnostic parts (TensorRef,
// ScanError, the KV and tensor-lookup helpers) live in gguf_util.h so other
// architectures can reuse them without inheriting BERT's assumptions.
//
// The resulting ScanResult owns the gguf_context. Inference code borrows it
// to read tensor data via ggml.

#pragma once

#include "gguf_util.h"

#include <string>
#include <vector>

// Forward-declare ggml types — keeps this header free of ggml.h.
struct gguf_context;
struct ggml_context;

namespace nanoembed {

// One BERT encoder block. v0 assumes post-LN BERT:
//   x_attn = LN_attn(x + Attn(x))
//   x_out  = LN_layer(x_attn + FFN(x_attn))
struct LayerSlot {
    TensorRef attn_q_w,       attn_q_b;
    TensorRef attn_k_w,       attn_k_b;
    TensorRef attn_v_w,       attn_v_b;
    TensorRef attn_o_w,       attn_o_b;
    TensorRef attn_norm_w,    attn_norm_b;     // post-attention LN
    TensorRef ffn_up_w,       ffn_up_b;
    TensorRef ffn_down_w,     ffn_down_b;
    TensorRef ffn_norm_w,     ffn_norm_b;      // post-FFN LN
};

// BERT hyperparameters pulled from GGUF metadata.
struct BertArch {
    int   n_layer        = 0;     // bert.block_count
    int   n_embed        = 0;     // bert.embedding_length (hidden_size)
    int   n_head         = 0;     // bert.attention.head_count
    int   n_ff           = 0;     // bert.feed_forward_length (intermediate)
    int   n_vocab        = 0;     // tokenizer vocab size
    int   max_seq_len    = 0;     // bert.context_length
    float layer_norm_eps = 1e-12f;

    // bert.pooling_type, in llama.cpp's numbering (1=mean, 2=cls, 3=last).
    // Defaults to mean when the file does not say: that is both the more
    // common BERT convention and what the public API returned before pooling
    // began following the model.
    int   pooling_type   = 1;
};

struct ModelManifest {
    BertArch                arch;
    std::vector<LayerSlot>  layers;

    // Embedding-stage tensors
    TensorRef tok_embed_w;       // token_embd.weight
    TensorRef pos_embed_w;       // position_embd.weight
    TensorRef type_embed_w;      // token_types.weight  (segment embeddings)
    TensorRef embed_norm_w,      // token_embd_norm.weight
              embed_norm_b;      // token_embd_norm.bias

    // Optional sentence-transformer pooler (often unused — kept for completeness)
    TensorRef pooler_w, pooler_b;
};

// Owns the gguf_context + ggml meta_context for the manifest's lifetime.
// gguf_init_from_file with no_alloc=true creates a ggml_context that holds
// tensor *metadata only* (shape, dtype, offset). M3+ inference reads tensor
// data via ggml_get_tensor on this context.
//
// RAII; non-copyable, movable.
class ScanResult {
public:
    ScanResult() = default;
    ~ScanResult();

    ScanResult(const ScanResult &)             = delete;
    ScanResult & operator=(const ScanResult &) = delete;
    ScanResult(ScanResult && other) noexcept;
    ScanResult & operator=(ScanResult && other) noexcept;

    const ModelManifest & manifest() const noexcept { return manifest_; }
    gguf_context *        gguf()     const noexcept { return gguf_ctx_; }
    ggml_context *        meta_ctx() const noexcept { return meta_ctx_; }

    // Internal: scan_gguf populates these.
    void reset(gguf_context * gguf, ggml_context * meta, ModelManifest m);

private:
    void release() noexcept;

    gguf_context * gguf_ctx_ = nullptr;
    ggml_context * meta_ctx_ = nullptr;
    ModelManifest  manifest_ = {};
};

// Scan a GGUF file into a typed BERT manifest. Throws ScanError on failure.
ScanResult scan_gguf(const std::string & path);

// Validate BERT metadata and required tensor names/shapes in an already-open
// metadata-only GGUF context.  The caller retains both contexts.  The mapped
// weight loader uses this overload so architecture validation and the later
// tensor-data borrows refer to one opened file identity.
ModelManifest scan_bert(gguf_context * gguf, ggml_context * meta);

} // namespace nanoembed
