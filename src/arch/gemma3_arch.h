// ModelArch implementation for `general.architecture == "gemma3"`.
//
// Target: microsoft/harrier-oss-v1-270m — a Gemma 3 decoder fine-tuned into a
// text embedding model. It shares nothing structural with BERT: rotary
// positions instead of a learned table, RMSNorm instead of LayerNorm, a gated
// GeGLU MLP instead of a two-matrix one, no bias anywhere, causal attention,
// multi-query attention, and last-token pooling.
//
// The conventions below were verified against the actual GGUF and against
// transformers' Gemma3TextModel. They are the kind of detail that silently
// corrupts every downstream number if guessed, so do not re-derive them:
//
//   * RMSNorm. HF computes `x * (1 + w)`, and the converter folds that +1
//     into the stored weights — every norm tensor in the GGUF has a mean
//     exactly 1.0 above its HF counterpart, and `output_norm`'s minimum moves
//     from -1.0 to 0.0. The graph therefore applies a plain
//     `rms_norm(x) * w` and must NOT add 1 a second time.
//
//   * Embedding scale. `sqrt(n_embed)` is NOT folded — token_embd matches the
//     HF tensor's statistics exactly — so the graph applies it after lookup.
//
//   * Attention scale is `1/sqrt(query_pre_attn_scalar)`, carried in the
//     non-standard key `gemma3.attention.key_length_scale`. It happens to
//     equal head_dim for this checkpoint, but upstream they are independent.
//
//   * Four RMSNorms per block (around attention, around the FFN) plus a final
//     `output_norm`. BERT has two per block and no final norm.
//
//   * head_dim (256) is not n_embed / n_head (640 / 4 = 160), and there is a
//     single KV head shared by all four query heads.
//
// One block, with every norm being RMSNorm:
//     h = x + post_attention_norm(attn(attn_norm(x)))
//     y = h + post_ffw_norm(geglu(ffn_norm(h)))

#pragma once

#include "arch/model_arch.h"
#include "gguf_util.h"

#include <string>
#include <vector>

namespace nanoembed {

// Per-block tensor slots, resolved at scan time. Every one is required;
// gemma3 has no optional per-layer tensor.
struct Gemma3LayerSlot {
    TensorRef attn_norm;             // blk.N.attn_norm.weight  (pre-attention)
    TensorRef attn_q, attn_k, attn_v, attn_output;
    TensorRef attn_q_norm, attn_k_norm;
    TensorRef post_attention_norm;
    TensorRef ffn_norm;              // blk.N.ffn_norm.weight   (pre-FFN)
    TensorRef post_ffw_norm;
    TensorRef ffn_gate, ffn_up, ffn_down;
};

struct Gemma3Manifest {
    ArchParams                   params;
    std::vector<Gemma3LayerSlot> layers;

    TensorRef tok_embed;             // token_embd.weight
    TensorRef output_norm;           // output_norm.weight (final, post-stack)

    float embed_scale = 0.0f;        // sqrt(n_embed); not folded into the weights
    float attn_scale  = 0.0f;        // 1/sqrt(query_pre_attn_scalar)

    PoolType pooling    = PoolType::Last;   // gemma3.pooling_type
    bool     normalize  = true;             // gemma3.normalize_embeddings
};

// Validate a gemma3 GGUF and resolve every tensor slot. Throws ScanError on
// a missing key, a missing tensor or a shape mismatch. Unlike scan_gguf this
// keeps no context open: the manifest is a pure validation artifact, and the
// weight data is read later through the Embedder's own GGUF handle.
Gemma3Manifest scan_gemma3(const std::string & gguf_path);

// Weight pointers resolved against the context that owns the tensor data.
struct Gemma3AttnWeights {
    ggml_tensor * norm;              // pre-attention RMSNorm gain
    ggml_tensor * q, * k, * v, * o;  // unbiased projections
    ggml_tensor * q_norm, * k_norm;  // QK-norm, applied before RoPE
    ggml_tensor * post_norm;         // post-attention RMSNorm gain
};

struct Gemma3FfnWeights {
    ggml_tensor * norm;              // pre-FFN RMSNorm gain
    ggml_tensor * gate, * up, * down;
    ggml_tensor * post_norm;         // post-FFN RMSNorm gain
};

struct Gemma3LayerWeights {
    Gemma3AttnWeights attn;
    Gemma3FfnWeights  ffn;
};

class Gemma3ModelArch : public ModelArch {
public:
    explicit Gemma3ModelArch(const std::string & gguf_path);

    const ArchParams & params() const noexcept override { return manifest_.params; }

    // RoPE consumes the same 0..S-1 ramp BERT uses to index its learned
    // position table, so the existing pos_ids input carries it. There are no
    // segment embeddings.
    InputRequirements inputs() const noexcept override {
        return InputRequirements{/*needs_pos_ids=*/true, /*needs_type_ids=*/false};
    }

    PoolType default_pooling() const noexcept override { return manifest_.pooling; }

    void bind_weights(ggml_context * model_ctx) override;

    ggml_tensor * build_graph(ggml_context *      gctx,
                              const GraphInputs & in) const override;

private:
    Gemma3Manifest                  manifest_;
    ggml_tensor *                   tok_embed_   = nullptr;
    ggml_tensor *                   output_norm_ = nullptr;
    std::vector<Gemma3LayerWeights> layer_w_;
};

} // namespace nanoembed
