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
#include "forward/gated_ffn.h"
#include "forward/gqa_attention.h"
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

    PoolType pooling = PoolType::Last;      // gemma3.pooling_type
};

// Validate a gemma3 GGUF and resolve every tensor slot. Throws ScanError on
// a missing key, a missing tensor or a shape mismatch. Unlike scan_gguf this
// keeps no context open: the manifest is a pure validation artifact, and the
// weight data is read later through the Embedder's own GGUF handle.
Gemma3Manifest scan_gemma3(const std::string & gguf_path);

// Same validation against caller-owned metadata-only contexts.  Used by the
// mapped preparation seam so validation and tensor borrows share one file
// identity and one metadata context.
Gemma3Manifest scan_gemma3(gguf_context * gguf, ggml_context * meta);

// Weight pointers resolved against the context that owns the tensor data.
// The four RMSNorm gains sit here rather than inside the sub-builders: the
// sandwich (normalize before *and* after each sub-layer) is this family's
// block structure, while the attention and FFN builders are shared.
struct Gemma3LayerWeights {
    ggml_tensor *                        attn_norm;       // before attention
    forward::GqaAttentionWeights         attn;
    ggml_tensor *                        attn_post_norm;  // after attention
    ggml_tensor *                        ffn_norm;        // before the FFN
    forward::GatedFfnWeights             ffn;
    ggml_tensor *                        ffn_post_norm;   // after the FFN
};

class Gemma3ModelArch : public ModelArch {
public:
    explicit Gemma3ModelArch(const std::string & gguf_path);
    explicit Gemma3ModelArch(Gemma3Manifest manifest);

    const ArchParams & params() const noexcept override { return manifest_.params; }

    // RoPE consumes one shared 0..S-1 vector. There are no learned position or
    // segment embeddings.
    InputRequirements inputs() const noexcept override {
        return InputRequirements{/*learned_pos=*/false, /*rope_pos=*/true,
                                 /*type_ids=*/false, /*kq_mask=*/true,
                                 /*seq_lengths=*/true, /*packed=*/true};
    }
    InputRequirements embedding_inputs() const noexcept override { return {}; }

    PoolType default_pooling() const noexcept override { return manifest_.pooling; }

    void bind_weights(ggml_context * model_ctx) override;

    ggml_tensor * build_graph(ggml_context *      gctx,
                              const GraphInputs & in) const override;

    ggml_tensor * build_embedding_phase(ggml_context *      gctx,
                                        const GraphInputs & in) const override;
    ggml_tensor * build_final_phase(ggml_context * gctx,
                                    ggml_tensor *  x) const override;

    StreamingLayerPlan  streaming_units(int layer) const override;
    StreamingCommonPlan streaming_common_plan() const override;

    // build_graph is the composition of these three. They are public so the
    // per-layer parity test can drive the real wiring — feeding HuggingFace's
    // captured activation for layer N-1 in and comparing layer N out — rather
    // than reimplementing the block and testing its own copy.
    ggml_tensor * build_embeddings(ggml_context * gctx, ggml_tensor * token_ids) const;
    ggml_tensor * build_block(ggml_context * gctx,
                              ggml_tensor *  x,
                              ggml_tensor *  pos,
                              int            layer,
                              ggml_tensor *  kq_mask = nullptr,
                              const int32_t * seq_lengths = nullptr,
                              const int32_t * seq_offsets = nullptr,
                              int64_t         n_seq = 0) const;
    ggml_tensor * build_final_norm(ggml_context * gctx, ggml_tensor * x) const;

    // Scales the graph applies that are not folded into the weights.
    float embed_scale() const noexcept { return manifest_.embed_scale; }
    float attn_scale()  const noexcept { return manifest_.attn_scale; }

private:
    // Shared by build_block and by the streaming units, so the two cannot
    // drift on scale, epsilon or head geometry.
    forward::GqaAttentionParams gqa_params() const noexcept;

    Gemma3Manifest                  manifest_;
    ggml_tensor *                   tok_embed_   = nullptr;
    ggml_tensor *                   output_norm_ = nullptr;
    std::vector<Gemma3LayerWeights> layer_w_;
};

} // namespace nanoembed
