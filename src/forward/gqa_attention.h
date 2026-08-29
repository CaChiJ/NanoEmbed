// Attention with decoupled head geometry, grouped/multi-query KV sharing,
// optional QK-norm and optional rotary positions.
//
// Generalizes three assumptions BERT's build_attention_block bakes in, each of
// which harrier-oss-v1-270m violates:
//
//   * head_dim is a parameter, not n_embed / n_head. Gemma 3 270M runs 4 heads
//     of width 256 over a 640-wide residual stream, so the projections are not
//     square and the width cannot be recovered by division.
//   * n_head_kv may be smaller than n_head. K and V are projected once per KV
//     head and shared across a group of query heads; ggml_mul_mat broadcasts
//     over ne[2] when the counts divide, so no explicit repeat is needed.
//   * Queries and keys may be RMSNormed before rotation (QK-norm), and the
//     rotation itself is optional.
//
//   q, k, v = linear(x)                       (no bias)
//   q, k    = rms_norm(q), rms_norm(k)        if QK-norm weights are given
//   q, k    = rope(q), rope(k)                if rope_freq_base > 0
//   scores  = softmax(scale * (K^T Q) + masks)
//   y       = linear(scores @ V)              (no bias)
//
// Shapes: x is [H, S, B]; output is [H, S, B]. Only the attention itself —
// surrounding normalization and the residual belong to the block.

#pragma once

#include <cstdint>

struct ggml_context;
struct ggml_tensor;

namespace nanoembed::forward {

struct GqaAttentionWeights {
    ggml_tensor * q;        // [H, n_head    * head_dim]
    ggml_tensor * k;        // [H, n_head_kv * head_dim]
    ggml_tensor * v;        // [H, n_head_kv * head_dim]
    ggml_tensor * o;        // [n_head * head_dim, H]
    ggml_tensor * q_norm;   // [head_dim], or null for no QK-norm
    ggml_tensor * k_norm;   // [head_dim], or null
};

struct GqaAttentionParams {
    int   n_head        = 0;
    int   n_head_kv     = 0;
    int   head_dim      = 0;
    // Multiplied into the scores before softmax. Usually 1/sqrt(head_dim), but
    // Gemma states it separately (query_pre_attn_scalar) so it is passed in.
    float scale         = 0.0f;
    float norm_eps      = 1e-6f;   // QK-norm epsilon
    float rope_freq_base = 0.0f;   // 0 disables rotary positions
    int   n_ctx_orig    = 0;
    bool  causal        = false;
};

// The raw Q/K/V projections, before any reshape. Streaming cuts here so the
// value crossing the boundary is a freshly allocated contiguous [*, S, B] --
// ggml_mul_mat blocks its reduction differently for non-contiguous src1, so a
// cut at a permuted tensor could change rounding.
struct GqaProjections {
    ggml_tensor * q;   // [n_head    * head_dim, S, B]
    ggml_tensor * k;   // [n_head_kv * head_dim, S, B]
    ggml_tensor * v;   // [n_head_kv * head_dim, S, B]
};

// Reads w.q, w.k, w.v only.
GqaProjections build_gqa_projections(ggml_context *              ctx,
                                     ggml_tensor *               x,      // [H, S, B]
                                     const GqaAttentionWeights & w);

// Everything after the projections. Reads w.q_norm, w.k_norm and w.o.
// `pos` is the I32 position ramp, required when rope_freq_base > 0.
// `kq_mask` is an optional additive padding mask, as in attention.h; the
// causal mask is generated internally and the two compose.
// When `seq_lengths` is non-null it holds this sub-batch's true token counts
// and takes precedence over `kq_mask`: the scores are then built one sentence
// at a time at that sentence's own length, so no padded key is ever scored and
// no mask is needed. Sentences never mix in either form -- the batch axis
// already keeps them apart -- so this only removes wasted work.
ggml_tensor * build_gqa_attention_core(ggml_context *              ctx,
                                       const GqaProjections &      proj,
                                       ggml_tensor *               pos,
                                       ggml_tensor *               kq_mask,
                                       const GqaAttentionWeights & w,
                                       const GqaAttentionParams &  p,
                                       const int32_t *             seq_lengths = nullptr);

// The composition of the two. The eager path calls this and is unaffected by
// the split: the same ggml calls are emitted in the same order.
ggml_tensor * build_gqa_attention(ggml_context *             ctx,
                                  ggml_tensor *              x,
                                  ggml_tensor *              pos,
                                  ggml_tensor *              kq_mask,
                                  const GqaAttentionWeights & w,
                                  const GqaAttentionParams &  p,
                                  const int32_t *            seq_lengths = nullptr);

} // namespace nanoembed::forward
