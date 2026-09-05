// BERT post-LN attention block.
//
//   q, k, v = linear(x; Wq), linear(x; Wk), linear(x; Wv)
//   split into n_head heads, head_dim = H / n_head
//   scores = softmax((K^T Q) / sqrt(head_dim) + mask)
//   ctx    = scores @ V   (per-head)
//   attn   = linear(ctx; Wo)
//   y      = LN(x + attn)
//
// Shape: x is [H, S, B]; output is [H, S, B].

#pragma once

struct ggml_context;
struct ggml_tensor;

namespace nanoembed::forward {

struct AttentionWeights {
    ggml_tensor * q_w;     // [H, H]
    ggml_tensor * q_b;     // [H]
    ggml_tensor * k_w;     // [H, H]
    ggml_tensor * k_b;     // [H]
    ggml_tensor * v_w;     // [H, H]
    ggml_tensor * v_b;     // [H]
    ggml_tensor * o_w;     // [H, H]   output projection
    ggml_tensor * o_b;     // [H]
    ggml_tensor * norm_w;  // [H]      post-attention LN gain
    ggml_tensor * norm_b;  // [H]      post-attention LN bias
};

// The raw Q/K/V projections, biases included, before any reshape. Streaming
// cuts here so what crosses the boundary is a freshly allocated contiguous
// [H, S, B] rather than a permuted view.
struct AttentionProjections {
    ggml_tensor * q;   // [H, S, B]
    ggml_tensor * k;   // [H, S, B]
    ggml_tensor * v;   // [H, S, B]
};

// Reads w.q_*, w.k_*, w.v_* only.
AttentionProjections build_attention_projections(ggml_context *           ctx,
                                                 ggml_tensor *            x,
                                                 const AttentionWeights & w);

// Scores through the post-attention LayerNorm. Reads w.o_* and w.norm_*.
// BERT adds the residual inside this sub-layer, and this half does not compute
// the block input, so the residual source is passed explicitly.
ggml_tensor * build_attention_output(ggml_context *                ctx,
                                     const AttentionProjections &  proj,
                                     ggml_tensor *                 x_residual,
                                     ggml_tensor *                 kq_mask,
                                     int                           n_head,
                                     const AttentionWeights &      w,
                                     float                         layer_norm_eps);

// kq_mask is optional (NULL when no padding). If provided it must be an
// additive mask broadcastable to the [S_k, S_q, 1, B] score tensor:
// 0 for valid positions, large-negative (or -INF) for masked.
//
// The composition of the two halves, emitting the same calls in the same order.
ggml_tensor * build_attention_block(
    ggml_context *           ctx,
    ggml_tensor *            x,
    ggml_tensor *            kq_mask,
    int                      n_head,
    const AttentionWeights & w,
    float                    layer_norm_eps);

} // namespace nanoembed::forward
