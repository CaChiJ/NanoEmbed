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

// kq_mask is optional (NULL when no padding). If provided it must be an
// additive mask broadcastable to the [S_k, S_q, 1, B] score tensor:
// 0 for valid positions, large-negative (or -INF) for masked.
ggml_tensor * build_attention_block(
    ggml_context *           ctx,
    ggml_tensor *            x,
    ggml_tensor *            kq_mask,
    int                      n_head,
    const AttentionWeights & w,
    float                    layer_norm_eps);

} // namespace nanoembed::forward
