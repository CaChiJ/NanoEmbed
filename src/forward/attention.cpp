#include "attention.h"

#include "ggml.h"

#include <cmath>

namespace nanoembed::forward {

AttentionProjections build_attention_projections(ggml_context *           ctx,
                                                 ggml_tensor *            x,
                                                 const AttentionWeights & w) {
    // Linear projections produce [H, S, B] each.
    AttentionProjections proj;
    proj.q = ggml_add(ctx, ggml_mul_mat(ctx, w.q_w, x), w.q_b);
    proj.k = ggml_add(ctx, ggml_mul_mat(ctx, w.k_w, x), w.k_b);
    proj.v = ggml_add(ctx, ggml_mul_mat(ctx, w.v_w, x), w.v_b);
    return proj;
}

ggml_tensor * build_attention_output(ggml_context *               ctx,
                                     const AttentionProjections & proj,
                                     ggml_tensor *                x_residual,
                                     ggml_tensor *                kq_mask,
                                     int                          n_head,
                                     const AttentionWeights &     w,
                                     float                        layer_norm_eps) {
    // Taken from the projections rather than the residual: ggml_mul_mat
    // preserves ne[1..2] and the projections are square, so these are the same
    // values the fused function read off x.
    const int H        = static_cast<int>(proj.q->ne[0]);
    const int S        = static_cast<int>(proj.q->ne[1]);
    const int B        = static_cast<int>(proj.q->ne[2]);
    const int head_dim = H / n_head;
    const float scale  = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Split heads: [H, S, B] -> [head_dim, n_head, S, B].
    ggml_tensor * q = ggml_reshape_4d(ctx, proj.q, head_dim, n_head, S, B);
    ggml_tensor * k = ggml_reshape_4d(ctx, proj.k, head_dim, n_head, S, B);
    ggml_tensor * v = ggml_reshape_4d(ctx, proj.v, head_dim, n_head, S, B);

    // Permute to [head_dim, S, n_head, B] so attention runs per-head.
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    // Scores = K^T Q (ggml_mul_mat reduces ne[0] = head_dim).
    // Result ne = [S_k, S_q, n_head, B].
    ggml_tensor * scores = ggml_mul_mat(ctx, k, q);

    // Scaled softmax over S_k (ne[0]). Optional additive mask.
    scores = ggml_soft_max_ext(ctx, scores, kq_mask, scale, /*max_bias=*/0.0f);

    // We need out[d, q, h, b] = sum_k v[d, k, h, b] * scores[k, q, h, b].
    // ggml_mul_mat reduces ne[0]; transpose v's first two axes so its ne[0]
    // becomes S_k.
    ggml_tensor * v_t = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
    // Result ne = [head_dim, S_q, n_head, B].
    ggml_tensor * attn = ggml_mul_mat(ctx, v_t, scores);

    // Reassemble heads: [head_dim, S, n_head, B] -> [H, S, B].
    attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
    attn = ggml_reshape_3d(ctx, attn, H, S, B);

    // Output projection.
    attn = ggml_add(ctx, ggml_mul_mat(ctx, w.o_w, attn), w.o_b);

    // Residual + post-attention LayerNorm.
    ggml_tensor * y      = ggml_add(ctx, x_residual, attn);
    ggml_tensor * normed = ggml_norm(ctx, y, layer_norm_eps);
    ggml_tensor * scaled = ggml_mul(ctx, normed, w.norm_w);
    return ggml_add(ctx, scaled, w.norm_b);
}

ggml_tensor * build_attention_block(
    ggml_context *           ctx,
    ggml_tensor *            x,
    ggml_tensor *            kq_mask,
    int                      n_head,
    const AttentionWeights & w,
    float                    layer_norm_eps) {
    return build_attention_output(ctx, build_attention_projections(ctx, x, w),
                                  x, kq_mask, n_head, w, layer_norm_eps);
}

} // namespace nanoembed::forward
