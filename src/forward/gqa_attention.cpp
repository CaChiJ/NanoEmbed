#include "gqa_attention.h"

#include "rms_norm.h"
#include "rope.h"

#include "ggml.h"

#include <stdexcept>

namespace nanoembed::forward {

GqaProjections build_gqa_projections(ggml_context *              ctx,
                                     ggml_tensor *               x,
                                     const GqaAttentionWeights & w) {
    // Projections. No bias anywhere in this family.
    GqaProjections proj;
    proj.q = ggml_mul_mat(ctx, w.q, x);   // [n_head    * D, S, B]
    proj.k = ggml_mul_mat(ctx, w.k, x);   // [n_head_kv * D, S, B]
    proj.v = ggml_mul_mat(ctx, w.v, x);   // [n_head_kv * D, S, B]
    return proj;
}

ggml_tensor * build_gqa_attention_core(ggml_context *              ctx,
                                       const GqaProjections &      proj,
                                       ggml_tensor *               pos,
                                       ggml_tensor *               kq_mask,
                                       const GqaAttentionWeights & w,
                                       const GqaAttentionParams &  p) {
    // S and B come from the projections rather than a residual stream the core
    // half never sees. ggml_mul_mat preserves ne[1..2], so these are the same
    // values the fused function read off x.
    const int64_t S = proj.q->ne[1];
    const int64_t B = proj.q->ne[2];
    const int64_t D = p.head_dim;

    if (p.rope_freq_base > 0.0f && pos == nullptr) {
        throw std::invalid_argument("build_gqa_attention: rope requested without positions");
    }

    // Head-major layout: rope's kernel requires the token axis at ne[2], and
    // RMSNorm normalizes along ne[0], which is head_dim here — exactly the
    // axis QK-norm is defined over.
    ggml_tensor * q = ggml_reshape_4d(ctx, proj.q, D, p.n_head,    S, B);
    ggml_tensor * k = ggml_reshape_4d(ctx, proj.k, D, p.n_head_kv, S, B);
    ggml_tensor * v = ggml_reshape_4d(ctx, proj.v, D, p.n_head_kv, S, B);

    if (w.q_norm != nullptr) q = build_rms_norm(ctx, q, w.q_norm, p.norm_eps);
    if (w.k_norm != nullptr) k = build_rms_norm(ctx, k, w.k_norm, p.norm_eps);

    if (p.rope_freq_base > 0.0f) {
        q = build_rope(ctx, q, pos, static_cast<int>(D), p.n_ctx_orig, p.rope_freq_base);
        k = build_rope(ctx, k, pos, static_cast<int>(D), p.n_ctx_orig, p.rope_freq_base);
    }

    // Token axis to ne[1] so the head axis becomes the broadcast axis.
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));   // [D, S, n_head,    B]
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));   // [D, S, n_head_kv, B]
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));   // [D, S, n_head_kv, B]

    // ggml_mul_mat broadcasts ne[2] when the smaller count divides the larger,
    // which is what makes one KV head serve a group of query heads without
    // materializing a repeated copy.
    ggml_tensor * scores = ggml_mul_mat(ctx, k, q);         // [S_k, S_q, n_head, B]

    // Causal masking. ne[0] indexes keys and ne[1] queries, so zeroing the
    // upper triangle is exactly "a token may not attend to later tokens".
    // Generating it here costs no graph input: no tensor has to be created,
    // filled or reserved by the caller.
    if (p.causal) {
        scores = ggml_diag_mask_inf(ctx, scores, /*n_past=*/0);
    }

    scores = ggml_soft_max_ext(ctx, scores, kq_mask, p.scale, /*max_bias=*/0.0f);

    ggml_tensor * v_t  = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
    ggml_tensor * attn = ggml_mul_mat(ctx, v_t, scores);    // [D, S_q, n_head, B]

    attn = ggml_cont(ctx, ggml_permute(ctx, attn, 0, 2, 1, 3));
    attn = ggml_reshape_3d(ctx, attn, D * p.n_head, S, B);

    return ggml_mul_mat(ctx, w.o, attn);                    // [H, S, B]
}

ggml_tensor * build_gqa_attention(ggml_context *              ctx,
                                  ggml_tensor *               x,
                                  ggml_tensor *               pos,
                                  ggml_tensor *               kq_mask,
                                  const GqaAttentionWeights & w,
                                  const GqaAttentionParams &  p) {
    // Kept here as well as in the core half so the eager path still rejects a
    // missing position ramp before emitting any node, exactly as it used to.
    if (p.rope_freq_base > 0.0f && pos == nullptr) {
        throw std::invalid_argument("build_gqa_attention: rope requested without positions");
    }
    return build_gqa_attention_core(
        ctx, build_gqa_projections(ctx, x, w), pos, kq_mask, w, p);
}

} // namespace nanoembed::forward
