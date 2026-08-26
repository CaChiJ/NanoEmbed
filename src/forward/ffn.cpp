#include "ffn.h"

#include "ggml.h"

namespace nanoembed::forward {

ggml_tensor * build_ffn_up(ggml_context *     ctx,
                           ggml_tensor *      x,
                           const FFNWeights & w) {
    // Up-projection: [H] -> [F]. mul_mat(up_w[H,F], x[H,S,B]) = [F, S, B].
    ggml_tensor * h = ggml_add(ctx, ggml_mul_mat(ctx, w.up_w, x), w.up_b);

    // Exact erf-based GeLU — matches HuggingFace BertConfig.hidden_act = "gelu".
    // Kept on this side of the boundary so the down half reads only its own
    // four weights.
    return ggml_gelu_erf(ctx, h);
}

ggml_tensor * build_ffn_down(ggml_context *     ctx,
                             ggml_tensor *      h,
                             ggml_tensor *      x_residual,
                             const FFNWeights & w,
                             float              layer_norm_eps) {
    // Down-projection: [F] -> [H].
    ggml_tensor * y = ggml_add(ctx, ggml_mul_mat(ctx, w.down_w, h), w.down_b);

    // Residual + post-FFN LayerNorm.
    y = ggml_add(ctx, y, x_residual);
    ggml_tensor * normed = ggml_norm(ctx, y, layer_norm_eps);
    ggml_tensor * scaled = ggml_mul(ctx, normed, w.norm_w);
    return ggml_add(ctx, scaled, w.norm_b);
}

ggml_tensor * build_ffn_block(
    ggml_context *     ctx,
    ggml_tensor *      x,
    const FFNWeights & w,
    float              layer_norm_eps) {
    return build_ffn_down(ctx, build_ffn_up(ctx, x, w), x, w, layer_norm_eps);
}

} // namespace nanoembed::forward
