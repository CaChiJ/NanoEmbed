#include "ffn.h"

#include "ggml.h"

namespace nanoembed::forward {

ggml_tensor * build_ffn_block(
    ggml_context *     ctx,
    ggml_tensor *      x,
    const FFNWeights & w,
    float              layer_norm_eps) {

    // Up-projection: [H] -> [F]. mul_mat(up_w[H,F], x[H,S,B]) = [F, S, B].
    ggml_tensor * h = ggml_add(ctx, ggml_mul_mat(ctx, w.up_w, x), w.up_b);

    // Exact erf-based GeLU — matches HuggingFace BertConfig.hidden_act = "gelu".
    h = ggml_gelu_erf(ctx, h);

    // Down-projection: [F] -> [H].
    h = ggml_add(ctx, ggml_mul_mat(ctx, w.down_w, h), w.down_b);

    // Residual + post-FFN LayerNorm.
    h = ggml_add(ctx, h, x);
    ggml_tensor * normed = ggml_norm(ctx, h, layer_norm_eps);
    ggml_tensor * scaled = ggml_mul(ctx, normed, w.norm_w);
    return ggml_add(ctx, scaled, w.norm_b);
}

} // namespace nanoembed::forward
