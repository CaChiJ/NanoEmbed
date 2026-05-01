#include "encoder_block.h"

namespace nanoembed::forward {

ggml_tensor * build_encoder_block(
    ggml_context *       ctx,
    ggml_tensor *        x,
    ggml_tensor *        kq_mask,
    int                  n_head,
    const LayerWeights & w,
    float                layer_norm_eps) {
    ggml_tensor * x_attn = build_attention_block(ctx, x, kq_mask, n_head, w.attn, layer_norm_eps);
    return build_ffn_block(ctx, x_attn, w.ffn, layer_norm_eps);
}

} // namespace nanoembed::forward
