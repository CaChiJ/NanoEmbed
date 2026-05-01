// BERT post-LN FFN block.
//
//   h = ffn_up(x)            [H] -> [F]
//   h = gelu(h)              (HF default = exact erf-based)
//   h = ffn_down(h)          [F] -> [H]
//   y = LN(x + h)            residual + post-FFN LN
//
// Shape: x is [H, S, B]; output is [H, S, B].

#pragma once

struct ggml_context;
struct ggml_tensor;

namespace nanoembed::forward {

struct FFNWeights {
    ggml_tensor * up_w;     // [H, F]
    ggml_tensor * up_b;     // [F]
    ggml_tensor * down_w;   // [F, H]
    ggml_tensor * down_b;   // [H]
    ggml_tensor * norm_w;   // [H]    post-FFN LN gain
    ggml_tensor * norm_b;   // [H]    post-FFN LN bias
};

ggml_tensor * build_ffn_block(
    ggml_context *     ctx,
    ggml_tensor *      x,
    const FFNWeights & w,
    float              layer_norm_eps);

} // namespace nanoembed::forward
