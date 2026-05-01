// One BERT encoder block: post-LN attention sub-layer + post-LN FFN sub-layer.

#pragma once

#include "attention.h"
#include "ffn.h"

struct ggml_context;
struct ggml_tensor;

namespace nanoembed::forward {

struct LayerWeights {
    AttentionWeights attn;
    FFNWeights       ffn;
};

ggml_tensor * build_encoder_block(
    ggml_context *       ctx,
    ggml_tensor *        x,        // [H, S, B]
    ggml_tensor *        kq_mask,  // optional, see attention.h
    int                  n_head,
    const LayerWeights & w,
    float                layer_norm_eps);

} // namespace nanoembed::forward
