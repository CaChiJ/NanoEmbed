#pragma once

#include "ggml.h"

namespace nanoembed::forward {

// Apply one shared 2-D weight matrix to every token in [I,S,B]. ggml's
// batched mul_mat treats B as separate matrix products; flattening the
// contiguous token/batch axes turns this into one larger GEMM without mixing
// sentences. The B=1 graph is deliberately left unchanged.
inline ggml_tensor * build_linear(ggml_context * ctx,
                                  ggml_tensor *  weight,
                                  ggml_tensor *  x) {
    if (x->ne[2] == 1 && x->ne[3] == 1) {
        return ggml_mul_mat(ctx, weight, x);
    }
    const int64_t S = x->ne[1];
    const int64_t B = x->ne[2] * x->ne[3];
    ggml_tensor * flat = ggml_reshape_2d(ctx, x, x->ne[0], S * B);
    ggml_tensor * projected = ggml_mul_mat(ctx, weight, flat);
    return ggml_reshape_3d(ctx, projected, projected->ne[0], S, B);
}

} // namespace nanoembed::forward
