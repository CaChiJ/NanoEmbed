#include "pool.h"

#include "ggml.h"

#include <stdexcept>

namespace nanoembed::forward {

ggml_tensor * build_mean_pool(ggml_context * ctx, ggml_tensor * x) {
    // x: [H, S, B]. ggml_mean reduces ne[0], so permute S to ne[0] first.
    ggml_tensor * xt   = ggml_cont(ctx, ggml_permute(ctx, x, 1, 0, 2, 3));  // [S, H, B]
    ggml_tensor * m    = ggml_mean(ctx, xt);                                // [1, H, B]
    return ggml_reshape_2d(ctx, m, m->ne[1], m->ne[2]);                     // [H, B]
}

ggml_tensor * build_cls_pool(ggml_context * ctx, ggml_tensor * x) {
    // x: [H, S, B]. Take the first token along S: x[:, 0, :] -> [H, B].
    // The B-stride between batch slices is x->nb[2]; offset 0 picks S=0.
    return ggml_cont(ctx, ggml_view_2d(ctx, x,
                                       x->ne[0], x->ne[2],
                                       x->nb[2], /*offset=*/0));
}

ggml_tensor * build_last_pool(ggml_context * ctx, ggml_tensor * x) {
    // x: [H, S, B]. Take the final token along S: x[:, S-1, :] -> [H, B].
    // No padding in the B=1 path, so the last position is always the real
    // final token; a mask-aware variant lands with M5 batching.
    const size_t offset = static_cast<size_t>(x->ne[1] - 1) * x->nb[1];
    return ggml_cont(ctx, ggml_view_2d(ctx, x,
                                       x->ne[0], x->ne[2],
                                       x->nb[2], offset));
}

ggml_tensor * build_masked_mean_pool(ggml_context * ctx, ggml_tensor * x,
                                     const PoolInputs & inputs) {
    if (inputs.valid_mask == nullptr || inputs.mean_scale == nullptr) {
        throw std::invalid_argument("masked mean pooling requires mask and scale");
    }
    ggml_tensor * masked = ggml_mul(ctx, x, inputs.valid_mask);
    ggml_tensor * xt = ggml_cont(ctx, ggml_permute(ctx, masked, 1, 0, 2, 3));
    ggml_tensor * sum = ggml_sum_rows(ctx, xt); // [1,H,B]
    sum = ggml_reshape_2d(ctx, sum, sum->ne[1], sum->ne[2]); // [H,B]
    return ggml_mul(ctx, sum, inputs.mean_scale);
}

ggml_tensor * build_indexed_last_pool(ggml_context * ctx, ggml_tensor * x,
                                      const PoolInputs & inputs) {
    if (inputs.last_indices == nullptr) {
        throw std::invalid_argument("indexed LAST pooling requires indices");
    }
    ggml_tensor * flat = ggml_reshape_2d(ctx, ggml_cont(ctx, x),
                                         x->ne[0], x->ne[1] * x->ne[2]);
    return ggml_get_rows(ctx, flat, inputs.last_indices);
}

ggml_tensor * build_pool(ggml_context * ctx, ggml_tensor * x, PoolType type) {
    switch (type) {
        case PoolType::Mean: return build_mean_pool(ctx, x);
        case PoolType::Cls:  return build_cls_pool(ctx, x);
        case PoolType::Last: return build_last_pool(ctx, x);
    }
    throw std::invalid_argument("unknown PoolType");
}

ggml_tensor * build_pool(ggml_context * ctx, ggml_tensor * x, PoolType type,
                         const PoolInputs & inputs) {
    switch (type) {
        case PoolType::Mean:
            return inputs.valid_mask == nullptr
                ? build_mean_pool(ctx, x)
                : build_masked_mean_pool(ctx, x, inputs);
        case PoolType::Cls:
            return build_cls_pool(ctx, x);
        case PoolType::Last:
            return inputs.last_indices == nullptr
                ? build_last_pool(ctx, x)
                : build_indexed_last_pool(ctx, x, inputs);
    }
    throw std::invalid_argument("unknown PoolType");
}

ggml_tensor * build_l2_normalize(ggml_context * ctx, ggml_tensor * x, float eps) {
    return ggml_l2_norm(ctx, x, eps);
}

} // namespace nanoembed::forward
