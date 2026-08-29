#include "pool.h"

#include "ggml.h"

#include <stdexcept>
#include <vector>

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

ggml_tensor * build_packed_pool(ggml_context * ctx,
                                ggml_tensor *  x,
                                PoolType       type,
                                const int32_t * offsets,
                                const int32_t * lengths,
                                int64_t         n_seq) {
    if (offsets == nullptr || lengths == nullptr || n_seq <= 0) {
        throw std::invalid_argument("packed pooling requires offsets and lengths");
    }
    const int64_t H = x->ne[0];
    std::vector<ggml_tensor *> parts;
    parts.reserve(static_cast<size_t>(n_seq));
    for (int64_t b = 0; b < n_seq; ++b) {
        const int64_t len = lengths[b];
        const int64_t start = offsets[b];
        if (len <= 0 || start < 0 || start + len > x->ne[1]) {
            throw std::invalid_argument("packed pooling slice is out of range");
        }
        ggml_tensor * part = nullptr;
        if (type == PoolType::Mean) {
            // Reduce this sentence's own tokens, then divide by its own count.
            ggml_tensor * slice = ggml_view_2d(
                ctx, x, H, len, x->nb[1], static_cast<size_t>(start) * x->nb[1]);
            ggml_tensor * t   = ggml_cont(ctx, ggml_transpose(ctx, slice)); // [len,H]
            ggml_tensor * sum = ggml_sum_rows(ctx, t);                      // [1,H]
            sum = ggml_scale(ctx, sum, 1.0f / static_cast<float>(len));
            part = ggml_reshape_2d(ctx, sum, H, 1);
        } else {
            // CLS is this sentence's first token, LAST its final one. Packing
            // removes the padding that made the final position ambiguous.
            const int64_t index = type == PoolType::Cls ? start : start + len - 1;
            part = ggml_cont(ctx, ggml_view_2d(
                ctx, x, H, 1, x->nb[1], static_cast<size_t>(index) * x->nb[1]));
        }
        parts.push_back(part);
    }
    ggml_tensor * out = parts[0];
    for (size_t i = 1; i < parts.size(); ++i) {
        out = ggml_concat(ctx, out, parts[i], /*dim=*/1);   // [H, n_seq]
    }
    return out;
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
