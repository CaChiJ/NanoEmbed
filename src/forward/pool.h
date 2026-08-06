// Pooling and L2 normalization graph builders.
//
//   Mean pooling: average over the S (sequence) axis of a [H, S, B] tensor,
//                 producing [H, B]. v0 assumes no padding (B=1 path); a
//                 mask-aware variant lands with M5 batching.
//   CLS  pooling: takes x[:, 0, :], producing [H, B].
//   LAST pooling: takes x[:, S-1, :], producing [H, B]. Used by decoder-style
//                 encoders (e.g. eurobert declares pooling_type=3).
//   L2   normalize: rescale each row to unit length along ne[0]=H.

#pragma once

struct ggml_context;
struct ggml_tensor;

namespace nanoembed::forward {

enum class PoolType { Mean, Cls, Last };

ggml_tensor * build_mean_pool(ggml_context * ctx, ggml_tensor * x);
ggml_tensor * build_cls_pool (ggml_context * ctx, ggml_tensor * x);
ggml_tensor * build_last_pool(ggml_context * ctx, ggml_tensor * x);

ggml_tensor * build_pool(ggml_context * ctx, ggml_tensor * x, PoolType type);

// L2 normalize each row: y = x / max(||x||_2, sqrt(eps)).
ggml_tensor * build_l2_normalize(ggml_context * ctx, ggml_tensor * x,
                                  float eps = 1e-12f);

} // namespace nanoembed::forward
