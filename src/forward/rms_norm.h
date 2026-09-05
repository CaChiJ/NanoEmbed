// RMSNorm with a learned gain and no bias.
//
//   y = x / sqrt(mean(x^2) + eps) * w
//
// Normalizes along ne[0], so it applies unchanged to a [H, S, B] residual
// stream and to a [head_dim, n_head, S, B] attention tensor (QK-norm).
//
// The gain is used as stored. Gemma's reference implementation computes
// x * (1 + w) and its GGUF converter folds that +1 into the weights, so
// adding it here as well would apply it twice — see arch/gemma3_arch.h.

#pragma once

struct ggml_context;
struct ggml_tensor;

namespace nanoembed::forward {

ggml_tensor * build_rms_norm(ggml_context * ctx,
                             ggml_tensor *  x,
                             ggml_tensor *  weight,   // [ne[0] of x]
                             float          eps);

} // namespace nanoembed::forward
