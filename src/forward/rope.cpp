#include "rope.h"

#include "ggml.h"

namespace nanoembed::forward {

ggml_tensor * build_rope(ggml_context * ctx,
                         ggml_tensor *  x,
                         ggml_tensor *  pos,
                         int            n_dims,
                         int            n_ctx_orig,
                         float          freq_base) {
    // No frequency scaling: ext_factor 0 disables YaRN, so n_ctx_orig,
    // beta_fast and beta_slow are inert and only freq_base matters.
    return ggml_rope_ext(ctx, x, pos, /*freq_factors=*/nullptr,
                         n_dims,
                         GGML_ROPE_TYPE_NEOX,
                         n_ctx_orig,
                         freq_base,
                         /*freq_scale=*/1.0f,
                         /*ext_factor=*/0.0f,
                         /*attn_factor=*/1.0f,
                         /*beta_fast=*/0.0f,
                         /*beta_slow=*/0.0f);
}

} // namespace nanoembed::forward
