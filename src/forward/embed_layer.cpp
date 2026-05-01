#include "embed_layer.h"

#include "ggml.h"

namespace nanoembed::forward {

ggml_tensor * build_embed_layer(
    ggml_context *       ctx,
    ggml_tensor *        token_ids,
    ggml_tensor *        position_ids,
    ggml_tensor *        type_ids,
    const EmbedWeights & w,
    float                layer_norm_eps) {

    // Embedding lookups. ggml_get_rows on a [H, N] table with I32 [S, B]
    // indices returns a tensor of shape [H, S, B].
    ggml_tensor * tok  = ggml_get_rows(ctx, w.tok,  token_ids);
    ggml_tensor * pos  = ggml_get_rows(ctx, w.pos,  position_ids);
    ggml_tensor * type = ggml_get_rows(ctx, w.type, type_ids);

    // Sum the three embedding contributions.
    ggml_tensor * sum = ggml_add(ctx, tok, pos);
    sum               = ggml_add(ctx, sum, type);

    // LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta
    // ggml_norm normalizes along ne[0] (the hidden dim, H).
    ggml_tensor * normed = ggml_norm(ctx, sum, layer_norm_eps);
    ggml_tensor * scaled = ggml_mul(ctx, normed, w.norm_w);  // broadcasts [H] over [H, S, B]
    ggml_tensor * biased = ggml_add(ctx, scaled, w.norm_b);

    return biased;
}

} // namespace nanoembed::forward
