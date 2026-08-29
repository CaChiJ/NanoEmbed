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
    // indices returns a tensor of shape [H, S, B] with the table's dtype.
    //
    // For bge-small-en-v1.5: tok/pos are F16, token_types is F32 (mixed dtype
    // is normal for BERT GGUFs from llama.cpp's converter). ggml_add promotes
    // internally; the layer-isolation parity test confirms the result is
    // bit-stable to within ~1.4e-6 vs HuggingFace.
    auto rows = [&](ggml_tensor * table, ggml_tensor * ids) {
        if (ids->ne[1] == 1) return ggml_get_rows(ctx, table, ids);
        const int64_t S = ids->ne[0];
        const int64_t B = ids->ne[1];
        // GET_ROWS in the vendored ggml requires its source and index batch
        // axes to match. The embedding table is shared by every sentence, so
        // flatten the contiguous [S,B] IDs, perform one lookup, and restore
        // the natural [H,S,B] activation shape. Both reshapes are views.
        ggml_tensor * indexed = ggml_get_rows(
            ctx, table, ggml_reshape_1d(ctx, ids, S * B));
        return ggml_reshape_3d(ctx, indexed, indexed->ne[0], S, B);
    };
    ggml_tensor * tok  = rows(w.tok,  token_ids);
    ggml_tensor * pos  = rows(w.pos,  position_ids);
    ggml_tensor * type = rows(w.type, type_ids);

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
