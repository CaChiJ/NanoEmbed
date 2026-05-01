// BERT embedding-stage graph builder.
//
//   acts = LN(tok_embed[token_ids] + pos_embed[positions] + type_embed[type_ids])
//
// All inputs/weights must be reachable from the graph context. Output is a
// new tensor in the same context (caller-owned via ctx lifetime).
//
// Shape conventions (ggml ne order, ne[0] is contiguous):
//   token_ids / position_ids / type_ids : I32 [S, B]   (B may be 1)
//   weights  (passed in EmbedWeights)   : F32/F16 — see field comments
//   output                              : F32 [H, S, B]

#pragma once

struct ggml_context;
struct ggml_tensor;

namespace nanoembed::forward {

struct EmbedWeights {
    ggml_tensor * tok;     // [H, V]      — token embedding table
    ggml_tensor * pos;     // [H, Smax]   — positional embedding table
    ggml_tensor * type;    // [H, n_types]— token-type embedding table
    ggml_tensor * norm_w;  // [H]         — LayerNorm gain
    ggml_tensor * norm_b;  // [H]         — LayerNorm bias
};

ggml_tensor * build_embed_layer(
    ggml_context *       ctx,
    ggml_tensor *        token_ids,
    ggml_tensor *        position_ids,
    ggml_tensor *        type_ids,
    const EmbedWeights & w,
    float                layer_norm_eps);

} // namespace nanoembed::forward
