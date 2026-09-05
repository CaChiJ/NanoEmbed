// Rotary position embedding.
//
// Expects the head-major layout ggml's rope kernel assumes:
//   x   : [head_dim, n_head, S, B]
//   pos : I32 with ne[0] == S   (the plain 0..S-1 ramp)
//
// Half-split (GPT-NeoX) rotation, which is what HuggingFace's rotate_half
// does — it takes the two halves of head_dim as the real and imaginary parts.
// ggml's default mode interleaves them instead and would rotate the wrong
// pairs, silently, producing a model that still runs and returns nonsense.

#pragma once

struct ggml_context;
struct ggml_tensor;

namespace nanoembed::forward {

ggml_tensor * build_rope(ggml_context * ctx,
                         ggml_tensor *  x,
                         ggml_tensor *  pos,
                         int            n_dims,        // rotated width; = head_dim
                         int            n_ctx_orig,    // training context length
                         float          freq_base);

} // namespace nanoembed::forward
