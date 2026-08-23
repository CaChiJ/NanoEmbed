#include "gated_ffn.h"

#include "ggml.h"

#include <stdexcept>

namespace nanoembed::forward {

ggml_tensor * build_gated_ffn(ggml_context *          ctx,
                              ggml_tensor *           x,
                              const GatedFfnWeights & w,
                              GateActivation          act) {
    ggml_tensor * gate = ggml_mul_mat(ctx, w.gate, x);   // [F, S, B]
    ggml_tensor * up   = ggml_mul_mat(ctx, w.up,   x);   // [F, S, B]

    // The *_split forms take the two halves as separate tensors and apply the
    // activation to the first, matching act(gate) * up.
    ggml_tensor * h = nullptr;
    switch (act) {
        case GateActivation::Gelu: h = ggml_geglu_split (ctx, gate, up); break;
        case GateActivation::Silu: h = ggml_swiglu_split(ctx, gate, up); break;
        default: throw std::invalid_argument("unknown GateActivation");
    }

    return ggml_mul_mat(ctx, w.down, h);                 // [H, S, B]
}

} // namespace nanoembed::forward
