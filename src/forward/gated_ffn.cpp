#include "gated_ffn.h"
#include "linear.h"

#include "ggml.h"

#include <stdexcept>

namespace nanoembed::forward {

ggml_tensor * build_gated_ffn_gate_up(ggml_context *          ctx,
                                      ggml_tensor *           x,
                                      const GatedFfnWeights & w,
                                      GateActivation          act) {
    ggml_tensor * gate = build_linear(ctx, w.gate, x);   // [F, S, B]
    ggml_tensor * up   = build_linear(ctx, w.up,   x);   // [F, S, B]

    // The *_split forms take the two halves as separate tensors and apply the
    // activation to the first, matching act(gate) * up.
    switch (act) {
        case GateActivation::Gelu: return ggml_geglu_split (ctx, gate, up);
        case GateActivation::Silu: return ggml_swiglu_split(ctx, gate, up);
        default: throw std::invalid_argument("unknown GateActivation");
    }
}

ggml_tensor * build_gated_ffn_down(ggml_context *          ctx,
                                   ggml_tensor *           h,
                                   const GatedFfnWeights & w) {
    return build_linear(ctx, w.down, h);                 // [H, S, B]
}

ggml_tensor * build_gated_ffn(ggml_context *          ctx,
                              ggml_tensor *           x,
                              const GatedFfnWeights & w,
                              GateActivation          act) {
    return build_gated_ffn_down(ctx, build_gated_ffn_gate_up(ctx, x, w, act), w);
}

} // namespace nanoembed::forward
