// Gated feed-forward network, unbiased.
//
//   y = down( act(gate(x)) * up(x) )
//
// Three matrices rather than BERT's two: the extra one produces a
// multiplicative gate. Which activation applies is a per-family choice —
// Gemma 3 uses the tanh approximation of GELU (`gelu_pytorch_tanh`), Llama
// uses SiLU — so it is a parameter rather than baked in.
//
// Only the FFN itself. The surrounding normalization and residual belong to
// the block, which differs between families (Gemma normalizes both before and
// after this), so they are the architecture's business.

#pragma once

struct ggml_context;
struct ggml_tensor;

namespace nanoembed::forward {

struct GatedFfnWeights {
    ggml_tensor * gate;   // [H, F]
    ggml_tensor * up;     // [H, F]
    ggml_tensor * down;   // [F, H]
};

enum class GateActivation {
    Gelu,   // tanh approximation — gelu_pytorch_tanh
    Silu,   // SwiGLU
};

// Up to and including the gating multiply. Reads w.gate and w.up.
//
// gate and up cannot be separated further without a boundary of their own:
// ggml_geglu_split fuses act(gate) * up into one op, so both [F, S, B] tensors
// have to be live at that point.
ggml_tensor * build_gated_ffn_gate_up(ggml_context *          ctx,
                                      ggml_tensor *           x,     // [H, S, B]
                                      const GatedFfnWeights & w,
                                      GateActivation          act);  // -> [F, S, B]

// The down-projection. Reads w.down.
ggml_tensor * build_gated_ffn_down(ggml_context *          ctx,
                                   ggml_tensor *           h,        // [F, S, B]
                                   const GatedFfnWeights & w);       // -> [H, S, B]

// The composition of the two, emitting the same calls in the same order.
ggml_tensor * build_gated_ffn(ggml_context *          ctx,
                              ggml_tensor *           x,       // [H, S, B]
                              const GatedFfnWeights & w,
                              GateActivation          act);

} // namespace nanoembed::forward
