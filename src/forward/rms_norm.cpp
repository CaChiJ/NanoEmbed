#include "rms_norm.h"

#include "ggml.h"

namespace nanoembed::forward {

ggml_tensor * build_rms_norm(ggml_context * ctx,
                             ggml_tensor *  x,
                             ggml_tensor *  weight,
                             float          eps) {
    ggml_tensor * y = ggml_rms_norm(ctx, x, eps);
    // weight is 1-D; ggml broadcasts it across every higher dimension.
    return ggml_mul(ctx, y, weight);
}

} // namespace nanoembed::forward
