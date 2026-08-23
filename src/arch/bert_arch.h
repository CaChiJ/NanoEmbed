// ModelArch implementation for `general.architecture == "bert"`.
//
// Learned absolute position embeddings + segment embeddings, post-LayerNorm
// blocks, biased projections, GELU MLP. Backed by the M2 scanner
// (scan_gguf/ModelManifest), which stays BERT-specific by design — it is this
// family's private detail, not a shared contract.

#pragma once

#include "arch/model_arch.h"

#include "forward/embed_layer.h"
#include "forward/encoder_block.h"
#include "gguf_scanner.h"

#include <string>
#include <vector>

namespace nanoembed {

class BertModelArch : public ModelArch {
public:
    explicit BertModelArch(const std::string & gguf_path);

    const ArchParams & params() const noexcept override { return params_; }

    InputRequirements inputs() const noexcept override {
        return InputRequirements{/*needs_pos_ids=*/true, /*needs_type_ids=*/true};
    }

    // Read from bert.pooling_type rather than assumed. BERT embedding models
    // disagree: bge-small is CLS, all-MiniLM and the e5 family are mean, and
    // the file states which. Hardcoding one of them was harmless only while
    // the public default was MEAN and callers had to opt in.
    PoolType default_pooling() const noexcept override { return default_pooling_; }

    void bind_weights(ggml_context * model_ctx) override;

    ggml_tensor * build_graph(ggml_context *      gctx,
                              const GraphInputs & in) const override;

private:
    ArchParams                          params_;
    PoolType                            default_pooling_ = PoolType::Mean;
    ModelManifest                       manifest_;
    forward::EmbedWeights               embed_w_{};
    std::vector<forward::LayerWeights>  layer_w_;
};

} // namespace nanoembed
