// Model architecture interface + registry.
//
// Everything that differs between GGUF model families lives behind this:
// which metadata keys hold the hyperparameters, which tensors must exist, and
// what graph to build from them. `bge-small` (arch "bert") and
// `jina-embeddings-v5-text-nano` (arch "eurobert") share almost nothing —
// learned vs rotary position encoding, LayerNorm vs RMSNorm, GELU MLP vs
// SwiGLU, biased vs unbiased projections — so the seam has to sit above the
// individual forward builders, not inside them.
//
// Adding a family means implementing ModelArch and adding one line to
// create_model_arch(). No existing family's code is touched.

#pragma once

#include <memory>
#include <string>
#include <vector>

struct gguf_context;
struct ggml_context;
struct ggml_tensor;

namespace nanoembed {

// The hyperparameters every supported encoder has, read from whichever
// metadata prefix the family uses ("bert.*", "eurobert.*", ...).
struct ArchParams {
    std::string name;              // general.architecture
    int   n_layer     = 0;
    int   n_embed     = 0;
    int   n_head      = 0;
    int   n_ff        = 0;
    int   n_vocab     = 0;
    int   max_seq_len = 0;         // model's own context length
    float norm_eps    = 1e-12f;
};

enum class PoolType { Mean, Cls, Last };

// Graph-time inputs. Which of these an architecture consumes varies: BERT
// needs segment IDs, eurobert (rotary, no segments) does not. The embedder
// creates every tensor an arch declares it wants and fills it after
// allocation.
struct GraphInputs {
    ggml_tensor * token_ids = nullptr;   // I32 [S, B]
    ggml_tensor * pos_ids   = nullptr;   // I32 [S, B], null if unused
    ggml_tensor * type_ids  = nullptr;   // I32 [S, B], null if unused
};

// What the embedder must supply for this family.
struct InputRequirements {
    bool needs_pos_ids  = false;
    bool needs_type_ids = false;
};

class ModelArch {
public:
    virtual ~ModelArch() = default;

    virtual const ArchParams & params() const noexcept = 0;

    virtual InputRequirements inputs() const noexcept = 0;

    // Pooling the model was trained for, used when the caller does not force
    // one. GGUF may state it (`<arch>.pooling_type`); otherwise the family
    // picks its convention.
    virtual PoolType default_pooling() const noexcept = 0;

    // Resolve weight tensors against the context that owns the weight data.
    // Called once, after the inference GGUF handle is open.
    virtual void bind_weights(ggml_context * model_ctx) = 0;

    // Build the encoder stack. Returns [H, S, B] hidden states; pooling and
    // normalization are shared and applied by the caller.
    virtual ggml_tensor * build_graph(ggml_context *      gctx,
                                      const GraphInputs & in) const = 0;
};

// Build the architecture a GGUF declares. `meta` is the no_alloc context that
// owns tensor metadata (from the same gguf handle). Throws ScanError when the
// architecture is unknown or the file does not match its declared family.
std::unique_ptr<ModelArch> create_model_arch(const std::string & gguf_path);

} // namespace nanoembed
