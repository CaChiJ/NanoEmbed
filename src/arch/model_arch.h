// Model architecture interface + registry.
//
// Everything that differs between GGUF model families lives behind this:
// which metadata keys hold the hyperparameters, which tensors must exist, and
// what graph to build from them. `bge-small` (arch "bert") and
// `harrier-oss-v1-270m` (arch "gemma3") share almost nothing — learned vs
// rotary position encoding, LayerNorm vs RMSNorm, GELU MLP vs gated GeGLU,
// biased vs unbiased projections, bidirectional vs causal attention — so the
// seam has to sit above the individual forward builders, not inside them.
//
// Adding a family means implementing ModelArch and adding one line to
// create_model_arch(). No existing family's code is touched.

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct gguf_context;
struct ggml_context;
struct ggml_tensor;

namespace nanoembed {

// The hyperparameters every supported encoder has, read from whichever
// metadata prefix the family uses ("bert.*", "gemma3.*", ...).
struct ArchParams {
    std::string name;              // general.architecture
    int   n_layer     = 0;
    int   n_embed     = 0;
    int   n_head      = 0;         // query heads
    int   n_ff        = 0;
    int   n_vocab     = 0;
    int   max_seq_len = 0;         // model's own context length
    float norm_eps    = 1e-12f;

    // Head geometry. BERT ties these together (head_dim = n_embed / n_head,
    // one KV head per query head), but that is a coincidence of its design,
    // not a rule: gemma3 270M is n_embed=640 with 4 query heads of width 256
    // and a single shared KV head. Every family sets these explicitly.
    int head_dim  = 0;             // width of one attention head
    int n_head_kv = 0;             // KV heads; < n_head means grouped/multi-query

    // Rotary position encoding. 0 means the family does not use RoPE and
    // carries learned position embeddings instead (BERT).
    float rope_freq_base = 0.0f;

    // Whether a token may attend to later tokens. Decoder-derived embedding
    // models (gemma3) are causal; encoders (BERT) are not.
    bool causal = false;
};

enum class PoolType { Mean, Cls, Last };

// Graph-time inputs. Which of these an architecture consumes varies: BERT
// needs segment IDs, gemma3 does not. `pos_ids` is the plain 0..S-1 ramp and
// serves both uses — BERT indexes its learned table with it, gemma3 feeds it
// to RoPE. The embedder creates every tensor an arch declares it wants and
// fills it after allocation.
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

// A value that crosses a graph boundary in the streaming runner. Architectures
// name their slots; the runner interns each name to a dense id once at model
// init, so nothing on the per-sentence path touches a string.
using SlotId = uint16_t;
inline constexpr SlotId kInvalidSlot = static_cast<SlotId>(-1);

// The coarsest thing a partition strategy is allowed to group by. Anything
// finer would mean the code that merges units understands what a weight is,
// which is the leak this whole seam exists to prevent.
enum class StreamingStage { Attention, Ffn, Other };

// A unit's window onto the graph it is building. Indices are positions in the
// unit's own `inputs`/`outputs` vectors, so the declaration and the code using
// it cannot drift apart: an index the unit did not declare is out of range, and
// an output it did declare has somewhere to go.
class SlotTable final {
public:
    SlotTable(ggml_tensor * const * inputs,  size_t n_inputs,
              ggml_tensor **        outputs, size_t n_outputs,
              const GraphInputs &   graph_inputs) noexcept
        : inputs_(inputs), n_inputs_(n_inputs),
          outputs_(outputs), n_outputs_(n_outputs), graph_in_(&graph_inputs) {}

    ggml_tensor * in(size_t index) const {
        if (index >= n_inputs_) {
            throw std::out_of_range("streaming unit read an input it did not declare");
        }
        return inputs_[index];
    }

    void out(size_t index, ggml_tensor * value) {
        if (index >= n_outputs_) {
            throw std::out_of_range("streaming unit wrote an output it did not declare");
        }
        if (value == nullptr) {
            throw std::invalid_argument("streaming unit produced a null output");
        }
        if (outputs_[index] != nullptr) {
            throw std::logic_error("streaming unit wrote the same output twice");
        }
        outputs_[index] = value;
    }

    const GraphInputs & graph_inputs() const noexcept { return *graph_in_; }

private:
    ggml_tensor * const * inputs_    = nullptr;
    size_t                n_inputs_  = 0;
    ggml_tensor **        outputs_   = nullptr;
    size_t                n_outputs_ = 0;
    const GraphInputs *   graph_in_  = nullptr;
};

// One streaming unit: the weights it needs resident, and the graph nodes that
// consume them, in a single object.
//
// These used to be two independent virtuals -- a name list on one side and
// build_layer_phase() on the other -- with nothing checking that they described
// the same weights. A mismatch was silent rather than fatal, because mmap makes
// every byte readable no matter what madvise was told: results stayed correct
// while residency control was quietly wrong. Holding both in one object removes
// the opportunity to disagree.
//
// A unit declares no shapes. What `build` produces is a fact `build` already
// knows, and a second copy of it could drift -- the failure this seam exists to
// prevent. The runner records ne[] when it drains a slot and replays it when it
// feeds the next graph.
struct StreamingUnit {
    std::string              name;      // lease-key suffix and bench label
    std::vector<std::string> weights;   // exact GGUF tensor names
    std::vector<std::string> inputs;    // slots consumed, in build order
    std::vector<std::string> outputs;   // slots produced, in build order
    InputRequirements        graph_inputs{};
    StreamingStage           stage = StreamingStage::Other;
    std::function<void(ggml_context *, SlotTable &)> build;
};

// One layer's units in topological order. A partition strategy splits that
// order into contiguous runs, and merging a contiguous run of a topological
// order is always valid: everything a later unit depends on is either inside
// the run or before it. So a strategy needs only indices, never names.
struct StreamingLayerPlan {
    std::vector<StreamingUnit> units;
    std::string                input_slot;   // residual stream entering the layer
    std::string                output_slot;  // and the one leaving it
};

// The weights that are not part of any layer: the token table, streamed by the
// rows a request actually touches, and the small tensors every request needs.
// File order and numeric adjacency are deliberately absent from the contract.
struct StreamingCommonPlan {
    std::string              token_embedding;
    std::vector<std::string> common;
};

class ModelArch {
public:
    virtual ~ModelArch() = default;

    virtual const ArchParams & params() const noexcept = 0;

    virtual InputRequirements inputs() const noexcept = 0;
    virtual InputRequirements embedding_inputs() const noexcept = 0;

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

    // Phase seam for the internal streaming runner. build_graph remains the
    // eager path and composes the same operations, preserving its behavior.
    virtual ggml_tensor * build_embedding_phase(ggml_context *      gctx,
                                                const GraphInputs & in) const = 0;
    virtual ggml_tensor * build_final_phase(ggml_context * gctx,
                                            ggml_tensor *  x) const = 0;

    // The encoder stack, decomposed. One call per layer, units in topological
    // order; the runner groups them and turns each group into one graph.
    virtual StreamingLayerPlan  streaming_units(int layer) const = 0;
    virtual StreamingCommonPlan streaming_common_plan() const = 0;
};

// Build the architecture a GGUF declares. `meta` is the no_alloc context that
// owns tensor metadata (from the same gguf handle). Throws ScanError when the
// architecture is unknown or the file does not match its declared family.
std::unique_ptr<ModelArch> create_model_arch(const std::string & gguf_path);

// Build from already-open metadata-only contexts.  This is the B2 mapped
// preparation seam; it preserves the path overload for the eager loader.
std::unique_ptr<ModelArch> create_model_arch(gguf_context * gguf,
                                             ggml_context * meta);

} // namespace nanoembed
