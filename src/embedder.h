// Embedder façade — owns model state (manifest, weights, tokenizer) and runs
// a single forward + pool + (optional) L2 normalize per call.
//
// M3: in-memory baseline, no streaming; one input at a time.
// M4: introduces StreamingRunner (per-layer load).
// M5: introduces BatchedRunner (per-layer over a batch).
//
// PImpl so this header is free of ggml/gguf — only callers under src/ pay
// the include cost.

#pragma once

#include "arch/model_arch.h"   // ArchParams, PoolType

#include <cstddef>
#include <memory>
#include <string>

namespace nanoembed {

struct EmbedderConfig {
    int      n_threads   = 0;            // 0 = auto
    int      max_seq_len = 0;            // 0 = use model's default
    PoolType pooling     = PoolType::Mean;
    bool     normalize   = true;         // L2 normalize the pooled output
};

// Per-caller compute workspace: graph-metadata arena, reusable graph
// allocator, and CPU backend. Held across calls so a steady-state embed()
// performs no large allocation.
//
// NOT thread-safe, and deliberately separate from Embedder: the C ABI
// promises that distinct nanoembed_context handles may be used concurrently
// against one model, so the mutable scratch belongs to the context, not to
// the (shared, read-only) model.
class ComputeScratch {
public:
    ComputeScratch();
    ~ComputeScratch();

    ComputeScratch(const ComputeScratch &)             = delete;
    ComputeScratch & operator=(const ComputeScratch &) = delete;

private:
    friend class Embedder;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class Embedder {
public:
    explicit Embedder(const std::string & gguf_path);
    ~Embedder();

    int n_embed()     const noexcept;
    int n_layer()     const noexcept;
    int max_seq_len() const noexcept;   // the model's own context length

    // general.architecture of the loaded file ("bert", ...).
    const std::string & architecture() const noexcept;

    // Pooling the loaded model was trained for.
    PoolType default_pooling() const noexcept;

    // Grow a context's activation reservation to cover max_seq_len.
    // Idempotent and monotonic. Called when a context declares its cap, since
    // reserving a long-context model's full window up front is prohibitive
    // (attention is O(S^2)).
    //
    // `pooling` and `normalize` must be the ones the context will actually
    // embed with. Reserving against a different graph shape silently defers
    // the allocation to the first embed() — where a larger graph makes the
    // allocator resize behind the caller's back, which is exactly the failure
    // this reservation exists to move forward.
    void reserve(ComputeScratch & scratch,
                 int              max_seq_len,
                 PoolType         pooling,
                 bool             normalize) const;

    // Sequence length this context's activation buffer is sized for. May be
    // below max_seq_len(): the reservation follows what contexts asked for,
    // not the model's full context window.
    int reserved_seq_len(const ComputeScratch & scratch) const noexcept;

    // Bytes reserved for this context's graph activations and reused by every
    // call. Reported by the inspect tool; the M4 budget is stated partly
    // against this.
    size_t graph_buffer_size(const ComputeScratch & scratch) const noexcept;

    // Tokenize, run forward, pool, normalize. out length = n_embed.
    // `scratch` carries the reusable compute buffers; it must not be shared
    // between threads. Throws std::runtime_error on tokenizer / forward
    // failures.
    void embed(ComputeScratch &       scratch,
               const std::string &    text,
               const EmbedderConfig & cfg,
               float *                out);

    Embedder(const Embedder &)             = delete;
    Embedder & operator=(const Embedder &) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nanoembed
