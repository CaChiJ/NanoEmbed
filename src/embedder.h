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

    // Grow the activation reservation to cover sequences up to max_seq_len.
    // Idempotent and monotonic. Called when a context declares its cap, since
    // reserving a long-context model's full window up front is prohibitive
    // (attention is O(S^2)).
    void reserve(int max_seq_len);

    // Sequence length the activation buffer is currently sized for. May be
    // below max_seq_len(): the reservation follows what contexts asked for,
    // not the model's full context window.
    int reserved_seq_len() const noexcept;

    // Bytes reserved for graph activations, fixed at construction against the
    // worst case (max_seq_len) and reused by every call. Reported by the bench
    // and inspect tools; the M4 budget is stated partly against this.
    size_t graph_buffer_size() const noexcept;

    // Tokenize, run forward, pool, normalize. out length = n_embed.
    // Throws std::runtime_error on tokenizer / forward failures.
    void embed(const std::string &    text,
               const EmbedderConfig & cfg,
               float *                out);

    Embedder(const Embedder &)             = delete;
    Embedder & operator=(const Embedder &) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nanoembed
