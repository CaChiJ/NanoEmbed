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

#include <memory>
#include <string>

namespace nanoembed {

enum class PoolType { Mean, Cls };

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
    int max_seq_len() const noexcept;

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
