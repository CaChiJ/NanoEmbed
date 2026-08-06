#include "embedder.h"

#include "arch/model_arch.h"
#include "forward/pool.h"
#include "tokenizer/tokenizer.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace nanoembed {

namespace {

forward::PoolType to_forward_pool(PoolType p) {
    switch (p) {
        case PoolType::Cls:  return forward::PoolType::Cls;
        case PoolType::Last: return forward::PoolType::Last;
        default:             return forward::PoolType::Mean;
    }
}

// Upper bound on tensors in one forward graph. A 12-block encoder comes to
// roughly 500 nodes (BERT) and a SwiGLU/rotary one somewhat more, so this is
// several times the real requirement. Not free, though: the buffer is
// zero-initialized and therefore fully resident, at 368 B/tensor -- 4096 costs
// 1.52 MiB and doubling it showed up as a 1.5 MiB rise in peak RSS.
constexpr size_t kMaxGraphTensors = 4096;

// Fallback cap for the activation reservation when the caller never states a
// max_seq_len. Long-context models make "just reserve the model maximum"
// untenable: attention is O(S^2), so eurobert's 8192 context would reserve
// gigabytes for inputs that are never that long. Callers who do want the full
// context ask for it explicitly and pay for it.
constexpr int kDefaultReserveSeqLen = 512;

// Everything the caller needs from a freshly built graph. The input tensors
// have no backing store until ggml_gallocr_alloc_graph runs, so they are
// handed back to be filled afterwards rather than written at build time.
struct GraphIO {
    ggml_cgraph * graph = nullptr;
    GraphInputs   in;
    ggml_tensor * out   = nullptr;
};

} // namespace

struct Embedder::Impl {
    std::unique_ptr<ModelArch> arch;
    std::unique_ptr<Tokenizer> tokenizer;

    gguf_context * gguf      = nullptr;
    ggml_context * model_ctx = nullptr;

    // Graph machinery, persistent for the handle's lifetime. `meta_buf` holds
    // tensor structs; `galloc` owns the one data buffer that every call reuses
    // under ggml's liveness analysis.
    ggml_backend_t       backend = nullptr;
    ggml_gallocr_t       galloc  = nullptr;
    std::vector<uint8_t> meta_buf;
    int                  reserved_seq_len = 0;

    GraphIO build_graph(ggml_context * gctx,
                        int64_t        S,
                        PoolType       pooling,
                        bool           normalize) const {
        const InputRequirements req = arch->inputs();
        const int64_t           B   = 1;
        GraphIO io;

        auto make_input = [&](ggml_tensor ** slot) {
            *slot = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, S, B);
            // Written after allocation, so they must not share storage with
            // anything the allocator would otherwise overlap them with.
            ggml_set_input(*slot);
        };

        make_input(&io.in.token_ids);
        if (req.needs_pos_ids)  make_input(&io.in.pos_ids);
        if (req.needs_type_ids) make_input(&io.in.type_ids);

        ggml_tensor * x = arch->build_graph(gctx, io.in);

        io.out = forward::build_pool(gctx, x, to_forward_pool(pooling));
        if (normalize) {
            io.out = forward::build_l2_normalize(gctx, io.out);
        }
        // Read back after compute, so it must survive the whole graph.
        ggml_set_output(io.out);

        io.graph = ggml_new_graph(gctx);
        ggml_build_forward_expand(io.graph, io.out);
        return io;
    }

    // A no_alloc context over the persistent meta buffer. Structs only.
    ggml_context * new_meta_ctx() {
        ggml_init_params p;
        p.mem_size   = meta_buf.size();
        p.mem_buffer = meta_buf.data();
        p.no_alloc   = true;
        ggml_context * c = ggml_init(p);
        if (!c) throw std::runtime_error("ggml_init failed for graph meta context");
        return c;
    }

    // Size the activation buffer for sequences up to `seq_len`. Idempotent and
    // monotonic: several contexts may share one model handle, and the buffer
    // has to satisfy the longest of them.
    void reserve_for(int seq_len) {
        const int want = std::min(std::max(seq_len, 1), arch->params().max_seq_len);
        if (want <= reserved_seq_len) return;

        ggml_context * gctx = new_meta_ctx();
        const GraphIO  io   = build_graph(gctx, want, arch->default_pooling(),
                                          /*normalize=*/true);
        const bool ok = ggml_gallocr_reserve(galloc, io.graph);
        ggml_free(gctx);
        if (!ok) {
            throw std::runtime_error(
                "failed to reserve the graph buffer for max_seq_len=" +
                std::to_string(want));
        }
        reserved_seq_len = want;
    }

    ~Impl() {
        if (galloc)    ggml_gallocr_free(galloc);
        if (backend)   ggml_backend_free(backend);
        if (model_ctx) ggml_free(model_ctx);
        if (gguf)      gguf_free(gguf);
    }
};

Embedder::Embedder(const std::string & gguf_path)
    : impl_(std::make_unique<Impl>()) {
    // Dispatch on general.architecture. This validates the file and reads the
    // family's hyperparameters before any weight data is touched.
    impl_->arch = create_model_arch(gguf_path);

    // Open a second handle that DOES allocate tensor data. This is the one
    // we forward through.
    gguf_init_params gp;
    gp.no_alloc = false;
    gp.ctx      = &impl_->model_ctx;
    impl_->gguf = gguf_init_from_file(gguf_path.c_str(), gp);
    if (!impl_->gguf || !impl_->model_ctx) {
        throw std::runtime_error("failed to open GGUF for inference: " + gguf_path);
    }

    impl_->tokenizer = create_tokenizer(impl_->gguf);
    impl_->arch->bind_weights(impl_->model_ctx);

    impl_->backend = ggml_backend_cpu_init();
    if (!impl_->backend) throw std::runtime_error("ggml_backend_cpu_init failed");

    impl_->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl_->backend));
    if (!impl_->galloc) throw std::runtime_error("ggml_gallocr_new failed");

    impl_->meta_buf.resize(kMaxGraphTensors * ggml_tensor_overhead() + ggml_graph_overhead());

    // Reserve up front so an unaffordable model fails at load rather than
    // mid-inference, and so peak RSS is fixed instead of drifting with input
    // length. reserve_for() grows this if a context asks for more.
    impl_->reserve_for(kDefaultReserveSeqLen);
}

Embedder::~Embedder() = default;

int Embedder::n_embed()     const noexcept { return impl_->arch->params().n_embed; }
int Embedder::n_layer()     const noexcept { return impl_->arch->params().n_layer; }
int Embedder::max_seq_len() const noexcept { return impl_->arch->params().max_seq_len; }

const std::string & Embedder::architecture() const noexcept {
    return impl_->arch->params().name;
}

PoolType Embedder::default_pooling() const noexcept {
    return impl_->arch->default_pooling();
}

int Embedder::reserved_seq_len() const noexcept { return impl_->reserved_seq_len; }

size_t Embedder::graph_buffer_size() const noexcept {
    return ggml_gallocr_get_buffer_size(impl_->galloc, 0);
}

void Embedder::reserve(int max_seq_len) {
    impl_->reserve_for(max_seq_len);
}

void Embedder::embed(const std::string &    text,
                     const EmbedderConfig & cfg,
                     float *                out) {
    const int n_embed   = impl_->arch->params().n_embed;
    const int n_threads = (cfg.n_threads <= 0) ? 4 : cfg.n_threads;

    const std::vector<int> ids = impl_->tokenizer->encode(text, cfg.max_seq_len);
    const int64_t S = static_cast<int64_t>(ids.size());

    ggml_backend_cpu_set_n_threads(impl_->backend, n_threads);

    // The context holds tensor structs only; tensor data comes from galloc's
    // reused buffer, which is why this costs a few hundred KB per call instead
    // of re-faulting a fresh arena.
    ggml_context * gctx = impl_->new_meta_ctx();
    try {
        const GraphIO io = impl_->build_graph(gctx, S, cfg.pooling, cfg.normalize);

        if (!ggml_gallocr_alloc_graph(impl_->galloc, io.graph)) {
            throw std::runtime_error("failed to allocate the graph buffer");
        }

        // Only now do the input tensors have storage.
        ggml_backend_tensor_set(io.in.token_ids, ids.data(), 0,
                                ids.size() * sizeof(int32_t));

        std::vector<int32_t> scratch(static_cast<size_t>(S));
        if (io.in.pos_ids) {
            for (int64_t s = 0; s < S; ++s) scratch[static_cast<size_t>(s)] = static_cast<int32_t>(s);
            ggml_backend_tensor_set(io.in.pos_ids, scratch.data(), 0,
                                    scratch.size() * sizeof(int32_t));
        }
        if (io.in.type_ids) {
            std::fill(scratch.begin(), scratch.end(), 0);
            ggml_backend_tensor_set(io.in.type_ids, scratch.data(), 0,
                                    scratch.size() * sizeof(int32_t));
        }

        const ggml_status st = ggml_backend_graph_compute(impl_->backend, io.graph);
        if (st != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_backend_graph_compute failed");
        }

        ggml_backend_tensor_get(io.out, out, 0,
                                static_cast<size_t>(n_embed) * sizeof(float));
    } catch (...) {
        ggml_free(gctx);
        throw;
    }
    ggml_free(gctx);
}

} // namespace nanoembed
