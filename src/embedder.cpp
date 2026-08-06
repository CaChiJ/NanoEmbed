#include "embedder.h"

#include "forward/embed_layer.h"
#include "forward/encoder_block.h"
#include "forward/pool.h"
#include "gguf_scanner.h"
#include "tokenizer/wordpiece.h"

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
    return (p == PoolType::Cls) ? forward::PoolType::Cls : forward::PoolType::Mean;
}

// Upper bound on tensors in one forward graph. 12 BERT blocks come to a few
// hundred nodes; the slack is cheap because this context holds ggml_tensor
// structs only (no_alloc), not tensor data.
constexpr size_t kMaxGraphTensors = 4096;

// Everything the caller needs from a freshly built graph. The input tensors
// have no backing store until ggml_gallocr_alloc_graph runs, so they are
// handed back to be filled afterwards rather than written at build time.
struct GraphIO {
    ggml_cgraph * graph     = nullptr;
    ggml_tensor * token_ids = nullptr;
    ggml_tensor * pos_ids   = nullptr;
    ggml_tensor * type_ids  = nullptr;
    ggml_tensor * out       = nullptr;
};

} // namespace

struct Embedder::Impl {
    ModelManifest                       manifest;
    WordPieceTokenizer                  tokenizer;
    gguf_context *                      gguf      = nullptr;
    ggml_context *                      model_ctx = nullptr;
    forward::EmbedWeights               embed_w;
    std::vector<forward::LayerWeights>  layer_w;

    // Graph machinery, persistent for the handle's lifetime. `meta_buf` holds
    // tensor structs; `galloc` owns the one data buffer that every call reuses
    // under ggml's liveness analysis.
    ggml_backend_t       backend = nullptr;
    ggml_gallocr_t       galloc  = nullptr;
    std::vector<uint8_t> meta_buf;

    // Build the forward graph into `gctx` (which must be no_alloc).
    GraphIO build_graph(ggml_context * gctx,
                        int64_t        S,
                        PoolType       pooling,
                        bool           normalize) const {
        const auto &  arch = manifest.arch;
        const int64_t B    = 1;
        GraphIO io;

        io.token_ids = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, S, B);
        io.pos_ids   = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, S, B);
        io.type_ids  = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, S, B);
        // Inputs are written after allocation, so they must not share storage
        // with anything the allocator would otherwise overlap them with.
        ggml_set_input(io.token_ids);
        ggml_set_input(io.pos_ids);
        ggml_set_input(io.type_ids);

        ggml_tensor * x = forward::build_embed_layer(
            gctx, io.token_ids, io.pos_ids, io.type_ids, embed_w, arch.layer_norm_eps);
        for (int li = 0; li < arch.n_layer; ++li) {
            x = forward::build_encoder_block(
                gctx, x, /*kq_mask=*/nullptr, arch.n_head,
                layer_w[static_cast<size_t>(li)], arch.layer_norm_eps);
        }

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

    ~Impl() {
        if (galloc)    ggml_gallocr_free(galloc);
        if (backend)   ggml_backend_free(backend);
        if (model_ctx) ggml_free(model_ctx);
        if (gguf)      gguf_free(gguf);
    }
};

Embedder::Embedder(const std::string & gguf_path)
    : impl_(std::make_unique<Impl>()) {
    // Validate the file with the scanner first; this builds a typed manifest
    // and surfaces missing tensors / wrong shapes before we touch weight data.
    {
        ScanResult scan = scan_gguf(gguf_path);
        impl_->manifest = scan.manifest();
    }

    // Open a second handle that DOES allocate tensor data. This is the one
    // we forward through.
    gguf_init_params gp;
    gp.no_alloc = false;
    gp.ctx      = &impl_->model_ctx;
    impl_->gguf = gguf_init_from_file(gguf_path.c_str(), gp);
    if (!impl_->gguf || !impl_->model_ctx) {
        throw std::runtime_error("failed to open GGUF for inference: " + gguf_path);
    }

    impl_->tokenizer = WordPieceTokenizer::from_gguf(impl_->gguf);

    auto T = [&](const std::string & name) -> ggml_tensor * {
        ggml_tensor * t = ggml_get_tensor(impl_->model_ctx, name.c_str());
        if (t == nullptr) {
            throw std::runtime_error("expected tensor missing in model_ctx: " + name);
        }
        return t;
    };

    impl_->embed_w.tok    = T("token_embd.weight");
    impl_->embed_w.pos    = T("position_embd.weight");
    impl_->embed_w.type   = T("token_types.weight");
    impl_->embed_w.norm_w = T("token_embd_norm.weight");
    impl_->embed_w.norm_b = T("token_embd_norm.bias");

    impl_->layer_w.resize(static_cast<size_t>(impl_->manifest.arch.n_layer));
    for (int li = 0; li < impl_->manifest.arch.n_layer; ++li) {
        forward::LayerWeights & lw = impl_->layer_w[static_cast<size_t>(li)];
        const std::string p = "blk." + std::to_string(li) + ".";
        lw.attn.q_w    = T(p + "attn_q.weight");        lw.attn.q_b    = T(p + "attn_q.bias");
        lw.attn.k_w    = T(p + "attn_k.weight");        lw.attn.k_b    = T(p + "attn_k.bias");
        lw.attn.v_w    = T(p + "attn_v.weight");        lw.attn.v_b    = T(p + "attn_v.bias");
        lw.attn.o_w    = T(p + "attn_output.weight");   lw.attn.o_b    = T(p + "attn_output.bias");
        lw.attn.norm_w = T(p + "attn_output_norm.weight");
        lw.attn.norm_b = T(p + "attn_output_norm.bias");
        lw.ffn.up_w    = T(p + "ffn_up.weight");        lw.ffn.up_b    = T(p + "ffn_up.bias");
        lw.ffn.down_w  = T(p + "ffn_down.weight");      lw.ffn.down_b  = T(p + "ffn_down.bias");
        lw.ffn.norm_w  = T(p + "layer_output_norm.weight");
        lw.ffn.norm_b  = T(p + "layer_output_norm.bias");
    }

    impl_->backend = ggml_backend_cpu_init();
    if (!impl_->backend) throw std::runtime_error("ggml_backend_cpu_init failed");

    impl_->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(impl_->backend));
    if (!impl_->galloc) throw std::runtime_error("ggml_gallocr_new failed");

    impl_->meta_buf.resize(kMaxGraphTensors * ggml_tensor_overhead() + ggml_graph_overhead());

    // Reserve against the worst case this handle can ever see, so an
    // unaffordable model fails here rather than mid-inference, and so peak RSS
    // is fixed at construction instead of drifting with input length.
    {
        ggml_context * gctx = impl_->new_meta_ctx();
        const GraphIO  io   = impl_->build_graph(
            gctx, impl_->manifest.arch.max_seq_len, PoolType::Mean, /*normalize=*/true);
        const bool ok = ggml_gallocr_reserve(impl_->galloc, io.graph);
        ggml_free(gctx);
        if (!ok) {
            throw std::runtime_error(
                "failed to reserve the graph buffer for max_seq_len=" +
                std::to_string(impl_->manifest.arch.max_seq_len));
        }
    }
}

size_t Embedder::graph_buffer_size() const noexcept {
    return ggml_gallocr_get_buffer_size(impl_->galloc, 0);
}

Embedder::~Embedder() = default;

int Embedder::n_embed()     const noexcept { return impl_->manifest.arch.n_embed; }
int Embedder::n_layer()     const noexcept { return impl_->manifest.arch.n_layer; }
int Embedder::max_seq_len() const noexcept { return impl_->manifest.arch.max_seq_len; }

void Embedder::embed(const std::string &    text,
                     const EmbedderConfig & cfg,
                     float *                out) {
    const auto & arch      = impl_->manifest.arch;
    const int    n_threads = (cfg.n_threads <= 0) ? 4 : cfg.n_threads;

    const std::vector<int> ids = impl_->tokenizer.encode(text, cfg.max_seq_len);
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
        ggml_backend_tensor_set(io.token_ids, ids.data(), 0,
                                ids.size() * sizeof(int32_t));

        std::vector<int32_t> scratch(static_cast<size_t>(S));
        for (int64_t s = 0; s < S; ++s) scratch[static_cast<size_t>(s)] = static_cast<int32_t>(s);
        ggml_backend_tensor_set(io.pos_ids, scratch.data(), 0, scratch.size() * sizeof(int32_t));

        std::fill(scratch.begin(), scratch.end(), 0);
        ggml_backend_tensor_set(io.type_ids, scratch.data(), 0, scratch.size() * sizeof(int32_t));

        const ggml_status st = ggml_backend_graph_compute(impl_->backend, io.graph);
        if (st != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_backend_graph_compute failed");
        }

        ggml_backend_tensor_get(io.out, out, 0,
                                static_cast<size_t>(arch.n_embed) * sizeof(float));
    } catch (...) {
        ggml_free(gctx);
        throw;
    }
    ggml_free(gctx);
}

} // namespace nanoembed
