#include "embedder.h"

#include "forward/embed_layer.h"
#include "forward/encoder_block.h"
#include "forward/pool.h"
#include "gguf_scanner.h"
#include "tokenizer/wordpiece.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace nanoembed {

struct Embedder::Impl {
    ModelManifest                       manifest;
    WordPieceTokenizer                  tokenizer;
    gguf_context *                      gguf      = nullptr;
    ggml_context *                      model_ctx = nullptr;
    forward::EmbedWeights               embed_w;
    std::vector<forward::LayerWeights>  layer_w;

    ~Impl() {
        if (model_ctx) ggml_free(model_ctx);
        if (gguf)      gguf_free(gguf);
    }
};

namespace {

forward::PoolType to_forward_pool(PoolType p) {
    return (p == PoolType::Cls) ? forward::PoolType::Cls : forward::PoolType::Mean;
}

// Sized for 12 BERT blocks at S<=512, B=1 with comfortable headroom.
// Generous on purpose — M3 baseline is "ignore the memory budget".
constexpr size_t kGraphMemSize = 256ull * 1024 * 1024;

} // namespace

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
    const int64_t B = 1;

    std::vector<uint8_t> graph_buf(kGraphMemSize);
    ggml_init_params gip;
    gip.mem_size   = graph_buf.size();
    gip.mem_buffer = graph_buf.data();
    gip.no_alloc   = false;
    ggml_context * gctx = ggml_init(gip);
    if (!gctx) throw std::runtime_error("ggml_init failed");

    // Inputs.
    ggml_tensor * token_ids = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, S, B);
    std::memcpy(token_ids->data, ids.data(), ids.size() * sizeof(int32_t));

    ggml_tensor * pos_ids = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, S, B);
    auto * pos_data = static_cast<int32_t *>(pos_ids->data);
    for (int64_t s = 0; s < S; ++s) pos_data[s] = static_cast<int32_t>(s);

    ggml_tensor * type_ids = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, S, B);
    std::memset(type_ids->data, 0, static_cast<size_t>(S * B) * sizeof(int32_t));

    // Forward.
    ggml_tensor * x = forward::build_embed_layer(
        gctx, token_ids, pos_ids, type_ids, impl_->embed_w, arch.layer_norm_eps);
    for (int li = 0; li < arch.n_layer; ++li) {
        x = forward::build_encoder_block(
            gctx, x, /*kq_mask=*/nullptr, arch.n_head,
            impl_->layer_w[static_cast<size_t>(li)], arch.layer_norm_eps);
    }

    ggml_tensor * pooled = forward::build_pool(gctx, x, to_forward_pool(cfg.pooling));
    if (cfg.normalize) {
        pooled = forward::build_l2_normalize(gctx, pooled);
    }

    ggml_cgraph * graph = ggml_new_graph(gctx);
    ggml_build_forward_expand(graph, pooled);
    const ggml_status st = ggml_graph_compute_with_ctx(gctx, graph, n_threads);
    if (st != GGML_STATUS_SUCCESS) {
        ggml_free(gctx);
        throw std::runtime_error("ggml_graph_compute failed");
    }

    std::memcpy(out, pooled->data,
                static_cast<size_t>(arch.n_embed) * sizeof(float));

    ggml_free(gctx);
}

} // namespace nanoembed
