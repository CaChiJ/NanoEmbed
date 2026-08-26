#include "gguf_scanner.h"

#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <string>

namespace nanoembed {

// The KV readers, tensor lookup and shape validators used below are
// architecture-agnostic and now live in gguf_util.h, so gemma3 (and any
// later family) gets the same error text without depending on this file.
using namespace gguf_util;

// ---- ScanResult --------------------------------------------------------

void ScanResult::release() noexcept {
    // ggml_context first — its tensors reference offsets the gguf_context tracks.
    if (meta_ctx_) { ggml_free(meta_ctx_); meta_ctx_ = nullptr; }
    if (gguf_ctx_) { gguf_free(gguf_ctx_); gguf_ctx_ = nullptr; }
}

ScanResult::~ScanResult() {
    release();
}

ScanResult::ScanResult(ScanResult && other) noexcept
    : gguf_ctx_(other.gguf_ctx_),
      meta_ctx_(other.meta_ctx_),
      manifest_(std::move(other.manifest_)) {
    other.gguf_ctx_ = nullptr;
    other.meta_ctx_ = nullptr;
}

ScanResult & ScanResult::operator=(ScanResult && other) noexcept {
    if (this != &other) {
        release();
        gguf_ctx_ = other.gguf_ctx_;
        meta_ctx_ = other.meta_ctx_;
        manifest_ = std::move(other.manifest_);
        other.gguf_ctx_ = nullptr;
        other.meta_ctx_ = nullptr;
    }
    return *this;
}

void ScanResult::reset(gguf_context * gguf,
                       ggml_context * meta,
                       ModelManifest m) {
    release();
    gguf_ctx_ = gguf;
    meta_ctx_ = meta;
    manifest_ = std::move(m);
}

// ---- scan_gguf ---------------------------------------------------------

ModelManifest scan_bert(gguf_context * gguf, ggml_context * meta) {
    if (gguf == nullptr || meta == nullptr) {
        fail("BERT scan requires GGUF and ggml metadata contexts");
    }
    // Architecture must be BERT in v0.
    const std::string arch_name = read_str(gguf, "general.architecture");
    if (arch_name != "bert") {
        fail("unsupported architecture (expected 'bert', got '" + arch_name + "')");
    }

    ModelManifest m;

    // Hyperparameters
    m.arch.n_layer        = read_u32_as_int(gguf, "bert.block_count");
    m.arch.n_embed        = read_u32_as_int(gguf, "bert.embedding_length");
    m.arch.n_head         = read_u32_as_int(gguf, "bert.attention.head_count");
    m.arch.n_ff           = read_u32_as_int(gguf, "bert.feed_forward_length");
    m.arch.max_seq_len    = read_u32_as_int(gguf, "bert.context_length");
    m.arch.layer_norm_eps = read_f32_or   (gguf, "bert.attention.layer_norm_epsilon", 1e-12f);
    m.arch.pooling_type   = read_u32_or   (gguf, "bert.pooling_type", 1);

    if (m.arch.n_layer <= 0 || m.arch.n_embed <= 0 ||
        m.arch.n_head <= 0  || m.arch.n_ff    <= 0 ||
        m.arch.max_seq_len <= 0) {
        fail("BERT hyperparameters out of range (got non-positive value)");
    }
    if (!(m.arch.layer_norm_eps > 0.0f) || !std::isfinite(m.arch.layer_norm_eps)) {
        fail("bert.attention.layer_norm_epsilon must be finite and positive");
    }
    if (m.arch.pooling_type < 1 || m.arch.pooling_type > 3) {
        fail("bert.pooling_type must be mean(1), cls(2), or last(3)");
    }
    if (m.arch.n_embed % m.arch.n_head != 0) {
        fail("hidden size is not divisible by head count");
    }

    // Vocab size = length of tokenizer.ggml.tokens array
    {
        const int64_t tk = require_kv(gguf, "tokenizer.ggml.tokens");
        if (gguf_get_kv_type(gguf, tk) != GGUF_TYPE_ARRAY) {
            fail("tokenizer.ggml.tokens is not an array");
        }
        const size_t n_vocab = gguf_get_arr_n(gguf, tk);
        if (n_vocab > static_cast<size_t>(std::numeric_limits<int>::max())) {
            fail("tokenizer vocabulary is too large");
        }
        m.arch.n_vocab = static_cast<int>(n_vocab);
        if (m.arch.n_vocab <= 0) fail("vocab size is non-positive");
    }

    const int64_t H = m.arch.n_embed;
    const int64_t F = m.arch.n_ff;
    const int64_t V = m.arch.n_vocab;
    const int64_t S = m.arch.max_seq_len;

    // Embedding-stage tensors
    m.tok_embed_w  = require_tensor(gguf, meta, "token_embd.weight");
    validate_shape_2d(m.tok_embed_w, H, V, "token_embd.weight");
    m.pos_embed_w  = require_tensor(gguf, meta, "position_embd.weight");
    validate_shape_2d(m.pos_embed_w, H, S, "position_embd.weight");
    m.type_embed_w = require_tensor(gguf, meta, "token_types.weight");
    // token_types.weight has shape [H, n_types] where n_types is typically 2.
    if (m.type_embed_w.ne[0] != H || m.type_embed_w.ne[1] < 1) {
        fail("token_types.weight has unexpected shape");
    }
    m.embed_norm_w = require_tensor(gguf, meta, "token_embd_norm.weight");
    validate_shape_1d(m.embed_norm_w, H, "token_embd_norm.weight");
    m.embed_norm_b = require_tensor(gguf, meta, "token_embd_norm.bias");
    validate_shape_1d(m.embed_norm_b, H, "token_embd_norm.bias");

    // Optional sentence-bert pooler dense layer.
    {
        const int64_t pw = gguf_find_tensor(gguf, "pooler.dense.weight");
        if (pw >= 0) {
            m.pooler_w = require_tensor(gguf, meta, "pooler.dense.weight");
            m.pooler_b = require_tensor(gguf, meta, "pooler.dense.bias");
        }
    }

    // Per-layer tensors
    m.layers.resize(static_cast<size_t>(m.arch.n_layer));
    for (int i = 0; i < m.arch.n_layer; ++i) {
        LayerSlot & s = m.layers[static_cast<size_t>(i)];

        const std::string nq_w  = layer_tensor_name(i, "attn_q.weight");
        const std::string nq_b  = layer_tensor_name(i, "attn_q.bias");
        const std::string nk_w  = layer_tensor_name(i, "attn_k.weight");
        const std::string nk_b  = layer_tensor_name(i, "attn_k.bias");
        const std::string nv_w  = layer_tensor_name(i, "attn_v.weight");
        const std::string nv_b  = layer_tensor_name(i, "attn_v.bias");
        const std::string no_w  = layer_tensor_name(i, "attn_output.weight");
        const std::string no_b  = layer_tensor_name(i, "attn_output.bias");
        const std::string nan_w = layer_tensor_name(i, "attn_output_norm.weight");
        const std::string nan_b = layer_tensor_name(i, "attn_output_norm.bias");
        const std::string nfu_w = layer_tensor_name(i, "ffn_up.weight");
        const std::string nfu_b = layer_tensor_name(i, "ffn_up.bias");
        const std::string nfd_w = layer_tensor_name(i, "ffn_down.weight");
        const std::string nfd_b = layer_tensor_name(i, "ffn_down.bias");
        const std::string nfn_w = layer_tensor_name(i, "layer_output_norm.weight");
        const std::string nfn_b = layer_tensor_name(i, "layer_output_norm.bias");

        s.attn_q_w    = require_tensor(gguf, meta, nq_w);
        s.attn_q_b    = require_tensor(gguf, meta, nq_b);
        s.attn_k_w    = require_tensor(gguf, meta, nk_w);
        s.attn_k_b    = require_tensor(gguf, meta, nk_b);
        s.attn_v_w    = require_tensor(gguf, meta, nv_w);
        s.attn_v_b    = require_tensor(gguf, meta, nv_b);
        s.attn_o_w    = require_tensor(gguf, meta, no_w);
        s.attn_o_b    = require_tensor(gguf, meta, no_b);
        s.attn_norm_w = require_tensor(gguf, meta, nan_w);
        s.attn_norm_b = require_tensor(gguf, meta, nan_b);
        s.ffn_up_w    = require_tensor(gguf, meta, nfu_w);
        s.ffn_up_b    = require_tensor(gguf, meta, nfu_b);
        s.ffn_down_w  = require_tensor(gguf, meta, nfd_w);
        s.ffn_down_b  = require_tensor(gguf, meta, nfd_b);
        s.ffn_norm_w  = require_tensor(gguf, meta, nfn_w);
        s.ffn_norm_b  = require_tensor(gguf, meta, nfn_b);

        // Shape spot-checks. Q/K/V/O are square H x H, FFN_up is H x F,
        // FFN_down is F x H, biases are 1D.
        validate_shape_2d(s.attn_q_w, H, H, nq_w);
        validate_shape_2d(s.attn_k_w, H, H, nk_w);
        validate_shape_2d(s.attn_v_w, H, H, nv_w);
        validate_shape_2d(s.attn_o_w, H, H, no_w);
        validate_shape_1d(s.attn_q_b, H, nq_b);
        validate_shape_1d(s.attn_k_b, H, nk_b);
        validate_shape_1d(s.attn_v_b, H, nv_b);
        validate_shape_1d(s.attn_o_b, H, no_b);
        validate_shape_1d(s.attn_norm_w, H, nan_w);
        validate_shape_1d(s.attn_norm_b, H, nan_b);
        validate_shape_2d(s.ffn_up_w,   H, F, nfu_w);
        validate_shape_1d(s.ffn_up_b,   F, nfu_b);
        validate_shape_2d(s.ffn_down_w, F, H, nfd_w);
        validate_shape_1d(s.ffn_down_b, H, nfd_b);
        validate_shape_1d(s.ffn_norm_w, H, nfn_w);
        validate_shape_1d(s.ffn_norm_b, H, nfn_b);
    }

    return m;
}

ScanResult scan_gguf(const std::string & path) {
    ggml_context * meta_raw = nullptr;
    gguf_init_params params;
    params.no_alloc = true;
    params.ctx      = &meta_raw;

    gguf_context * gguf_raw = gguf_init_from_file(path.c_str(), params);
    if (gguf_raw == nullptr) {
        fail("failed to open GGUF file: " + path);
    }
    GgufPtr gguf_owner(gguf_raw);
    GgmlPtr meta_owner(meta_raw);

    ModelManifest manifest = scan_bert(gguf_owner.get(), meta_owner.get());
    ScanResult result;
    result.reset(gguf_owner.release(), meta_owner.release(), std::move(manifest));
    return result;
}

} // namespace nanoembed
