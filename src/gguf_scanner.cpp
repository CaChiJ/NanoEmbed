#include "gguf_scanner.h"

#include "ggml.h"
#include "gguf.h"

#include <memory>
#include <sstream>
#include <string>

namespace nanoembed {

namespace {

// ---- Cleanup helpers ---------------------------------------------------

struct GgufDeleter {
    void operator()(gguf_context * p) const noexcept { if (p) gguf_free(p); }
};
struct GgmlDeleter {
    void operator()(ggml_context * p) const noexcept { if (p) ggml_free(p); }
};

using GgufPtr = std::unique_ptr<gguf_context, GgufDeleter>;
using GgmlPtr = std::unique_ptr<ggml_context, GgmlDeleter>;

[[noreturn]] void fail(const std::string & msg) {
    throw ScanError(msg);
}

// ---- KV helpers --------------------------------------------------------

int64_t require_kv(gguf_context * ctx, const char * key) {
    int64_t i = gguf_find_key(ctx, key);
    if (i < 0) fail(std::string("missing required metadata key: ") + key);
    return i;
}

int read_u32_as_int(gguf_context * ctx, const char * key) {
    int64_t i = require_kv(ctx, key);
    if (gguf_get_kv_type(ctx, i) != GGUF_TYPE_UINT32) {
        fail(std::string("metadata key has wrong type (expected u32): ") + key);
    }
    return static_cast<int>(gguf_get_val_u32(ctx, i));
}

float read_f32_or(gguf_context * ctx, const char * key, float fallback) {
    int64_t i = gguf_find_key(ctx, key);
    if (i < 0) return fallback;
    if (gguf_get_kv_type(ctx, i) != GGUF_TYPE_FLOAT32) return fallback;
    return gguf_get_val_f32(ctx, i);
}

std::string read_str(gguf_context * ctx, const char * key) {
    int64_t i = require_kv(ctx, key);
    if (gguf_get_kv_type(ctx, i) != GGUF_TYPE_STRING) {
        fail(std::string("metadata key has wrong type (expected string): ") + key);
    }
    return std::string(gguf_get_val_str(ctx, i));
}

// ---- Tensor helpers ----------------------------------------------------

TensorRef require_tensor(gguf_context * gguf,
                         ggml_context * meta,
                         const std::string & name) {
    const int64_t idx = gguf_find_tensor(gguf, name.c_str());
    if (idx < 0) fail("missing required tensor: " + name);

    ggml_tensor * t = ggml_get_tensor(meta, name.c_str());
    if (t == nullptr) {
        fail("tensor present in gguf but not in ggml meta: " + name);
    }

    TensorRef r;
    r.gguf_index  = idx;
    r.ggml_type   = static_cast<int>(gguf_get_tensor_type(gguf, idx));
    r.size_bytes  = gguf_get_tensor_size(gguf, idx);
    r.data_offset = gguf_get_tensor_offset(gguf, idx);
    for (int d = 0; d < 4; ++d) {
        r.ne[d] = t->ne[d];
    }
    return r;
}

std::string layer_tensor_name(int layer_idx, const char * suffix) {
    std::ostringstream s;
    s << "blk." << layer_idx << "." << suffix;
    return s.str();
}

void validate_shape_2d(const TensorRef & r,
                       int64_t expected_rows,
                       int64_t expected_cols,
                       const std::string & name) {
    // ggml stores a 2D matrix W of shape [in, out] with ne = {in, out, 1, 1}.
    // We pass (rows, cols) in the matmul-natural order; convert here.
    if (r.ne[0] != expected_rows || r.ne[1] != expected_cols ||
        r.ne[2] != 1 || r.ne[3] != 1) {
        std::ostringstream s;
        s << "tensor shape mismatch for " << name
          << ": expected ne=[" << expected_rows << "," << expected_cols << ",1,1]"
          << " got ne=[" << r.ne[0] << "," << r.ne[1] << "," << r.ne[2] << "," << r.ne[3] << "]";
        fail(s.str());
    }
}

void validate_shape_1d(const TensorRef & r,
                       int64_t expected,
                       const std::string & name) {
    if (r.ne[0] != expected || r.ne[1] != 1 || r.ne[2] != 1 || r.ne[3] != 1) {
        std::ostringstream s;
        s << "tensor shape mismatch for " << name
          << ": expected ne=[" << expected << ",1,1,1]"
          << " got ne=[" << r.ne[0] << "," << r.ne[1] << "," << r.ne[2] << "," << r.ne[3] << "]";
        fail(s.str());
    }
}

} // namespace

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

    // Architecture must be BERT in v0.
    const std::string arch_name = read_str(gguf_owner.get(), "general.architecture");
    if (arch_name != "bert") {
        fail("unsupported architecture (expected 'bert', got '" + arch_name + "')");
    }

    ModelManifest m;

    // Hyperparameters
    m.arch.n_layer        = read_u32_as_int(gguf_owner.get(), "bert.block_count");
    m.arch.n_embed        = read_u32_as_int(gguf_owner.get(), "bert.embedding_length");
    m.arch.n_head         = read_u32_as_int(gguf_owner.get(), "bert.attention.head_count");
    m.arch.n_ff           = read_u32_as_int(gguf_owner.get(), "bert.feed_forward_length");
    m.arch.max_seq_len    = read_u32_as_int(gguf_owner.get(), "bert.context_length");
    m.arch.layer_norm_eps = read_f32_or   (gguf_owner.get(), "bert.attention.layer_norm_epsilon", 1e-12f);

    if (m.arch.n_layer <= 0 || m.arch.n_embed <= 0 ||
        m.arch.n_head <= 0  || m.arch.n_ff    <= 0 ||
        m.arch.max_seq_len <= 0) {
        fail("BERT hyperparameters out of range (got non-positive value)");
    }
    if (m.arch.n_embed % m.arch.n_head != 0) {
        fail("hidden size is not divisible by head count");
    }

    // Vocab size = length of tokenizer.ggml.tokens array
    {
        const int64_t tk = require_kv(gguf_owner.get(), "tokenizer.ggml.tokens");
        if (gguf_get_kv_type(gguf_owner.get(), tk) != GGUF_TYPE_ARRAY) {
            fail("tokenizer.ggml.tokens is not an array");
        }
        m.arch.n_vocab = static_cast<int>(gguf_get_arr_n(gguf_owner.get(), tk));
        if (m.arch.n_vocab <= 0) fail("vocab size is non-positive");
    }

    const int64_t H = m.arch.n_embed;
    const int64_t F = m.arch.n_ff;
    const int64_t V = m.arch.n_vocab;
    const int64_t S = m.arch.max_seq_len;

    // Embedding-stage tensors
    m.tok_embed_w  = require_tensor(gguf_owner.get(), meta_owner.get(), "token_embd.weight");
    validate_shape_2d(m.tok_embed_w, H, V, "token_embd.weight");
    m.pos_embed_w  = require_tensor(gguf_owner.get(), meta_owner.get(), "position_embd.weight");
    validate_shape_2d(m.pos_embed_w, H, S, "position_embd.weight");
    m.type_embed_w = require_tensor(gguf_owner.get(), meta_owner.get(), "token_types.weight");
    // token_types.weight has shape [H, n_types] where n_types is typically 2.
    if (m.type_embed_w.ne[0] != H || m.type_embed_w.ne[1] < 1) {
        fail("token_types.weight has unexpected shape");
    }
    m.embed_norm_w = require_tensor(gguf_owner.get(), meta_owner.get(), "token_embd_norm.weight");
    validate_shape_1d(m.embed_norm_w, H, "token_embd_norm.weight");
    m.embed_norm_b = require_tensor(gguf_owner.get(), meta_owner.get(), "token_embd_norm.bias");
    validate_shape_1d(m.embed_norm_b, H, "token_embd_norm.bias");

    // Optional sentence-bert pooler dense layer.
    {
        const int64_t pw = gguf_find_tensor(gguf_owner.get(), "pooler.dense.weight");
        if (pw >= 0) {
            m.pooler_w = require_tensor(gguf_owner.get(), meta_owner.get(), "pooler.dense.weight");
            m.pooler_b = require_tensor(gguf_owner.get(), meta_owner.get(), "pooler.dense.bias");
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

        s.attn_q_w    = require_tensor(gguf_owner.get(), meta_owner.get(), nq_w);
        s.attn_q_b    = require_tensor(gguf_owner.get(), meta_owner.get(), nq_b);
        s.attn_k_w    = require_tensor(gguf_owner.get(), meta_owner.get(), nk_w);
        s.attn_k_b    = require_tensor(gguf_owner.get(), meta_owner.get(), nk_b);
        s.attn_v_w    = require_tensor(gguf_owner.get(), meta_owner.get(), nv_w);
        s.attn_v_b    = require_tensor(gguf_owner.get(), meta_owner.get(), nv_b);
        s.attn_o_w    = require_tensor(gguf_owner.get(), meta_owner.get(), no_w);
        s.attn_o_b    = require_tensor(gguf_owner.get(), meta_owner.get(), no_b);
        s.attn_norm_w = require_tensor(gguf_owner.get(), meta_owner.get(), nan_w);
        s.attn_norm_b = require_tensor(gguf_owner.get(), meta_owner.get(), nan_b);
        s.ffn_up_w    = require_tensor(gguf_owner.get(), meta_owner.get(), nfu_w);
        s.ffn_up_b    = require_tensor(gguf_owner.get(), meta_owner.get(), nfu_b);
        s.ffn_down_w  = require_tensor(gguf_owner.get(), meta_owner.get(), nfd_w);
        s.ffn_down_b  = require_tensor(gguf_owner.get(), meta_owner.get(), nfd_b);
        s.ffn_norm_w  = require_tensor(gguf_owner.get(), meta_owner.get(), nfn_w);
        s.ffn_norm_b  = require_tensor(gguf_owner.get(), meta_owner.get(), nfn_b);

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

    ScanResult result;
    result.reset(gguf_owner.release(), meta_owner.release(), std::move(m));
    return result;
}

} // namespace nanoembed
