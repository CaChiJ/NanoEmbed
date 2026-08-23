#include "arch/gemma3_arch.h"

#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace nanoembed {

using namespace gguf_util;

namespace {

// llama.cpp's pooling enum, as written into <arch>.pooling_type.
PoolType pooling_from_gguf(int v) {
    switch (v) {
        case 1:  return PoolType::Mean;
        case 2:  return PoolType::Cls;
        case 3:  return PoolType::Last;
        default:
            fail("gemma3.pooling_type " + std::to_string(v) +
                 " is not one of mean(1) / cls(2) / last(3)");
    }
}

} // namespace

Gemma3Manifest scan_gemma3(const std::string & gguf_path) {
    ggml_context * meta_raw = nullptr;
    gguf_init_params gp;
    gp.no_alloc = true;
    gp.ctx      = &meta_raw;

    gguf_context * gguf_raw = gguf_init_from_file(gguf_path.c_str(), gp);
    if (gguf_raw == nullptr) {
        fail("failed to open GGUF file: " + gguf_path);
    }
    GgufPtr gguf(gguf_raw);
    GgmlPtr meta(meta_raw);

    const std::string arch_name = read_str(gguf.get(), "general.architecture");
    if (arch_name != "gemma3") {
        fail("unsupported architecture (expected 'gemma3', got '" + arch_name + "')");
    }

    Gemma3Manifest m;
    ArchParams &   a = m.params;
    a.name = "gemma3";

    a.n_layer     = read_u32_as_int(gguf.get(), "gemma3.block_count");
    a.n_embed     = read_u32_as_int(gguf.get(), "gemma3.embedding_length");
    a.n_head      = read_u32_as_int(gguf.get(), "gemma3.attention.head_count");
    a.n_head_kv   = read_u32_as_int(gguf.get(), "gemma3.attention.head_count_kv");
    a.n_ff        = read_u32_as_int(gguf.get(), "gemma3.feed_forward_length");
    a.max_seq_len = read_u32_as_int(gguf.get(), "gemma3.context_length");
    a.norm_eps    = read_f32_or(gguf.get(), "gemma3.attention.layer_norm_rms_epsilon", 1e-6f);
    a.rope_freq_base = read_f32_or(gguf.get(), "gemma3.rope.freq_base", 1e6f);
    a.causal      = true;   // decoder-derived; not stated in the file

    // head_dim is independent of n_embed here (640 hidden, 4 heads of 256),
    // so it must come from the file rather than a division.
    const int key_len   = read_u32_as_int(gguf.get(), "gemma3.attention.key_length");
    const int value_len = read_u32_as_int(gguf.get(), "gemma3.attention.value_length");
    if (key_len != value_len) {
        fail("gemma3 key_length (" + std::to_string(key_len) + ") != value_length (" +
             std::to_string(value_len) + "); split K/V widths are not supported");
    }
    a.head_dim = key_len;

    if (a.n_layer <= 0 || a.n_embed <= 0 || a.n_head <= 0 || a.n_head_kv <= 0 ||
        a.n_ff <= 0 || a.max_seq_len <= 0 || a.head_dim <= 0) {
        fail("gemma3 hyperparameters out of range (got non-positive value)");
    }
    if (a.n_head % a.n_head_kv != 0) {
        fail("query head count (" + std::to_string(a.n_head) +
             ") is not a multiple of KV head count (" + std::to_string(a.n_head_kv) + ")");
    }

    // Vocab size = length of tokenizer.ggml.tokens array.
    {
        const int64_t tk = require_kv(gguf.get(), "tokenizer.ggml.tokens");
        if (gguf_get_kv_type(gguf.get(), tk) != GGUF_TYPE_ARRAY) {
            fail("tokenizer.ggml.tokens is not an array");
        }
        a.n_vocab = static_cast<int>(gguf_get_arr_n(gguf.get(), tk));
        if (a.n_vocab <= 0) fail("vocab size is non-positive");
    }

    // Scales. The embedding one is a property of the architecture; the
    // attention one is stated per-file because Gemma decouples it from
    // head_dim upstream (query_pre_attn_scalar).
    m.embed_scale = std::sqrt(static_cast<float>(a.n_embed));
    const float qk_scalar =
        read_f32_or(gguf.get(), "gemma3.attention.key_length_scale",
                    static_cast<float>(a.head_dim));
    if (!(qk_scalar > 0.0f)) {
        fail("gemma3.attention.key_length_scale must be positive");
    }
    m.attn_scale = 1.0f / std::sqrt(qk_scalar);

    m.pooling   = pooling_from_gguf(read_u32_as_int(gguf.get(), "gemma3.pooling_type"));
    m.normalize = read_bool_or(gguf.get(), "gemma3.normalize_embeddings", true);

    const int64_t H  = a.n_embed;
    const int64_t F  = a.n_ff;
    const int64_t V  = a.n_vocab;
    const int64_t D  = a.head_dim;
    const int64_t QW = static_cast<int64_t>(a.n_head)    * D;   // packed query width
    const int64_t KW = static_cast<int64_t>(a.n_head_kv) * D;   // packed K/V width

    m.tok_embed = require_tensor(gguf.get(), meta.get(), "token_embd.weight");
    validate_shape_2d(m.tok_embed, H, V, "token_embd.weight");

    m.output_norm = require_tensor(gguf.get(), meta.get(), "output_norm.weight");
    validate_shape_1d(m.output_norm, H, "output_norm.weight");

    m.layers.resize(static_cast<size_t>(a.n_layer));
    for (int li = 0; li < a.n_layer; ++li) {
        Gemma3LayerSlot & L = m.layers[static_cast<size_t>(li)];

        auto req = [&](const char * suffix) {
            const std::string n = layer_tensor_name(li, suffix);
            return std::pair<TensorRef, std::string>{
                require_tensor(gguf.get(), meta.get(), n), n};
        };
        auto req_2d = [&](const char * suffix, int64_t rows, int64_t cols) {
            auto [t, n] = req(suffix);
            validate_shape_2d(t, rows, cols, n);
            return t;
        };
        auto req_1d = [&](const char * suffix, int64_t len) {
            auto [t, n] = req(suffix);
            validate_shape_1d(t, len, n);
            return t;
        };

        L.attn_norm           = req_1d("attn_norm.weight", H);
        L.attn_q              = req_2d("attn_q.weight",      H,  QW);
        L.attn_k              = req_2d("attn_k.weight",      H,  KW);
        L.attn_v              = req_2d("attn_v.weight",      H,  KW);
        L.attn_output         = req_2d("attn_output.weight", QW, H);
        L.attn_q_norm         = req_1d("attn_q_norm.weight", D);
        L.attn_k_norm         = req_1d("attn_k_norm.weight", D);
        L.post_attention_norm = req_1d("post_attention_norm.weight", H);
        L.ffn_norm            = req_1d("ffn_norm.weight", H);
        L.post_ffw_norm       = req_1d("post_ffw_norm.weight", H);
        L.ffn_gate            = req_2d("ffn_gate.weight", H, F);
        L.ffn_up              = req_2d("ffn_up.weight",   H, F);
        L.ffn_down            = req_2d("ffn_down.weight", F, H);
    }

    return m;
}

Gemma3ModelArch::Gemma3ModelArch(const std::string & gguf_path)
    : manifest_(scan_gemma3(gguf_path)) {}

void Gemma3ModelArch::bind_weights(ggml_context * model_ctx) {
    auto T = [&](const std::string & name) -> ggml_tensor * {
        ggml_tensor * t = ggml_get_tensor(model_ctx, name.c_str());
        if (t == nullptr) {
            throw std::runtime_error("expected tensor missing in model_ctx: " + name);
        }
        return t;
    };

    tok_embed_   = T("token_embd.weight");
    output_norm_ = T("output_norm.weight");

    const int n_layer = manifest_.params.n_layer;
    layer_w_.resize(static_cast<size_t>(n_layer));
    for (int li = 0; li < n_layer; ++li) {
        const std::string p = "blk." + std::to_string(li) + ".";
        Gemma3LayerWeights & w = layer_w_[static_cast<size_t>(li)];

        w.attn.norm      = T(p + "attn_norm.weight");
        w.attn.q         = T(p + "attn_q.weight");
        w.attn.k         = T(p + "attn_k.weight");
        w.attn.v         = T(p + "attn_v.weight");
        w.attn.o         = T(p + "attn_output.weight");
        w.attn.q_norm    = T(p + "attn_q_norm.weight");
        w.attn.k_norm    = T(p + "attn_k_norm.weight");
        w.attn.post_norm = T(p + "post_attention_norm.weight");

        w.ffn.norm       = T(p + "ffn_norm.weight");
        w.ffn.gate       = T(p + "ffn_gate.weight");
        w.ffn.up         = T(p + "ffn_up.weight");
        w.ffn.down       = T(p + "ffn_down.weight");
        w.ffn.post_norm  = T(p + "post_ffw_norm.weight");
    }
}

ggml_tensor * Gemma3ModelArch::build_graph(ggml_context *      gctx,
                                           const GraphInputs & in) const {
    (void) gctx;
    (void) in;
    // The rotary / RMSNorm / GeGLU / grouped-attention builders land with
    // PLAN.md M3.6 Phase 3. Scanning and weight binding are already live, so
    // nanoembed-inspect reports this model correctly even now.
    throw std::runtime_error(
        "gemma3 forward graph is not implemented yet (PLAN.md M3.6)");
}

} // namespace nanoembed
