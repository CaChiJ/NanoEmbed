#include "arch/gemma3_arch.h"

#include "forward/rms_norm.h"

#include "ggml.h"
#include "gguf.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

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

Gemma3Manifest scan_gemma3(gguf_context * gguf, ggml_context * meta) {
    if (gguf == nullptr || meta == nullptr) {
        fail("gemma3 scan requires GGUF and ggml metadata contexts");
    }

    const std::string arch_name = read_str(gguf, "general.architecture");
    if (arch_name != "gemma3") {
        fail("unsupported architecture (expected 'gemma3', got '" + arch_name + "')");
    }

    Gemma3Manifest m;
    ArchParams &   a = m.params;
    a.name = "gemma3";

    a.n_layer     = read_u32_as_int(gguf, "gemma3.block_count");
    a.n_embed     = read_u32_as_int(gguf, "gemma3.embedding_length");
    a.n_head      = read_u32_as_int(gguf, "gemma3.attention.head_count");
    a.n_head_kv   = read_u32_as_int(gguf, "gemma3.attention.head_count_kv");
    a.n_ff        = read_u32_as_int(gguf, "gemma3.feed_forward_length");
    a.max_seq_len = read_u32_as_int(gguf, "gemma3.context_length");
    a.norm_eps    = read_f32_or(gguf, "gemma3.attention.layer_norm_rms_epsilon", 1e-6f);
    a.rope_freq_base = read_f32_or(gguf, "gemma3.rope.freq_base", 1e6f);
    a.causal      = true;   // decoder-derived; not stated in the file

    // head_dim is independent of n_embed here (640 hidden, 4 heads of 256),
    // so it must come from the file rather than a division.
    const int key_len   = read_u32_as_int(gguf, "gemma3.attention.key_length");
    const int value_len = read_u32_as_int(gguf, "gemma3.attention.value_length");
    if (key_len != value_len) {
        fail("gemma3 key_length (" + std::to_string(key_len) + ") != value_length (" +
             std::to_string(value_len) + "); split K/V widths are not supported");
    }
    a.head_dim = key_len;

    if (a.n_layer <= 0 || a.n_embed <= 0 || a.n_head <= 0 || a.n_head_kv <= 0 ||
        a.n_ff <= 0 || a.max_seq_len <= 0 || a.head_dim <= 0) {
        fail("gemma3 hyperparameters out of range (got non-positive value)");
    }
    if (!(a.norm_eps > 0.0f) || !std::isfinite(a.norm_eps)) {
        fail("gemma3.attention.layer_norm_rms_epsilon must be finite and positive");
    }
    if (!(a.rope_freq_base > 0.0f) || !std::isfinite(a.rope_freq_base)) {
        fail("gemma3.rope.freq_base must be finite and positive");
    }
    if (a.n_head % a.n_head_kv != 0) {
        fail("query head count (" + std::to_string(a.n_head) +
             ") is not a multiple of KV head count (" + std::to_string(a.n_head_kv) + ")");
    }

    // Vocab size = length of tokenizer.ggml.tokens array.
    {
        const int64_t tk = require_kv(gguf, "tokenizer.ggml.tokens");
        if (gguf_get_kv_type(gguf, tk) != GGUF_TYPE_ARRAY) {
            fail("tokenizer.ggml.tokens is not an array");
        }
        const size_t n_vocab = gguf_get_arr_n(gguf, tk);
        if (n_vocab > static_cast<size_t>(std::numeric_limits<int>::max())) {
            fail("tokenizer vocabulary is too large");
        }
        a.n_vocab = static_cast<int>(n_vocab);
        if (a.n_vocab <= 0) fail("vocab size is non-positive");
    }

    // Scales. The embedding one is a property of the architecture; the
    // attention one is stated per-file because Gemma decouples it from
    // head_dim upstream (query_pre_attn_scalar).
    m.embed_scale = std::sqrt(static_cast<float>(a.n_embed));
    const float qk_scalar =
        read_f32_or(gguf, "gemma3.attention.key_length_scale",
                    static_cast<float>(a.head_dim));
    if (!(qk_scalar > 0.0f) || !std::isfinite(qk_scalar)) {
        fail("gemma3.attention.key_length_scale must be finite and positive");
    }
    m.attn_scale = 1.0f / std::sqrt(qk_scalar);

    m.pooling = pooling_from_gguf(read_u32_as_int(gguf, "gemma3.pooling_type"));

    // Stock Gemma 3 alternates local (sliding-window) and global attention
    // layers. build_block applies full attention to every layer, so a file
    // that declares a window would compute silently wrong results for the
    // local ones. harrier's export states no window — every layer is global —
    // and anything that does is refused rather than approximated.
    const int window = read_u32_or(gguf, "gemma3.attention.sliding_window", 0);
    if (window > 0) {
        fail("gemma3.attention.sliding_window=" + std::to_string(window) +
             " is not supported: this implementation applies full attention to "
             "every layer, which would be wrong for sliding-window layers");
    }

    const int64_t H  = a.n_embed;
    const int64_t F  = a.n_ff;
    const int64_t V  = a.n_vocab;
    const int64_t D  = a.head_dim;
    const int64_t QW = static_cast<int64_t>(a.n_head)    * D;   // packed query width
    const int64_t KW = static_cast<int64_t>(a.n_head_kv) * D;   // packed K/V width

    m.tok_embed = require_tensor(gguf, meta, "token_embd.weight");
    validate_shape_2d(m.tok_embed, H, V, "token_embd.weight");

    m.output_norm = require_tensor(gguf, meta, "output_norm.weight");
    validate_shape_1d(m.output_norm, H, "output_norm.weight");

    m.layers.resize(static_cast<size_t>(a.n_layer));
    for (int li = 0; li < a.n_layer; ++li) {
        Gemma3LayerSlot & L = m.layers[static_cast<size_t>(li)];

        auto req = [&](const char * suffix) {
            const std::string n = layer_tensor_name(li, suffix);
            return std::pair<TensorRef, std::string>{
                require_tensor(gguf, meta, n), n};
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

Gemma3Manifest scan_gemma3(const std::string & gguf_path) {
    ggml_context * meta_raw = nullptr;
    gguf_init_params gp;
    gp.no_alloc = true;
    gp.ctx      = &meta_raw;

    gguf_context * gguf_raw = gguf_init_from_file(gguf_path.c_str(), gp);
    if (gguf_raw == nullptr) {
        fail("failed to open GGUF file: " + gguf_path);
    }
    GgufPtr gguf_owner(gguf_raw);
    GgmlPtr meta_owner(meta_raw);
    return scan_gemma3(gguf_owner.get(), meta_owner.get());
}

Gemma3ModelArch::Gemma3ModelArch(const std::string & gguf_path)
    : manifest_(scan_gemma3(gguf_path)) {}

Gemma3ModelArch::Gemma3ModelArch(Gemma3Manifest manifest)
    : manifest_(std::move(manifest)) {}

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

        w.attn_norm      = T(p + "attn_norm.weight");
        w.attn.q         = T(p + "attn_q.weight");
        w.attn.k         = T(p + "attn_k.weight");
        w.attn.v         = T(p + "attn_v.weight");
        w.attn.o         = T(p + "attn_output.weight");
        w.attn.q_norm    = T(p + "attn_q_norm.weight");
        w.attn.k_norm    = T(p + "attn_k_norm.weight");
        w.attn_post_norm = T(p + "post_attention_norm.weight");

        w.ffn_norm       = T(p + "ffn_norm.weight");
        w.ffn.gate       = T(p + "ffn_gate.weight");
        w.ffn.up         = T(p + "ffn_up.weight");
        w.ffn.down       = T(p + "ffn_down.weight");
        w.ffn_post_norm  = T(p + "post_ffw_norm.weight");
    }
}

ggml_tensor * Gemma3ModelArch::build_embeddings(ggml_context * gctx,
                                                ggml_tensor *  token_ids) const {
    // Token lookup, then the sqrt(n_embed) scale the converter left unfolded.
    // It survives into the residual stream: the first thing each block does is
    // RMSNorm, which would erase it, but the residual branch carries the
    // unnormalized value forward.
    ggml_tensor * x = ggml_get_rows(gctx, tok_embed_, token_ids);
    return ggml_scale(gctx, x, manifest_.embed_scale);
}

forward::GqaAttentionParams Gemma3ModelArch::gqa_params() const noexcept {
    const ArchParams & a = manifest_.params;
    forward::GqaAttentionParams ap;
    ap.n_head         = a.n_head;
    ap.n_head_kv      = a.n_head_kv;
    ap.head_dim       = a.head_dim;
    ap.scale          = manifest_.attn_scale;
    ap.norm_eps       = a.norm_eps;
    ap.rope_freq_base = a.rope_freq_base;
    ap.n_ctx_orig     = a.max_seq_len;
    ap.causal         = a.causal;
    return ap;
}

ggml_tensor * Gemma3ModelArch::build_block(ggml_context * gctx,
                                           ggml_tensor *  x,
                                           ggml_tensor *  pos,
                                           int            layer) const {
    const ArchParams &         a = manifest_.params;
    const Gemma3LayerWeights & w = layer_w_[static_cast<size_t>(layer)];

    const forward::GqaAttentionParams ap = gqa_params();

    // Attention sub-layer. Normalized going in and coming out, with the
    // residual added last — BERT normalizes only after the residual, and
    // Llama only before the sub-layer.
    ggml_tensor * h = forward::build_rms_norm(gctx, x, w.attn_norm, a.norm_eps);
    h = forward::build_gqa_attention(gctx, h, pos, /*kq_mask=*/nullptr, w.attn, ap);
    h = forward::build_rms_norm(gctx, h, w.attn_post_norm, a.norm_eps);
    x = ggml_add(gctx, x, h);

    ggml_tensor * f = forward::build_rms_norm(gctx, x, w.ffn_norm, a.norm_eps);
    f = forward::build_gated_ffn(gctx, f, w.ffn, forward::GateActivation::Gelu);
    f = forward::build_rms_norm(gctx, f, w.ffn_post_norm, a.norm_eps);
    return ggml_add(gctx, x, f);
}

ggml_tensor * Gemma3ModelArch::build_final_norm(ggml_context * gctx,
                                                ggml_tensor *  x) const {
    // BERT has no equivalent; here it is the last thing before pooling.
    return forward::build_rms_norm(gctx, x, output_norm_, manifest_.params.norm_eps);
}

ggml_tensor * Gemma3ModelArch::build_graph(ggml_context *      gctx,
                                           const GraphInputs & in) const {
    ggml_tensor * x = build_embedding_phase(gctx, in);
    for (int li = 0; li < manifest_.params.n_layer; ++li) {
        x = build_block(gctx, x, in.pos_ids, li);
    }
    return build_final_phase(gctx, x);
}

ggml_tensor * Gemma3ModelArch::build_embedding_phase(
    ggml_context * gctx, const GraphInputs & in) const {
    return build_embeddings(gctx, in.token_ids);
}

ggml_tensor * Gemma3ModelArch::build_final_phase(
    ggml_context * gctx, ggml_tensor * x) const {
    return build_final_norm(gctx, x);
}

StreamingCommonPlan Gemma3ModelArch::streaming_common_plan() const {
    StreamingCommonPlan plan;
    plan.token_embedding = "token_embd.weight";
    plan.common = {"output_norm.weight"};
    return plan;
}

StreamingLayerPlan Gemma3ModelArch::streaming_units(int layer) const {
    if (layer < 0 || layer >= manifest_.params.n_layer) {
        throw std::out_of_range("Gemma3 streaming layer index out of range");
    }
    const std::string p = "blk." + std::to_string(layer) + ".";
    const size_t li = static_cast<size_t>(layer);
    const float eps = manifest_.params.norm_eps;

    StreamingLayerPlan plan;
    // Same slot in and out: every layer reads the residual stream and writes it
    // back, so layer N's output is layer N+1's input with nothing to rename.
    // Intermediates inside a layer take distinct names.
    plan.input_slot  = "x";
    plan.output_slot = "x";

    // Declared finest-first, in topological order. A partition strategy merges
    // contiguous runs of this; the norms are units of their own so they can be
    // grouped either way, and the default preset folds them into a neighbour at
    // no runtime cost.
    auto unit = [&](const char * name, std::vector<std::string> weights,
                    std::vector<std::string> inputs, std::vector<std::string> outputs,
                    StreamingStage stage,
                    std::function<void(ggml_context *, SlotTable &)> build) {
        StreamingUnit u;
        u.name    = name;
        u.weights = std::move(weights);
        u.inputs  = std::move(inputs);
        u.outputs = std::move(outputs);
        u.stage   = stage;
        u.build   = std::move(build);
        plan.units.push_back(std::move(u));
    };

    unit("attn_norm", {p + "attn_norm.weight"}, {"x"}, {"xn"}, StreamingStage::Attention,
         [this, li, eps](ggml_context * ctx, SlotTable & s) {
             s.out(0, forward::build_rms_norm(ctx, s.in(0), layer_w_[li].attn_norm, eps));
         });

    unit("attn_qkv", {p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight"},
         {"xn"}, {"q", "k", "v"}, StreamingStage::Attention,
         [this, li](ggml_context * ctx, SlotTable & s) {
             const auto proj = forward::build_gqa_projections(ctx, s.in(0), layer_w_[li].attn);
             s.out(0, proj.q);
             s.out(1, proj.k);
             s.out(2, proj.v);
         });

    // Everything between the projections and the output projection: QK-norm,
    // RoPE, scores, softmax, the value product. The [S, S, n_head] score tensor
    // lives entirely inside this unit and never crosses a boundary.
    StreamingUnit core;
    core.name    = "attn_core";
    core.weights = {p + "attn_q_norm.weight", p + "attn_k_norm.weight",
                    p + "attn_output.weight"};
    core.inputs  = {"q", "k", "v"};
    core.outputs = {"h"};
    core.stage   = StreamingStage::Attention;
    core.graph_inputs = InputRequirements{/*needs_pos_ids=*/true, /*needs_type_ids=*/false};
    core.build = [this, li](ggml_context * ctx, SlotTable & s) {
        forward::GqaProjections proj{s.in(0), s.in(1), s.in(2)};
        s.out(0, forward::build_gqa_attention_core(
                     ctx, proj, s.graph_inputs().pos_ids, /*kq_mask=*/nullptr,
                     layer_w_[li].attn, gqa_params()));
    };
    plan.units.push_back(std::move(core));

    unit("attn_post", {p + "post_attention_norm.weight"}, {"h", "x"}, {"x1"},
         StreamingStage::Attention,
         [this, li, eps](ggml_context * ctx, SlotTable & s) {
             ggml_tensor * h = forward::build_rms_norm(
                 ctx, s.in(0), layer_w_[li].attn_post_norm, eps);
             s.out(0, ggml_add(ctx, s.in(1), h));
         });

    unit("ffn_norm", {p + "ffn_norm.weight"}, {"x1"}, {"fn"}, StreamingStage::Ffn,
         [this, li, eps](ggml_context * ctx, SlotTable & s) {
             s.out(0, forward::build_rms_norm(ctx, s.in(0), layer_w_[li].ffn_norm, eps));
         });

    // gate and up stay together: ggml_geglu_split fuses act(gate) * up, so
    // separating them would need both [F, S] tensors live across a boundary.
    unit("ffn_gate_up", {p + "ffn_gate.weight", p + "ffn_up.weight"},
         {"fn"}, {"fh"}, StreamingStage::Ffn,
         [this, li](ggml_context * ctx, SlotTable & s) {
             s.out(0, forward::build_gated_ffn_gate_up(
                          ctx, s.in(0), layer_w_[li].ffn, forward::GateActivation::Gelu));
         });

    unit("ffn_down", {p + "ffn_down.weight"}, {"fh"}, {"fd"}, StreamingStage::Ffn,
         [this, li](ggml_context * ctx, SlotTable & s) {
             s.out(0, forward::build_gated_ffn_down(ctx, s.in(0), layer_w_[li].ffn));
         });

    unit("ffn_post", {p + "post_ffw_norm.weight"}, {"fd", "x1"}, {"x"},
         StreamingStage::Ffn,
         [this, li, eps](ggml_context * ctx, SlotTable & s) {
             ggml_tensor * f = forward::build_rms_norm(
                 ctx, s.in(0), layer_w_[li].ffn_post_norm, eps);
             s.out(0, ggml_add(ctx, s.in(1), f));
         });

    return plan;
}

} // namespace nanoembed
