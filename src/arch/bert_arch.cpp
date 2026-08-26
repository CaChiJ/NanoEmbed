#include "arch/bert_arch.h"

#include "ggml.h"

#include <stdexcept>
#include <utility>

namespace nanoembed {

BertModelArch::BertModelArch(const std::string & gguf_path)
    : BertModelArch(scan_gguf(gguf_path).manifest()) {}

BertModelArch::BertModelArch(ModelManifest manifest)
    : manifest_(std::move(manifest)) {
    // The scanner validates the architecture tag, hyperparameters and every
    // required tensor shape, so anything that reaches bind_weights is known
    // to be a well-formed BERT.
    const auto & a = manifest_.arch;
    params_.name        = "bert";
    params_.n_layer     = a.n_layer;
    params_.n_embed     = a.n_embed;
    params_.n_head      = a.n_head;
    params_.n_ff        = a.n_ff;
    params_.n_vocab     = a.n_vocab;
    params_.max_seq_len = a.max_seq_len;
    params_.norm_eps    = a.layer_norm_eps;

    // BERT ties head geometry to the hidden size and uses one KV head per
    // query head; the scanner has already rejected files where n_embed is not
    // divisible by n_head. Learned position embeddings, not RoPE, and the
    // encoder attends in both directions.
    params_.head_dim       = a.n_embed / a.n_head;
    params_.n_head_kv      = a.n_head;
    params_.rope_freq_base = 0.0f;
    params_.causal         = false;

    switch (a.pooling_type) {
        case 1:  default_pooling_ = PoolType::Mean; break;
        case 2:  default_pooling_ = PoolType::Cls;  break;
        case 3:  default_pooling_ = PoolType::Last; break;
    }
}

void BertModelArch::bind_weights(ggml_context * model_ctx) {
    auto T = [&](const std::string & name) -> ggml_tensor * {
        ggml_tensor * t = ggml_get_tensor(model_ctx, name.c_str());
        if (t == nullptr) {
            throw std::runtime_error("expected tensor missing in model_ctx: " + name);
        }
        return t;
    };

    embed_w_.tok    = T("token_embd.weight");
    embed_w_.pos    = T("position_embd.weight");
    embed_w_.type   = T("token_types.weight");
    embed_w_.norm_w = T("token_embd_norm.weight");
    embed_w_.norm_b = T("token_embd_norm.bias");

    layer_w_.resize(static_cast<size_t>(params_.n_layer));
    for (int li = 0; li < params_.n_layer; ++li) {
        forward::LayerWeights & lw = layer_w_[static_cast<size_t>(li)];
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

ggml_tensor * BertModelArch::build_graph(ggml_context *      gctx,
                                         const GraphInputs & in) const {
    ggml_tensor * x = build_embedding_phase(gctx, in);

    for (int li = 0; li < params_.n_layer; ++li) {
        x = forward::build_encoder_block(
            gctx, x, /*kq_mask=*/nullptr, params_.n_head,
            layer_w_[static_cast<size_t>(li)], params_.norm_eps);
    }
    return build_final_phase(gctx, x);
}

ggml_tensor * BertModelArch::build_embedding_phase(
    ggml_context * gctx, const GraphInputs & in) const {
    return forward::build_embed_layer(
        gctx, in.token_ids, in.pos_ids, in.type_ids, embed_w_, params_.norm_eps);
}

ggml_tensor * BertModelArch::build_final_phase(ggml_context *, ggml_tensor * x) const {
    return x;
}

StreamingCommonPlan BertModelArch::streaming_common_plan() const {
    StreamingCommonPlan plan;
    plan.token_embedding = "token_embd.weight";
    plan.common = {
        "position_embd.weight", "token_types.weight",
        "token_embd_norm.weight", "token_embd_norm.bias",
    };
    return plan;
}

StreamingLayerPlan BertModelArch::streaming_units(int layer) const {
    if (layer < 0 || layer >= params_.n_layer) {
        throw std::out_of_range("BERT streaming layer index out of range");
    }
    const std::string p = "blk." + std::to_string(layer) + ".";
    const size_t li = static_cast<size_t>(layer);
    const float eps = params_.norm_eps;
    const int n_head = params_.n_head;

    StreamingLayerPlan plan;
    // Same slot in and out: every layer reads the residual stream and writes it
    // back, so layer N's output is layer N+1's input with nothing to rename.
    plan.input_slot  = "x";
    plan.output_slot = "x";

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

    // BERT is post-LN: each sub-layer adds the residual and normalizes at its
    // end, so the normalization gains belong to the unit that closes the
    // sub-layer rather than to one of their own.
    //
    // No unit declares a graph input: BERT's learned positions are consumed
    // once by the embedding phase, not by every block.
    unit("attn_qkv",
         {p + "attn_q.weight", p + "attn_q.bias",
          p + "attn_k.weight", p + "attn_k.bias",
          p + "attn_v.weight", p + "attn_v.bias"},
         {"x"}, {"q", "k", "v"}, StreamingStage::Attention,
         [this, li](ggml_context * ctx, SlotTable & s) {
             const auto proj = forward::build_attention_projections(
                 ctx, s.in(0), layer_w_[li].attn);
             s.out(0, proj.q);
             s.out(1, proj.k);
             s.out(2, proj.v);
         });

    unit("attn_out",
         {p + "attn_output.weight", p + "attn_output.bias",
          p + "attn_output_norm.weight", p + "attn_output_norm.bias"},
         {"q", "k", "v", "x"}, {"xa"}, StreamingStage::Attention,
         [this, li, n_head, eps](ggml_context * ctx, SlotTable & s) {
             forward::AttentionProjections proj{s.in(0), s.in(1), s.in(2)};
             s.out(0, forward::build_attention_output(
                          ctx, proj, s.in(3), /*kq_mask=*/nullptr, n_head,
                          layer_w_[li].attn, eps));
         });

    unit("ffn_up", {p + "ffn_up.weight", p + "ffn_up.bias"},
         {"xa"}, {"fh"}, StreamingStage::Ffn,
         [this, li](ggml_context * ctx, SlotTable & s) {
             s.out(0, forward::build_ffn_up(ctx, s.in(0), layer_w_[li].ffn));
         });

    unit("ffn_down",
         {p + "ffn_down.weight", p + "ffn_down.bias",
          p + "layer_output_norm.weight", p + "layer_output_norm.bias"},
         {"fh", "xa"}, {"x"}, StreamingStage::Ffn,
         [this, li, eps](ggml_context * ctx, SlotTable & s) {
             s.out(0, forward::build_ffn_down(
                          ctx, s.in(0), s.in(1), layer_w_[li].ffn, eps));
         });

    return plan;
}

} // namespace nanoembed
