#include "arch/bert_arch.h"

#include "ggml.h"

#include <stdexcept>

namespace nanoembed {

BertModelArch::BertModelArch(const std::string & gguf_path) {
    // The scanner validates the architecture tag, hyperparameters and every
    // required tensor shape, so anything that reaches bind_weights is known
    // to be a well-formed BERT.
    ScanResult scan = scan_gguf(gguf_path);
    manifest_ = scan.manifest();

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
    ggml_tensor * x = forward::build_embed_layer(
        gctx, in.token_ids, in.pos_ids, in.type_ids, embed_w_, params_.norm_eps);

    for (int li = 0; li < params_.n_layer; ++li) {
        x = forward::build_encoder_block(
            gctx, x, /*kq_mask=*/nullptr, params_.n_head,
            layer_w_[static_cast<size_t>(li)], params_.norm_eps);
    }
    return x;
}

} // namespace nanoembed
