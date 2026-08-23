// Layer-isolation parity tests for src/forward/* graph builders.
//
// Loads bge-small-en-v1.5 with all weights resident, replays each
// HuggingFace-captured activation fixture (tests/fixtures/activations/*.nemb),
// and compares our builder output against the reference.
//
// Skipped if NANOEMBED_TEST_MODEL or NANOEMBED_ACTIVATION_FIXTURES is unset.

#include "nemb_fixture.h"

#include "forward/embed_layer.h"
#include "forward/encoder_block.h"
#include "gguf_scanner.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace nanoembed::test;

// ---- Test framework helpers -------------------------------------------------

int g_failures = 0;

#define EXPECT_TRUE(cond)                                                              \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::fprintf(stderr, "FAIL: %s (%s:%d)\n", #cond, __FILE__, __LINE__);     \
            ++g_failures;                                                              \
        }                                                                              \
    } while (0)

bool fixtures_available(const char *& model_path, const char *& fix_dir) {
    model_path = std::getenv("NANOEMBED_TEST_MODEL");
    fix_dir    = std::getenv("NANOEMBED_ACTIVATION_FIXTURES");
    return model_path && fix_dir;
}

// ---- Embed-layer parity test ------------------------------------------------

bool test_embed_layer() {
    const char * model_path = nullptr;
    const char * fix_dir    = nullptr;
    if (!fixtures_available(model_path, fix_dir)) {
        std::fprintf(stderr, "[forward_test] skip embed_layer: env not set\n");
        return true;
    }

    ModelHarness mh(model_path);

    const int64_t eps_kv = gguf_find_key(mh.gguf, "bert.attention.layer_norm_epsilon");
    EXPECT_TRUE(eps_kv >= 0);
    const float eps = gguf_get_val_f32(mh.gguf, eps_kv);

    nanoembed::forward::EmbedWeights ew{};
    ew.tok    = ggml_get_tensor(mh.model_ctx, "token_embd.weight");
    ew.pos    = ggml_get_tensor(mh.model_ctx, "position_embd.weight");
    ew.type   = ggml_get_tensor(mh.model_ctx, "token_types.weight");
    ew.norm_w = ggml_get_tensor(mh.model_ctx, "token_embd_norm.weight");
    ew.norm_b = ggml_get_tensor(mh.model_ctx, "token_embd_norm.bias");
    EXPECT_TRUE(ew.tok && ew.pos && ew.type && ew.norm_w && ew.norm_b);

    int n_pass = 0;
    int n_fail = 0;
    constexpr float TOL = 5e-3f;  // F16 weights + LN reduction noise

    for (int idx = 0; idx < 3; ++idx) {
        char path[512];
        std::snprintf(path, sizeof(path), "%s/sample_%02d.nemb", fix_dir, idx);
        NembFile fix(path);

        const auto & ids_t       = fix.require("input_ids");
        const auto & embed_out_t = fix.require("embed_out");
        EXPECT_TRUE(ids_t.shape.size()       == 2);  // [B, S]
        EXPECT_TRUE(embed_out_t.shape.size() == 3);  // [B, S, H]

        const int64_t B = ids_t.shape[0];
        const int64_t S = ids_t.shape[1];
        const int64_t H = embed_out_t.shape[2];
        EXPECT_TRUE(B == 1);  // M3 baseline only handles single-input
        EXPECT_TRUE(B == embed_out_t.shape[0] && S == embed_out_t.shape[1]);

        GraphCtx gctx(kGraphMemSize);

        // ggml ne convention reverses numpy: numpy [B, S] → ggml ne=[S, B].
        ggml_tensor * token_ids = ggml_new_tensor_2d(gctx.get(), GGML_TYPE_I32, S, B);
        ggml_tensor * pos_ids   = ggml_new_tensor_2d(gctx.get(), GGML_TYPE_I32, S, B);
        ggml_tensor * type_ids  = ggml_new_tensor_2d(gctx.get(), GGML_TYPE_I32, S, B);

        // input_ids comes from fixture as int32 in numpy [B, S] layout; with
        // B=1 the layout is identical to ggml [S, B] so a direct memcpy works.
        std::memcpy(token_ids->data, ids_t.data.data(), ids_t.data.size());

        // Position IDs: 0..S-1 per row.
        std::vector<int32_t> pos(static_cast<size_t>(S * B));
        for (int64_t b = 0; b < B; ++b) {
            for (int64_t s = 0; s < S; ++s) {
                pos[static_cast<size_t>(b * S + s)] = static_cast<int32_t>(s);
            }
        }
        std::memcpy(pos_ids->data, pos.data(), pos.size() * sizeof(int32_t));

        // Token-type IDs: 0 (single segment).
        std::memset(type_ids->data, 0, static_cast<size_t>(S * B) * sizeof(int32_t));

        ggml_tensor * out = nanoembed::forward::build_embed_layer(
            gctx.get(), token_ids, pos_ids, type_ids, ew, eps);
        gctx.compute(out);

        EXPECT_TRUE(out->ne[0] == H);
        EXPECT_TRUE(out->ne[1] == S);
        EXPECT_TRUE(out->ne[2] == B);

        // Both tensors are H-fastest in memory ([B,S,H] numpy and [H,S,B] ggml).
        const float * got = static_cast<const float *>(out->data);
        const float * exp = reinterpret_cast<const float *>(embed_out_t.data.data());
        const auto    d   = compute_diff(got, exp, static_cast<size_t>(B * S * H));

        std::fprintf(stderr,
            "[forward_test] sample[%d] embed_layer S=%lld max_abs=%g mean_abs=%g\n",
            idx, static_cast<long long>(S), d.max_abs, d.mean_abs);

        if (d.max_abs > TOL) ++n_fail; else ++n_pass;
    }

    EXPECT_TRUE(n_fail == 0);
    std::printf("[forward_test] embed_layer: %d/%d samples within tol=%g\n",
                n_pass, n_pass + n_fail, TOL);

    return g_failures == 0;
}

// ---- Per-layer encoder-block parity test ------------------------------------
//
// Each block is fed the *HuggingFace* activation for its input (not our own
// previous-layer output) so per-layer drift stays isolated. This is the core
// debugging tool from PLAN.md §9.

bool test_encoder_blocks() {
    const char * model_path = nullptr;
    const char * fix_dir    = nullptr;
    if (!fixtures_available(model_path, fix_dir)) {
        std::fprintf(stderr, "[forward_test] skip encoder_blocks: env not set\n");
        return true;
    }

    ModelHarness mh(model_path);

    nanoembed::ScanResult scan = nanoembed::scan_gguf(model_path);
    const auto & arch = scan.manifest().arch;
    const int    n_layer = arch.n_layer;
    const int    n_head  = arch.n_head;
    const int    H       = arch.n_embed;
    const float  eps     = arch.layer_norm_eps;

    auto load_layer = [&](int li) {
        nanoembed::forward::LayerWeights lw{};
        auto T = [&](const std::string & suffix) -> ggml_tensor * {
            const std::string full = "blk." + std::to_string(li) + "." + suffix;
            return ggml_get_tensor(mh.model_ctx, full.c_str());
        };
        lw.attn.q_w    = T("attn_q.weight");        lw.attn.q_b    = T("attn_q.bias");
        lw.attn.k_w    = T("attn_k.weight");        lw.attn.k_b    = T("attn_k.bias");
        lw.attn.v_w    = T("attn_v.weight");        lw.attn.v_b    = T("attn_v.bias");
        lw.attn.o_w    = T("attn_output.weight");   lw.attn.o_b    = T("attn_output.bias");
        lw.attn.norm_w = T("attn_output_norm.weight");
        lw.attn.norm_b = T("attn_output_norm.bias");
        lw.ffn.up_w    = T("ffn_up.weight");        lw.ffn.up_b    = T("ffn_up.bias");
        lw.ffn.down_w  = T("ffn_down.weight");      lw.ffn.down_b  = T("ffn_down.bias");
        lw.ffn.norm_w  = T("layer_output_norm.weight");
        lw.ffn.norm_b  = T("layer_output_norm.bias");
        return lw;
    };

    int   n_pass = 0;
    int   n_fail = 0;
    float worst_max = 0.0f;
    constexpr float TOL = 5e-3f;

    for (int sample_idx = 0; sample_idx < 3; ++sample_idx) {
        char path[512];
        std::snprintf(path, sizeof(path), "%s/sample_%02d.nemb", fix_dir, sample_idx);
        NembFile fix(path);

        const auto & embed_out_t = fix.require("embed_out");
        EXPECT_TRUE(embed_out_t.shape.size() == 3);
        const int64_t B = embed_out_t.shape[0];
        const int64_t S = embed_out_t.shape[1];
        EXPECT_TRUE(B == 1);
        EXPECT_TRUE(static_cast<int64_t>(embed_out_t.shape[2]) == H);

        for (int li = 0; li < n_layer; ++li) {
            const NembTensor & in_t  = (li == 0)
                ? embed_out_t
                : fix.require("layer_" + std::to_string(li - 1) + "_out");
            const NembTensor & exp_t = fix.require("layer_" + std::to_string(li) + "_out");

            GraphCtx gctx(kGraphMemSize);

            ggml_tensor * x = ggml_new_tensor_3d(gctx.get(), GGML_TYPE_F32, H, S, B);
            std::memcpy(x->data, in_t.data.data(), in_t.data.size());

            const auto lw = load_layer(li);
            ggml_tensor * out = nanoembed::forward::build_encoder_block(
                gctx.get(), x, /*kq_mask=*/nullptr, n_head, lw, eps);
            gctx.compute(out);

            EXPECT_TRUE(out->ne[0] == H && out->ne[1] == S && out->ne[2] == B);

            const float * got = static_cast<const float *>(out->data);
            const float * exp = reinterpret_cast<const float *>(exp_t.data.data());
            const auto    d   = compute_diff(got, exp, static_cast<size_t>(B * S * H));

            if (d.max_abs > worst_max) worst_max = d.max_abs;
            if (d.max_abs > TOL) {
                std::fprintf(stderr,
                    "[forward_test] sample[%d] layer[%2d] FAIL max_abs=%g mean_abs=%g (tol=%g)\n",
                    sample_idx, li, d.max_abs, d.mean_abs, TOL);
                ++n_fail;
            } else {
                ++n_pass;
            }
        }
    }

    EXPECT_TRUE(n_fail == 0);
    std::printf("[forward_test] encoder_blocks: %d/%d layer-isolated checks within tol=%g (worst max_abs=%g)\n",
                n_pass, n_pass + n_fail, TOL, worst_max);

    return g_failures == 0;
}

} // namespace

int main() {
    int rc = 0;
    g_failures = 0; if (!test_embed_layer())     rc = 1;
    g_failures = 0; if (!test_encoder_blocks())  rc = 1;
    std::printf("forward_test: %s\n", rc == 0 ? "ok" : "FAIL");
    return rc;
}
