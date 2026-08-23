// Layer-isolation parity tests for the gemma3 forward path.
//
// Loads harrier-oss-v1-270m with weights resident and replays each
// HuggingFace-captured fixture stage by stage. Every stage is fed the
// reference's own input rather than our previous stage's output, so an error
// in block 7 shows up as block 7 and does not smear across the ones after it.
//
// The stages come from Gemma3ModelArch itself — build_embeddings / build_block
// / build_final_norm are exactly what build_graph composes — so this exercises
// the real block wiring (the two-sided RMSNorm sandwich and where the residual
// joins), not a reimplementation of it that could agree with itself and still
// be wrong.
//
// Skipped if NANOEMBED_TEST_MODEL_GEMMA3 or
// NANOEMBED_ACTIVATION_FIXTURES_GEMMA3 is unset.

#include "nemb_fixture.h"

#include "arch/gemma3_arch.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace nanoembed::test;

namespace {

int g_failures = 0;

#define EXPECT_TRUE(cond)                                                              \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            std::fprintf(stderr, "FAIL: %s (%s:%d)\n", #cond, __FILE__, __LINE__);     \
            ++g_failures;                                                              \
        }                                                                              \
    } while (0)

// Relative, not absolute: gemma3 activations run far larger than BERT's. The
// embedding scale alone is sqrt(640) ~= 25, and some RMSNorm gains exceed 200,
// so a fixed epsilon would be vacuous at one stage and impossible at another.
constexpr float kTolRel = 1e-4f;

float max_abs(const float * v, size_t n) {
    float m = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        const float a = std::fabs(v[i]);
        if (a > m) m = a;
    }
    return m;
}

// numpy [B, S, H] -> ggml ne = [H, S, B]. Both are H-fastest in memory, so the
// bytes transfer directly.
ggml_tensor * activation_input(ggml_context * c, const NembTensor & t) {
    const int64_t B = t.shape[0];
    const int64_t S = t.shape[1];
    const int64_t H = t.shape[2];
    ggml_tensor * x = ggml_new_tensor_3d(c, GGML_TYPE_F32, H, S, B);
    std::memcpy(x->data, t.data.data(), t.data.size());
    return x;
}

// Compare a computed tensor against its reference. Returns true on match and
// prints the measured error either way, so a tightening or loosening of
// kTolRel can be argued from the log rather than guessed.
bool check(const char * stage, int sample, ggml_tensor * got, const NembTensor & exp) {
    const size_t n = exp.elements();
    if (static_cast<size_t>(ggml_nelements(got)) != n) {
        std::fprintf(stderr, "FAIL sample[%d] %s: %lld elements vs %zu expected\n",
                     sample, stage, static_cast<long long>(ggml_nelements(got)), n);
        ++g_failures;
        return false;
    }

    const float * g = static_cast<const float *>(got->data);
    const float * e = exp.f32();
    const DiffStats d     = compute_diff(g, e, n);
    const float     scale = std::max(max_abs(e, n), 1e-6f);
    const float     rel   = d.max_abs / scale;

    const bool ok = rel <= kTolRel;
    std::fprintf(stderr, "[gemma3_forward] sample[%d] %-16s max_abs=%.3e rel=%.3e %s\n",
                 sample, stage, d.max_abs, rel, ok ? "ok" : "FAIL");
    if (!ok) ++g_failures;
    return ok;
}

bool run() {
    const char * model_path = std::getenv("NANOEMBED_TEST_MODEL_GEMMA3");
    const char * fix_dir    = std::getenv("NANOEMBED_ACTIVATION_FIXTURES_GEMMA3");
    if (!model_path || !fix_dir) {
        std::fprintf(stderr,
            "[gemma3_forward] skip: NANOEMBED_TEST_MODEL_GEMMA3 / "
            "NANOEMBED_ACTIVATION_FIXTURES_GEMMA3 not set\n");
        return true;
    }

    nanoembed::Gemma3ModelArch arch(model_path);
    ModelHarness               mh(model_path);
    arch.bind_weights(mh.model_ctx);

    const int n_layer = arch.params().n_layer;

    for (int sample = 0; sample < 3; ++sample) {
        char path[512];
        std::snprintf(path, sizeof(path), "%s/sample_%02d.nemb", fix_dir, sample);
        NembFile fix(path);

        const NembTensor & ids_t   = fix.require("input_ids");
        const NembTensor & embed_t = fix.require("embed_out");
        EXPECT_TRUE(ids_t.shape.size() == 2);      // [B, S]
        EXPECT_TRUE(embed_t.shape.size() == 3);    // [B, S, H]

        const int64_t B = ids_t.shape[0];
        const int64_t S = ids_t.shape[1];
        EXPECT_TRUE(B == 1);

        GraphCtx gctx(kGraphMemSize);

        // Position ramp, shared by every block's RoPE.
        ggml_tensor * pos = ggml_new_tensor_2d(gctx.get(), GGML_TYPE_I32, S, B);
        {
            std::vector<int32_t> ramp(static_cast<size_t>(S * B));
            for (int64_t b = 0; b < B; ++b) {
                for (int64_t s = 0; s < S; ++s) {
                    ramp[static_cast<size_t>(b * S + s)] = static_cast<int32_t>(s);
                }
            }
            std::memcpy(pos->data, ramp.data(), ramp.size() * sizeof(int32_t));
        }

        // Stage 1: token lookup + the unfolded sqrt(n_embed) scale. If the
        // scale were already folded into token_embd this would come out ~25x
        // too large, which is the cheapest place to notice.
        {
            ggml_tensor * token_ids = ggml_new_tensor_2d(gctx.get(), GGML_TYPE_I32, S, B);
            std::memcpy(token_ids->data, ids_t.data.data(), ids_t.data.size());

            ggml_tensor * out = arch.build_embeddings(gctx.get(), token_ids);
            gctx.compute(out);
            check("embeddings", sample, out, embed_t);
        }

        // Stage 2: each block, fed the reference's input for that block.
        for (int li = 0; li < n_layer; ++li) {
            const NembTensor & in_t =
                (li == 0) ? embed_t : fix.require("layer_" + std::to_string(li - 1) + "_out");
            const NembTensor & out_t = fix.require("layer_" + std::to_string(li) + "_out");

            ggml_tensor * x   = activation_input(gctx.get(), in_t);
            ggml_tensor * out = arch.build_block(gctx.get(), x, pos, li);
            gctx.compute(out);

            char stage[32];
            std::snprintf(stage, sizeof(stage), "block[%d]", li);
            check(stage, sample, out, out_t);
        }

        // Stage 3: the post-stack norm, which BERT has no equivalent of and
        // which is the last thing to run before pooling.
        {
            const NembTensor & in_t  = fix.require("layer_" + std::to_string(n_layer - 1) + "_out");
            const NembTensor & out_t = fix.require("final_norm_out");

            ggml_tensor * x   = activation_input(gctx.get(), in_t);
            ggml_tensor * out = arch.build_final_norm(gctx.get(), x);
            gctx.compute(out);
            check("final_norm", sample, out, out_t);
        }
    }

    return g_failures == 0;
}

} // namespace

int main() {
    const bool ok = run();
    std::printf("gemma3_forward_test: %s\n", ok ? "ok" : "FAIL");
    return ok ? 0 : 1;
}
