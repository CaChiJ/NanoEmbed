// Layer-isolation parity tests for src/forward/* graph builders.
//
// Loads bge-small-en-v1.5 with all weights resident, replays each
// HuggingFace-captured activation fixture (tests/fixtures/activations/*.nemb),
// and compares our builder output against the reference.
//
// Skipped if NANOEMBED_TEST_MODEL or NANOEMBED_ACTIVATION_FIXTURES is unset.

#include "forward/embed_layer.h"
#include "forward/encoder_block.h"
#include "gguf_scanner.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// ---- NEMB (multi-tensor binary) parser --------------------------------------
//
// Format documented in tools/dump_hf_activations.py. All numbers are
// little-endian; the host is assumed LE (macOS arm64 + Linux x86_64).

struct NembTensor {
    std::string          name;
    int                  dtype;  // 0=F32, 1=I32, 2=I64
    std::vector<int64_t> shape;  // numpy order (outermost first)
    std::vector<uint8_t> data;
};

class NembFile {
public:
    explicit NembFile(const std::string & path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open NEMB: " + path);

        auto must_read = [&](void * dst, std::streamsize n) {
            f.read(static_cast<char *>(dst), n);
            if (f.gcount() != n) {
                throw std::runtime_error("NEMB truncated or unreadable: " + path);
            }
        };

        char magic[4];
        must_read(magic, 4);
        if (std::string(magic, 4) != "NEMB") throw std::runtime_error("bad NEMB magic");

        uint32_t version, n_tensors;
        must_read(&version,  4);
        must_read(&n_tensors, 4);
        if (version != 1) throw std::runtime_error("unsupported NEMB version");

        for (uint32_t i = 0; i < n_tensors; ++i) {
            NembTensor t;
            uint32_t name_len, dtype, n_dims;
            must_read(&name_len, 4);
            t.name.resize(name_len);
            must_read(t.name.data(), name_len);
            must_read(&dtype, 4);
            t.dtype = static_cast<int>(dtype);
            must_read(&n_dims, 4);
            t.shape.resize(n_dims);
            must_read(t.shape.data(), static_cast<std::streamsize>(n_dims * sizeof(int64_t)));

            const size_t elem = (dtype == 2) ? 8 : 4;  // F32/I32 = 4, I64 = 8
            size_t total = elem;
            for (int64_t d : t.shape) total *= static_cast<size_t>(d);
            t.data.resize(total);
            must_read(t.data.data(), static_cast<std::streamsize>(total));

            tensors_.push_back(std::move(t));
        }
    }

    const NembTensor & require(const std::string & name) const {
        for (const auto & t : tensors_) if (t.name == name) return t;
        throw std::runtime_error("NEMB missing tensor: " + name);
    }

private:
    std::vector<NembTensor> tensors_;
};

// ---- ggml RAII helpers ------------------------------------------------------

// Owns (gguf_context, ggml_context) loaded from a model file with weights
// resident. Used by every forward parity test.
struct ModelHarness {
    gguf_context * gguf      = nullptr;
    ggml_context * model_ctx = nullptr;

    explicit ModelHarness(const std::string & path) {
        gguf_init_params p;
        p.no_alloc = false;
        p.ctx      = &model_ctx;
        gguf = gguf_init_from_file(path.c_str(), p);
        if (!gguf || !model_ctx) {
            if (gguf) gguf_free(gguf);
            if (model_ctx) ggml_free(model_ctx);
            throw std::runtime_error("failed to load model: " + path);
        }
    }
    ~ModelHarness() {
        if (model_ctx) ggml_free(model_ctx);
        if (gguf)      gguf_free(gguf);
    }

    ModelHarness(const ModelHarness &)             = delete;
    ModelHarness & operator=(const ModelHarness &) = delete;
};

// Transient ggml_context backed by a stack-friendly buffer. Sized for
// "single forward pass of one BERT block at S<=64": 128 MB is comfortably
// over the largest scratch we observed (~30 MB for layer 0 at S=12), and
// doubles as headroom for ggml_graph_compute's internal work buffer.
class GraphCtx {
public:
    explicit GraphCtx(size_t mem_size) : buf_(mem_size) {
        ggml_init_params gip;
        gip.mem_size   = buf_.size();
        gip.mem_buffer = buf_.data();
        gip.no_alloc   = false;
        ctx_ = ggml_init(gip);
        if (!ctx_) throw std::runtime_error("ggml_init failed");
    }
    ~GraphCtx() { if (ctx_) ggml_free(ctx_); }

    GraphCtx(const GraphCtx &)             = delete;
    GraphCtx & operator=(const GraphCtx &) = delete;

    ggml_context * get() const { return ctx_; }

    // Build the forward graph rooted at `out` and run it. Throws on any
    // ggml status other than SUCCESS so silent garbage never reaches a
    // comparison.
    void compute(ggml_tensor * out, int n_threads = 1) const {
        ggml_cgraph * graph = ggml_new_graph(ctx_);
        ggml_build_forward_expand(graph, out);
        const ggml_status st = ggml_graph_compute_with_ctx(ctx_, graph, n_threads);
        if (st != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_graph_compute failed (status="
                                     + std::to_string(static_cast<int>(st)) + ")");
        }
    }

private:
    std::vector<uint8_t> buf_;
    ggml_context *       ctx_ = nullptr;
};

constexpr size_t kGraphMemSize = 128ull * 1024 * 1024;

// ---- Comparison helpers -----------------------------------------------------

struct DiffStats {
    float  max_abs  = 0.0f;
    float  mean_abs = 0.0f;
    size_t n        = 0;
};

DiffStats compute_diff(const float * got, const float * exp, size_t n) {
    DiffStats s;
    s.n = n;
    double accum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const float d = std::fabs(got[i] - exp[i]);
        if (d > s.max_abs) s.max_abs = d;
        accum += d;
    }
    s.mean_abs = static_cast<float>(accum / static_cast<double>(n));
    return s;
}

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
