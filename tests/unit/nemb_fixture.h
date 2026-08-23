// Shared scaffolding for the per-layer activation parity tests.
//
// Each model family needs its own parity test — the tensor names, the block
// structure and the hyperparameter keys all differ — but they all replay the
// same NEMB fixture format against the same kind of throwaway ggml graph.
// That part lives here so a second family does not fork it and drift.

#pragma once

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nanoembed::test {

// ---- NEMB (multi-tensor binary) parser --------------------------------------
//
// Format documented in tools/dump_hf_activations.py. All numbers are
// little-endian; the host is assumed LE (macOS arm64 + Linux x86_64).

struct NembTensor {
    std::string          name;
    int                  dtype;  // 0=F32, 1=I32, 2=I64
    std::vector<int64_t> shape;  // numpy order (outermost first)
    std::vector<uint8_t> data;

    const float * f32() const { return reinterpret_cast<const float *>(data.data()); }

    size_t elements() const {
        size_t n = 1;
        for (int64_t d : shape) n *= static_cast<size_t>(d);
        return n;
    }
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

    bool has(const std::string & name) const {
        for (const auto & t : tensors_) if (t.name == name) return true;
        return false;
    }

private:
    std::vector<NembTensor> tensors_;
};

// ---- ggml RAII helpers ------------------------------------------------------

// Owns (gguf_context, ggml_context) loaded from a model file with weights
// resident.
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

    ggml_tensor * tensor(const std::string & name) const {
        ggml_tensor * t = ggml_get_tensor(model_ctx, name.c_str());
        if (t == nullptr) throw std::runtime_error("missing tensor: " + name);
        return t;
    }
};

// Transient ggml_context backed by a plain buffer, sized for a single forward
// pass over one block at small S. Generous on purpose: it also backs
// ggml_graph_compute's internal work buffer.
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

    // Build the forward graph rooted at `out` and run it. Throws on any ggml
    // status other than SUCCESS so silent garbage never reaches a comparison.
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

inline DiffStats compute_diff(const float * got, const float * exp, size_t n) {
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

} // namespace nanoembed::test
