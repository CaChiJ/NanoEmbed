#include "embedder.h"

#include "arch/model_arch.h"
#include "batch.h"
#include "forward/pool.h"
#include "tokenizer/tokenizer.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <vector>

namespace nanoembed {

namespace {

forward::PoolType to_forward_pool(PoolType p) {
    switch (p) {
        case PoolType::Cls:  return forward::PoolType::Cls;
        case PoolType::Last: return forward::PoolType::Last;
        default:             return forward::PoolType::Mean;
    }
}

// Upper bound on tensors in one forward graph. A 12-block encoder comes to
// roughly 500 nodes (BERT) and a SwiGLU/rotary one somewhat more, so this is
// several times the real requirement. Not free, though: the buffer is
// zero-initialized and therefore fully resident, at 368 B/tensor -- 4096 costs
// 1.52 MiB and doubling it showed up as a 1.5 MiB rise in peak RSS.
constexpr size_t kMaxGraphTensors = 4096;

// Upper bound on the auto-selected thread count.
//
// ggml synchronizes its CPU workers at a barrier per graph node, and these
// graphs have many small nodes, so past a point each added thread costs more
// in barrier waiting than it contributes. Measured on an 8-core M-series Mac
// (6 performance cores), ms per embed:
//
//                       1       2       4       6       8
//   bge, 12 tokens     9.4     6.0     4.9    10.9   688.2
//   bge, 460 tokens  1033.6   976.1   888.9  1597.6  3917.7
//   gemma3, 12 tokens 236.1   179.7   272.4   394.3  4893.9
//
// Four is the best or near-best column in every row, including the one with
// plenty of work to divide, and the cliff past the performance-core count is
// severe — a worker scheduled onto an efficiency core makes every barrier wait
// for it. Batched execution (M5) changes the work-per-node profile completely,
// so this should be re-measured then rather than assumed to still hold.
constexpr int kMaxAutoThreads = 4;

// Count of cores worth scheduling on. On a hybrid CPU that means the
// performance cores only, for the barrier reason above.
int usable_cores() {
#if defined(__APPLE__)
    int    n   = 0;
    size_t len = sizeof(n);
    if (sysctlbyname("hw.perflevel0.logicalcpu", &n, &len, nullptr, 0) == 0 && n > 0) {
        return n;
    }
#endif
    const unsigned hw = std::thread::hardware_concurrency();
    return hw == 0 ? 1 : static_cast<int>(hw);
}

int resolve_n_threads(int requested) {
    if (requested > 0) return requested;
    return std::min(usable_cores(), kMaxAutoThreads);
}

// Everything the caller needs from a freshly built graph. The input tensors
// have no backing store until ggml_gallocr_alloc_graph runs, so they are
// handed back to be filled afterwards rather than written at build time.
struct GraphIO {
    ggml_cgraph * graph = nullptr;
    GraphInputs   in;
    forward::PoolInputs pool;
    ggml_tensor * out   = nullptr;
    bool          packed = false;
};

} // namespace

struct Embedder::Impl {
    std::unique_ptr<ModelArch> arch;
    std::unique_ptr<Tokenizer> tokenizer;

    gguf_context * gguf      = nullptr;
    ggml_context * model_ctx = nullptr;

    GraphIO build_graph(ggml_context *  gctx,
                        int64_t         S,
                        int64_t         B,
                        bool            padded,
                        PoolType        pooling,
                        bool            normalize,
                        const MaterializedBatch * batch = nullptr) const {
        const InputRequirements req = arch->inputs();
        GraphIO io;

        // Packed mode drops the padded cells from every token-wise operator.
        // It needs the architecture to opt in, because attention and pooling
        // must then slice by sentence offset rather than by batch stride.
        const bool packed = padded && batch != nullptr &&
                            req.consumes_seq_lengths && req.consumes_packed_batch;
        if (packed) {
            const int64_t T = batch->total_tokens;
            io.in.token_ids = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, T, 1);
            ggml_set_input(io.in.token_ids);
            if (req.needs_rope_pos_ids) {
                io.in.rope_pos_ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, T);
                ggml_set_input(io.in.rope_pos_ids);
            }
            io.in.seq_lengths = batch->lengths.data();
            io.in.seq_offsets = batch->offsets.data();
            io.in.n_seq       = batch->batch_size;
            io.packed         = true;

            ggml_tensor * x = arch->build_graph(gctx, io.in);
            io.out = forward::build_packed_pool(
                gctx, x, to_forward_pool(pooling),
                batch->offsets.data(), batch->lengths.data(), batch->batch_size);
            if (normalize) io.out = forward::build_l2_normalize(gctx, io.out);
            ggml_set_output(io.out);
            io.graph = ggml_new_graph(gctx);
            ggml_build_forward_expand(io.graph, io.out);
            return io;
        }

        auto make_input = [&](ggml_tensor ** slot) {
            *slot = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, S, B);
            // Written after allocation, so they must not share storage with
            // anything the allocator would otherwise overlap them with.
            ggml_set_input(*slot);
        };

        make_input(&io.in.token_ids);
        if (req.needs_learned_pos_ids) make_input(&io.in.learned_pos_ids);
        if (req.needs_rope_pos_ids) {
            io.in.rope_pos_ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, S);
            ggml_set_input(io.in.rope_pos_ids);
        }
        if (req.needs_type_ids) make_input(&io.in.type_ids);
        // Per-sentence attention needs no mask, so only build one when the
        // architecture has not been handed the lengths it would replace.
        io.in.seq_lengths = (padded && req.consumes_seq_lengths && batch != nullptr)
            ? batch->lengths.data() : nullptr;
        if (padded && req.uses_kq_mask && io.in.seq_lengths == nullptr) {
            io.in.kq_mask = ggml_new_tensor_4d(
                gctx, GGML_TYPE_F16, S, S, 1, B);
            ggml_set_input(io.in.kq_mask);
        }

        if (padded && pooling == PoolType::Mean) {
            io.pool.valid_mask = ggml_new_tensor_3d(
                gctx, GGML_TYPE_F32, 1, S, B);
            io.pool.mean_scale = ggml_new_tensor_2d(
                gctx, GGML_TYPE_F32, 1, B);
            ggml_set_input(io.pool.valid_mask);
            ggml_set_input(io.pool.mean_scale);
        } else if (padded && pooling == PoolType::Last) {
            io.pool.last_indices = ggml_new_tensor_1d(
                gctx, GGML_TYPE_I32, B);
            ggml_set_input(io.pool.last_indices);
        }

        ggml_tensor * x = arch->build_graph(gctx, io.in);

        io.out = forward::build_pool(gctx, x, to_forward_pool(pooling), io.pool);
        if (normalize) {
            io.out = forward::build_l2_normalize(gctx, io.out);
        }
        // Read back after compute, so it must survive the whole graph.
        ggml_set_output(io.out);

        io.graph = ggml_new_graph(gctx);
        ggml_build_forward_expand(io.graph, io.out);
        return io;
    }

    ~Impl() {
        if (model_ctx) ggml_free(model_ctx);
        if (gguf)      gguf_free(gguf);
    }
};

struct ComputeScratch::Impl {
    ggml_backend_t       backend = nullptr;
    ggml_gallocr_t       galloc  = nullptr;
    std::vector<uint8_t> meta_buf;
    int                  reserved_seq_len = 0;

    Impl() {
        backend = ggml_backend_cpu_init();
        if (!backend) throw std::runtime_error("ggml_backend_cpu_init failed");
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (!galloc) {
            ggml_backend_free(backend);
            throw std::runtime_error("ggml_gallocr_new failed");
        }
        meta_buf.resize(kMaxGraphTensors * ggml_tensor_overhead() + ggml_graph_overhead());
    }

    ~Impl() {
        if (galloc)  ggml_gallocr_free(galloc);
        if (backend) ggml_backend_free(backend);
    }

    ggml_context * new_meta_ctx() {
        ggml_init_params p;
        p.mem_size   = meta_buf.size();
        p.mem_buffer = meta_buf.data();
        p.no_alloc   = true;
        ggml_context * c = ggml_init(p);
        if (!c) throw std::runtime_error("ggml_init failed for graph meta context");
        return c;
    }
};

ComputeScratch::ComputeScratch() : impl_(std::make_unique<Impl>()) {}
ComputeScratch::~ComputeScratch() = default;

Embedder::Embedder(const std::string & gguf_path)
    : impl_(std::make_unique<Impl>()) {
    // Dispatch on general.architecture. This validates the file and reads the
    // family's hyperparameters before any weight data is touched.
    impl_->arch = create_model_arch(gguf_path);

    // Open a second handle that DOES allocate tensor data. This is the one
    // we forward through.
    gguf_init_params gp;
    gp.no_alloc = false;
    gp.ctx      = &impl_->model_ctx;
    impl_->gguf = gguf_init_from_file(gguf_path.c_str(), gp);
    if (!impl_->gguf || !impl_->model_ctx) {
        throw std::runtime_error("failed to open GGUF for inference: " + gguf_path);
    }

    impl_->tokenizer = create_tokenizer(impl_->gguf);
    discard_consumed_tokenizer_metadata(impl_->gguf);
    impl_->arch->bind_weights(impl_->model_ctx);

}

Embedder::~Embedder() = default;

int Embedder::n_embed()     const noexcept { return impl_->arch->params().n_embed; }
int Embedder::n_layer()     const noexcept { return impl_->arch->params().n_layer; }
int Embedder::max_seq_len() const noexcept { return impl_->arch->params().max_seq_len; }

const std::string & Embedder::architecture() const noexcept {
    return impl_->arch->params().name;
}

PoolType Embedder::default_pooling() const noexcept {
    return impl_->arch->default_pooling();
}

int Embedder::reserved_seq_len(const ComputeScratch & scratch) const noexcept {
    return scratch.impl_->reserved_seq_len;
}

size_t Embedder::graph_buffer_size(const ComputeScratch & scratch) const noexcept {
    return ggml_gallocr_get_buffer_size(scratch.impl_->galloc, 0);
}

void Embedder::reserve(ComputeScratch & scratch,
                       int              max_seq_len,
                       PoolType         pooling,
                       bool             normalize) const {
    ComputeScratch::Impl & sc = *scratch.impl_;
    const int want = std::min(std::max(max_seq_len, 1), impl_->arch->params().max_seq_len);
    if (want <= sc.reserved_seq_len) return;

    ggml_context * gctx = sc.new_meta_ctx();
    try {
        const GraphIO io = impl_->build_graph(
            gctx, want, /*B=*/1, /*padded=*/false, pooling, normalize);
        if (!ggml_gallocr_reserve(sc.galloc, io.graph)) {
            throw AllocationError(
                "failed to reserve the graph buffer for max_seq_len=" +
                std::to_string(want));
        }
    } catch (...) {
        ggml_free(gctx);
        throw;
    }
    ggml_free(gctx);
    sc.reserved_seq_len = want;
}

void Embedder::embed(ComputeScratch &       scratch,
                     const std::string &    text,
                     const EmbedderConfig & cfg,
                     float *                out) {
    ComputeScratch::Impl & sc = *scratch.impl_;
    const int n_embed   = impl_->arch->params().n_embed;
    const int n_threads = resolve_n_threads(cfg.n_threads);

    int limit = cfg.max_seq_len > 0 ? cfg.max_seq_len : impl_->arch->params().max_seq_len;
    limit = std::min(limit, impl_->arch->params().max_seq_len);
    const std::vector<int> ids = impl_->tokenizer->encode(text, limit);
    const int64_t S = static_cast<int64_t>(ids.size());

    if (S > sc.reserved_seq_len) {
        reserve(scratch, static_cast<int>(S), cfg.pooling, cfg.normalize);
    }

    ggml_backend_cpu_set_n_threads(sc.backend, n_threads);

    // The context holds tensor structs only; tensor data comes from galloc's
    // reused buffer, which is why this costs a few hundred KB per call instead
    // of re-faulting a fresh arena.
    ggml_context * gctx = sc.new_meta_ctx();
    try {
        const GraphIO io = impl_->build_graph(
            gctx, S, /*B=*/1, /*padded=*/false, cfg.pooling, cfg.normalize);

        if (!ggml_gallocr_alloc_graph(sc.galloc, io.graph)) {
            throw AllocationError("failed to allocate the graph buffer");
        }
        // Only now do the input tensors have storage.
        ggml_backend_tensor_set(io.in.token_ids, ids.data(), 0,
                                ids.size() * sizeof(int32_t));

        std::vector<int32_t> input_scratch(static_cast<size_t>(S));
        if (io.in.learned_pos_ids || io.in.rope_pos_ids) {
            for (int64_t s = 0; s < S; ++s) {
                input_scratch[static_cast<size_t>(s)] = static_cast<int32_t>(s);
            }
            if (io.in.learned_pos_ids) {
                ggml_backend_tensor_set(io.in.learned_pos_ids, input_scratch.data(), 0,
                                        input_scratch.size() * sizeof(int32_t));
            }
            if (io.in.rope_pos_ids) {
                ggml_backend_tensor_set(io.in.rope_pos_ids, input_scratch.data(), 0,
                                        input_scratch.size() * sizeof(int32_t));
            }
        }
        if (io.in.type_ids) {
            std::fill(input_scratch.begin(), input_scratch.end(), 0);
            ggml_backend_tensor_set(io.in.type_ids, input_scratch.data(), 0,
                                    input_scratch.size() * sizeof(int32_t));
        }

        const ggml_status st = ggml_backend_graph_compute(sc.backend, io.graph);
        if (st != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("ggml_backend_graph_compute failed");
        }

        ggml_backend_tensor_get(io.out, out, 0,
                                static_cast<size_t>(n_embed) * sizeof(float));
    } catch (...) {
        ggml_free(gctx);
        throw;
    }
    ggml_free(gctx);
}

void Embedder::embed_batch(ComputeScratch &                 scratch,
                           const std::vector<std::string> & texts,
                           const EmbedderConfig &           cfg,
                           float *                          out) {
    if (out == nullptr) throw std::invalid_argument("batch output pointer is null");
    if (texts.empty()) return;

    ComputeScratch::Impl & sc = *scratch.impl_;
    const int H = impl_->arch->params().n_embed;
    const int n_threads = resolve_n_threads(cfg.n_threads);
    int limit = cfg.max_seq_len > 0 ? cfg.max_seq_len : impl_->arch->params().max_seq_len;
    limit = std::min(limit, impl_->arch->params().max_seq_len);

    const BatchPlan plan = make_batch_plan(
        *impl_->tokenizer, texts, limit, cfg.max_batch);
    ggml_backend_cpu_set_n_threads(sc.backend, n_threads);

    for (size_t batch_index = 0; batch_index < plan.subbatch_count(); ++batch_index) {
        const MaterializedBatch batch = materialize_batch(
            plan, batch_index, impl_->tokenizer->padding_id(),
            impl_->tokenizer->vocab_size());
        ggml_context * gctx = sc.new_meta_ctx();
        try {
            const GraphIO io = impl_->build_graph(
                gctx, batch.seq_len, batch.batch_size, batch.padded,
                cfg.pooling, cfg.normalize, &batch);
            if (!ggml_gallocr_alloc_graph(sc.galloc, io.graph)) {
                throw AllocationError("failed to allocate the batched graph buffer");
            }

            auto set = [](ggml_tensor * tensor, const void * data, size_t bytes) {
                if (tensor != nullptr) ggml_backend_tensor_set(tensor, data, 0, bytes);
            };
            if (io.packed) {
                set(io.in.token_ids, batch.packed_token_ids.data(),
                    batch.packed_token_ids.size() * sizeof(int32_t));
                set(io.in.rope_pos_ids, batch.packed_positions.data(),
                    batch.packed_positions.size() * sizeof(int32_t));
            } else {
            set(io.in.token_ids, batch.token_ids.data(),
                batch.token_ids.size() * sizeof(int32_t));
            set(io.in.learned_pos_ids, batch.learned_positions.data(),
                batch.learned_positions.size() * sizeof(int32_t));
            set(io.in.rope_pos_ids, batch.rope_positions.data(),
                batch.rope_positions.size() * sizeof(int32_t));
            set(io.in.type_ids, batch.type_ids.data(),
                batch.type_ids.size() * sizeof(int32_t));
            set(io.in.kq_mask, batch.attention_mask.data(),
                batch.attention_mask.size() * sizeof(ggml_fp16_t));
            set(io.pool.valid_mask, batch.valid_mask.data(),
                batch.valid_mask.size() * sizeof(float));
            set(io.pool.mean_scale, batch.mean_scale.data(),
                batch.mean_scale.size() * sizeof(float));
            set(io.pool.last_indices, batch.last_indices.data(),
                batch.last_indices.size() * sizeof(int32_t));
            }

            if (ggml_backend_graph_compute(sc.backend, io.graph) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("batched ggml graph compute failed");
            }
            const size_t B = static_cast<size_t>(batch.batch_size);
            std::vector<float> sorted_output;
            try {
                if (B > std::numeric_limits<size_t>::max() / static_cast<size_t>(H)) {
                    throw AllocationError("batched output staging size overflow");
                }
                sorted_output.resize(B * static_cast<size_t>(H));
            } catch (const std::bad_alloc &) {
                throw AllocationError("failed to allocate batched output staging");
            }
            ggml_backend_tensor_get(io.out, sorted_output.data(), 0,
                                    sorted_output.size() * sizeof(float));
            for (size_t b = 0; b < B; ++b) {
                std::memcpy(out + batch.original_indices[b] * static_cast<size_t>(H),
                            sorted_output.data() + b * static_cast<size_t>(H),
                            static_cast<size_t>(H) * sizeof(float));
            }
        } catch (...) {
            ggml_free(gctx);
            throw;
        }
        ggml_free(gctx);
    }
}

} // namespace nanoembed
