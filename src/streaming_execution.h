// Internal B3 phase-separated mapped inference.  This header is intentionally
// private to nanoembed_core/tests; B4 decides whether/how the frozen C ABI
// selects it.

#pragma once

#include "arch/model_arch.h"
#include "embedder.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace nanoembed {

struct StreamingByteRange {
    size_t offset = 0;
    size_t length = 0;
};

bool operator==(const StreamingByteRange & lhs,
                const StreamingByteRange & rhs) noexcept;

namespace streaming_detail {

std::vector<StreamingByteRange> page_align_and_coalesce(
    const std::vector<StreamingByteRange> & ranges,
    size_t mapping_size,
    size_t page_size);

std::vector<StreamingByteRange> token_row_ranges(
    size_t tensor_offset,
    size_t tensor_nbytes,
    size_t row_stride,
    int64_t row_count,
    const std::vector<int> & token_ids,
    size_t mapping_size,
    size_t page_size);

// Slot bookkeeping for merged units, kept free of ggml and of Linux so the
// liveness rule can be tested directly. That rule is the one piece of this
// design whose failure is silent: get it wrong and a slot is never marked as a
// graph output, gets eliminated as dead, is never drained, and the group that
// reads it next uploads the previous sentence's value.
struct UnitSlots {
    std::vector<SlotId> inputs;
    std::vector<SlotId> outputs;
};

// What a merged run of units exchanges with the world outside it.
struct GroupSlots {
    std::vector<SlotId> inputs;
    std::vector<SlotId> outputs;
};

// Cut `units` (topological order) into the contiguous runs `run_lengths`
// describes and report each run's external slots.
//
// The two rules are deliberately asymmetric. A run's inputs are what it reads
// without having written first -- last writer *within* the run. Its outputs are
// what it writes that anything *after* it reads, plus the layer's own output.
// "Read later", not "not read here": a residual stream is read inside a run and
// again beyond it, and belongs in both sets.
std::vector<GroupSlots> resolve_group_slots(
    const std::vector<UnitSlots> & units,
    const std::vector<size_t> &    run_lengths,
    SlotId                         layer_output_slot);

// Cut a unit sequence into runs. A strategy sees only run structure, the coarse
// stage tag and resolved byte sizes -- never a weight name -- which is what
// keeps the grouping swappable without leaking a layer's internals.
//
//   "layer"     one run: today's whole-block behavior
//   "attn-ffn"  cut where StreamingStage changes
//   "unit"      no merging at all
//   "budget:N"  greedy, extend a run while its weights stay within N bytes
std::vector<size_t> partition_runs(const std::vector<uint64_t> &       weight_bytes,
                                   const std::vector<StreamingStage> & stages,
                                   const std::string &                 preset);

enum class AdviceKind { WillNeed, DontNeed };
using AdviceFunction = std::function<int(
    size_t offset, size_t length, AdviceKind kind, std::string & error)>;

struct ResidencyDiagnostics {
    uint64_t common_class_bytes = 0;
    uint64_t common_class_pages = 0;
    uint64_t token_class_bytes = 0;
    uint64_t token_class_pages = 0;
    uint64_t layer_class_bytes = 0;
    uint64_t layer_class_pages = 0;
    uint64_t token_advised_bytes = 0;
    uint64_t token_advised_pages = 0;
    // Peak bytes under an active lease, plus the retained common ranges. This
    // is the number the partitioning exists to lower; lease_high_water below
    // counts leases, not bytes, and cannot answer that question.
    uint64_t advised_bytes_high_water = 0;
    uint64_t willneed_calls = 0;
    uint64_t willneed_failures = 0;
    uint64_t dontneed_calls = 0;
    uint64_t dontneed_failures = 0;
    uint64_t leases_acquired = 0;
    uint64_t leases_released = 0;
    uint64_t active_leases = 0;
    uint64_t lease_high_water = 0;
    uint64_t compute_completions = 0;
    uint64_t premature_release_attempts = 0;
    bool poisoned = false;
    std::string resolved_mode;
};

class ResidencyCoordinator final {
public:
    class Lease;

    ResidencyCoordinator(size_t mapping_size,
                         size_t page_size,
                         AdviceFunction advice);
    ~ResidencyCoordinator();

    ResidencyCoordinator(const ResidencyCoordinator &) = delete;
    ResidencyCoordinator & operator=(const ResidencyCoordinator &) = delete;

    void retain_common(const std::vector<StreamingByteRange> & ranges);
    void set_classification(const std::vector<StreamingByteRange> & common,
                            const std::vector<StreamingByteRange> & token,
                            const std::vector<std::vector<StreamingByteRange>> & layers);
    Lease acquire(const std::string & key,
                  const std::vector<StreamingByteRange> & ranges,
                  bool token_rows);
    ResidencyDiagnostics diagnostics() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    void release(uint64_t id, const std::string & key,
                 bool compute_complete, bool token_rows);
    friend class Lease;
};

class ResidencyCoordinator::Lease final {
public:
    Lease() = default;
    ~Lease();
    Lease(Lease && other) noexcept;
    Lease & operator=(Lease && other) noexcept;
    Lease(const Lease &) = delete;
    Lease & operator=(const Lease &) = delete;

    void mark_compute_complete() noexcept { compute_complete_ = true; }
    void release();

private:
    Lease(ResidencyCoordinator * owner, uint64_t id,
          std::string key, bool token_rows);
    ResidencyCoordinator * owner_ = nullptr;
    uint64_t id_ = 0;
    std::string key_;
    bool token_rows_ = false;
    bool compute_complete_ = false;
    friend class ResidencyCoordinator;
};

} // namespace streaming_detail

struct InternalStreamingDiagnostics {
    streaming_detail::ResidencyDiagnostics residency;
    uint64_t phase_graph_computes = 0;
    uint64_t activation_copy_bytes = 0;
    // Host memory held for values crossing graph boundaries. Reported next to
    // advised_bytes_high_water on purpose: finer partitions trade weight
    // residency for slot buffers, and either number alone is half the story.
    uint64_t slot_resident_bytes = 0;
    // ggml_gallocr re-plans its allocation whenever a graph's node or leaf
    // count differs from the last one, so a finer partition raises this from
    // ~3 per sentence to one per group. Measured before it is optimized.
    uint64_t graph_replans = 0;
    uint64_t batches_processed = 0;
    uint64_t items_processed = 0;
    uint64_t valid_tokens_processed = 0;
    uint64_t padding_tokens_processed = 0;
};

// How a model's layers are currently cut into graphs. A property of the model
// and its selected preset, not of any one context.
struct StreamingPartitionInfo {
    std::string preset;
    uint64_t    groups_per_sentence = 0;    // 2 + the per-layer groups summed
    uint64_t    groups_per_layer = 0;       // 0 when layers disagree
    uint64_t    max_group_weight_bytes = 0;
};

class InternalStreamingContext final {
public:
    InternalStreamingContext();
    ~InternalStreamingContext();
    InternalStreamingContext(const InternalStreamingContext &) = delete;
    InternalStreamingContext & operator=(const InternalStreamingContext &) = delete;

    // Internal deterministic failure injection. After `successful_creations`
    // graph metadata contexts have been created, the next creation fails once.
    // This is deliberately outside the public C ABI and exists only to verify
    // lease/output rollback at graph-construction boundaries.
    void diagnostic_fail_graph_context_after(uint64_t successful_creations);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend class InternalStreamingModel;
};

class InternalStreamingModel final {
public:
    explicit InternalStreamingModel(const std::string & gguf_path);
    ~InternalStreamingModel();
    InternalStreamingModel(const InternalStreamingModel &) = delete;
    InternalStreamingModel & operator=(const InternalStreamingModel &) = delete;

    int n_embed() const noexcept;
    int n_layer() const noexcept;
    int max_seq_len() const noexcept;
    PoolType default_pooling() const noexcept;
    const std::string & architecture() const noexcept;

    void embed(InternalStreamingContext & context,
               const std::string & text,
               const EmbedderConfig & config,
               float * output) const;
    void embed_batch(InternalStreamingContext & context,
                     const std::vector<std::string> & texts,
                     const EmbedderConfig & config,
                     float * output) const;
    InternalStreamingDiagnostics diagnostics(
        const InternalStreamingContext & context) const;
    StreamingPartitionInfo partition_info() const;
    // Explicit test diagnostic. Never called by embed/timed execution.
    size_t diagnostic_resident_pages() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nanoembed
