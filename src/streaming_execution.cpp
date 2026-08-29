#include "streaming_execution.h"

#include "arch/model_arch.h"
#include "batch.h"
#include "forward/pool.h"
#include "mapped_weight_store.h"
#include "tokenizer/tokenizer.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace nanoembed {

bool operator==(const StreamingByteRange & lhs,
                const StreamingByteRange & rhs) noexcept {
    return lhs.offset == rhs.offset && lhs.length == rhs.length;
}

namespace {

size_t checked_add(size_t lhs, size_t rhs, const char * subject) {
    if (rhs > std::numeric_limits<size_t>::max() - lhs) {
        throw std::runtime_error(std::string("streaming range overflow: ") + subject);
    }
    return lhs + rhs;
}

size_t page_count(const std::vector<StreamingByteRange> & ranges, size_t page_size) {
    size_t pages = 0;
    for (const auto & range : ranges) {
        const size_t range_pages = range.length / page_size +
            (range.length % page_size == 0 ? 0 : 1);
        pages = checked_add(pages, range_pages,
                            "page count");
    }
    return pages;
}

size_t byte_count(const std::vector<StreamingByteRange> & ranges) {
    size_t bytes = 0;
    for (const auto & range : ranges) {
        bytes = checked_add(bytes, range.length, "byte count");
    }
    return bytes;
}

std::vector<StreamingByteRange> subtract_ranges(
    const std::vector<StreamingByteRange> & source,
    const std::vector<StreamingByteRange> & protected_ranges) {
    std::vector<StreamingByteRange> result;
    for (const auto & item : source) {
        size_t cursor = item.offset;
        const size_t end = checked_add(item.offset, item.length, "subtract source");
        for (const auto & guard : protected_ranges) {
            const size_t guard_end = checked_add(guard.offset, guard.length,
                                                 "subtract guard");
            if (guard_end <= cursor) continue;
            if (guard.offset >= end) break;
            if (guard.offset > cursor) {
                result.push_back({cursor, guard.offset - cursor});
            }
            cursor = std::max(cursor, guard_end);
            if (cursor >= end) break;
        }
        if (cursor < end) result.push_back({cursor, end - cursor});
    }
    return result;
}

#if defined(__linux__)
forward::PoolType to_forward_pool(PoolType p) {
    switch (p) {
        case PoolType::Cls:  return forward::PoolType::Cls;
        case PoolType::Last: return forward::PoolType::Last;
        default:             return forward::PoolType::Mean;
    }
}

constexpr size_t kPhaseGraphTensors = 2048;
constexpr int kMaxAutoThreads = 4;

int resolve_threads(int requested) {
    if (requested > 0) return requested;
    const unsigned hw = std::thread::hardware_concurrency();
    return std::min(static_cast<int>(hw == 0 ? 1 : hw), kMaxAutoThreads);
}
#endif

} // namespace

namespace streaming_detail {

std::vector<StreamingByteRange> page_align_and_coalesce(
    const std::vector<StreamingByteRange> & ranges,
    size_t mapping_size,
    size_t page_size) {
    if (mapping_size == 0 || page_size == 0 || (page_size & (page_size - 1)) != 0) {
        throw std::runtime_error("streaming range planner requires non-zero mapping and power-of-two page size");
    }
    std::vector<StreamingByteRange> aligned;
    aligned.reserve(ranges.size());
    for (const auto & range : ranges) {
        if (range.length == 0) throw std::runtime_error("streaming tensor range is empty");
        const size_t end = checked_add(range.offset, range.length, "tensor end");
        if (range.offset >= mapping_size || end > mapping_size) {
            throw std::runtime_error("streaming tensor range exceeds mapped file");
        }
        const size_t begin_page = range.offset & ~(page_size - 1);
        size_t end_page = end;
        const size_t remainder = end_page & (page_size - 1);
        if (remainder != 0) {
            const size_t padding = page_size - remainder;
            end_page = padding > mapping_size - end_page
                ? mapping_size : end_page + padding;
        }
        end_page = std::min(end_page, mapping_size);
        aligned.push_back({begin_page, end_page - begin_page});
    }
    std::sort(aligned.begin(), aligned.end(), [](const auto & lhs, const auto & rhs) {
        return lhs.offset < rhs.offset ||
               (lhs.offset == rhs.offset && lhs.length < rhs.length);
    });
    std::vector<StreamingByteRange> result;
    for (const auto & range : aligned) {
        if (result.empty()) {
            result.push_back(range);
            continue;
        }
        auto & tail = result.back();
        const size_t tail_end = checked_add(tail.offset, tail.length, "coalesce tail");
        const size_t range_end = checked_add(range.offset, range.length, "coalesce range");
        if (range.offset <= tail_end) {
            tail.length = std::max(tail_end, range_end) - tail.offset;
        } else {
            result.push_back(range);
        }
    }
    return result;
}

std::vector<StreamingByteRange> token_row_ranges(
    size_t tensor_offset,
    size_t tensor_nbytes,
    size_t row_stride,
    int64_t row_count,
    const std::vector<int> & token_ids,
    size_t mapping_size,
    size_t page_size) {
    if (row_stride == 0 || row_count <= 0 || token_ids.empty()) {
        throw std::runtime_error("invalid/empty token row range request");
    }
    const size_t rows = static_cast<size_t>(row_count);
    if (rows > tensor_nbytes / row_stride || rows * row_stride > tensor_nbytes) {
        throw std::runtime_error("token tensor row stride exceeds validated tensor bounds");
    }
    std::set<int> unique_ids;
    std::vector<StreamingByteRange> raw;
    for (int id : token_ids) {
        if (id < 0 || static_cast<int64_t>(id) >= row_count) {
            throw std::runtime_error("token ID is outside validated token table");
        }
        if (!unique_ids.insert(id).second) continue;
        const size_t relative = static_cast<size_t>(id) * row_stride;
        const size_t start = checked_add(tensor_offset, relative, "token row offset");
        const size_t tensor_end = checked_add(tensor_offset, tensor_nbytes, "token tensor end");
        const size_t row_end = checked_add(start, row_stride, "token row end");
        if (row_end > tensor_end) throw std::runtime_error("token row exceeds tensor bounds");
        raw.push_back({start, row_stride});
    }
    return page_align_and_coalesce(raw, mapping_size, page_size);
}

namespace {

// Append `id` unless it is already there. Slot counts per group are tiny (a
// handful), so a linear scan beats any container and keeps declaration order,
// which is what makes the upload/download sequence reproducible.
void push_unique(std::vector<SlotId> & out, SlotId id) {
    if (std::find(out.begin(), out.end(), id) == out.end()) out.push_back(id);
}

} // namespace

std::vector<GroupSlots> resolve_group_slots(
    const std::vector<UnitSlots> & units,
    const std::vector<size_t> &    run_lengths,
    SlotId                         layer_output_slot) {
    size_t total = 0;
    for (size_t length : run_lengths) {
        if (length == 0) throw std::runtime_error("streaming partition produced an empty run");
        total = checked_add(total, length, "partition run length");
    }
    if (total != units.size()) {
        throw std::runtime_error("streaming partition does not cover the unit sequence");
    }

    std::vector<GroupSlots> groups;
    groups.reserve(run_lengths.size());
    size_t begin = 0;
    for (size_t length : run_lengths) {
        const size_t end = begin + length;
        GroupSlots group;

        for (size_t i = begin; i < end; ++i) {
            for (SlotId id : units[i].inputs) {
                // Produced earlier in this same run? Then it never leaves it.
                bool internal = false;
                for (size_t j = begin; j < i && !internal; ++j) {
                    const auto & produced = units[j].outputs;
                    internal = std::find(produced.begin(), produced.end(), id) != produced.end();
                }
                if (!internal) push_unique(group.inputs, id);
            }
        }

        for (size_t i = begin; i < end; ++i) {
            for (SlotId id : units[i].outputs) {
                bool needed = id == layer_output_slot;
                for (size_t k = end; k < units.size() && !needed; ++k) {
                    const auto & consumed = units[k].inputs;
                    needed = std::find(consumed.begin(), consumed.end(), id) != consumed.end();
                }
                if (needed) push_unique(group.outputs, id);
            }
        }

        if (group.outputs.empty()) {
            throw std::runtime_error("streaming group produces nothing any later group reads");
        }
        groups.push_back(std::move(group));
        begin = end;
    }
    return groups;
}

std::vector<size_t> partition_runs(const std::vector<uint64_t> &       weight_bytes,
                                   const std::vector<StreamingStage> & stages,
                                   const std::string &                 preset) {
    if (weight_bytes.size() != stages.size()) {
        throw std::runtime_error("streaming partition inputs disagree on unit count");
    }
    const size_t n = weight_bytes.size();
    if (n == 0) throw std::runtime_error("streaming layer declared no units");

    if (preset == "layer") return {n};
    if (preset == "unit")  return std::vector<size_t>(n, 1);

    if (preset == "attn-ffn") {
        std::vector<size_t> runs{1};
        for (size_t i = 1; i < n; ++i) {
            if (stages[i] != stages[i - 1]) runs.push_back(1);
            else ++runs.back();
        }
        return runs;
    }

    static const std::string kBudget = "budget:";
    if (preset.compare(0, kBudget.size(), kBudget) == 0) {
        const std::string digits = preset.substr(kBudget.size());
        if (digits.empty() ||
            digits.find_first_not_of("0123456789") != std::string::npos) {
            throw std::runtime_error("streaming budget preset needs a byte count: " + preset);
        }
        const uint64_t budget = std::strtoull(digits.c_str(), nullptr, 10);
        if (budget == 0) throw std::runtime_error("streaming budget preset must be positive");

        // Greedy, and deliberately never splits a unit: a unit is already one
        // whole GGUF tensor at minimum, so a budget below the largest unit
        // yields that unit alone rather than an error.
        std::vector<size_t> runs{1};
        uint64_t running = weight_bytes[0];
        for (size_t i = 1; i < n; ++i) {
            if (running + weight_bytes[i] <= budget) {
                running += weight_bytes[i];
                ++runs.back();
            } else {
                runs.push_back(1);
                running = weight_bytes[i];
            }
        }
        return runs;
    }

    throw std::runtime_error("unknown streaming partition preset: " + preset);
}

struct ResidencyCoordinator::Impl {
    struct ActiveLease { std::string key; std::vector<StreamingByteRange> ranges; };
    struct Region { uint64_t count = 0; std::vector<StreamingByteRange> pending; };

    size_t mapping_size;
    size_t page_size;
    AdviceFunction advice;
    mutable std::mutex mutex;
    uint64_t next_id = 1;
    std::map<uint64_t, ActiveLease> active;
    std::map<std::string, Region> regions;
    std::vector<StreamingByteRange> common;
    uint64_t common_bytes = 0;
    ResidencyDiagnostics diag;

    uint64_t current_advised_bytes() const {
        std::vector<StreamingByteRange> ranges;
        for (const auto & entry : active) {
            ranges.insert(ranges.end(), entry.second.ranges.begin(), entry.second.ranges.end());
        }
        if (!ranges.empty()) {
            ranges = page_align_and_coalesce(ranges, mapping_size, page_size);
            ranges = subtract_ranges(ranges, common);
        }
        return byte_count(ranges);
    }

    void note_advised_peak() {
        diag.advised_bytes_high_water =
            std::max(diag.advised_bytes_high_water,
                     current_advised_bytes() + common_bytes);
    }

    Impl(size_t mapped, size_t page, AdviceFunction fn)
        : mapping_size(mapped), page_size(page), advice(std::move(fn)) {
        diag.resolved_mode = "internal-streaming-linux";
    }

    void call_advice(const StreamingByteRange & range, AdviceKind kind) {
        std::string error;
        uint64_t & calls = kind == AdviceKind::WillNeed
            ? diag.willneed_calls : diag.dontneed_calls;
        uint64_t & failures = kind == AdviceKind::WillNeed
            ? diag.willneed_failures : diag.dontneed_failures;
        ++calls;
        if (advice(range.offset, range.length, kind, error) != 0) {
            ++failures;
            const char * action = kind == AdviceKind::WillNeed
                ? "MADV_WILLNEED" : "MADV_DONTNEED";
            throw std::runtime_error(std::string(action) + " failed: " + error);
        }
    }
};

ResidencyCoordinator::ResidencyCoordinator(
    size_t mapping_size, size_t page_size, AdviceFunction advice)
    : impl_(std::make_unique<Impl>(mapping_size, page_size, std::move(advice))) {
    if (!impl_->advice) throw std::runtime_error("streaming advice function is empty");
}

ResidencyCoordinator::~ResidencyCoordinator() = default;

void ResidencyCoordinator::set_classification(
    const std::vector<StreamingByteRange> & common,
    const std::vector<StreamingByteRange> & token,
    const std::vector<std::vector<StreamingByteRange>> & layers) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    impl_->diag.common_class_bytes = byte_count(common);
    impl_->diag.common_class_pages = page_count(common, impl_->page_size);
    impl_->diag.token_class_bytes = byte_count(token);
    impl_->diag.token_class_pages = page_count(token, impl_->page_size);
    for (const auto & layer : layers) {
        impl_->diag.layer_class_bytes += byte_count(layer);
        impl_->diag.layer_class_pages += page_count(layer, impl_->page_size);
    }
}

void ResidencyCoordinator::retain_common(
    const std::vector<StreamingByteRange> & ranges) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (!impl_->common.empty()) throw std::runtime_error("common ranges already retained");
    for (const auto & range : ranges) impl_->call_advice(range, AdviceKind::WillNeed);
    impl_->common = ranges;
    impl_->common_bytes = byte_count(ranges);
    impl_->note_advised_peak();
}

ResidencyCoordinator::Lease ResidencyCoordinator::acquire(
    const std::string & key,
    const std::vector<StreamingByteRange> & ranges,
    bool token_rows) {
    if (ranges.empty()) throw std::runtime_error("cannot lease an empty streaming range");
    std::lock_guard<std::mutex> lock(impl_->mutex);
    if (impl_->diag.poisoned) throw std::runtime_error("streaming residency coordinator is poisoned");

    // Mandatory precompute advice. A failed acquisition is rolled back and
    // fails this call, but postcompute failures below poison the model.
    for (const auto & range : ranges) impl_->call_advice(range, AdviceKind::WillNeed);
    const uint64_t id = impl_->next_id++;
    impl_->active.emplace(id, Impl::ActiveLease{key, ranges});
    auto & region = impl_->regions[key];
    ++region.count;
    region.pending.insert(region.pending.end(), ranges.begin(), ranges.end());
    ++impl_->diag.leases_acquired;
    ++impl_->diag.active_leases;
    impl_->diag.lease_high_water = std::max(
        impl_->diag.lease_high_water, impl_->diag.active_leases);
    impl_->note_advised_peak();
    if (token_rows) {
        impl_->diag.token_advised_bytes += byte_count(ranges);
        impl_->diag.token_advised_pages += page_count(ranges, impl_->page_size);
    }
    return Lease(this, id, key, token_rows);
}

void ResidencyCoordinator::release(
    uint64_t id, const std::string & key, bool compute_complete, bool) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    const auto active_it = impl_->active.find(id);
    if (active_it == impl_->active.end()) return;
    if (!compute_complete) {
        ++impl_->diag.premature_release_attempts;
        impl_->diag.poisoned = true;
        throw std::runtime_error("streaming lease released before synchronous compute completed");
    }
    ++impl_->diag.compute_completions;
    const auto region_it = impl_->regions.find(key);
    if (region_it == impl_->regions.end() || region_it->second.count == 0) {
        impl_->diag.poisoned = true;
        throw std::runtime_error("streaming lease accounting underflow");
    }
    Impl::Region & region = region_it->second;
    --region.count;
    impl_->active.erase(active_it);
    --impl_->diag.active_leases;
    ++impl_->diag.leases_released;
    if (region.count != 0) return;
    const auto pending = page_align_and_coalesce(
        region.pending, impl_->mapping_size, impl_->page_size);
    region.pending.clear();

    std::vector<StreamingByteRange> protected_ranges = impl_->common;
    for (const auto & entry : impl_->active) {
        protected_ranges.insert(protected_ranges.end(),
                                entry.second.ranges.begin(), entry.second.ranges.end());
    }
    if (!protected_ranges.empty()) {
        protected_ranges = page_align_and_coalesce(
            protected_ranges, impl_->mapping_size, impl_->page_size);
    }
    const auto releasable = subtract_ranges(pending, protected_ranges);
    try {
        for (const auto & range : releasable) {
            impl_->call_advice(range, AdviceKind::DontNeed);
        }
    } catch (...) {
        impl_->diag.poisoned = true;
        throw;
    }
}

ResidencyDiagnostics ResidencyCoordinator::diagnostics() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    return impl_->diag;
}

ResidencyCoordinator::Lease::Lease(
    ResidencyCoordinator * owner, uint64_t id, std::string key, bool token_rows)
    : owner_(owner), id_(id), key_(std::move(key)), token_rows_(token_rows) {}

ResidencyCoordinator::Lease::~Lease() {
    if (owner_ != nullptr) {
        try { release(); } catch (...) { /* coordinator is poisoned */ }
    }
}

ResidencyCoordinator::Lease::Lease(Lease && other) noexcept
    : owner_(other.owner_), id_(other.id_), key_(std::move(other.key_)),
      token_rows_(other.token_rows_), compute_complete_(other.compute_complete_) {
    other.owner_ = nullptr;
}

ResidencyCoordinator::Lease & ResidencyCoordinator::Lease::operator=(Lease && other) noexcept {
    if (this == &other) return *this;
    if (owner_ != nullptr) {
        try { release(); } catch (...) {}
    }
    owner_ = other.owner_;
    id_ = other.id_;
    key_ = std::move(other.key_);
    token_rows_ = other.token_rows_;
    compute_complete_ = other.compute_complete_;
    other.owner_ = nullptr;
    return *this;
}

void ResidencyCoordinator::Lease::release() {
    if (owner_ == nullptr) return;
    ResidencyCoordinator * owner = owner_;
    owner_ = nullptr;
    owner->release(id_, key_, compute_complete_, token_rows_);
}

} // namespace streaming_detail

// A declared unit with its names resolved: slot ids, byte ranges, and a
// borrowed pointer to the closure the architecture owns.
struct ResolvedUnit {
    std::string                     name;
    std::vector<SlotId>             inputs;
    std::vector<SlotId>             outputs;
    InputRequirements               graph_inputs;
    StreamingStage                  stage = StreamingStage::Other;
    uint64_t                        weight_bytes = 0;
    std::vector<StreamingByteRange> raw_ranges;
    const std::function<void(ggml_context *, SlotTable &)> * build = nullptr;
};

// A merged contiguous run of units: one lease, one graph, one compute. Units
// are closed under merging, so a run of one takes exactly the same path as a
// run of eight and the runner has a single case to handle.
struct ResolvedGroup {
    std::string                     key;
    std::vector<StreamingByteRange> ranges;
    std::vector<SlotId>             inputs;
    std::vector<SlotId>             outputs;
    InputRequirements               graph_inputs;
    std::vector<const ResolvedUnit *> members;
    uint64_t                        weight_bytes = 0;
};

struct ResolvedLayer {
    std::vector<ResolvedUnit>  units;    // owns; groups point into this
    std::vector<ResolvedGroup> groups;
    SlotId                     input_slot = kInvalidSlot;
    SlotId                     output_slot = kInvalidSlot;
};

// One growable F32 buffer per slot, reused across groups, layers and sentences.
// Sizes only grow, so a warm embed() allocates nothing -- the property the two
// ping-pong activation buffers had, generalized to a variable set of values of
// different shapes.
//
// Shapes are recorded when a value is drained and replayed when it is fed back
// in, rather than declared by the architecture. A declared shape would be a
// second copy of something the builder already knows, free to drift from it --
// the exact failure this refactor exists to remove.
class SlotStore final {
public:
    void resize(size_t n_slots) { entries_.resize(n_slots); }

    // Between sentences a slot's contents mean nothing. Clearing the valid bits
    // while keeping the capacity turns "a group read a slot nothing wrote this
    // sentence" from a silent stale-value read into an exception.
    void invalidate_all() noexcept {
        for (Entry & entry : entries_) entry.valid = false;
    }

    void download(SlotId id, const ggml_tensor * tensor) {
        if (tensor == nullptr) {
            throw std::runtime_error("streaming slot download of a null tensor");
        }
        if (tensor->type != GGML_TYPE_F32 || !ggml_is_contiguous(tensor)) {
            throw std::runtime_error("streaming slots carry contiguous F32 only");
        }
        Entry & entry = at(id);
        entry.data.resize(static_cast<size_t>(ggml_nelements(tensor)));
        for (int i = 0; i < 4; ++i) entry.ne[i] = tensor->ne[i];
        ggml_backend_tensor_get(tensor, entry.data.data(), 0, ggml_nbytes(tensor));
        entry.valid = true;
    }

    ggml_tensor * create_input(SlotId id, ggml_context * ctx) const {
        const Entry & entry = at(id);
        if (!entry.valid) {
            throw std::runtime_error("streaming group reads a slot nothing produced");
        }
        ggml_tensor * tensor = ggml_new_tensor_4d(
            ctx, GGML_TYPE_F32, entry.ne[0], entry.ne[1], entry.ne[2], entry.ne[3]);
        ggml_set_input(tensor);
        return tensor;
    }

    const void * data(SlotId id) const { return at(id).data.data(); }
    size_t nbytes(SlotId id) const { return at(id).data.size() * sizeof(float); }

    uint64_t resident_bytes() const noexcept {
        uint64_t total = 0;
        for (const Entry & entry : entries_) {
            total += static_cast<uint64_t>(entry.data.capacity()) * sizeof(float);
        }
        return total;
    }

private:
    struct Entry {
        std::vector<float> data;
        int64_t            ne[4] = {0, 0, 0, 0};
        bool               valid = false;
    };

    Entry & at(SlotId id) {
        if (static_cast<size_t>(id) >= entries_.size()) {
            throw std::runtime_error("streaming slot id out of range");
        }
        return entries_[static_cast<size_t>(id)];
    }
    const Entry & at(SlotId id) const {
        if (static_cast<size_t>(id) >= entries_.size()) {
            throw std::runtime_error("streaming slot id out of range");
        }
        return entries_[static_cast<size_t>(id)];
    }

    std::vector<Entry> entries_;
};

struct InternalStreamingContext::Impl {
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t galloc = nullptr;
    std::vector<uint8_t> meta;
    SlotStore slots;
    // Scratch mapping slot id -> the tensor currently carrying it in the graph
    // under construction. Sized once per model; cleared per group.
    std::vector<ggml_tensor *> slot_tensors;
    uint64_t phase_graph_computes = 0;
    uint64_t activation_copy_bytes = 0;
    // ggml_gallocr re-plans whenever a graph's node or leaf count differs from
    // the previous one. Only the node count is reachable through the public
    // API, so this is a lower bound -- enough to see whether finer partitions
    // make re-planning matter before deciding to give each group its own
    // allocator.
    uint64_t graph_replans = 0;
    uint64_t batches_processed = 0;
    uint64_t items_processed = 0;
    uint64_t valid_tokens_processed = 0;
    uint64_t padding_tokens_processed = 0;
    int previous_graph_nodes = -1;
    int64_t graph_context_failure_countdown = -1;

    Impl() {
#if !defined(__linux__)
        throw std::runtime_error("internal streaming execution is unsupported outside Linux");
#else
        backend = ggml_backend_cpu_init();
        if (backend == nullptr) throw std::runtime_error("streaming CPU backend initialization failed");
        galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (galloc == nullptr) {
            ggml_backend_free(backend);
            backend = nullptr;
            throw std::runtime_error("streaming graph allocator initialization failed");
        }
        meta.resize(kPhaseGraphTensors * ggml_tensor_overhead() + ggml_graph_overhead());
#endif
    }

    ~Impl() {
        if (galloc != nullptr) ggml_gallocr_free(galloc);
        if (backend != nullptr) ggml_backend_free(backend);
    }

    ggml_context * new_graph_context() {
        if (graph_context_failure_countdown == 0) {
            graph_context_failure_countdown = -1;
            throw AllocationError("injected streaming graph context allocation failure");
        }
        if (graph_context_failure_countdown > 0) {
            --graph_context_failure_countdown;
        }
        ggml_init_params params{meta.size(), meta.data(), true};
        ggml_context * result = ggml_init(params);
        if (result == nullptr) {
            throw AllocationError("streaming graph metadata arena exhausted");
        }
        return result;
    }

    void set_tensor(ggml_tensor * tensor, const void * data, size_t bytes,
                    const char * subject) {
        if (tensor == nullptr || tensor->buffer == nullptr || tensor->data == nullptr) {
            throw std::runtime_error(std::string("streaming allocator did not back input: ") + subject);
        }
        ggml_backend_tensor_set(tensor, data, 0, bytes);
    }


    // Build, allocate, compute and drain one group's graph.
    //
    // Threading values through `slot_tensors` is the whole of "merging": no
    // weight name is ever consulted, and a group of one member takes the same
    // path as a group of eight.
    void run_group(const ResolvedGroup &        group,
                   ggml_context *               gctx,
                   const MaterializedBatch &     batch) {
        const int64_t S = batch.seq_len;
        const int64_t B = batch.batch_size;
        GraphInputs graph_in;
        if (group.graph_inputs.needs_learned_pos_ids) {
            graph_in.learned_pos_ids = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, S, B);
            ggml_set_input(graph_in.learned_pos_ids);
        }
        if (group.graph_inputs.needs_rope_pos_ids) {
            graph_in.rope_pos_ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, S);
            ggml_set_input(graph_in.rope_pos_ids);
        }
        if (group.graph_inputs.needs_type_ids) {
            graph_in.type_ids = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, S, B);
            ggml_set_input(graph_in.type_ids);
        }
        // Per-sentence attention replaces the mask; see GraphInputs.
        graph_in.seq_lengths =
            (batch.padded && group.graph_inputs.consumes_seq_lengths)
                ? batch.lengths.data() : nullptr;
        if (batch.padded && group.graph_inputs.uses_kq_mask &&
            graph_in.seq_lengths == nullptr) {
            graph_in.kq_mask = ggml_new_tensor_4d(
                gctx, GGML_TYPE_F16, S, S, 1, B);
            ggml_set_input(graph_in.kq_mask);
        }

        // Keep the created input tensors aside. A slot is routinely both an
        // input and an output of the same group -- the residual stream is read
        // at the top of a layer and rewritten at the bottom -- and by upload
        // time slot_tensors[id] points at the value the group produced, not the
        // one it needs to be given.
        std::vector<std::pair<SlotId, ggml_tensor *>> uploads;
        uploads.reserve(group.inputs.size());
        for (SlotId id : group.inputs) {
            ggml_tensor * tensor = slots.create_input(id, gctx);
            slot_tensors[id] = tensor;
            uploads.emplace_back(id, tensor);
        }

        std::vector<ggml_tensor *> in_buf;
        std::vector<ggml_tensor *> out_buf;
        for (const ResolvedUnit * unit : group.members) {
            in_buf.clear();
            for (SlotId id : unit->inputs) {
                ggml_tensor * value = slot_tensors[id];
                if (value == nullptr) {
                    throw std::runtime_error("streaming unit reads a slot not yet built: " + unit->name);
                }
                in_buf.push_back(value);
            }
            out_buf.assign(unit->outputs.size(), nullptr);
            SlotTable table(in_buf.data(), in_buf.size(),
                            out_buf.data(), out_buf.size(), graph_in);
            (*unit->build)(gctx, table);
            for (size_t i = 0; i < unit->outputs.size(); ++i) {
                if (out_buf[i] == nullptr) {
                    throw std::runtime_error("streaming unit did not produce a declared output: " +
                                             unit->name);
                }
                slot_tensors[unit->outputs[i]] = out_buf[i];
            }
        }

        ggml_cgraph * graph = ggml_new_graph(gctx);
        for (SlotId id : group.outputs) {
            ggml_tensor * value = slot_tensors[id];
            if (value == nullptr) {
                throw std::runtime_error("streaming group lost a declared output: " + group.key);
            }
            ggml_set_output(value);
            ggml_build_forward_expand(graph, value);
        }
        allocate(graph, group.key.c_str());

        // Inputs can only be filled once the allocator has backed them.
        for (const auto & upload : uploads) {
            const size_t bytes = slots.nbytes(upload.first);
            set_tensor(upload.second, slots.data(upload.first), bytes, "slot input");
            activation_copy_bytes += bytes;
        }
        if (graph_in.learned_pos_ids != nullptr) {
            set_tensor(graph_in.learned_pos_ids, batch.learned_positions.data(),
                       batch.learned_positions.size() * sizeof(int32_t),
                       "group_learned_position_ids");
        }
        if (graph_in.rope_pos_ids != nullptr) {
            set_tensor(graph_in.rope_pos_ids, batch.rope_positions.data(),
                       batch.rope_positions.size() * sizeof(int32_t),
                       "group_rope_position_ids");
        }
        if (graph_in.type_ids != nullptr) {
            set_tensor(graph_in.type_ids, batch.type_ids.data(),
                       batch.type_ids.size() * sizeof(int32_t), "group_type_ids");
        }
        if (graph_in.kq_mask != nullptr) {
            set_tensor(graph_in.kq_mask, batch.attention_mask.data(),
                       batch.attention_mask.size() * sizeof(ggml_fp16_t),
                       "group_attention_mask");
        }

        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS) {
            throw std::runtime_error("streaming group compute failed: " + group.key);
        }
        ++phase_graph_computes;

        for (SlotId id : group.outputs) {
            slots.download(id, slot_tensors[id]);
            activation_copy_bytes += slots.nbytes(id);
        }
    }

    // Allocate a graph, counting the re-plans a shape change forces.
    void allocate(ggml_cgraph * graph, const char * subject) {
        const int nodes = ggml_graph_n_nodes(graph);
        if (nodes != previous_graph_nodes) {
            ++graph_replans;
            previous_graph_nodes = nodes;
        }
        if (!ggml_gallocr_alloc_graph(galloc, graph)) {
            throw AllocationError(std::string("streaming graph allocation failed: ") + subject);
        }
    }
};

InternalStreamingContext::InternalStreamingContext()
    : impl_(std::make_unique<Impl>()) {}
InternalStreamingContext::~InternalStreamingContext() = default;

void InternalStreamingContext::diagnostic_fail_graph_context_after(
    uint64_t successful_creations) {
    if (successful_creations > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument("graph context failure countdown exceeds int64");
    }
    impl_->graph_context_failure_countdown =
        static_cast<int64_t>(successful_creations);
}

struct InternalStreamingModel::Impl {
    // preparation must come first: the unit closures below capture the
    // ModelArch it owns, so reverse-order member destruction has to tear them
    // down before it. Nothing enforced this while the resolved state was
    // integers only.
    MappedModelPreparation preparation;
    size_t page_size = 0;
    std::vector<StreamingByteRange> common_ranges;
    MappedTensorInfo token;
    std::string partition_preset = "layer";
    std::vector<StreamingLayerPlan> plans;      // owns the declared closures
    std::vector<ResolvedLayer> layers;
    std::map<std::string, SlotId> slot_ids;
    uint64_t groups_per_sentence = 0;
    uint64_t max_group_weight_bytes = 0;
    std::unique_ptr<streaming_detail::ResidencyCoordinator> residency;

    SlotId intern(const std::string & name) {
        const auto it = slot_ids.find(name);
        if (it != slot_ids.end()) return it->second;
        if (slot_ids.size() >= static_cast<size_t>(kInvalidSlot)) {
            throw std::runtime_error("streaming model declared too many slots");
        }
        const SlotId id = static_cast<SlotId>(slot_ids.size());
        slot_ids.emplace(name, id);
        return id;
    }

    explicit Impl(const std::string & path) : preparation(require_linux(path)) {
#if defined(__linux__)
        const long raw_page = sysconf(_SC_PAGESIZE);
        if (raw_page <= 0) throw std::runtime_error("sysconf(_SC_PAGESIZE) failed");
        page_size = static_cast<size_t>(raw_page);
        const MappedWeightStore & store = preparation.store();
        const ModelArch & arch = preparation.arch();
        const int n_layer = arch.params().n_layer;

        if (const char * env = std::getenv("NANOEMBED_STREAMING_PARTITION")) {
            if (*env != '\0') partition_preset = env;
        }

        const StreamingCommonPlan common_plan = arch.streaming_common_plan();
        if (common_plan.token_embedding.empty()) {
            throw std::runtime_error("architecture returned an incomplete streaming weight plan");
        }

        std::map<std::string, MappedTensorInfo> infos;
        for (const auto & info : store.tensor_infos()) {
            infos.emplace(info.name, info);
        }
        std::map<std::string, std::string> owner;   // tensor -> the unit that claimed it
        auto take = [&](const std::string & name,
                        const std::string & claimant) -> MappedTensorInfo {
            const auto it = infos.find(name);
            if (it == infos.end()) throw std::runtime_error("required streaming tensor missing: " + name);
            const auto claimed = owner.find(name);
            if (claimed != owner.end()) {
                throw std::runtime_error("streaming tensor classified twice: " + name +
                                         " (by " + claimed->second + " and " + claimant + ")");
            }
            owner.emplace(name, claimant);
            return it->second;
        };

        token = take(common_plan.token_embedding, "token");
        std::vector<StreamingByteRange> raw_common;
        for (const auto & name : common_plan.common) {
            const auto info = take(name, "common");
            raw_common.push_back({info.absolute_offset, info.nbytes});
        }
        common_ranges = streaming_detail::page_align_and_coalesce(
            raw_common, store.mapped_size(), page_size);

        plans.reserve(static_cast<size_t>(n_layer));
        layers.resize(static_cast<size_t>(n_layer));
        std::vector<std::vector<StreamingByteRange>> layer_ranges(
            static_cast<size_t>(n_layer));

        for (int layer = 0; layer < n_layer; ++layer) {
            plans.push_back(arch.streaming_units(layer));
            const StreamingLayerPlan & plan = plans.back();
            if (plan.units.empty() || plan.input_slot.empty() || plan.output_slot.empty()) {
                throw std::runtime_error("architecture returned an incomplete streaming layer plan");
            }
            // Every layer reads and writes the same residual slot, so layer N's
            // output really is layer N+1's input. Asserted rather than assumed:
            // a mismatch would silently chain a stale slot.
            if (plan.input_slot != plan.output_slot) {
                throw std::runtime_error("streaming layer plan does not return its residual slot");
            }
            if (layer > 0 && plan.input_slot != plans[static_cast<size_t>(layer) - 1].output_slot) {
                throw std::runtime_error("streaming layers disagree on the residual slot");
            }

            ResolvedLayer & resolved = layers[static_cast<size_t>(layer)];
            resolved.input_slot  = intern(plan.input_slot);
            resolved.output_slot = intern(plan.output_slot);
            resolved.units.reserve(plan.units.size());

            const std::string layer_key = "layer:" + std::to_string(layer);
            std::vector<StreamingByteRange> layer_raw;
            for (const StreamingUnit & unit : plan.units) {
                if (!unit.build) {
                    throw std::runtime_error("streaming unit declared no builder: " + unit.name);
                }
                ResolvedUnit item;
                item.name         = unit.name;
                item.graph_inputs = unit.graph_inputs;
                item.stage        = unit.stage;
                item.build        = &unit.build;
                for (const auto & slot : unit.inputs)  item.inputs.push_back(intern(slot));
                for (const auto & slot : unit.outputs) item.outputs.push_back(intern(slot));
                if (item.outputs.empty()) {
                    throw std::runtime_error("streaming unit produces nothing: " + unit.name);
                }
                for (const auto & name : unit.weights) {
                    const auto info = take(name, layer_key + "/" + unit.name);
                    item.raw_ranges.push_back({info.absolute_offset, info.nbytes});
                    item.weight_bytes += info.nbytes;
                    layer_raw.push_back({info.absolute_offset, info.nbytes});
                }
                resolved.units.push_back(std::move(item));
            }
            layer_ranges[static_cast<size_t>(layer)] =
                streaming_detail::page_align_and_coalesce(
                    layer_raw, store.mapped_size(), page_size);

            // Partition, then resolve what each run exchanges with the world.
            std::vector<uint64_t> weight_bytes;
            std::vector<StreamingStage> stages;
            std::vector<streaming_detail::UnitSlots> unit_slots;
            weight_bytes.reserve(resolved.units.size());
            stages.reserve(resolved.units.size());
            unit_slots.reserve(resolved.units.size());
            for (const ResolvedUnit & unit : resolved.units) {
                weight_bytes.push_back(unit.weight_bytes);
                stages.push_back(unit.stage);
                unit_slots.push_back({unit.inputs, unit.outputs});
            }
            const auto runs = streaming_detail::partition_runs(
                weight_bytes, stages, partition_preset);
            const auto group_slots = streaming_detail::resolve_group_slots(
                unit_slots, runs, resolved.output_slot);

            size_t begin = 0;
            for (size_t g = 0; g < runs.size(); ++g) {
                ResolvedGroup group;
                group.inputs  = group_slots[g].inputs;
                group.outputs = group_slots[g].outputs;
                std::vector<StreamingByteRange> raw;
                std::string label;
                for (size_t i = begin; i < begin + runs[g]; ++i) {
                    const ResolvedUnit & unit = resolved.units[i];
                    group.members.push_back(&unit);
                    group.weight_bytes += unit.weight_bytes;
                    group.graph_inputs.needs_learned_pos_ids |=
                        unit.graph_inputs.needs_learned_pos_ids;
                    group.graph_inputs.needs_rope_pos_ids |=
                        unit.graph_inputs.needs_rope_pos_ids;
                    group.graph_inputs.needs_type_ids |= unit.graph_inputs.needs_type_ids;
                    group.graph_inputs.uses_kq_mask |= unit.graph_inputs.uses_kq_mask;
                    group.graph_inputs.consumes_seq_lengths |=
                        unit.graph_inputs.consumes_seq_lengths;
                    raw.insert(raw.end(), unit.raw_ranges.begin(), unit.raw_ranges.end());
                    label += (label.empty() ? "" : "+") + unit.name;
                }
                group.key = layer_key + ":" + label;
                group.ranges = streaming_detail::page_align_and_coalesce(
                    raw, store.mapped_size(), page_size);
                max_group_weight_bytes = std::max(max_group_weight_bytes, group.weight_bytes);
                resolved.groups.push_back(std::move(group));
                begin += runs[g];
            }
            groups_per_sentence += resolved.groups.size();
        }
        groups_per_sentence += 2;   // embedding and final phases

        if (owner.size() != infos.size()) {
            for (const auto & item : infos) {
                if (owner.count(item.first) == 0) {
                    throw std::runtime_error("unclassified mapped tensor in streaming plan: " + item.first);
                }
            }
        }

        const unsigned char * base = static_cast<const unsigned char *>(store.mapping_base());
        auto advice = [base](size_t offset, size_t length,
                             streaming_detail::AdviceKind kind, std::string & error) {
            const int native = kind == streaming_detail::AdviceKind::WillNeed
                ? MADV_WILLNEED : MADV_DONTNEED;
            if (madvise(const_cast<unsigned char *>(base + offset), length, native) != 0) {
                error = std::strerror(errno);
                return -1;
            }
            return 0;
        };
        residency = std::make_unique<streaming_detail::ResidencyCoordinator>(
            store.mapped_size(), page_size, advice);
        const auto token_class = streaming_detail::page_align_and_coalesce(
            {{token.absolute_offset, token.nbytes}}, store.mapped_size(), page_size);
        residency->set_classification(common_ranges, token_class, layer_ranges);
        residency->retain_common(common_ranges);
#endif
    }

    static const std::string & require_linux(const std::string & path) {
#if !defined(__linux__)
        (void) path;
        throw std::runtime_error("internal streaming model creation is unsupported outside Linux");
#else
        return path;
#endif
    }
};

InternalStreamingModel::InternalStreamingModel(const std::string & path)
    : impl_(std::make_unique<Impl>(path)) {}
InternalStreamingModel::~InternalStreamingModel() = default;

int InternalStreamingModel::n_embed() const noexcept {
    return impl_->preparation.arch().params().n_embed;
}
int InternalStreamingModel::n_layer() const noexcept {
    return impl_->preparation.arch().params().n_layer;
}
int InternalStreamingModel::max_seq_len() const noexcept {
    return impl_->preparation.arch().params().max_seq_len;
}
PoolType InternalStreamingModel::default_pooling() const noexcept {
    return impl_->preparation.arch().default_pooling();
}
const std::string & InternalStreamingModel::architecture() const noexcept {
    return impl_->preparation.arch().params().name;
}

void InternalStreamingModel::embed(
    InternalStreamingContext & context,
    const std::string & text,
    const EmbedderConfig & config,
    float * output) const {
    embed_batch(context, std::vector<std::string>{text}, config, output);
}

void InternalStreamingModel::embed_batch(
    InternalStreamingContext & context,
    const std::vector<std::string> & texts,
    const EmbedderConfig & config,
    float * output) const {
#if !defined(__linux__)
    (void) context; (void) texts; (void) config; (void) output;
    throw std::runtime_error("internal streaming execute is unsupported outside Linux");
#else
    if (output == nullptr) throw std::runtime_error("streaming output pointer is null");
    if (texts.empty()) return;
    auto & sc = *context.impl_;
    const ModelArch & arch = impl_->preparation.arch();
    const Tokenizer & tokenizer = impl_->preparation.tokenizer();
    const ArchParams & params = arch.params();
    int limit = config.max_seq_len > 0 ? config.max_seq_len : params.max_seq_len;
    limit = std::min(limit, params.max_seq_len);
    const BatchPlan plan = make_batch_plan(tokenizer, texts, limit, config.max_batch);
    ggml_backend_cpu_set_n_threads(sc.backend, resolve_threads(config.n_threads));

    sc.slots.resize(impl_->slot_ids.size());
    sc.slot_tensors.assign(impl_->slot_ids.size(), nullptr);

    auto poison_output = [&] {
        std::fill(output, output + texts.size() * static_cast<size_t>(params.n_embed),
                  std::numeric_limits<float>::quiet_NaN());
    };
    try {
      for (size_t batch_index = 0; batch_index < plan.subbatch_count(); ++batch_index) {
        const MaterializedBatch batch = materialize_batch(
            plan, batch_index, tokenizer.padding_id(), tokenizer.vocab_size());
        const int64_t S = batch.seq_len;
        const int64_t B = batch.batch_size;
        if (static_cast<size_t>(B) > std::numeric_limits<size_t>::max() /
                                      static_cast<size_t>(params.n_embed)) {
            throw AllocationError("streaming batch output size overflow");
        }
        const size_t output_floats = static_cast<size_t>(params.n_embed) *
                                     static_cast<size_t>(B);
        std::vector<float> batch_output(output_floats);

        // Previous sub-batch values mean nothing to this one. Retain capacity
        // but invalidate contents so a liveness error cannot read stale data.
        sc.slots.invalidate_all();

        ggml_context * gctx = sc.new_graph_context();
        try {
            // Finish every failure-prone host allocation before publishing a
            // residency lease. A graph-context/range failure therefore cannot
            // leave an active lease behind or poison the shared coordinator.
            std::vector<int> range_ids(batch.token_ids.begin(), batch.token_ids.end());
            const auto rows = streaming_detail::token_row_ranges(
                impl_->token.absolute_offset, impl_->token.nbytes,
                impl_->token.row_stride, impl_->token.row_count, range_ids,
                impl_->preparation.store().mapped_size(), impl_->page_size);
            auto lease = impl_->residency->acquire("token", rows, true);
            try {
                GraphInputs in;
                in.token_ids = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, S, B);
                ggml_set_input(in.token_ids);
                const InputRequirements req = arch.embedding_inputs();
                if (req.needs_learned_pos_ids) {
                    in.learned_pos_ids = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, S, B);
                    ggml_set_input(in.learned_pos_ids);
                }
                if (req.needs_rope_pos_ids) {
                    in.rope_pos_ids = ggml_new_tensor_1d(gctx, GGML_TYPE_I32, S);
                    ggml_set_input(in.rope_pos_ids);
                }
                if (req.needs_type_ids) {
                    in.type_ids = ggml_new_tensor_2d(gctx, GGML_TYPE_I32, S, B);
                    ggml_set_input(in.type_ids);
                }
                ggml_tensor * phase_out = arch.build_embedding_phase(gctx, in);
                ggml_set_output(phase_out);
                ggml_cgraph * graph = ggml_new_graph(gctx);
                ggml_build_forward_expand(graph, phase_out);
                sc.allocate(graph, "embedding");
                sc.set_tensor(in.token_ids, batch.token_ids.data(),
                              batch.token_ids.size() * sizeof(int32_t), "token_ids");
                if (in.learned_pos_ids != nullptr) {
                    sc.set_tensor(in.learned_pos_ids, batch.learned_positions.data(),
                                  batch.learned_positions.size() * sizeof(int32_t),
                                  "learned_position_ids");
                }
                if (in.rope_pos_ids != nullptr) {
                    sc.set_tensor(in.rope_pos_ids, batch.rope_positions.data(),
                                  batch.rope_positions.size() * sizeof(int32_t),
                                  "rope_position_ids");
                }
                if (in.type_ids != nullptr) {
                    sc.set_tensor(in.type_ids, batch.type_ids.data(),
                                  batch.type_ids.size() * sizeof(int32_t), "type_ids");
                }
                if (ggml_backend_graph_compute(sc.backend, graph) != GGML_STATUS_SUCCESS) {
                    throw std::runtime_error("streaming embedding compute failed");
                }
                ++sc.phase_graph_computes;
                const SlotId entry = impl_->layers.front().input_slot;
                sc.slots.download(entry, phase_out);
                sc.activation_copy_bytes += sc.slots.nbytes(entry);
                lease.mark_compute_complete();
                lease.release();
            } catch (...) {
                // Backend execution is synchronous; by the time it returns or
                // throws, mapped pages are safe to release.
                lease.mark_compute_complete();
                throw;
            }
        } catch (...) {
            ggml_free(gctx);
            throw;
        }
        ggml_free(gctx);

        for (size_t layer_index = 0; layer_index < impl_->layers.size(); ++layer_index) {
            const ResolvedLayer & layer = impl_->layers[layer_index];
            for (size_t group_index = 0; group_index < layer.groups.size(); ++group_index) {
                const ResolvedGroup & group = layer.groups[group_index];
                ggml_context * group_ctx = sc.new_graph_context();
                try {
                    auto group_lease =
                        impl_->residency->acquire(group.key, group.ranges, false);
                    try {
                        std::fill(sc.slot_tensors.begin(), sc.slot_tensors.end(), nullptr);
                        sc.run_group(group, group_ctx, batch);
                        group_lease.mark_compute_complete();
                        group_lease.release();
                    } catch (...) {
                        group_lease.mark_compute_complete();
                        throw;
                    }
                } catch (...) {
                    ggml_free(group_ctx);
                    throw;
                }
                ggml_free(group_ctx);
            }
        }

        // Final common phase: architecture final norm, pool and normalization.
        ggml_context * final_ctx = sc.new_graph_context();
        try {
            const SlotId exit = impl_->layers.back().output_slot;
            ggml_tensor * activation = sc.slots.create_input(exit, final_ctx);
            ggml_tensor * phase_out = arch.build_final_phase(final_ctx, activation);
            forward::PoolInputs pool;
            if (batch.padded && config.pooling == PoolType::Mean) {
                pool.valid_mask = ggml_new_tensor_3d(
                    final_ctx, GGML_TYPE_F32, 1, S, B);
                pool.mean_scale = ggml_new_tensor_2d(
                    final_ctx, GGML_TYPE_F32, 1, B);
                ggml_set_input(pool.valid_mask);
                ggml_set_input(pool.mean_scale);
            } else if (batch.padded && config.pooling == PoolType::Last) {
                pool.last_indices = ggml_new_tensor_1d(
                    final_ctx, GGML_TYPE_I32, B);
                ggml_set_input(pool.last_indices);
            }
            phase_out = forward::build_pool(
                final_ctx, phase_out, to_forward_pool(config.pooling), pool);
            if (config.normalize) phase_out = forward::build_l2_normalize(final_ctx, phase_out);
            ggml_set_output(phase_out);
            ggml_cgraph * graph = ggml_new_graph(final_ctx);
            ggml_build_forward_expand(graph, phase_out);
            sc.allocate(graph, "final");
            sc.set_tensor(activation, sc.slots.data(exit), sc.slots.nbytes(exit),
                          "final_activation");
            sc.activation_copy_bytes += sc.slots.nbytes(exit);
            if (pool.valid_mask != nullptr) {
                sc.set_tensor(pool.valid_mask, batch.valid_mask.data(),
                              batch.valid_mask.size() * sizeof(float), "pool_valid_mask");
                sc.set_tensor(pool.mean_scale, batch.mean_scale.data(),
                              batch.mean_scale.size() * sizeof(float), "pool_mean_scale");
            }
            if (pool.last_indices != nullptr) {
                sc.set_tensor(pool.last_indices, batch.last_indices.data(),
                              batch.last_indices.size() * sizeof(int32_t),
                              "pool_last_indices");
            }
            if (ggml_backend_graph_compute(sc.backend, graph) != GGML_STATUS_SUCCESS) {
                throw std::runtime_error("streaming final compute failed");
            }
            ++sc.phase_graph_computes;
            ggml_backend_tensor_get(phase_out, batch_output.data(), 0,
                                    batch_output.size() * sizeof(float));
            sc.activation_copy_bytes += batch_output.size() * sizeof(float);
        } catch (...) {
            ggml_free(final_ctx);
            throw;
        }
        ggml_free(final_ctx);
        for (size_t b = 0; b < static_cast<size_t>(B); ++b) {
            std::memcpy(output + batch.original_indices[b] * static_cast<size_t>(params.n_embed),
                        batch_output.data() + b * static_cast<size_t>(params.n_embed),
                        static_cast<size_t>(params.n_embed) * sizeof(float));
        }
        ++sc.batches_processed;
        sc.items_processed += static_cast<uint64_t>(B);
        sc.valid_tokens_processed += batch.valid_tokens;
        sc.padding_tokens_processed += batch.padding_tokens;
      }
    } catch (...) {
        poison_output();
        throw;
    }
#endif
}

StreamingPartitionInfo InternalStreamingModel::partition_info() const {
    StreamingPartitionInfo info;
    info.preset                 = impl_->partition_preset;
    info.groups_per_sentence    = impl_->groups_per_sentence;
    info.max_group_weight_bytes = impl_->max_group_weight_bytes;
    info.groups_per_layer       = impl_->layers.empty() ? 0 : impl_->layers.front().groups.size();
    for (const ResolvedLayer & layer : impl_->layers) {
        if (layer.groups.size() != info.groups_per_layer) {
            info.groups_per_layer = 0;   // layers disagree; the caller must not assume
            break;
        }
    }
    return info;
}

InternalStreamingDiagnostics InternalStreamingModel::diagnostics(
    const InternalStreamingContext & context) const {
    InternalStreamingDiagnostics result;
    result.residency = impl_->residency->diagnostics();
    result.phase_graph_computes = context.impl_->phase_graph_computes;
    result.activation_copy_bytes = context.impl_->activation_copy_bytes;
    result.slot_resident_bytes = context.impl_->slots.resident_bytes();
    result.graph_replans = context.impl_->graph_replans;
    result.batches_processed = context.impl_->batches_processed;
    result.items_processed = context.impl_->items_processed;
    result.valid_tokens_processed = context.impl_->valid_tokens_processed;
    result.padding_tokens_processed = context.impl_->padding_tokens_processed;
    return result;
}

size_t InternalStreamingModel::diagnostic_resident_pages() const {
#if !defined(__linux__)
    throw std::runtime_error("mincore streaming diagnostic is unsupported outside Linux");
#else
    const MappedWeightStore & store = impl_->preparation.store();
    const size_t pages = (store.mapped_size() + impl_->page_size - 1) / impl_->page_size;
    std::vector<unsigned char> state(pages);
    if (mincore(const_cast<void *>(store.mapping_base()), store.mapped_size(), state.data()) != 0) {
        throw std::runtime_error(std::string("mincore streaming diagnostic failed: ") +
                                 std::strerror(errno));
    }
    return static_cast<size_t>(std::count_if(
        state.begin(), state.end(), [](unsigned char value) { return (value & 1U) != 0; }));
#endif
}

} // namespace nanoembed
