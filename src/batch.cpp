#include "batch.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>

namespace nanoembed {

namespace {

size_t checked_mul(size_t lhs, size_t rhs, const char * subject) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        throw AllocationError(std::string("batch size overflow: ") + subject);
    }
    return lhs * rhs;
}

} // namespace

size_t BatchPlan::subbatch_count() const noexcept {
    return items.empty() ? 0 : 1 + (items.size() - 1) / max_batch;
}

BatchPlan make_batch_plan(const Tokenizer & tokenizer,
                          const std::vector<std::string> & texts,
                          int max_seq_len,
                          int max_batch) {
    if (max_batch <= 0) throw std::invalid_argument("max_batch must be positive");

    BatchPlan plan;
    plan.max_batch = static_cast<size_t>(max_batch);
    try {
        plan.items.reserve(texts.size());
        for (size_t i = 0; i < texts.size(); ++i) {
            BatchItem item;
            item.ids = tokenizer.encode(texts[i], max_seq_len);
            if (item.ids.empty()) {
                throw TokenizerError("tokenizer produced no IDs for a batch item");
            }
            item.original_index = i;
            plan.items.push_back(std::move(item));
        }
        std::stable_sort(plan.items.begin(), plan.items.end(),
            [](const BatchItem & lhs, const BatchItem & rhs) {
                return lhs.ids.size() < rhs.ids.size();
            });
    } catch (const std::bad_alloc &) {
        throw AllocationError("failed to allocate the batch plan");
    } catch (const std::length_error &) {
        throw AllocationError("batch plan exceeds addressable memory");
    }
    return plan;
}

MaterializedBatch materialize_batch(const BatchPlan & plan,
                                    size_t subbatch_index,
                                    int padding_id,
                                    int vocab_size) {
    if (plan.max_batch == 0 || subbatch_index >= plan.subbatch_count()) {
        throw std::out_of_range("batch slice index is out of range");
    }
    if (vocab_size <= 0) throw TokenizerError("tokenizer vocabulary is empty");
    if (padding_id < 0 || padding_id >= vocab_size) padding_id = 0;

    const size_t begin = subbatch_index * plan.max_batch;
    const size_t end = std::min(begin + plan.max_batch, plan.items.size());
    const size_t B = end - begin;
    const size_t S = plan.items[end - 1].ids.size();
    if (B == 0 || S == 0 ||
        B > static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
        S > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
        throw AllocationError("batch tensor dimensions are not representable");
    }

    MaterializedBatch out;
    out.seq_len = static_cast<int64_t>(S);
    out.batch_size = static_cast<int64_t>(B);
    out.padded = plan.items[begin].ids.size() != S;

    try {
        const size_t cells = checked_mul(S, B, "S*B");
        out.token_ids.assign(cells, static_cast<int32_t>(padding_id));
        out.learned_positions.resize(cells);
        out.rope_positions.resize(S);
        out.type_ids.assign(cells, 0);
        out.original_indices.resize(B);
        out.lengths.resize(B);

        for (size_t s = 0; s < S; ++s) {
            out.rope_positions[s] = static_cast<int32_t>(s);
        }
        for (size_t b = 0; b < B; ++b) {
            const BatchItem & item = plan.items[begin + b];
            const size_t length = item.ids.size();
            if (length > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
                throw AllocationError("batch sequence length exceeds int32");
            }
            out.original_indices[b] = item.original_index;
            out.lengths[b] = static_cast<int32_t>(length);
            out.valid_tokens += length;
            out.padding_tokens += S - length;
            for (size_t s = 0; s < S; ++s) {
                const size_t index = s + S * b;
                out.learned_positions[index] = static_cast<int32_t>(s);
                if (s < length) out.token_ids[index] = static_cast<int32_t>(item.ids[s]);
            }
        }

        if (out.padded) {
            // Padding-free copy. Sentences keep plan order, so slicing sentence
            // b is offsets[b]..offsets[b+1] and nothing has to be moved later.
            out.offsets.resize(B + 1);
            out.packed_token_ids.reserve(out.valid_tokens);
            out.packed_positions.reserve(out.valid_tokens);
            int32_t cursor = 0;
            for (size_t b = 0; b < B; ++b) {
                out.offsets[b] = cursor;
                const BatchItem & item = plan.items[begin + b];
                for (size_t s = 0; s < item.ids.size(); ++s) {
                    out.packed_token_ids.push_back(static_cast<int32_t>(item.ids[s]));
                    out.packed_positions.push_back(static_cast<int32_t>(s));
                }
                if (item.ids.size() > static_cast<size_t>(
                        std::numeric_limits<int32_t>::max() - cursor)) {
                    throw AllocationError("packed token count exceeds int32");
                }
                cursor += static_cast<int32_t>(item.ids.size());
            }
            out.offsets[B] = cursor;
            out.total_tokens = cursor;

            const size_t mask_cells = checked_mul(checked_mul(S, S, "S*S"), B, "S*S*B");
            out.attention_mask.resize(mask_cells);
            out.valid_mask.resize(cells);
            out.mean_scale.resize(B);
            out.last_indices.resize(B);
            const ggml_fp16_t zero = ggml_fp32_to_fp16(0.0f);
            const ggml_fp16_t neg_inf = ggml_fp32_to_fp16(-INFINITY);
            for (size_t b = 0; b < B; ++b) {
                const size_t length = static_cast<size_t>(out.lengths[b]);
                out.mean_scale[b] = 1.0f / static_cast<float>(length);
                const size_t flat_last = b * S + length - 1;
                if (flat_last > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
                    throw AllocationError("LAST pooling index exceeds int32");
                }
                out.last_indices[b] = static_cast<int32_t>(flat_last);
                for (size_t s = 0; s < S; ++s) {
                    out.valid_mask[s + S * b] = s < length ? 1.0f : 0.0f;
                }
                for (size_t q = 0; q < S; ++q) {
                    for (size_t k = 0; k < S; ++k) {
                        out.attention_mask[k + S * (q + S * b)] =
                            k < length ? zero : neg_inf;
                    }
                }
            }
        }
    } catch (const std::bad_alloc &) {
        throw AllocationError("failed to allocate padded batch tensors");
    } catch (const std::length_error &) {
        throw AllocationError("padded batch tensors exceed addressable memory");
    }
    return out;
}

} // namespace nanoembed
