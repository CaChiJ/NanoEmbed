// Shared tokenizer-side planning for eager and mapped streaming batches.
//
// This module owns only request-local data: token IDs, stable length ordering,
// padding and masks. Model weights, graph allocators and activation slots stay
// in their existing owners.

#pragma once

#include "tokenizer/tokenizer.h"

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace nanoembed {

class AllocationError : public std::runtime_error {
public:
    explicit AllocationError(const std::string & what) : std::runtime_error(what) {}
};

struct BatchItem {
    std::vector<int> ids;
    size_t original_index = 0;
};

struct BatchPlan {
    std::vector<BatchItem> items;  // stable-sorted by token length
    size_t max_batch = 1;

    size_t subbatch_count() const noexcept;
};

struct MaterializedBatch {
    int64_t seq_len = 0;
    int64_t batch_size = 0;
    bool padded = false;

    // ggml ne[0] is contiguous, hence token index s + S*b.
    std::vector<int32_t> token_ids;          // [S,B]
    std::vector<int32_t> learned_positions;  // [S,B]
    std::vector<int32_t> rope_positions;     // [S]
    std::vector<int32_t> type_ids;           // [S,B], all zero

    // Present only when `padded` is true.
    std::vector<ggml_fp16_t> attention_mask; // [S,S,1,B]
    std::vector<float>       valid_mask;     // [1,S,B]
    std::vector<float>       mean_scale;     // [1,B] = 1/length
    std::vector<int32_t>     last_indices;   // [B], flattened S*B index

    // Padding-free view of the same sub-batch, built only when `padded`.
    // Token-wise operators -- embedding lookup, norms, the Q/K/V/O linear
    // transforms and the FFN -- read one token and write one token, so they
    // can run over `total_tokens` real tokens and skip the padded cells
    // entirely. Attention is the only operator that needs sentence bounds,
    // and it takes them from `offsets`.
    std::vector<int32_t> packed_token_ids;  // [T]
    std::vector<int32_t> packed_positions;  // [T], each sentence restarts at 0
    std::vector<int32_t> offsets;           // [B+1], first token of each sentence
    int64_t total_tokens = 0;               // T = sum of lengths

    std::vector<size_t> original_indices;
    std::vector<int32_t> lengths;
    uint64_t valid_tokens = 0;
    uint64_t padding_tokens = 0;
};

BatchPlan make_batch_plan(const Tokenizer & tokenizer,
                          const std::vector<std::string> & texts,
                          int max_seq_len,
                          int max_batch);

MaterializedBatch materialize_batch(const BatchPlan & plan,
                                    size_t subbatch_index,
                                    int padding_id,
                                    int vocab_size);

} // namespace nanoembed
