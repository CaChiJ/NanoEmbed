// Unit coverage for the two pure pieces of the streaming partitioner.
//
// The liveness rule in resolve_group_slots is the one part of this design whose
// failure is silent: get it wrong and a slot is never marked as a graph output,
// gets eliminated as dead code, is never drained, and the group that reads it
// next uploads the previous sentence's value. An end-to-end test cannot be
// relied on to catch that -- consecutive fixture sentences can be similar enough
// to clear a cosine gate -- so the rule is exercised directly here.

#include "streaming_execution.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

#define EXPECT_TRUE(condition) do {                                             \
    if (!(condition)) {                                                        \
        std::fprintf(stderr, "FAIL: %s (%s:%d)\n", #condition, __FILE__, __LINE__); \
        ++failures;                                                            \
    }                                                                          \
} while (0)

template <typename F>
void expect_throw(F && fn, const char * needle) {
    try {
        fn();
        std::fprintf(stderr, "FAIL: expected exception containing %s\n", needle);
        ++failures;
    } catch (const std::exception & error) {
        if (std::string(error.what()).find(needle) == std::string::npos) {
            std::fprintf(stderr, "FAIL: '%s' lacks '%s'\n", error.what(), needle);
            ++failures;
        }
    }
}

using nanoembed::SlotId;
using nanoembed::StreamingStage;
using nanoembed::streaming_detail::GroupSlots;
using nanoembed::streaming_detail::UnitSlots;
using nanoembed::streaming_detail::partition_runs;
using nanoembed::streaming_detail::resolve_group_slots;

// The gemma3 declaration, as slot ids. x=0 is the residual stream; the rest are
// intermediates that must never escape a group that owns both ends of them.
enum : SlotId { kX = 0, kXn, kQ, kK, kV, kH, kX1, kFn, kFh, kFd };

std::vector<UnitSlots> gemma3_units() {
    return {
        /* attn_norm   */ {{kX},        {kXn}},
        /* attn_qkv    */ {{kXn},       {kQ, kK, kV}},
        /* attn_core   */ {{kQ, kK, kV},{kH}},
        /* attn_post   */ {{kH, kX},    {kX1}},
        /* ffn_norm    */ {{kX1},       {kFn}},
        /* ffn_gate_up */ {{kFn},       {kFh}},
        /* ffn_down    */ {{kFh},       {kFd}},
        /* ffn_post    */ {{kFd, kX1},  {kX}},
    };
}

void test_full_merge() {
    // Everything in one group: only the layer's own residual slot may escape.
    // A rule of "produced and not consumed inside" would also yield {x} here --
    // which is exactly why the asymmetric cases below matter.
    const auto groups = resolve_group_slots(gemma3_units(), {8}, kX);
    EXPECT_TRUE(groups.size() == 1);
    EXPECT_TRUE(groups[0].inputs == std::vector<SlotId>({kX}));
    EXPECT_TRUE(groups[0].outputs == std::vector<SlotId>({kX}));
}

void test_read_inside_and_after() {
    // x is read by unit 0 inside the first run and by unit 3 beyond it, so it
    // must be both an input and an output of the run that also rewrites it.
    // Cutting {0,1} | {2,3} | {4,5} | {6,7} is the four-way split.
    const auto groups = resolve_group_slots(gemma3_units(), {2, 2, 2, 2}, kX);
    EXPECT_TRUE(groups.size() == 4);

    // {attn_norm, attn_qkv}: reads x, hands on q/k/v -- and x, which unit 3 in
    // the next run still needs. "Read later", not "not read here".
    EXPECT_TRUE(groups[0].inputs == std::vector<SlotId>({kX}));
    EXPECT_TRUE(groups[0].outputs == std::vector<SlotId>({kQ, kK, kV}));

    // {attn_core, attn_post}: consumes q/k/v and the still-live x.
    EXPECT_TRUE(groups[1].inputs == std::vector<SlotId>({kQ, kK, kV, kX}));
    EXPECT_TRUE(groups[1].outputs == std::vector<SlotId>({kX1}));

    // {ffn_norm, ffn_gate_up}: x1 is read here and again by ffn_post later.
    EXPECT_TRUE(groups[2].inputs == std::vector<SlotId>({kX1}));
    EXPECT_TRUE(groups[2].outputs == std::vector<SlotId>({kFh}));

    EXPECT_TRUE(groups[3].inputs == std::vector<SlotId>({kFh, kX1}));
    EXPECT_TRUE(groups[3].outputs == std::vector<SlotId>({kX}));
}

void test_x1_survives_its_first_reader() {
    // The trap case. x1 is produced by unit 3, read by unit 4 immediately, and
    // read again by unit 7. A group ending at unit 4 must still publish x1.
    const auto groups = resolve_group_slots(gemma3_units(), {5, 3}, kX);
    EXPECT_TRUE(groups.size() == 2);
    // x1 (unit 3) is read immediately by unit 4 inside the run and again by
    // unit 7 outside it, so it escapes; fn (unit 4) escapes because unit 5
    // reads it. Both, in declaration order.
    EXPECT_TRUE(groups[0].outputs == std::vector<SlotId>({kX1, kFn}));
    EXPECT_TRUE(groups[1].inputs == std::vector<SlotId>({kFn, kX1}));
}

void test_dead_output_is_dropped() {
    // A slot nothing downstream reads, and which is not the layer output, has
    // no reason to cross a boundary.
    const std::vector<UnitSlots> units = {
        {{kX}, {kXn, kQ}},   // kQ is never read again
        {{kXn}, {kX}},
    };
    const auto groups = resolve_group_slots(units, {1, 1}, kX);
    EXPECT_TRUE(groups[0].outputs == std::vector<SlotId>({kXn}));
}

void test_partition_rejects_bad_runs() {
    expect_throw([] { resolve_group_slots(gemma3_units(), {3, 3}, kX); },
                 "does not cover the unit sequence");
    expect_throw([] { resolve_group_slots(gemma3_units(), {8, 0}, kX); },
                 "empty run");
    // A run whose every output dies is a partition bug, not a valid grouping.
    expect_throw([] {
        const std::vector<UnitSlots> units = {{{kX}, {kQ}}, {{kX}, {kX}}};
        resolve_group_slots(units, {1, 1}, kX);
    }, "produces nothing");
}

void test_presets() {
    // harrier F32 byte counts, in the declared order.
    const std::vector<uint64_t> bytes = {
        2560,            // attn_norm
        3932160,         // attn_q + attn_k + attn_v
        2623488,         // q_norm + k_norm + attn_output
        2560,            // post_attention_norm
        2560,            // ffn_norm
        10485760,        // ffn_gate + ffn_up
        5242880,         // ffn_down
        2560,            // post_ffw_norm
    };
    using S = StreamingStage;
    const std::vector<S> stages = {S::Attention, S::Attention, S::Attention, S::Attention,
                                   S::Ffn, S::Ffn, S::Ffn, S::Ffn};

    EXPECT_TRUE(partition_runs(bytes, stages, "layer") == std::vector<size_t>({8}));
    EXPECT_TRUE(partition_runs(bytes, stages, "unit") == std::vector<size_t>(8, 1));
    EXPECT_TRUE(partition_runs(bytes, stages, "attn-ffn") == std::vector<size_t>({4, 4}));

    auto peak_of = [&](const std::vector<size_t> & runs) {
        uint64_t peak = 0, index = 0;
        for (size_t run : runs) {
            uint64_t total = 0;
            for (size_t i = 0; i < run; ++i) total += bytes[index++];
            peak = total > peak ? total : peak;
        }
        return peak;
    };

    // Peak group weight is what the split is for: one block is 22,294,528 B
    // (21.26 MiB) and every byte of it is resident while that graph runs.
    EXPECT_TRUE(peak_of({8}) == 22294528);
    // attn|ffn halves it far less than the byte split suggests, because the FFN
    // carries 71% of the block.
    EXPECT_TRUE(peak_of({4, 4}) == 15733760);

    // A 10 MiB budget hits the target peak without anyone naming a group count
    // -- and lands on three groups rather than the four a hand-drawn attention/
    // FFN split would give, which is strictly better: same residency, one fewer
    // graph and one fewer madvise pair per layer.
    const auto budget = partition_runs(bytes, stages, "budget:10485760");
    EXPECT_TRUE(budget == std::vector<size_t>({5, 1, 2}));
    EXPECT_TRUE(peak_of(budget) == 10485760);
    EXPECT_TRUE(peak_of(budget) < peak_of({8}) / 2);

    // Never splits a unit: a budget under the largest one still yields it alone.
    const auto tiny = partition_runs(bytes, stages, "budget:1");
    EXPECT_TRUE(tiny == std::vector<size_t>(8, 1));

    // Monotone in the budget -- a bigger allowance never makes more groups.
    EXPECT_TRUE(partition_runs(bytes, stages, "budget:20971520").size() <= budget.size());

    expect_throw([&] { partition_runs(bytes, stages, "nonsense"); }, "unknown streaming partition");
    expect_throw([&] { partition_runs(bytes, stages, "budget:"); }, "needs a byte count");
    expect_throw([&] { partition_runs(bytes, stages, "budget:0"); }, "must be positive");
}

} // namespace

int main() {
    test_full_merge();
    test_read_inside_and_after();
    test_x1_survives_its_first_reader();
    test_dead_output_is_dropped();
    test_partition_rejects_bad_runs();
    test_presets();
    std::printf("streaming_units_test: %s\n", failures == 0 ? "ok" : "FAIL");
    return failures == 0 ? 0 : 1;
}
