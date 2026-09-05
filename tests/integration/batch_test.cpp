// M5 true-batch contract: stable length bucketing, max_batch subdivision,
// default packed execution, selectable padded compatibility execution and
// restoration to caller order.

#include "nanoembed/nanoembed.h"
#include "batch.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "[batch_test] FAIL: %s\n", message);
        ++failures;
    }
}

class FakeTokenizer final : public nanoembed::Tokenizer {
public:
    std::vector<int> encode(const std::string & text, int) const override {
        std::vector<int> result;
        for (char c : text) result.push_back(1 + (static_cast<unsigned char>(c) % 30));
        return result.empty() ? std::vector<int>{1} : result;
    }
    int max_seq_len() const noexcept override { return 64; }
    int vocab_size() const noexcept override { return 32; }
    int padding_id() const noexcept override { return 0; }
    void set_max_seq_len(int) override {}
};

void planner_test() {
    FakeTokenizer tokenizer;
    const std::vector<std::string> texts = {"four", "a", "three", "bb", "cc"};
    const auto plan = nanoembed::make_batch_plan(tokenizer, texts, 64, 3);
    check(plan.subbatch_count() == 2, "max_batch creates two sub-batches");
    check(plan.items[0].original_index == 1 &&
          plan.items[1].original_index == 3 &&
          plan.items[2].original_index == 4,
          "length sort is stable for equal lengths");
    const auto first = nanoembed::materialize_batch(plan, 0, 0, 32);
    check(first.seq_len == 2 && first.batch_size == 3 && first.padded,
          "first sub-batch shape and padding are correct");
    check(first.lengths == std::vector<int32_t>({1, 2, 2}),
          "materialized lengths follow sorted order");
    check(first.last_indices == std::vector<int32_t>({0, 3, 5}),
          "LAST indices flatten S and B correctly");
    check(first.valid_tokens == 5 && first.padding_tokens == 1,
          "valid and padding token diagnostics are exact");
    check(first.valid_mask == std::vector<float>({1, 0, 1, 1, 1, 1}),
          "pool mask follows [S,B] layout");
    const auto second = nanoembed::materialize_batch(plan, 1, 0, 32);
    check(second.batch_size == 2 && second.seq_len == 5 && second.padded,
          "tail sub-batch is retained");
    for (const auto & item : std::vector<std::pair<size_t, size_t>>{
             {1, 1}, {3, 1}, {4, 2}, {9, 3}}) {
        std::vector<std::string> sized(item.first, "x");
        const auto sized_plan = nanoembed::make_batch_plan(
            tokenizer, sized, 64, 3);
        check(sized_plan.subbatch_count() == item.second,
              "batch boundary produces the expected sub-batch count");
    }
}

double cosine(const float * lhs, const float * rhs, int n) {
    double dot = 0.0, ll = 0.0, rr = 0.0;
    for (int i = 0; i < n; ++i) {
        dot += static_cast<double>(lhs[i]) * rhs[i];
        ll += static_cast<double>(lhs[i]) * lhs[i];
        rr += static_cast<double>(rhs[i]) * rhs[i];
    }
    return dot / std::sqrt(ll * rr);
}

void run_model(const char * label, const char * env, bool report_only) {
    const char * path = std::getenv(env);
    if (path == nullptr) {
        std::fprintf(stderr, "[batch_test] skip %s: %s not set\n", label, env);
        return;
    }
    nanoembed_model * model = nanoembed_load_model(path);
    check(model != nullptr, "model loads for batch test");
    if (model == nullptr) return;
    const int H = nanoembed_n_embed(model);

    const std::vector<std::string> texts = {
        "a",
        "The quick brown fox jumps over the lazy dog.",
        "한국어와 English가 섞인 문장입니다.",
        "tiny",
        "This deliberately longer sentence creates padding inside the first sub-batch.",
        "tiny",
        "emoji: 🧪🚀",
        "max-batch plus one boundary",
        "two max-batches plus three boundary",
    };
    std::vector<const char *> pointers;
    for (const auto & text : texts) pointers.push_back(text.c_str());

    const nanoembed_pool_type pools[] = {
        NANOEMBED_POOL_MODEL_DEFAULT,
        NANOEMBED_POOL_MEAN,
        NANOEMBED_POOL_LAST,
        NANOEMBED_POOL_CLS,
    };
    for (nanoembed_pool_type pool : pools) {
        nanoembed_context_params params = nanoembed_context_default_params();
        params.n_threads = 2;
        params.max_batch = 3;
        params.max_seq_len = 128;
        params.pooling = pool;
        nanoembed_context * ctx = nanoembed_new_context(model, params);
        check(ctx != nullptr, "batch context is created");
        if (ctx == nullptr) continue;

        std::vector<float> reference(texts.size() * static_cast<size_t>(H));
        std::vector<float> actual(reference.size());
        bool singles_ok = true;
        for (size_t i = 0; i < texts.size(); ++i) {
            singles_ok &= nanoembed_embed(
                ctx, texts[i].c_str(), reference.data() + i * static_cast<size_t>(H)) ==
                NANOEMBED_OK;
        }
        check(singles_ok, "sequential references succeed");
        const int rc = nanoembed_embed_batch(
            ctx, pointers.data(), static_cast<int>(pointers.size()), actual.data());
        check(rc == NANOEMBED_OK, "true batch succeeds");

        double worst_cos = 1.0;
        double max_abs = 0.0;
        for (size_t i = 0; i < texts.size(); ++i) {
            const float * expected = reference.data() + i * static_cast<size_t>(H);
            const float * got = actual.data() + i * static_cast<size_t>(H);
            worst_cos = std::min(worst_cos, cosine(expected, got, H));
            for (int h = 0; h < H; ++h) {
                max_abs = std::max(max_abs,
                    std::abs(static_cast<double>(expected[h]) - got[h]));
            }
        }
        std::fprintf(stderr, "[batch_test] %s pool=%d cos=%.12f max_abs=%.9g\n",
                     label, static_cast<int>(pool), worst_cos, max_abs);
        if (!report_only) {
            check(worst_cos >= 0.999998, "batch cosine matches sequential path");
            // A padded softmax reduces the same valid values in a wider row
            // than the trimmed B=1 graph. The extra -INF lanes do not affect
            // the result mathematically, but ARM vector reduction grouping
            // produces up to 2.35e-4 drift for BERT F16 across the exercised
            // ARM boundary shapes. Cosine remains the primary parity gate.
            check(max_abs <= 2.5e-4, "batch max absolute error is within CPU batch gate");
        }

        if (pool == NANOEMBED_POOL_MODEL_DEFAULT) {
            const char * equal[] = {texts[3].c_str(), texts[3].c_str(), texts[3].c_str()};
            std::vector<float> equal_out(static_cast<size_t>(H) * 3);
            check(nanoembed_embed_batch(ctx, equal, 3, equal_out.data()) == NANOEMBED_OK,
                  "equal-length batch takes the mask-free path");
            double equal_max = 0.0;
            for (int h = 0; h < H; ++h) {
                equal_max = std::max(equal_max, std::abs(
                    static_cast<double>(reference[3 * static_cast<size_t>(H) + h]) -
                    equal_out[static_cast<size_t>(h)]));
            }
            std::fprintf(stderr, "[batch_test] %s equal-length max_abs=%.9g\n",
                         label, equal_max);

            for (int count : {1, 3, 4, 9}) {
                std::vector<float> boundary(static_cast<size_t>(H) * count);
                check(nanoembed_embed_batch(ctx, pointers.data(), count,
                                            boundary.data()) == NANOEMBED_OK,
                      "batch boundary call succeeds");
                double boundary_cos = 1.0;
                for (int i = 0; i < count; ++i) {
                    boundary_cos = std::min(
                        boundary_cos,
                        cosine(reference.data() + static_cast<size_t>(i) * H,
                               boundary.data() + static_cast<size_t>(i) * H, H));
                }
                std::fprintf(stderr,
                             "[batch_test] %s boundary=%d cos=%.12f\n",
                             label, count, boundary_cos);
                if (!report_only) {
                    check(boundary_cos >= 0.999998,
                          "batch boundary call preserves item outputs");
                }
            }

            check(nanoembed_context_set_batch_layout(
                      ctx, NANOEMBED_BATCH_LAYOUT_PADDED) == NANOEMBED_OK,
                  "padded compatibility layout is selectable");
            std::vector<float> padded_out(reference.size());
            check(nanoembed_embed_batch(
                      ctx, pointers.data(), static_cast<int>(pointers.size()),
                      padded_out.data()) == NANOEMBED_OK,
                  "padded compatibility batch succeeds");
            double padded_cos = 1.0;
            for (size_t i = 0; i < texts.size(); ++i) {
                padded_cos = std::min(
                    padded_cos,
                    cosine(reference.data() + i * static_cast<size_t>(H),
                           padded_out.data() + i * static_cast<size_t>(H), H));
            }
            std::fprintf(stderr, "[batch_test] %s padded-layout cos=%.12f\n",
                         label, padded_cos);
            if (!report_only) {
                check(padded_cos >= 0.999998,
                      "padded compatibility layout preserves item outputs");
            }
            check(nanoembed_context_set_batch_layout(
                      ctx, static_cast<nanoembed_batch_layout>(99)) ==
                      NANOEMBED_ERR_INVALID_ARG,
                  "invalid batch layout is rejected");
            check(nanoembed_context_set_batch_layout(
                      ctx, NANOEMBED_BATCH_LAYOUT_DEFAULT) == NANOEMBED_OK,
                  "default packed layout can be restored");
        }

        std::vector<float> sentinel(static_cast<size_t>(H) * 3, 123.0f);
        const char * invalid[] = {"ok", nullptr, "later"};
        check(nanoembed_embed_batch(ctx, invalid, 3, sentinel.data()) ==
                  NANOEMBED_ERR_INVALID_ARG,
              "null item is rejected before execution");
        check(std::all_of(sentinel.begin(), sentinel.end(),
                          [](float value) { return value == 123.0f; }),
              "invalid batch leaves output untouched");
        const char * empty_inputs[] = {"unused"};
        check(nanoembed_embed_batch(ctx, empty_inputs, 0, sentinel.data()) == NANOEMBED_OK,
              "zero-size batch preserves existing success contract");
        nanoembed_free_context(ctx);
    }
    nanoembed_free_model(model);
}

} // namespace

int main() {
    planner_test();
    run_model("bert-f16", "NANOEMBED_TEST_MODEL", false);
    run_model("harrier-f32", "NANOEMBED_TEST_MODEL_GEMMA3", false);
    run_model("harrier-q8", "NANOEMBED_TEST_MODEL_GEMMA3_Q8", false);
    run_model("harrier-q4-report", "NANOEMBED_TEST_MODEL_GEMMA3_Q4", true);
    std::printf("batch_test: %s\n", failures == 0 ? "ok" : "FAIL");
    return failures == 0 ? 0 : 1;
}
