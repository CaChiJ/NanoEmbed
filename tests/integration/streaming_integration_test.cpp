// Linux-only B3 internal-path proof. No public use_streaming selection occurs.

#include "embedder.h"
#include "batch.h"
#include "streaming_execution.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

#if defined(__linux__)

struct Sample { std::string text; std::vector<float> embedding; };

std::vector<Sample> load_fixture(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open fixture: " + path);
    auto read = [&](void * output, size_t bytes) {
        file.read(static_cast<char *>(output), static_cast<std::streamsize>(bytes));
        if (file.gcount() != static_cast<std::streamsize>(bytes)) {
            throw std::runtime_error("truncated fixture: " + path);
        }
    };
    char magic[4]; uint32_t version, count, dimension;
    read(magic, 4); read(&version, 4); read(&count, 4); read(&dimension, 4);
    if (std::string(magic, 4) != "NEGD" || version != 1 || count == 0 || dimension == 0) {
        throw std::runtime_error("invalid fixture header: " + path);
    }
    std::vector<Sample> result;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t length = 0; read(&length, 4);
        Sample sample; sample.text.resize(length); read(sample.text.data(), length);
        sample.embedding.resize(dimension);
        read(sample.embedding.data(), dimension * sizeof(float));
        result.push_back(std::move(sample));
    }
    return result;
}

double cosine(const std::vector<float> & lhs, const std::vector<float> & rhs) {
    if (lhs.size() != rhs.size() || lhs.empty()) throw std::runtime_error("vector dimension mismatch");
    double dot = 0.0, nl = 0.0, nr = 0.0;
    for (size_t i = 0; i < lhs.size(); ++i) {
        dot += static_cast<double>(lhs[i]) * rhs[i];
        nl += static_cast<double>(lhs[i]) * lhs[i];
        nr += static_cast<double>(rhs[i]) * rhs[i];
    }
    return dot / std::sqrt(nl * nr);
}

struct Case {
    const char * label;
    const char * model_env;
    const char * fixture_env;
    double sample_gate;
    double mean_gate;
    bool enforce;
    // Graphs per layer the active partition preset should produce, asserted
    // exactly rather than as a lower bound -- a preset that silently collapsed
    // back to one group per layer would still satisfy ">= 1". Zero waives the
    // check, for presets like budget:N where the two architectures legitimately
    // land on different counts; the derived consistency checks still apply.
    uint64_t expected_groups_per_layer;
};

int run_case(const Case & item) {
    const char * model_path = std::getenv(item.model_env);
    const char * fixture_path = std::getenv(item.fixture_env);
    if (model_path == nullptr || fixture_path == nullptr) {
        std::fprintf(stderr, "[streaming] skip %s: missing %s/%s\n",
                     item.label, item.model_env, item.fixture_env);
        return 0;
    }
    const auto samples = load_fixture(fixture_path);
    nanoembed::Embedder eager(model_path);
    nanoembed::ComputeScratch eager_context;
    nanoembed::InternalStreamingModel streaming(model_path);
    nanoembed::InternalStreamingContext streaming_context;
    if (eager.n_embed() != streaming.n_embed() || eager.architecture() != streaming.architecture()) {
        throw std::runtime_error("eager/streaming model identity mismatch");
    }
    nanoembed::EmbedderConfig config;
    config.n_threads = 2;
    config.pooling = streaming.default_pooling();
    config.normalize = true;
    std::vector<float> eager_out(static_cast<size_t>(streaming.n_embed()));
    std::vector<float> stream_out(eager_out.size());
    double sum_fixture = 0.0;
    double minimum_fixture = 1.0;
    double minimum_eager = 1.0;
    const size_t before_pages = streaming.diagnostic_resident_pages();
    const size_t tested_samples = std::min<size_t>(samples.size(), 8);
    for (size_t sample_index = 0; sample_index < tested_samples; ++sample_index) {
        const auto & sample = samples[sample_index];
        eager.embed(eager_context, sample.text, config, eager_out.data());
        streaming.embed(streaming_context, sample.text, config, stream_out.data());
        const double vs_eager = cosine(stream_out, eager_out);
        const double vs_fixture = cosine(stream_out, sample.embedding);
        minimum_eager = std::min(minimum_eager, vs_eager);
        minimum_fixture = std::min(minimum_fixture, vs_fixture);
        sum_fixture += vs_fixture;
        if (vs_eager < 0.999999) {
            throw std::runtime_error(std::string(item.label) + " streaming/eager parity failed");
        }
        if (item.enforce && vs_fixture < item.sample_gate) {
            throw std::runtime_error(std::string(item.label) + " streaming/PyTorch sample gate failed");
        }
    }
    const double mean_fixture = sum_fixture / tested_samples;
    if (item.enforce && mean_fixture < item.mean_gate) {
        throw std::runtime_error(std::string(item.label) + " streaming/PyTorch mean gate failed");
    }

    // One public batch call must execute one copy of each phase per
    // sub-batch, not per item. Deliberately vary lengths so attention and
    // pooling masks are exercised as well as stable output scattering.
    const std::vector<std::string> batch_texts = {
        "x",
        "a somewhat longer streaming batch sentence",
        "한국어와 English batch",
        "same",
        "same",
    };
    std::vector<float> eager_batch(
        batch_texts.size() * static_cast<size_t>(streaming.n_embed()));
    std::vector<float> stream_batch(eager_batch.size());
    config.max_batch = 3;
    const auto before_batch_diag = streaming.diagnostics(streaming_context);
    eager.embed_batch(eager_context, batch_texts, config, eager_batch.data());
    streaming.embed_batch(streaming_context, batch_texts, config, stream_batch.data());
    for (size_t i = 0; i < batch_texts.size(); ++i) {
        const size_t offset = i * static_cast<size_t>(streaming.n_embed());
        std::vector<float> expected(
            eager_batch.begin() + static_cast<std::ptrdiff_t>(offset),
            eager_batch.begin() + static_cast<std::ptrdiff_t>(offset + streaming.n_embed()));
        std::vector<float> actual(
            stream_batch.begin() + static_cast<std::ptrdiff_t>(offset),
            stream_batch.begin() + static_cast<std::ptrdiff_t>(offset + streaming.n_embed()));
        if (cosine(expected, actual) < 0.999999) {
            throw std::runtime_error(std::string(item.label) + " batched streaming/eager parity failed");
        }
    }
    const auto after_batch_diag = streaming.diagnostics(streaming_context);
    const auto batch_part = streaming.partition_info();
    constexpr uint64_t expected_subbatches = 2;
    if (after_batch_diag.phase_graph_computes - before_batch_diag.phase_graph_computes !=
            expected_subbatches * batch_part.groups_per_sentence ||
        after_batch_diag.batches_processed - before_batch_diag.batches_processed !=
            expected_subbatches ||
        after_batch_diag.items_processed - before_batch_diag.items_processed !=
            batch_texts.size() ||
        after_batch_diag.padding_tokens_processed <=
            before_batch_diag.padding_tokens_processed) {
        throw std::runtime_error(std::string(item.label) + " batch diagnostics scale per item");
    }

    // Freeze the serial residency peak before failure-injection or concurrent
    // contexts add model-wide diagnostics. In particular, token_advised_bytes
    // participates in the page-slack bound below and must describe only the
    // canonical serial path.
    const auto serial_residency =
        streaming.diagnostics(streaming_context).residency;

    // A graph metadata failure must happen before its corresponding residency
    // lease is published. Verify both the embedding boundary (countdown 0) and
    // the first layer-group boundary (countdown 1), then reuse the same context
    // to prove the failure did not leave stale slots or a poisoned coordinator.
    auto verify_graph_context_failure = [&](uint64_t countdown,
                                            const char * boundary) {
        nanoembed::InternalStreamingContext failure_context;
        std::vector<float> failed_output(eager_batch.size(), 0.0f);
        failure_context.diagnostic_fail_graph_context_after(countdown);
        bool saw_allocation_error = false;
        try {
            streaming.embed_batch(
                failure_context, batch_texts, config, failed_output.data());
        } catch (const nanoembed::AllocationError &) {
            saw_allocation_error = true;
        }
        if (!saw_allocation_error ||
            !std::all_of(failed_output.begin(), failed_output.end(),
                         [](float value) { return std::isnan(value); })) {
            throw std::runtime_error(
                std::string(item.label) + " " + boundary +
                " failure did not return the poisoned-output contract");
        }
        const auto failed_diag = streaming.diagnostics(failure_context);
        if (failed_diag.residency.poisoned ||
            failed_diag.residency.active_leases != 0 ||
            failed_diag.residency.leases_acquired !=
                failed_diag.residency.leases_released ||
            failed_diag.residency.premature_release_attempts != 0) {
            throw std::runtime_error(
                std::string(item.label) + " " + boundary +
                " failure leaked or poisoned a residency lease");
        }

        std::fill(failed_output.begin(), failed_output.end(), 0.0f);
        streaming.embed_batch(
            failure_context, batch_texts, config, failed_output.data());
        for (size_t i = 0; i < batch_texts.size(); ++i) {
            const size_t offset = i * static_cast<size_t>(streaming.n_embed());
            std::vector<float> expected(
                eager_batch.begin() + static_cast<std::ptrdiff_t>(offset),
                eager_batch.begin() +
                    static_cast<std::ptrdiff_t>(offset + streaming.n_embed()));
            std::vector<float> actual(
                failed_output.begin() + static_cast<std::ptrdiff_t>(offset),
                failed_output.begin() +
                    static_cast<std::ptrdiff_t>(offset + streaming.n_embed()));
            if (cosine(expected, actual) < 0.999999) {
                throw std::runtime_error(
                    std::string(item.label) + " " + boundary +
                    " failure left the context unusable");
            }
        }
    };
    verify_graph_context_failure(/*countdown=*/0, "embedding-context");
    verify_graph_context_failure(/*countdown=*/1, "group-context");

    // Two genuinely distinct contexts share only the mapped read-only model
    // and residency coordinator. Repeat concurrently to expose scratch/advice
    // races and require deterministic output.
    nanoembed::InternalStreamingContext context_a;
    nanoembed::InternalStreamingContext context_b;
    std::vector<float> output_a(stream_out.size()), output_b(stream_out.size());
    std::vector<float> concurrency_reference(stream_out.size());
    eager.embed(eager_context, samples[0].text, config, concurrency_reference.data());
    std::mutex error_mutex;
    std::exception_ptr thread_error;
    auto run = [&](nanoembed::InternalStreamingContext & context, std::vector<float> & output) {
        try {
            for (int repetition = 0; repetition < 2; ++repetition) {
                streaming.embed(context, samples[0].text, config, output.data());
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!thread_error) thread_error = std::current_exception();
        }
    };
    std::thread thread_a(run, std::ref(context_a), std::ref(output_a));
    std::thread thread_b(run, std::ref(context_b), std::ref(output_b));
    thread_a.join(); thread_b.join();
    if (thread_error) std::rethrow_exception(thread_error);
    if (cosine(output_a, output_b) < 0.999999999 ||
        cosine(output_a, concurrency_reference) < 0.999999) {
        throw std::runtime_error(std::string(item.label) + " concurrent outputs differ/parity failed");
    }
    const auto diag = streaming.diagnostics(streaming_context);
    const auto part = streaming.partition_info();
    const size_t after_pages = streaming.diagnostic_resident_pages();
    if (diag.residency.poisoned || diag.residency.active_leases != 0 ||
        diag.residency.dontneed_failures != 0 ||
        diag.residency.compute_completions != diag.residency.leases_released ||
        diag.residency.premature_release_attempts != 0) {
        throw std::runtime_error(std::string(item.label) + " residency diagnostics invalid");
    }
    // One graph per resolved group, plus the embedding and final phases. Derived
    // from the model's own partition instead of assuming one graph per layer, so
    // changing the preset does not mean editing this line -- while the shape
    // checks below still pin down what that partition must be.
    if ((item.expected_groups_per_layer != 0 &&
         part.groups_per_layer != item.expected_groups_per_layer) ||
        part.groups_per_layer == 0 ||
        part.groups_per_sentence !=
            2 + part.groups_per_layer * static_cast<uint64_t>(streaming.n_layer()) ||
        diag.phase_graph_computes !=
            (tested_samples + expected_subbatches) * part.groups_per_sentence ||
        diag.batches_processed != tested_samples + expected_subbatches ||
        diag.items_processed != tested_samples + batch_texts.size()) {
        throw std::runtime_error(std::string(item.label) + " phase/partition diagnostics invalid");
    }
    // The number the partitioning exists to lower. common is retained for the
    // model's lifetime, and one group is resident on top of it; the page slack
    // covers ranges rounded out to page boundaries at each end of a group.
    // One group on top of the lifetime-retained common ranges. The slack covers
    // ranges rounded out to page boundaries at each end of a group, and the
    // token-row lease of the embedding phase, which is bounded by the pages the
    // sentence's own token IDs land on.
    const uint64_t page_slack =
        2ULL * 4096ULL * (part.groups_per_layer + 1) + serial_residency.token_advised_bytes;
    if (serial_residency.advised_bytes_high_water >
        serial_residency.common_class_bytes + part.max_group_weight_bytes + page_slack) {
        std::fprintf(stderr,
                     "[streaming] %s peak=%llu common=%llu group=%llu slack=%llu\n",
                     item.label,
                     static_cast<unsigned long long>(serial_residency.advised_bytes_high_water),
                     static_cast<unsigned long long>(serial_residency.common_class_bytes),
                     static_cast<unsigned long long>(part.max_group_weight_bytes),
                     static_cast<unsigned long long>(page_slack));
        throw std::runtime_error(std::string(item.label) + " resident peak exceeds one group");
    }
    std::printf(
        "[streaming] %s samples=%zu min_eager=%.9f min_pytorch=%.9f mean_pytorch=%.9f "
        "willneed=%llu dontneed=%llu lease_hwm=%llu replans=%llu "
        "preset=%s groups/layer=%llu peak_group_kib=%llu advised_peak_kib=%llu "
        "slot_kib=%llu mincore_before=%zu mincore_after=%zu\n",
        item.label, tested_samples, minimum_eager, minimum_fixture, mean_fixture,
        static_cast<unsigned long long>(diag.residency.willneed_calls),
        static_cast<unsigned long long>(diag.residency.dontneed_calls),
        static_cast<unsigned long long>(diag.residency.lease_high_water),
        static_cast<unsigned long long>(diag.graph_replans),
        part.preset.c_str(),
        static_cast<unsigned long long>(part.groups_per_layer),
        static_cast<unsigned long long>(part.max_group_weight_bytes / 1024),
        static_cast<unsigned long long>(serial_residency.advised_bytes_high_water / 1024),
        static_cast<unsigned long long>(diag.slot_resident_bytes / 1024),
        before_pages, after_pages);
    return 0;
}

#endif

} // namespace

int main() {
#if !defined(__linux__)
    std::printf("SKIP: internal streaming integration requires Linux\n");
    return 0;
#else
    // Read from the environment because the preset is what this test is
    // parameterized over. Both architectures give one group under the default
    // "layer" preset and two under "attn-ffn"; set 0 for presets where they
    // differ.
    uint64_t expected_groups = 1;
    if (const char * env = std::getenv("NANOEMBED_STREAMING_EXPECTED_GROUPS")) {
        expected_groups = std::strtoull(env, nullptr, 10);
    }
    const Case cases[] = {
        {"bert-f16", "NANOEMBED_TEST_MODEL", "NANOEMBED_GOLDEN_FIXTURE", 0.9999, 0.99999, true, expected_groups},
        {"harrier-f32", "NANOEMBED_TEST_MODEL_GEMMA3", "NANOEMBED_GOLDEN_FIXTURE_GEMMA3", 0.9999, 0.99999, true, expected_groups},
        {"harrier-q8", "NANOEMBED_TEST_MODEL_GEMMA3_Q8", "NANOEMBED_GOLDEN_FIXTURE_GEMMA3", 0.9985, 0.9995, true, expected_groups},
        {"harrier-q4-report-only", "NANOEMBED_TEST_MODEL_GEMMA3_Q4", "NANOEMBED_GOLDEN_FIXTURE_GEMMA3", 0.0, 0.0, false, expected_groups},
    };
    try {
        for (const auto & item : cases) if (run_case(item) != 0) return 1;
        std::printf("streaming_integration_test: ok\n");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "streaming_integration_test: FAIL: %s\n", error.what());
        return 1;
    }
#endif
}
