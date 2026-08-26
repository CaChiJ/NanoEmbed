// End-to-end parity against an integrity-checked PyTorch/sentence-transformers
// fixture. Existing cosine gates are unchanged; richer error metrics are
// report-only and written after inference, outside performance benchmark paths.

#include "golden_accuracy.h"
#include "nanoembed/nanoembed.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using golden_accuracy::DistributionSummary;
using golden_accuracy::ProvenanceInfo;
using golden_accuracy::QualityDirection;
using golden_accuracy::VectorMetrics;

struct GoldenSample {
    std::string text;
    std::vector<float> embedding;
};

struct SampleAccuracy {
    size_t index = 0;
    std::string text;
    VectorMetrics pytorch_reference_error;
    bool has_quantization_loss = false;
    VectorMetrics quantization_loss;
    bool has_execution_mode_error = false;
    VectorMetrics execution_mode_error;
};

struct AccuracyReport {
    std::string label;
    std::string family;
    std::string status;
    std::string error;
    std::string comparison_kind;
    std::string subject_precision;
    std::string quantization_baseline_label;
    std::string execution_mode;
    std::string execution_baseline_label;
    ProvenanceInfo provenance;
    std::vector<SampleAccuracy> samples;
    std::vector<std::vector<float>> outputs;
    size_t cosine_gate_failures = 0;
    size_t non_finite_failures = 0;
    double per_sample_cosine_gate = 0.0;
    double mean_cosine_gate = 0.0;
    double measured_mean_cosine = 0.0;
    bool enforce_cosine_gate = true;
    bool strict_mode_context_creation_succeeded = false;
};

struct ModelUnderTest {
    const char * label;
    const char * family;
    const char * model_env;
    const char * fixture_env;
    const char * comparison_kind;
    const char * subject_precision;
    const char * quantization_baseline_label;
    float per_sample_tol;
    float mean_tol;
    bool enforce_cosine_gate;
    bool streaming;
};

std::vector<GoldenSample> load_negd(const std::string & path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open NEGD: " + path);

    auto must_read = [&](void * dst, std::streamsize n) {
        f.read(static_cast<char *>(dst), n);
        if (f.gcount() != n) {
            throw std::runtime_error("NEGD truncated or unreadable: " + path);
        }
    };

    char magic[4];
    must_read(magic, 4);
    if (std::string(magic, 4) != "NEGD") throw std::runtime_error("bad NEGD magic");

    uint32_t version = 0;
    uint32_t n_samples = 0;
    uint32_t n_embed = 0;
    must_read(&version, 4);
    must_read(&n_samples, 4);
    must_read(&n_embed, 4);
    if (version != 1) throw std::runtime_error("unsupported NEGD version");
    if (n_samples == 0 || n_embed == 0) throw std::runtime_error("empty NEGD dimensions");

    std::vector<GoldenSample> samples;
    samples.reserve(n_samples);
    for (uint32_t i = 0; i < n_samples; ++i) {
        GoldenSample sample;
        uint32_t text_len = 0;
        must_read(&text_len, 4);
        sample.text.resize(text_len);
        must_read(sample.text.data(), text_len);
        sample.embedding.resize(n_embed);
        must_read(
            sample.embedding.data(),
            static_cast<std::streamsize>(n_embed)
                * static_cast<std::streamsize>(sizeof(float)));
        samples.push_back(std::move(sample));
    }
    if (f.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("NEGD has unexpected trailing bytes: " + path);
    }
    return samples;
}

std::vector<double> metric_values(
    const std::vector<SampleAccuracy> & samples,
    double VectorMetrics::* member,
    bool quantization) {
    std::vector<double> values;
    values.reserve(samples.size());
    for (const SampleAccuracy & sample : samples) {
        if (quantization && !sample.has_quantization_loss) continue;
        const VectorMetrics & metrics = quantization
            ? sample.quantization_loss
            : sample.pytorch_reference_error;
        values.push_back(metrics.*member);
    }
    return values;
}

void write_summary(
    std::ostream & output,
    const char * name,
    const std::vector<double> & values,
    QualityDirection direction,
    bool last) {
    const DistributionSummary summary = golden_accuracy::summarize(values, direction);
    output << "          \"" << name << "\": {\n"
           << "            \"count\": " << summary.count << ",\n"
           << "            \"quality_direction\": \""
           << golden_accuracy::quality_direction_name(direction) << "\",\n"
           << "            \"mean\": " << summary.mean << ",\n"
           << "            \"worst\": " << summary.worst << ",\n"
           << "            \"p50\": " << summary.p50 << ",\n"
           << "            \"p90\": " << summary.p90 << ",\n"
           << "            \"p95\": " << summary.p95 << ",\n"
           << "            \"p99\": " << summary.p99 << "\n"
           << "          }" << (last ? "\n" : ",\n");
}

void write_aggregate(
    std::ostream & output,
    const std::vector<SampleAccuracy> & samples,
    bool quantization) {
    write_summary(output, "cosine_similarity",
        metric_values(samples, &VectorMetrics::cosine, quantization),
        QualityDirection::higher_is_better, false);
    write_summary(output, "maximum_absolute_error",
        metric_values(samples, &VectorMetrics::max_absolute_error, quantization),
        QualityDirection::lower_is_better, false);
    write_summary(output, "mean_absolute_error",
        metric_values(samples, &VectorMetrics::mean_absolute_error, quantization),
        QualityDirection::lower_is_better, false);
    write_summary(output, "rmse",
        metric_values(samples, &VectorMetrics::rmse, quantization),
        QualityDirection::lower_is_better, false);
    write_summary(output, "output_norm_difference",
        metric_values(samples, &VectorMetrics::norm_difference, quantization),
        QualityDirection::lower_is_better, false);
    write_summary(output, "relative_output_norm_difference",
        metric_values(samples, &VectorMetrics::relative_norm_difference, quantization),
        QualityDirection::lower_is_better, true);
}

void write_vector_metrics(std::ostream & output, const VectorMetrics & metrics, size_t indent) {
    const std::string spaces(indent, ' ');
    output << "{\n"
           << spaces << "  \"cosine_similarity\": " << metrics.cosine << ",\n"
           << spaces << "  \"maximum_absolute_error\": "
           << metrics.max_absolute_error << ",\n"
           << spaces << "  \"mean_absolute_error\": "
           << metrics.mean_absolute_error << ",\n"
           << spaces << "  \"rmse\": " << metrics.rmse << ",\n"
           << spaces << "  \"output_norm\": " << metrics.output_norm << ",\n"
           << spaces << "  \"reference_norm\": " << metrics.reference_norm << ",\n"
           << spaces << "  \"output_norm_difference\": "
           << metrics.norm_difference << ",\n"
           << spaces << "  \"relative_output_norm_difference\": "
           << metrics.relative_norm_difference << "\n"
           << spaces << "}";
}

void write_accuracy_json(const std::string & path, const std::vector<AccuracyReport> & reports) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create accuracy JSON: " + path);
    output << std::setprecision(17);
    output << "{\n"
           << "  \"schema_version\": 1,\n"
           << "  \"measurement_scope\": "
              "\"accuracy_only; generated outside performance benchmark timing\",\n"
           << "  \"reference_contract\": {\n"
           << "    \"framework\": \"sentence-transformers/pytorch\",\n"
           << "    \"device\": \"cpu\",\n"
           << "    \"dtype\": \"float32\",\n"
           << "    \"new_metrics_gate_status\": \"report_only\"\n"
           << "  },\n"
           << "  \"execution_mode_comparison_contract\": {\n"
           << "    \"scope\": \"accuracy_only_outside_performance_timing\",\n"
           << "    \"binary\": \"same nanoembed_golden_test process\",\n"
           << "    \"controlled_dimensions\": \"same model path, fixture corpus, pooling, normalization, context limits and thread policy; only use_streaming differs\",\n"
           << "    \"gate_status\": \"report_only; historical PyTorch cosine gates unchanged\"\n"
           << "  },\n"
           << "  \"percentile_method\": "
              "\"ascending linear interpolation at (n-1)*q; direction is stated per metric\",\n"
           << "  \"comparisons\": [\n";

    for (size_t report_index = 0; report_index < reports.size(); ++report_index) {
        const AccuracyReport & report = reports[report_index];
        output << "    {\n"
               << "      \"label\": \"" << golden_accuracy::json_escape(report.label) << "\",\n"
               << "      \"family\": \"" << golden_accuracy::json_escape(report.family) << "\",\n"
               << "      \"status\": \"" << golden_accuracy::json_escape(report.status) << "\",\n"
               << "      \"comparison_kind\": \""
               << golden_accuracy::json_escape(report.comparison_kind) << "\",\n"
               << "      \"subject_precision\": \""
               << golden_accuracy::json_escape(report.subject_precision) << "\",\n";
        output << "      \"execution_mode\": \"" << report.execution_mode << "\",\n"
               << "      \"execution_mode_resolution\": {\n"
               << "        \"contract_version\": 1,\n"
               << "        \"strict_no_fallback\": true,\n"
               << "        \"context_creation_succeeded_with_exact_request\": "
               << (report.strict_mode_context_creation_succeeded ? "true" : "false")
               << "\n"
               << "      },\n";
        if (report.error.empty()) {
            output << "      \"error\": null,\n";
        } else {
            output << "      \"error\": \"" << golden_accuracy::json_escape(report.error)
                   << "\",\n";
        }
        if (report.provenance.status.empty()) {
            output << "      \"provenance\": null,\n";
        } else {
            output << "      \"provenance\": {\n"
                   << "        \"status\": \"" << report.provenance.status << "\",\n"
                   << "        \"model_id\": \""
                   << golden_accuracy::json_escape(report.provenance.model_id) << "\",\n"
                   << "        \"resolved_revision\": ";
            if (report.provenance.resolved_revision.empty()) output << "null,\n";
            else output << "\"" << report.provenance.resolved_revision << "\",\n";
            output << "        \"fixture_sha256\": \""
                   << report.provenance.fixture_sha256 << "\",\n"
                   << "        \"manifest_sha256\": \""
                   << report.provenance.manifest_sha256 << "\",\n"
                   << "        \"manifest_path\": \""
                   << golden_accuracy::json_escape(report.provenance.manifest_path) << "\"\n"
                   << "      },\n";
        }
        output << "      \"cosine_gates\": {\n"
               << "        \"scope\": \"pytorch_reference_error.cosine_similarity\",\n"
               << "        \"enforcement\": \""
               << (report.enforce_cosine_gate ? "active" : "report_only") << "\",\n"
               << "        \"per_sample_minimum\": " << report.per_sample_cosine_gate << ",\n"
               << "        \"mean_minimum\": " << report.mean_cosine_gate << ",\n"
               << "        \"measured_mean\": ";
        if (report.samples.empty()) output << "null,\n";
        else output << report.measured_mean_cosine << ",\n";
        output << "        \"failed_samples\": " << report.cosine_gate_failures << ",\n"
               << "        \"non_finite_samples\": " << report.non_finite_failures << "\n"
               << "      },\n";

        if (report.samples.empty()) {
            output << "      \"pytorch_reference_error\": null,\n"
                   << "      \"quantization_loss_vs_native_f32\": null,\n"
                   << "      \"samples\": []\n";
        } else {
            output << "      \"pytorch_reference_error\": {\n"
                   << "        \"reference_precision\": \"float32\",\n"
                   << "        \"aggregate\": {\n";
            write_aggregate(output, report.samples, false);
            output << "        }\n"
                   << "      },\n";

            const bool has_quantization = std::any_of(
                report.samples.begin(), report.samples.end(),
                [](const SampleAccuracy & sample) { return sample.has_quantization_loss; });
            if (has_quantization) {
                output << "      \"quantization_loss_vs_native_f32\": {\n"
                       << "        \"baseline_label\": \""
                       << golden_accuracy::json_escape(report.quantization_baseline_label)
                       << "\",\n"
                       << "        \"gate_status\": \"report_only\",\n"
                       << "        \"aggregate\": {\n";
                write_aggregate(output, report.samples, true);
                output << "        }\n"
                       << "      },\n";
            } else {
                output << "      \"quantization_loss_vs_native_f32\": null,\n";
            }

            const bool has_execution_mode_error = std::any_of(
                report.samples.begin(), report.samples.end(),
                [](const SampleAccuracy & sample) {
                    return sample.has_execution_mode_error;
                });
            if (has_execution_mode_error) {
                output << "      \"execution_mode_error_vs_eager\": {\n"
                       << "        \"baseline_label\": \""
                       << golden_accuracy::json_escape(report.execution_baseline_label)
                       << "\",\n"
                       << "        \"gate_status\": \"report_only\",\n"
                       << "        \"aggregate\": {\n";
                std::vector<SampleAccuracy> execution_samples = report.samples;
                for (SampleAccuracy & sample : execution_samples) {
                    sample.quantization_loss = sample.execution_mode_error;
                    sample.has_quantization_loss = sample.has_execution_mode_error;
                }
                write_aggregate(output, execution_samples, true);
                output << "        }\n"
                       << "      },\n";
            } else {
                output << "      \"execution_mode_error_vs_eager\": null,\n";
            }

            output << "      \"samples\": [\n";
            for (size_t sample_index = 0; sample_index < report.samples.size(); ++sample_index) {
                const SampleAccuracy & sample = report.samples[sample_index];
                output << "        {\n"
                       << "          \"index\": " << sample.index << ",\n"
                       << "          \"text\": \""
                       << golden_accuracy::json_escape(sample.text) << "\",\n"
                       << "          \"pytorch_reference_error\": ";
                write_vector_metrics(output, sample.pytorch_reference_error, 10);
                output << ",\n"
                       << "          \"quantization_loss_vs_native_f32\": ";
                if (sample.has_quantization_loss) {
                    write_vector_metrics(output, sample.quantization_loss, 10);
                    output << "\n";
                } else {
                    output << "null\n";
                }
                output << ",\n"
                       << "          \"execution_mode_error_vs_eager\": ";
                if (sample.has_execution_mode_error) {
                    write_vector_metrics(output, sample.execution_mode_error, 10);
                    output << "\n";
                } else {
                    output << "null\n";
                }
                output << "        }" << (sample_index + 1 == report.samples.size() ? "\n" : ",\n");
            }
            output << "      ]\n";
        }
        output << "    }" << (report_index + 1 == reports.size() ? "\n" : ",\n");
    }
    output << "  ]\n}\n";
    if (!output) throw std::runtime_error("cannot write accuracy JSON: " + path);
}

// Returns 0 on success or skip, 1 on failure. The cosine gates below are the
// historical gates; new metrics never participate in this return value.
int run_model(
    const ModelUnderTest & model_under_test,
    const std::vector<std::vector<float>> * quantization_baseline,
    const std::vector<std::vector<float>> * eager_execution_baseline,
    AccuracyReport & report) {
    report.label = model_under_test.label;
    report.family = model_under_test.family;
    report.comparison_kind = model_under_test.comparison_kind;
    report.subject_precision = model_under_test.subject_precision;
    report.quantization_baseline_label = model_under_test.quantization_baseline_label;
    report.per_sample_cosine_gate = model_under_test.per_sample_tol;
    report.mean_cosine_gate = model_under_test.mean_tol;
    report.enforce_cosine_gate = model_under_test.enforce_cosine_gate;
    report.execution_mode = model_under_test.streaming ? "streaming" : "eager";
    report.execution_baseline_label = eager_execution_baseline
        ? std::string(model_under_test.label).substr(
              0, std::string(model_under_test.label).find("-streaming"))
        : "";

    const char * model_path = std::getenv(model_under_test.model_env);
    const char * golden_path = std::getenv(model_under_test.fixture_env);
    if (!model_path || !golden_path) {
        report.status = "skipped";
        report.error = std::string("missing ") + model_under_test.model_env + "/"
            + model_under_test.fixture_env;
        std::fprintf(
            stderr, "[golden_test] skip %s: %s/%s not set\n",
            model_under_test.label, model_under_test.model_env, model_under_test.fixture_env);
        return 0;
    }

    std::vector<GoldenSample> samples;
    try {
        report.provenance = golden_accuracy::verify_fixture_provenance(golden_path);
        samples = load_negd(golden_path);
    } catch (const std::exception & exception) {
        report.status = "failed";
        report.error = exception.what();
        std::fprintf(stderr, "[golden_test] %s: %s\n", model_under_test.label, exception.what());
        return 1;
    }
    if (report.provenance.status == "legacy_unverified") {
        std::fprintf(
            stderr,
            "[golden_test] warning %s: fixture integrity verified, but generation "
            "provenance is legacy_unverified; regenerate with an exact HF revision\n",
            model_under_test.label);
    }

    nanoembed_model * model = nanoembed_load_model(model_path);
    if (!model) {
        report.status = "failed";
        report.error = nanoembed_last_error();
        std::fprintf(
            stderr, "load_model failed (%s): %s\n",
            model_under_test.label, nanoembed_last_error());
        return 1;
    }

    nanoembed_context_params params = nanoembed_context_default_params();
    params.pooling = NANOEMBED_POOL_MODEL_DEFAULT;
    params.normalize = 1;
    params.use_streaming = model_under_test.streaming ? 1 : 0;
    nanoembed_context * context = nanoembed_new_context(model, params);
    if (!context) {
        report.status = "failed";
        report.error = nanoembed_last_error();
        std::fprintf(
            stderr, "new_context failed (%s): %s\n",
            model_under_test.label, nanoembed_last_error());
        nanoembed_free_model(model);
        return 1;
    }
    report.strict_mode_context_creation_succeeded = true;

    const int embedding_dimension = nanoembed_n_embed(model);
    if (embedding_dimension <= 0
        || static_cast<int>(samples[0].embedding.size()) != embedding_dimension) {
        report.status = "failed";
        report.error = "model/fixture embedding dimension mismatch";
        std::fprintf(
            stderr, "[golden_test] %s dim mismatch: model=%d fixture=%zu\n",
            model_under_test.label, embedding_dimension, samples[0].embedding.size());
        nanoembed_free_context(context);
        nanoembed_free_model(model);
        return 1;
    }
    if (quantization_baseline && quantization_baseline->size() != samples.size()) {
        report.status = "failed";
        report.error = "quantized/F32 baseline sample count mismatch";
        nanoembed_free_context(context);
        nanoembed_free_model(model);
        return 1;
    }
    if (eager_execution_baseline &&
        eager_execution_baseline->size() != samples.size()) {
        report.status = "failed";
        report.error = "streaming/eager baseline sample count mismatch";
        nanoembed_free_context(context);
        nanoembed_free_model(model);
        return 1;
    }

    std::vector<float> output(static_cast<size_t>(embedding_dimension));
    double sum_cosine = 0.0;
    report.samples.reserve(samples.size());
    report.outputs.reserve(samples.size());
    for (size_t i = 0; i < samples.size(); ++i) {
        const GoldenSample & sample = samples[i];
        const int rc = nanoembed_embed(context, sample.text.c_str(), output.data());
        if (rc != NANOEMBED_OK) {
            report.status = "failed";
            report.error = nanoembed_last_error();
            std::fprintf(
                stderr, "embed failed (%s) sample[%zu]: %s\n",
                model_under_test.label, i, nanoembed_last_error());
            nanoembed_free_context(context);
            nanoembed_free_model(model);
            return 1;
        }

        SampleAccuracy accuracy;
        accuracy.index = i;
        accuracy.text = sample.text;
        accuracy.pytorch_reference_error = golden_accuracy::compare_vectors(
            output.data(), sample.embedding.data(), static_cast<size_t>(embedding_dimension));
        if (quantization_baseline) {
            const std::vector<float> & baseline = quantization_baseline->at(i);
            if (baseline.size() != output.size()) {
                report.status = "failed";
                report.error = "quantized/F32 baseline embedding dimension mismatch";
                nanoembed_free_context(context);
                nanoembed_free_model(model);
                return 1;
            }
            accuracy.has_quantization_loss = true;
            accuracy.quantization_loss = golden_accuracy::compare_vectors(
                output.data(), baseline.data(), output.size());
        }
        if (eager_execution_baseline) {
            const std::vector<float> & baseline = eager_execution_baseline->at(i);
            if (baseline.size() != output.size()) {
                report.status = "failed";
                report.error = "streaming/eager baseline embedding dimension mismatch";
                nanoembed_free_context(context);
                nanoembed_free_model(model);
                return 1;
            }
            accuracy.has_execution_mode_error = true;
            accuracy.execution_mode_error = golden_accuracy::compare_vectors(
                output.data(), baseline.data(), output.size());
        }
        // Checked before, and independently of, the cosine gate. Every error
        // metric is blind to NaN -- std::max(x, NaN) is x, and NaN < tol is
        // false -- so a NaN-poisoned vector scores zero error and clears the
        // cosine gate by failing its comparison. This is the only assertion
        // that sees it, and it applies even to report-only models.
        if (!accuracy.pytorch_reference_error.all_finite ||
            (accuracy.has_execution_mode_error &&
             !accuracy.execution_mode_error.all_finite)) {
            ++report.non_finite_failures;
            if (report.non_finite_failures <= 3) {
                std::fprintf(stderr, "FAIL %s sample[%zu] produced a non-finite embedding\n",
                             model_under_test.label, i);
            }
        }
        sum_cosine += accuracy.pytorch_reference_error.cosine;
        if (model_under_test.enforce_cosine_gate
            && accuracy.pytorch_reference_error.cosine < model_under_test.per_sample_tol) {
            ++report.cosine_gate_failures;
            if (report.cosine_gate_failures <= 3) {
                std::fprintf(
                    stderr,
                    "FAIL %s sample[%zu] cosine=%.6f (tol=%.4f) text=\"%.80s\"\n",
                    model_under_test.label,
                    i,
                    accuracy.pytorch_reference_error.cosine,
                    static_cast<double>(model_under_test.per_sample_tol),
                    sample.text.c_str());
            }
        }
        report.samples.push_back(std::move(accuracy));
        report.outputs.push_back(output);
    }

    report.measured_mean_cosine = sum_cosine / static_cast<double>(samples.size());
    const auto cosine_values = metric_values(
        report.samples, &VectorMetrics::cosine, false);
    const double minimum_cosine = *std::min_element(cosine_values.begin(), cosine_values.end());
    std::printf(
        "[golden_test] %s: %zu/%zu samples vs sentence-transformers, "
        "min cosine=%.6f mean cosine=%.6f\n",
        model_under_test.label,
        samples.size() - report.cosine_gate_failures,
        samples.size(),
        minimum_cosine,
        report.measured_mean_cosine);

    nanoembed_free_context(context);
    nanoembed_free_model(model);
    // A non-finite embedding fails every model, report-only ones included:
    // "we do not gate this model's accuracy" is not "we accept NaN from it".
    const bool passed = report.non_finite_failures == 0
        && (!model_under_test.enforce_cosine_gate
            || (report.cosine_gate_failures == 0
                && report.measured_mean_cosine >= model_under_test.mean_tol));
    report.status = passed ? "passed" : "failed";
    return passed ? 0 : 1;
}

const ModelUnderTest kModels[] = {
    // Historical thresholds stay unchanged. The BERT GGUF stores F16 weights
    // but computes in F32, so precision is explicit rather than hidden.
    {"bert", "bert", "NANOEMBED_TEST_MODEL", "NANOEMBED_GOLDEN_FIXTURE",
     "implementation_agreement", "f16_weights_f32_compute", "", 0.9999f, 0.99999f, true, false},
    {"gemma3-f32", "gemma3", "NANOEMBED_TEST_MODEL_GEMMA3",
     "NANOEMBED_GOLDEN_FIXTURE_GEMMA3", "f32_implementation_agreement",
     "f32_weights_f32_compute", "", 0.9999f, 0.99999f, true, false},

    // This gate remains against the PyTorch F32 oracle. Separately, the report
    // computes report-only quantization loss against gemma3-f32 native output.
    {"gemma3-q8", "gemma3", "NANOEMBED_TEST_MODEL_GEMMA3_Q8",
     "NANOEMBED_GOLDEN_FIXTURE_GEMMA3", "quantization_loss",
     "q8_0_weights_f32_compute", "gemma3-f32", 0.9985f, 0.9995f, true, false},

    // Q4 did not have a historical CI gate. It is measured against both the
    // PyTorch fixture and native F32 output, but remains report-only.
    {"gemma3-q4", "gemma3", "NANOEMBED_TEST_MODEL_GEMMA3_Q4",
     "NANOEMBED_GOLDEN_FIXTURE_GEMMA3", "quantization_loss",
     "q4_k_weights_f32_compute", "gemma3-f32", 0.0f, 0.0f, false, false},
};

#if defined(__linux__)
const ModelUnderTest kStreamingModels[] = {
    {"bert-streaming", "bert", "NANOEMBED_TEST_MODEL", "NANOEMBED_GOLDEN_FIXTURE",
     "implementation_agreement", "f16_weights_f32_compute", "", 0.9999f, 0.99999f, true, true},
    {"gemma3-f32-streaming", "gemma3", "NANOEMBED_TEST_MODEL_GEMMA3",
     "NANOEMBED_GOLDEN_FIXTURE_GEMMA3", "f32_implementation_agreement",
     "f32_weights_f32_compute", "", 0.9999f, 0.99999f, true, true},
    {"gemma3-q8-streaming", "gemma3", "NANOEMBED_TEST_MODEL_GEMMA3_Q8",
     "NANOEMBED_GOLDEN_FIXTURE_GEMMA3", "quantization_loss",
     "q8_0_weights_f32_compute", "gemma3-f32-streaming", 0.9985f, 0.9995f, true, true},
    {"gemma3-q4-streaming", "gemma3", "NANOEMBED_TEST_MODEL_GEMMA3_Q4",
     "NANOEMBED_GOLDEN_FIXTURE_GEMMA3", "quantization_loss",
     "q4_k_weights_f32_compute", "gemma3-f32-streaming", 0.0f, 0.0f, false, true},
};
#endif

} // namespace

int main() {
    int return_code = 0;
    std::vector<AccuracyReport> reports;
#if defined(__linux__)
    reports.reserve(sizeof(kModels) / sizeof(kModels[0]) +
                    sizeof(kStreamingModels) / sizeof(kStreamingModels[0]));
#else
    reports.reserve(sizeof(kModels) / sizeof(kModels[0]));
#endif

    const std::vector<std::vector<float>> * gemma3_f32_outputs = nullptr;
    for (const ModelUnderTest & model : kModels) {
        reports.emplace_back();
        const std::vector<std::vector<float>> * baseline = nullptr;
        if (std::string(model.comparison_kind) == "quantization_loss") {
            baseline = gemma3_f32_outputs;
            // If the F32 model was intentionally unavailable, keep the existing
            // PyTorch gate runnable but report why a true quantization delta is
            // unavailable instead of substituting a different reference.
        }
        if (run_model(model, baseline, nullptr, reports.back()) != 0) return_code = 1;
        if (std::string(model.label) == "gemma3-f32"
            && reports.back().status == "passed") {
            gemma3_f32_outputs = &reports.back().outputs;
        }
    }


#if defined(__linux__)
    const size_t eager_report_count = reports.size();
    const std::vector<std::vector<float>> * streaming_f32_outputs = nullptr;
    for (size_t i = 0; i < sizeof(kStreamingModels) / sizeof(kStreamingModels[0]); ++i) {
        const ModelUnderTest & model = kStreamingModels[i];
        const std::vector<std::vector<float>> * quantization_baseline = nullptr;
        if (std::string(model.comparison_kind) == "quantization_loss") {
            quantization_baseline = streaming_f32_outputs;
        }
        const std::vector<std::vector<float>> * eager_baseline = nullptr;
        if (i < eager_report_count && reports[i].status == "passed") {
            eager_baseline = &reports[i].outputs;
        }
        reports.emplace_back();
        if (run_model(model, quantization_baseline, eager_baseline, reports.back()) != 0) {
            return_code = 1;
        }
        if (std::string(model.label) == "gemma3-f32-streaming"
            && reports.back().status == "passed") {
            streaming_f32_outputs = &reports.back().outputs;
        }
    }
#endif

    const char * accuracy_json = std::getenv("NANOEMBED_ACCURACY_JSON");
    if (accuracy_json && *accuracy_json) {
        try {
            write_accuracy_json(accuracy_json, reports);
            std::printf("[golden_test] accuracy JSON: %s\n", accuracy_json);
        } catch (const std::exception & exception) {
            std::fprintf(stderr, "[golden_test] accuracy JSON failed: %s\n", exception.what());
            return_code = 1;
        }
    }
    std::printf("golden_test: %s\n", return_code == 0 ? "ok" : "FAIL");
    return return_code;
}
