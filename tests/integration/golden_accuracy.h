#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace golden_accuracy {

struct VectorMetrics {
    double cosine              = 0.0;
    double max_absolute_error  = 0.0;
    double mean_absolute_error = 0.0;
    double rmse                = 0.0;
    double output_norm         = 0.0;
    double reference_norm      = 0.0;
    double norm_difference     = 0.0;
    double relative_norm_difference = 0.0;
    // Every error metric above is blind to NaN: std::max(x, NaN) returns x, and
    // any comparison against a threshold is false. A NaN-poisoned output would
    // otherwise score a perfect max_absolute_error of 0 and clear a cosine gate
    // by failing the "< tolerance" test. Callers must check this first.
    bool   all_finite          = true;
    // Raw bit equality of the two float arrays. Stricter than a zero error --
    // it separates +0.0 from -0.0 -- which is what an identity claim wants.
    bool   bitwise_identical   = false;
};

enum class QualityDirection {
    higher_is_better,
    lower_is_better,
};

struct DistributionSummary {
    size_t count = 0;
    double mean  = 0.0;
    double worst = 0.0;
    double p50   = 0.0;
    double p90   = 0.0;
    double p95   = 0.0;
    double p99   = 0.0;
    QualityDirection direction = QualityDirection::lower_is_better;
};

struct ProvenanceInfo {
    std::string manifest_path;
    std::string manifest_sha256;
    std::string fixture_sha256;
    std::string status;
    std::string model_id;
    std::string resolved_revision;
};

VectorMetrics compare_vectors(const float * output, const float * reference, size_t count);
double percentile(std::vector<double> values, double quantile);
DistributionSummary summarize(
    const std::vector<double> & values, QualityDirection direction);

std::string sha256_bytes(const void * data, size_t size);
std::string sha256_file(const std::string & path);

// Verifies the companion .provenance.sha256 entries for both the fixture and
// JSON manifest, then validates the manifest's internal fixture hash and
// required provenance fields. Legacy fixtures remain loadable only when their
// manifest explicitly says legacy_unverified.
ProvenanceInfo verify_fixture_provenance(const std::string & fixture_path);

std::string json_escape(const std::string & value);
const char * quality_direction_name(QualityDirection direction);

} // namespace golden_accuracy
