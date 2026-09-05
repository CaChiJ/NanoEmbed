#include "golden_accuracy.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

void require_near(double actual, double expected, double tolerance, const char * message) {
    if (std::abs(actual - expected) > tolerance) {
        std::fprintf(stderr, "%s: actual=%.12g expected=%.12g\n", message, actual, expected);
        throw std::runtime_error(message);
    }
}

void test_sha256() {
    const std::string abc = "abc";
    require(
        golden_accuracy::sha256_bytes(abc.data(), abc.size())
            == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "SHA-256 known vector mismatch");
    require(
        golden_accuracy::sha256_bytes(nullptr, 0)
            == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "SHA-256 empty vector mismatch");
}

void test_vector_metrics() {
    const float output[] = {0.0f, 4.0f};
    const float reference[] = {3.0f, 4.0f};
    const auto metrics = golden_accuracy::compare_vectors(output, reference, 2);
    require_near(metrics.cosine, 0.8, 1e-12, "cosine");
    require_near(metrics.max_absolute_error, 3.0, 1e-12, "max absolute error");
    require_near(metrics.mean_absolute_error, 1.5, 1e-12, "mean absolute error");
    require_near(metrics.rmse, std::sqrt(4.5), 1e-12, "RMSE");
    require_near(metrics.output_norm, 4.0, 1e-12, "output norm");
    require_near(metrics.reference_norm, 5.0, 1e-12, "reference norm");
    require_near(metrics.norm_difference, 1.0, 1e-12, "norm difference");
    require_near(metrics.relative_norm_difference, 0.2, 1e-12, "relative norm difference");
}

void test_summary_direction_and_percentiles() {
    const std::vector<double> values = {0.0, 1.0, 2.0, 3.0, 4.0};
    const auto errors = golden_accuracy::summarize(
        values, golden_accuracy::QualityDirection::lower_is_better);
    const auto quality = golden_accuracy::summarize(
        values, golden_accuracy::QualityDirection::higher_is_better);
    require_near(errors.mean, 2.0, 1e-12, "summary mean");
    require_near(errors.worst, 4.0, 1e-12, "lower-is-better worst");
    require_near(quality.worst, 0.0, 1e-12, "higher-is-better worst");
    require_near(errors.p50, 2.0, 1e-12, "p50");
    require_near(errors.p90, 3.6, 1e-12, "p90");
    require_near(errors.p95, 3.8, 1e-12, "p95");
    require_near(errors.p99, 3.96, 1e-12, "p99");
}

void test_json_escape() {
    require(
        golden_accuracy::json_escape("quote=\" newline=\n") == "quote=\\\" newline=\\n",
        "JSON escaping mismatch");
}

void test_committed_fixture_provenance_if_configured() {
    const char * fixture = std::getenv("NANOEMBED_GOLDEN_PROVENANCE_TEST_FIXTURE");
    if (!fixture || !*fixture) return;
    const auto provenance = golden_accuracy::verify_fixture_provenance(fixture);
    require(provenance.status == "legacy_unverified" || provenance.status == "verified",
            "unexpected committed fixture provenance status");
    require(provenance.fixture_sha256.size() == 64, "fixture SHA-256 missing");
    require(provenance.manifest_sha256.size() == 64, "manifest SHA-256 missing");
}

class TemporaryDirectory {
public:
    TemporaryDirectory() {
        const auto nonce = std::chrono::high_resolution_clock::now()
            .time_since_epoch().count();
        path_ = std::filesystem::temp_directory_path()
            / ("nanoembed-golden-accuracy-" + std::to_string(nonce));
        if (!std::filesystem::create_directory(path_)) {
            throw std::runtime_error("cannot create temporary provenance test directory");
        }
    }

    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path_, error);
    }

    const std::filesystem::path & path() const { return path_; }

    TemporaryDirectory(const TemporaryDirectory &) = delete;
    TemporaryDirectory & operator=(const TemporaryDirectory &) = delete;

private:
    std::filesystem::path path_;
};

void write_text(const std::filesystem::path & path, const std::string & contents) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create test artifact: " + path.string());
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    if (!output) throw std::runtime_error("cannot write test artifact: " + path.string());
}

template<typename Function>
void require_runtime_error(Function function, const std::string & expected_message) {
    try {
        function();
    } catch (const std::runtime_error & error) {
        require(
            std::string(error.what()).find(expected_message) != std::string::npos,
            "provenance rejection reported an unexpected error");
        return;
    }
    throw std::runtime_error("tampered provenance artifact was accepted");
}

void test_provenance_rejects_tampering_and_duplicate_integrity_entry() {
    TemporaryDirectory temporary;
    const std::filesystem::path fixture = temporary.path() / "fixture.bin";
    const std::filesystem::path manifest = fixture.string() + ".provenance.json";
    const std::filesystem::path integrity = fixture.string() + ".provenance.sha256";
    const std::string fixture_contents = "test fixture bytes";
    write_text(fixture, fixture_contents);
    const std::string fixture_hash = golden_accuracy::sha256_file(fixture.string());

    const std::string manifest_contents =
        "{\n"
        "  \"schema_version\": 1,\n"
        "  \"provenance_status\": \"legacy_unverified\",\n"
        "  \"fixture_sha256\": \"" + fixture_hash + "\",\n"
        "  \"model_id\": \"test/model\",\n"
        "  \"resolved_revision\": null\n"
        "}\n";
    write_text(manifest, manifest_contents);
    const std::string manifest_hash = golden_accuracy::sha256_file(manifest.string());
    const std::string valid_integrity =
        "fixture " + fixture_hash + "\n"
        "manifest " + manifest_hash + "\n";
    write_text(integrity, valid_integrity);

    const auto valid = golden_accuracy::verify_fixture_provenance(fixture.string());
    require(valid.fixture_sha256 == fixture_hash, "valid temporary fixture was not verified");
    require(valid.manifest_sha256 == manifest_hash, "valid temporary manifest was not verified");

    write_text(fixture, fixture_contents + " tampered");
    require_runtime_error(
        [&] { (void) golden_accuracy::verify_fixture_provenance(fixture.string()); },
        "fixture SHA-256 mismatch");

    write_text(fixture, fixture_contents);
    write_text(manifest, manifest_contents + "tampered");
    require_runtime_error(
        [&] { (void) golden_accuracy::verify_fixture_provenance(fixture.string()); },
        "manifest SHA-256 mismatch");

    write_text(manifest, manifest_contents);
    write_text(integrity, valid_integrity + "fixture " + fixture_hash + "\n");
    require_runtime_error(
        [&] { (void) golden_accuracy::verify_fixture_provenance(fixture.string()); },
        "invalid integrity entry");
}

} // namespace

int main() {
    try {
        test_sha256();
        test_vector_metrics();
        test_summary_direction_and_percentiles();
        test_json_escape();
        test_committed_fixture_provenance_if_configured();
        test_provenance_rejects_tampering_and_duplicate_integrity_entry();
    } catch (const std::exception & error) {
        std::fprintf(stderr, "golden_accuracy_test: FAIL: %s\n", error.what());
        return EXIT_FAILURE;
    }
    std::printf("golden_accuracy_test: ok\n");
    return EXIT_SUCCESS;
}
