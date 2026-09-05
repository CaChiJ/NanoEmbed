#include "golden_accuracy.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace golden_accuracy {
namespace {

constexpr std::array<uint32_t, 64> kRoundConstants = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

uint32_t rotate_right(uint32_t value, uint32_t count) {
    return (value >> count) | (value << (32U - count));
}

class Sha256 {
public:
    void update(const uint8_t * data, size_t size) {
        total_size_ += static_cast<uint64_t>(size);
        while (size > 0) {
            const size_t copy = std::min(size, block_.size() - block_size_);
            std::memcpy(block_.data() + block_size_, data, copy);
            block_size_ += copy;
            data += copy;
            size -= copy;
            if (block_size_ == block_.size()) {
                transform(block_.data());
                block_size_ = 0;
            }
        }
    }

    std::array<uint8_t, 32> finish() {
        const uint64_t message_bits = total_size_ * 8U;
        block_[block_size_++] = 0x80U;
        if (block_size_ > 56) {
            std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_), block_.end(), 0U);
            transform(block_.data());
            block_size_ = 0;
        }
        std::fill(
            block_.begin() + static_cast<std::ptrdiff_t>(block_size_),
            block_.begin() + 56,
            0U);
        for (size_t i = 0; i < 8; ++i) {
            block_[63 - i] = static_cast<uint8_t>(message_bits >> (i * 8U));
        }
        transform(block_.data());

        std::array<uint8_t, 32> digest{};
        for (size_t i = 0; i < state_.size(); ++i) {
            digest[i * 4]     = static_cast<uint8_t>(state_[i] >> 24U);
            digest[i * 4 + 1] = static_cast<uint8_t>(state_[i] >> 16U);
            digest[i * 4 + 2] = static_cast<uint8_t>(state_[i] >> 8U);
            digest[i * 4 + 3] = static_cast<uint8_t>(state_[i]);
        }
        return digest;
    }

private:
    void transform(const uint8_t * block) {
        std::array<uint32_t, 64> words{};
        for (size_t i = 0; i < 16; ++i) {
            words[i] =
                (static_cast<uint32_t>(block[i * 4]) << 24U)
                | (static_cast<uint32_t>(block[i * 4 + 1]) << 16U)
                | (static_cast<uint32_t>(block[i * 4 + 2]) << 8U)
                | static_cast<uint32_t>(block[i * 4 + 3]);
        }
        for (size_t i = 16; i < words.size(); ++i) {
            const uint32_t s0 = rotate_right(words[i - 15], 7U)
                ^ rotate_right(words[i - 15], 18U) ^ (words[i - 15] >> 3U);
            const uint32_t s1 = rotate_right(words[i - 2], 17U)
                ^ rotate_right(words[i - 2], 19U) ^ (words[i - 2] >> 10U);
            words[i] = words[i - 16] + s0 + words[i - 7] + s1;
        }

        uint32_t a = state_[0];
        uint32_t b = state_[1];
        uint32_t c = state_[2];
        uint32_t d = state_[3];
        uint32_t e = state_[4];
        uint32_t f = state_[5];
        uint32_t g = state_[6];
        uint32_t h = state_[7];

        for (size_t i = 0; i < words.size(); ++i) {
            const uint32_t sigma1 = rotate_right(e, 6U) ^ rotate_right(e, 11U)
                ^ rotate_right(e, 25U);
            const uint32_t choose = (e & f) ^ ((~e) & g);
            const uint32_t temporary1 = h + sigma1 + choose + kRoundConstants[i] + words[i];
            const uint32_t sigma0 = rotate_right(a, 2U) ^ rotate_right(a, 13U)
                ^ rotate_right(a, 22U);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temporary2 = sigma0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temporary1;
            d = c;
            c = b;
            b = a;
            a = temporary1 + temporary2;
        }

        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<uint32_t, 8> state_ = {
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
    };
    std::array<uint8_t, 64> block_{};
    size_t block_size_ = 0;
    uint64_t total_size_ = 0;
};

std::string hex_digest(const std::array<uint8_t, 32> & digest) {
    std::ostringstream stream;
    stream << std::hex << std::setfill('0');
    for (uint8_t byte : digest) stream << std::setw(2) << static_cast<unsigned>(byte);
    return stream.str();
}

std::string read_text_file(const std::string & path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open provenance artifact: " + path);
    std::ostringstream contents;
    contents << stream.rdbuf();
    if (!stream.good() && !stream.eof()) {
        throw std::runtime_error("cannot read provenance artifact: " + path);
    }
    return contents.str();
}

std::string require_json_string(const std::string & json, const std::string & key) {
    const std::regex expression(
        "\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
    std::smatch match;
    if (!std::regex_search(json, match, expression)) {
        throw std::runtime_error("provenance manifest missing string field: " + key);
    }
    return match[1].str();
}

bool has_json_null(const std::string & json, const std::string & key) {
    return std::regex_search(
        json, std::regex("\\\"" + key + "\\\"\\s*:\\s*null"));
}

bool is_lower_hex(const std::string & value, size_t length) {
    return value.size() == length
        && std::all_of(value.begin(), value.end(), [](char ch) {
            return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
        });
}

} // namespace

VectorMetrics compare_vectors(const float * output, const float * reference, size_t count) {
    if (!output || !reference || count == 0) {
        throw std::invalid_argument("compare_vectors requires non-empty vectors");
    }
    double output_squared = 0.0;
    double reference_squared = 0.0;
    double dot = 0.0;
    double absolute_sum = 0.0;
    double squared_sum = 0.0;
    double maximum = 0.0;
    bool all_finite = true;
    for (size_t i = 0; i < count; ++i) {
        const double actual = output[i];
        const double expected = reference[i];
        if (!std::isfinite(actual) || !std::isfinite(expected)) all_finite = false;
        const double difference = actual - expected;
        const double absolute = std::abs(difference);
        output_squared += actual * actual;
        reference_squared += expected * expected;
        dot += actual * expected;
        absolute_sum += absolute;
        squared_sum += difference * difference;
        maximum = std::max(maximum, absolute);
    }

    VectorMetrics result;
    result.output_norm = std::sqrt(output_squared);
    result.reference_norm = std::sqrt(reference_squared);
    result.cosine = (result.output_norm > 0.0 && result.reference_norm > 0.0)
        ? dot / (result.output_norm * result.reference_norm)
        : 0.0;
    result.max_absolute_error = maximum;
    result.all_finite = all_finite;
    result.bitwise_identical =
        std::memcmp(output, reference, count * sizeof(float)) == 0;
    result.mean_absolute_error = absolute_sum / static_cast<double>(count);
    result.rmse = std::sqrt(squared_sum / static_cast<double>(count));
    result.norm_difference = std::abs(result.output_norm - result.reference_norm);
    result.relative_norm_difference = result.reference_norm > 0.0
        ? result.norm_difference / result.reference_norm
        : (result.norm_difference == 0.0 ? 0.0 : std::numeric_limits<double>::infinity());
    return result;
}

double percentile(std::vector<double> values, double quantile) {
    if (values.empty()) throw std::invalid_argument("percentile requires values");
    if (quantile < 0.0 || quantile > 1.0) {
        throw std::invalid_argument("percentile quantile must be in [0, 1]");
    }
    std::sort(values.begin(), values.end());
    const double position = quantile * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(std::floor(position));
    const size_t upper = static_cast<size_t>(std::ceil(position));
    const double fraction = position - static_cast<double>(lower);
    return values[lower] + (values[upper] - values[lower]) * fraction;
}

DistributionSummary summarize(
    const std::vector<double> & values, QualityDirection direction) {
    if (values.empty()) throw std::invalid_argument("summarize requires values");
    DistributionSummary result;
    result.count = values.size();
    result.direction = direction;
    double sum = 0.0;
    for (double value : values) sum += value;
    result.mean = sum / static_cast<double>(values.size());
    result.worst = direction == QualityDirection::higher_is_better
        ? *std::min_element(values.begin(), values.end())
        : *std::max_element(values.begin(), values.end());
    result.p50 = percentile(values, 0.50);
    result.p90 = percentile(values, 0.90);
    result.p95 = percentile(values, 0.95);
    result.p99 = percentile(values, 0.99);
    return result;
}

std::string sha256_bytes(const void * data, size_t size) {
    if (!data && size != 0) throw std::invalid_argument("sha256_bytes received null data");
    Sha256 hash;
    hash.update(static_cast<const uint8_t *>(data), size);
    return hex_digest(hash.finish());
}

std::string sha256_file(const std::string & path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("cannot open file for SHA-256: " + path);
    Sha256 hash;
    std::array<char, 1024 * 1024> buffer{};
    while (stream) {
        stream.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = stream.gcount();
        if (count > 0) {
            hash.update(
                reinterpret_cast<const uint8_t *>(buffer.data()),
                static_cast<size_t>(count));
        }
    }
    if (!stream.eof()) throw std::runtime_error("cannot read file for SHA-256: " + path);
    return hex_digest(hash.finish());
}

ProvenanceInfo verify_fixture_provenance(const std::string & fixture_path) {
    ProvenanceInfo result;
    result.manifest_path = fixture_path + ".provenance.json";
    const std::string integrity_path = fixture_path + ".provenance.sha256";

    std::ifstream integrity(integrity_path);
    if (!integrity) {
        throw std::runtime_error(
            "missing fixture integrity sidecar: " + integrity_path
            + " (legacy fixtures need an explicit legacy_unverified manifest)");
    }
    std::unordered_map<std::string, std::string> expected;
    std::string name;
    std::string digest;
    while (integrity >> name >> digest) {
        if (name != "fixture" && name != "manifest") {
            throw std::runtime_error("unknown integrity entry in " + integrity_path + ": " + name);
        }
        if (!is_lower_hex(digest, 64) || !expected.emplace(name, digest).second) {
            throw std::runtime_error("invalid integrity entry in " + integrity_path + ": " + name);
        }
    }
    if (expected.size() != 2 || !expected.count("fixture") || !expected.count("manifest")) {
        throw std::runtime_error("integrity sidecar must contain fixture and manifest SHA-256");
    }

    result.fixture_sha256 = sha256_file(fixture_path);
    result.manifest_sha256 = sha256_file(result.manifest_path);
    if (result.fixture_sha256 != expected.at("fixture")) {
        throw std::runtime_error("fixture SHA-256 mismatch: " + fixture_path);
    }
    if (result.manifest_sha256 != expected.at("manifest")) {
        throw std::runtime_error("manifest SHA-256 mismatch: " + result.manifest_path);
    }

    const std::string json = read_text_file(result.manifest_path);
    if (!std::regex_search(json, std::regex("\\\"schema_version\\\"\\s*:\\s*1"))) {
        throw std::runtime_error("unsupported or missing provenance schema_version");
    }
    if (require_json_string(json, "fixture_sha256") != result.fixture_sha256) {
        throw std::runtime_error("manifest fixture_sha256 does not match loaded fixture");
    }
    result.status = require_json_string(json, "provenance_status");
    result.model_id = require_json_string(json, "model_id");
    if (result.status == "verified") {
        result.resolved_revision = require_json_string(json, "resolved_revision");
        if (!is_lower_hex(result.resolved_revision, 40)) {
            throw std::runtime_error("verified provenance requires an exact 40-character revision");
        }
        (void) require_json_string(json, "command");
        if (!std::regex_search(
                json, std::regex("\\\"deterministic_algorithms\\\"\\s*:\\s*true"))) {
            throw std::runtime_error("verified provenance must assert deterministic algorithms");
        }
        if (!std::regex_search(json, std::regex("\\\"dtype\\\"\\s*:\\s*\\\"float32\\\""))
            || !std::regex_search(json, std::regex("\\\"device\\\"\\s*:\\s*\\\"cpu\\\""))) {
            throw std::runtime_error("verified provenance must be CPU FP32");
        }
        if (!std::regex_search(json, std::regex("\\\"batch_size\\\"\\s*:\\s*[1-9][0-9]*"))
            || !std::regex_search(json, std::regex("\\\"max_length\\\"\\s*:\\s*[1-9][0-9]*"))
            || !std::regex_search(json, std::regex("\\\"seed\\\"\\s*:\\s*[0-9]+"))) {
            throw std::runtime_error("verified provenance requires batch, length and seed settings");
        }
        (void) require_json_string(json, "pooling");
        if (!std::regex_search(json, std::regex("\\\"normalize\\\"\\s*:\\s*(true|false)"))
            || !std::regex_search(json, std::regex("\\\"truncation\\\"\\s*:\\s*true"))
            || !std::regex_search(json, std::regex("\\\"package_versions\\\"\\s*:\\s*\\{"))
            || !std::regex_search(json, std::regex("\\\"requirements_lock\\\"\\s*:\\s*\\{"))
            || !std::regex_search(json, std::regex("\\\"model_artifacts\\\"\\s*:\\s*\\["))) {
            throw std::runtime_error("verified provenance is missing reproducibility fields");
        }
    } else if (result.status == "legacy_unverified") {
        if (!has_json_null(json, "resolved_revision")) {
            throw std::runtime_error(
                "legacy_unverified provenance must not claim a resolved revision");
        }
    } else {
        throw std::runtime_error("unsupported provenance_status: " + result.status);
    }
    return result;
}

std::string json_escape(const std::string & value) {
    std::ostringstream output;
    for (unsigned char ch : value) {
        switch (ch) {
            case '\"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (ch < 0x20U) {
                    output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                           << static_cast<unsigned>(ch) << std::dec;
                } else {
                    output << static_cast<char>(ch);
                }
        }
    }
    return output.str();
}

const char * quality_direction_name(QualityDirection direction) {
    return direction == QualityDirection::higher_is_better
        ? "higher_is_better; worst=min"
        : "lower_is_better; worst=max";
}

} // namespace golden_accuracy
