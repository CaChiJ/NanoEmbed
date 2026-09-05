#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace nanoembed::detail {

using Sha256Digest = std::array<uint8_t, 32>;

class Sha256 {
public:
    Sha256();

    void update(const void * data, size_t size);
    Sha256Digest finish();

private:
    void transform(const uint8_t block[64]);

    uint32_t state_[8];
    uint64_t total_bytes_ = 0;
    uint8_t  buffer_[64] = {};
    size_t   buffered_ = 0;
    bool     finished_ = false;
};

Sha256Digest sha256(const void * data, size_t size);
std::string sha256_hex(const Sha256Digest & digest);

} // namespace nanoembed::detail
