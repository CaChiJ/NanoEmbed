#include "sha256.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace nanoembed::detail {

namespace {

constexpr uint32_t kRound[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

uint32_t rotate_right(uint32_t value, unsigned count) {
    return (value >> count) | (value << (32u - count));
}

uint32_t read_be32(const uint8_t * p) {
    return (static_cast<uint32_t>(p[0]) << 24) |
           (static_cast<uint32_t>(p[1]) << 16) |
           (static_cast<uint32_t>(p[2]) << 8) |
           static_cast<uint32_t>(p[3]);
}

void write_be64(uint8_t * p, uint64_t value) {
    for (int i = 7; i >= 0; --i) {
        p[i] = static_cast<uint8_t>(value);
        value >>= 8;
    }
}

} // namespace

Sha256::Sha256()
    : state_{0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
             0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u} {}

void Sha256::transform(const uint8_t block[64]) {
    uint32_t w[64];
    for (size_t i = 0; i < 16; ++i) w[i] = read_be32(block + i * 4);
    for (size_t i = 16; i < 64; ++i) {
        const uint32_t s0 = rotate_right(w[i - 15], 7) ^
                            rotate_right(w[i - 15], 18) ^ (w[i - 15] >> 3);
        const uint32_t s1 = rotate_right(w[i - 2], 17) ^
                            rotate_right(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }

    uint32_t a = state_[0];
    uint32_t b = state_[1];
    uint32_t c = state_[2];
    uint32_t d = state_[3];
    uint32_t e = state_[4];
    uint32_t f = state_[5];
    uint32_t g = state_[6];
    uint32_t h = state_[7];

    for (size_t i = 0; i < 64; ++i) {
        const uint32_t s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
        const uint32_t choose = (e & f) ^ (~e & g);
        const uint32_t temp1 = h + s1 + choose + kRound[i] + w[i];
        const uint32_t s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
        const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        const uint32_t temp2 = s0 + majority;
        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }

    state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
    state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
}

void Sha256::update(const void * data, size_t size) {
    if (finished_) throw std::logic_error("SHA-256 update after finish");
    if (size != 0 && data == nullptr) throw std::invalid_argument("SHA-256 null input");
    const auto * p = static_cast<const uint8_t *>(data);
    total_bytes_ += size;

    if (buffered_ != 0) {
        const size_t take = std::min(size, sizeof(buffer_) - buffered_);
        std::memcpy(buffer_ + buffered_, p, take);
        buffered_ += take;
        p += take;
        size -= take;
        if (buffered_ == sizeof(buffer_)) {
            transform(buffer_);
            buffered_ = 0;
        }
    }
    while (size >= sizeof(buffer_)) {
        transform(p);
        p += sizeof(buffer_);
        size -= sizeof(buffer_);
    }
    if (size != 0) {
        std::memcpy(buffer_, p, size);
        buffered_ = size;
    }
}

Sha256Digest Sha256::finish() {
    if (finished_) throw std::logic_error("SHA-256 finish called twice");
    const uint64_t bit_count = total_bytes_ * 8u;
    buffer_[buffered_++] = 0x80;
    if (buffered_ > 56) {
        std::fill(buffer_ + buffered_, buffer_ + 64, 0);
        transform(buffer_);
        buffered_ = 0;
    }
    std::fill(buffer_ + buffered_, buffer_ + 56, 0);
    write_be64(buffer_ + 56, bit_count);
    transform(buffer_);
    finished_ = true;

    Sha256Digest result{};
    for (size_t i = 0; i < 8; ++i) {
        result[i * 4]     = static_cast<uint8_t>(state_[i] >> 24);
        result[i * 4 + 1] = static_cast<uint8_t>(state_[i] >> 16);
        result[i * 4 + 2] = static_cast<uint8_t>(state_[i] >> 8);
        result[i * 4 + 3] = static_cast<uint8_t>(state_[i]);
    }
    return result;
}

Sha256Digest sha256(const void * data, size_t size) {
    Sha256 h;
    h.update(data, size);
    return h.finish();
}

std::string sha256_hex(const Sha256Digest & digest) {
    constexpr char kHex[] = "0123456789abcdef";
    std::string result(64, '0');
    for (size_t i = 0; i < digest.size(); ++i) {
        result[i * 2] = kHex[digest[i] >> 4];
        result[i * 2 + 1] = kHex[digest[i] & 0x0f];
    }
    return result;
}

} // namespace nanoembed::detail
