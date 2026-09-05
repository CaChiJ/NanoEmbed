#pragma once

#include "sha256.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

struct gguf_context;

namespace nanoembed {

class DiskMergeIndex {
public:
    struct Rule {
        int rank = 0;
        int merged = -1;
    };

    struct LookupScratch {
        struct Slot {
            uint64_t page = UINT64_MAX;
            std::array<uint8_t, 4096> bytes{};
        };
        std::array<Slot, 8> slots{};
        uint64_t next_slot = 0;
        uint64_t page_reads = 0;
    };

    DiskMergeIndex();
    ~DiskMergeIndex();
    DiskMergeIndex(DiskMergeIndex &&) noexcept;
    DiskMergeIndex & operator=(DiskMergeIndex &&) noexcept;
    DiskMergeIndex(const DiskMergeIndex &) = delete;
    DiskMergeIndex & operator=(const DiskMergeIndex &) = delete;

    static DiskMergeIndex from_gguf(gguf_context * ctx,
                                    int64_t tokens_key,
                                    int64_t merges_key);

    bool find(uint64_t pair_key, LookupScratch & scratch, Rule & result) const;

    const std::filesystem::path & cache_path() const noexcept { return cache_path_; }
    bool cache_hit() const noexcept { return cache_hit_; }
    size_t fence_bytes() const noexcept;
    uint64_t record_count() const noexcept { return record_count_; }

private:
    struct Impl;
    struct Fence {
        uint64_t first_key;
        uint64_t offset;
    };

    static bool open_validated(const std::filesystem::path & path,
                               const detail::Sha256Digest & source_digest,
                               int64_t vocab_size,
                               DiskMergeIndex & result,
                               std::string & reason);
    static void create_cache(gguf_context * ctx,
                             int64_t tokens_key,
                             int64_t merges_key,
                             const detail::Sha256Digest & source_digest,
                             const std::filesystem::path & path);

    std::unique_ptr<Impl> impl_;
    std::vector<Fence> fences_;
    std::filesystem::path cache_path_;
    uint64_t record_count_ = 0;
    // Every merged ID this index can emit becomes a row index into the token
    // embedding table. ggml asserts that range and aborts the process on a
    // miss, which is not something a caller can catch, so the bound is checked
    // here instead.
    int64_t vocab_size_ = 0;
    bool cache_hit_ = false;
};

} // namespace nanoembed
