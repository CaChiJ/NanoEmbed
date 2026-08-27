// Linux-only construction-phase probe for tokenizer metadata lifetime.
//
// It deliberately does not execute a forward graph. Its purpose is to show
// the memory state immediately before and after release of the GGUF's large
// source tokenizer arrays, using the same mapped-model construction path as
// production.

#include "arch/model_arch.h"
#include "mapped_weight_store.h"
#include "tokenizer/spm_bpe.h"
#include "tokenizer/tokenizer.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__GLIBC__)
#include <malloc.h>
#endif

namespace {

size_t read_kib_line(const std::string & contents, const std::string & key) {
    const std::string prefix = key + ":";
    const size_t begin = contents.find(prefix);
    if (begin == std::string::npos) return 0;

    const char * p = contents.c_str() + begin + prefix.size();
    char * end = nullptr;
    const unsigned long long kib = std::strtoull(p, &end, 10);
    return end == p ? 0 : static_cast<size_t>(kib);
}

std::string read_file(const char * path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error(std::string("cannot read ") + path);
    return std::string(std::istreambuf_iterator<char>(in),
                       std::istreambuf_iterator<char>());
}

struct Snapshot {
    std::string phase;
    size_t vm_rss_kib = 0;
    size_t rss_kib = 0;
    size_t pss_kib = 0;
    size_t pss_anon_kib = 0;
    size_t allocated_heap_kib = 0;
    size_t allocated_mmap_kib = 0;
};

Snapshot snapshot(const char * phase) {
    const std::string status = read_file("/proc/self/status");
    const std::string rollup = read_file("/proc/self/smaps_rollup");

    Snapshot result;
    result.phase = phase;
    result.vm_rss_kib = read_kib_line(status, "VmRSS");
    result.rss_kib = read_kib_line(rollup, "Rss");
    result.pss_kib = read_kib_line(rollup, "Pss");
    result.pss_anon_kib = read_kib_line(rollup, "Pss_Anon");
#if defined(__GLIBC__)
    const struct mallinfo2 allocator = mallinfo2();
    result.allocated_heap_kib = static_cast<size_t>(allocator.uordblks / 1024);
    result.allocated_mmap_kib = static_cast<size_t>(allocator.hblkhd / 1024);
#endif
    return result;
}

void print_snapshot(const Snapshot & s) {
    std::cout << s.phase
              << "\tVmRSS_KiB=" << s.vm_rss_kib
              << "\tRss_KiB=" << s.rss_kib
              << "\tPss_KiB=" << s.pss_kib
              << "\tPss_Anon_KiB=" << s.pss_anon_kib
              << "\tallocated_heap_KiB=" << s.allocated_heap_kib
              << "\tallocated_mmap_KiB=" << s.allocated_mmap_kib
              << '\n';
}

void print_delta(const Snapshot & before, const Snapshot & after) {
    const auto delta = [](size_t lhs, size_t rhs) -> long long {
        return static_cast<long long>(rhs) - static_cast<long long>(lhs);
    };
    std::cout << "delta(" << before.phase << "->" << after.phase << ')'
              << "\tVmRSS_KiB=" << delta(before.vm_rss_kib, after.vm_rss_kib)
              << "\tRss_KiB=" << delta(before.rss_kib, after.rss_kib)
              << "\tPss_KiB=" << delta(before.pss_kib, after.pss_kib)
              << "\tPss_Anon_KiB=" << delta(before.pss_anon_kib, after.pss_anon_kib)
              << "\tallocated_heap_KiB="
              << delta(before.allocated_heap_kib, after.allocated_heap_kib)
              << "\tallocated_mmap_KiB="
              << delta(before.allocated_mmap_kib, after.allocated_mmap_kib)
              << '\n';
}

} // namespace

int main(int argc, char ** argv) {
    if (argc != 2) {
        std::cerr << "usage: nanoembed-tokenizer-memory-probe MODEL.gguf\n";
        return 2;
    }

    try {
        const Snapshot initial = snapshot("initial");

        nanoembed::MappedWeightStore store(argv[1]);
        const Snapshot parsed = snapshot("after_gguf_parse");

        std::unique_ptr<nanoembed::ModelArch> arch =
            nanoembed::create_model_arch(store.gguf(), store.model_context());
        const Snapshot architecture = snapshot("after_architecture");

        std::unique_ptr<nanoembed::Tokenizer> tokenizer =
            nullptr;
        const auto load_begin = std::chrono::steady_clock::now();
        tokenizer = nanoembed::create_tokenizer(store.gguf());
        const auto load_end = std::chrono::steady_clock::now();
        const std::vector<int> before = tokenizer->encode("metadata lifetime probe");
        const Snapshot tokenizer_built = snapshot("after_tokenizer_build");

        nanoembed::discard_consumed_tokenizer_metadata(store.gguf());
        const std::vector<int> after = tokenizer->encode("metadata lifetime probe");
        if (after != before) {
            throw std::runtime_error("metadata release changed tokenizer output");
        }
        const Snapshot discarded = snapshot("after_metadata_discard");

        print_snapshot(initial);
        print_snapshot(parsed);
        print_snapshot(architecture);
        print_snapshot(tokenizer_built);
        print_snapshot(discarded);
        print_delta(tokenizer_built, discarded);
        if (const auto * bpe = dynamic_cast<const nanoembed::SpmBpeTokenizer *>(tokenizer.get())) {
            using Microseconds = std::chrono::duration<double, std::micro>;
            std::vector<double> encode_us;
            encode_us.reserve(100);
            uint64_t page_reads = 0;
            for (int i = 0; i < 100; ++i) {
                const auto begin = std::chrono::steady_clock::now();
                const auto encoded = bpe->encode_with_diagnostics(
                    "The quick brown fox jumps over the lazy dog.");
                const auto end = std::chrono::steady_clock::now();
                encode_us.push_back(Microseconds(end - begin).count());
                page_reads += encoded.page_reads;
            }
            std::sort(encode_us.begin(), encode_us.end());
            const uintmax_t cache_bytes = std::filesystem::file_size(bpe->merge_cache_path());
            std::cout << "merge_index"
                      << "\tcache_hit=" << (bpe->merge_cache_hit() ? 1 : 0)
                      << "\tload_ms="
                      << std::chrono::duration<double, std::milli>(load_end - load_begin).count()
                      << "\tcache_bytes=" << cache_bytes
                      << "\tfence_bytes=" << bpe->merge_fence_bytes()
                      << "\trecords=" << bpe->merge_record_count()
                      << "\tencode_p50_us=" << encode_us[49]
                      << "\tencode_p95_us=" << encode_us[94]
                      << "\tpage_reads_per_encode="
                      << static_cast<double>(page_reads) / encode_us.size()
                      << '\n';
        }
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "nanoembed-tokenizer-memory-probe: " << e.what() << '\n';
        return 1;
    }
}
