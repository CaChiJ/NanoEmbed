// B1 design proof: validate that this vendored GGUF representation can point
// metadata-only ggml tensors directly into a read-only file mapping.
//
// This deliberately does not use NanoEmbed's production loader and does not
// execute inference. It proves only the prerequisites that B2 is allowed to
// rely on: offset/alignment/size arithmetic, block layout, mapping lifetime,
// and byte identity between mmap and positioned file reads.

#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace {

constexpr size_t kProbeBytes = 64;

bool checked_add(size_t a, size_t b, size_t & out) {
    if (b > std::numeric_limits<size_t>::max() - a) return false;
    out = a + b;
    return true;
}

bool checked_pad(size_t value, size_t alignment, size_t & out) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) return false;
    const size_t mask = alignment - 1;
    if (value > std::numeric_limits<size_t>::max() - mask) return false;
    out = (value + mask) & ~mask;
    return true;
}

bool read_exact_at(int fd, void * dst, size_t size, size_t offset) {
    auto * p = static_cast<unsigned char *>(dst);
    size_t done = 0;
    while (done < size) {
        size_t absolute = 0;
        if (!checked_add(offset, done, absolute) ||
            absolute > static_cast<size_t>(std::numeric_limits<off_t>::max())) {
            return false;
        }
        const ssize_t n = pread(fd, p + done, size - done, static_cast<off_t>(absolute));
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return false;
        done += static_cast<size_t>(n);
    }
    return true;
}

bool parse_layer(const char * name, int & layer) {
    constexpr char prefix[] = "blk.";
    if (std::strncmp(name, prefix, sizeof(prefix) - 1) != 0) return false;
    const char * p = name + sizeof(prefix) - 1;
    if (*p < '0' || *p > '9') return false;
    int value = 0;
    while (*p >= '0' && *p <= '9') {
        const int digit = *p - '0';
        if (value > (std::numeric_limits<int>::max() - digit) / 10) return false;
        value = value * 10 + digit;
        ++p;
    }
    if (*p != '.') return false;
    layer = value;
    return true;
}

struct RegionStats {
    size_t tensor_count = 0;
    size_t payload_bytes = 0;
    size_t min_file_offset = std::numeric_limits<size_t>::max();
    size_t max_file_end = 0;

    bool add(size_t start, size_t size) {
        size_t end = 0;
        if (!checked_add(start, size, end) ||
            size > std::numeric_limits<size_t>::max() - payload_bytes) {
            return false;
        }
        ++tensor_count;
        payload_bytes += size;
        if (start < min_file_offset) min_file_offset = start;
        if (end > max_file_end) max_file_end = end;
        return true;
    }
};

int fail(const std::string & message) {
    std::fprintf(stderr, "mmap_gguf_proof: FAIL: %s\n", message.c_str());
    return 1;
}

int run(const char * path, const std::set<std::string> & required_types) {
    const int fd = open(path, O_RDONLY);
    if (fd < 0) return fail(std::string("open failed: ") + std::strerror(errno));

    struct stat st {};
    if (fstat(fd, &st) != 0) {
        const std::string error = std::strerror(errno);
        close(fd);
        return fail("fstat failed: " + error);
    }
    if (st.st_size <= 0 ||
        static_cast<uintmax_t>(st.st_size) >
            static_cast<uintmax_t>(std::numeric_limits<size_t>::max())) {
        close(fd);
        return fail("file size is zero or not representable as size_t");
    }
    const size_t file_size = static_cast<size_t>(st.st_size);

    void * mapping = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED) {
        const std::string error = std::strerror(errno);
        close(fd);
        return fail("mmap failed: " + error);
    }

    ggml_context * meta = nullptr;
    gguf_init_params params;
    params.no_alloc = true;
    params.ctx = &meta;
    const int metadata_fd = dup(fd);
    if (metadata_fd < 0) {
        const std::string error = std::strerror(errno);
        munmap(mapping, file_size);
        close(fd);
        return fail("dup for GGUF metadata failed: " + error);
    }
    FILE * metadata_file = fdopen(metadata_fd, "rb");
    if (metadata_file == nullptr) {
        const std::string error = std::strerror(errno);
        close(metadata_fd);
        munmap(mapping, file_size);
        close(fd);
        return fail("fdopen for GGUF metadata failed: " + error);
    }
    gguf_context * gguf = gguf_init_from_file_ptr(metadata_file, params);
    const int metadata_close_result = fclose(metadata_file);
    if (gguf == nullptr || meta == nullptr || metadata_close_result != 0) {
        if (meta != nullptr) ggml_free(meta);
        if (gguf != nullptr) gguf_free(gguf);
        munmap(mapping, file_size);
        close(fd);
        return fail(metadata_close_result == 0
                        ? "gguf metadata load failed"
                        : "closing GGUF metadata stream failed");
    }

    int result = 0;
    std::string error;
    const size_t data_offset = gguf_get_data_offset(gguf);
    const size_t gguf_alignment = gguf_get_alignment(gguf);
    const size_t cpu_alignment =
        ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
    const int64_t n_tensors = gguf_get_n_tensors(gguf);
    struct TypeStats {
        size_t count = 0;
        int64_t block_size = 0;
        size_t type_size = 0;
    };
    std::map<std::string, TypeStats> type_stats;
    std::map<int, RegionStats> layer_stats;
    RegionStats common_stats;
    RegionStats token_embedding_stats;
    size_t expected_relative_offset = 0;

    if (data_offset > file_size) {
        error = "data section begins after end of file";
    } else if (gguf_alignment == 0 ||
               (gguf_alignment & (gguf_alignment - 1)) != 0) {
        error = "GGUF alignment is not a non-zero power of two";
    } else if (cpu_alignment == 0 ||
               (cpu_alignment & (cpu_alignment - 1)) != 0) {
        error = "CPU backend alignment is not a non-zero power of two";
    } else if (reinterpret_cast<uintptr_t>(mapping) % cpu_alignment != 0) {
        error = "mapping base does not meet the CPU backend alignment";
    } else if (n_tensors <= 0) {
        error = "GGUF has no tensors";
    }

    for (int64_t i = 0; error.empty() && i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gguf, i);
        const ggml_type type = gguf_get_tensor_type(gguf, i);
        const size_t relative_offset = gguf_get_tensor_offset(gguf, i);
        const size_t size = gguf_get_tensor_size(gguf, i);
        ggml_tensor * tensor = ggml_get_tensor(meta, name);

        if (tensor == nullptr) {
            error = std::string("metadata tensor missing: ") + name;
            break;
        }
        if (tensor->data != nullptr || tensor->buffer != nullptr) {
            error = std::string("no_alloc tensor unexpectedly owns data/buffer: ") + name;
            break;
        }
        if (tensor->type != type || ggml_nbytes(tensor) != size) {
            error = std::string("GGUF/ggml type or size mismatch: ") + name;
            break;
        }

        const int64_t block_size = ggml_blck_size(type);
        const size_t type_size = ggml_type_size(type);
        if (block_size <= 0 || type_size == 0 || tensor->ne[0] % block_size != 0) {
            error = std::string("invalid type/block/row layout: ") + name;
            break;
        }
        if (relative_offset != expected_relative_offset ||
            relative_offset % gguf_alignment != 0) {
            error = std::string("non-canonical or unaligned tensor offset: ") + name;
            break;
        }

        size_t file_offset = 0;
        size_t file_end = 0;
        if (!checked_add(data_offset, relative_offset, file_offset) ||
            !checked_add(file_offset, size, file_end) || file_end > file_size) {
            error = std::string("tensor range is outside the mapped file: ") + name;
            break;
        }
        if (file_offset % gguf_alignment != 0 || file_offset % cpu_alignment != 0) {
            error = std::string("absolute tensor pointer is insufficiently aligned: ") + name;
            break;
        }

        // This is the direct-pointer operation B2 would perform. The metadata
        // context still owns only tensor structs; the mapping owns every byte.
        tensor->data = static_cast<unsigned char *>(mapping) + file_offset;

        std::vector<size_t> probes{0};
        if (size > kProbeBytes) probes.push_back((size - kProbeBytes) / 2);
        if (size > kProbeBytes) probes.push_back(size - kProbeBytes);
        for (const size_t tensor_offset : probes) {
            const size_t count = std::min(kProbeBytes, size - tensor_offset);
            std::vector<unsigned char> positioned(count);
            size_t absolute = 0;
            if (!checked_add(file_offset, tensor_offset, absolute) ||
                !read_exact_at(fd, positioned.data(), positioned.size(), absolute) ||
                std::memcmp(static_cast<unsigned char *>(tensor->data) + tensor_offset,
                            positioned.data(), positioned.size()) != 0) {
                error = std::string("mmap/pread byte mismatch: ") + name;
                break;
            }
        }
        if (!error.empty()) break;

        const std::string type_name = ggml_type_name(type);
        TypeStats & observed = type_stats[type_name];
        ++observed.count;
        observed.block_size = block_size;
        observed.type_size = type_size;

        RegionStats * region = &common_stats;
        int layer = -1;
        if (std::strcmp(name, "token_embd.weight") == 0) {
            region = &token_embedding_stats;
        } else if (parse_layer(name, layer)) {
            region = &layer_stats[layer];
        }
        if (!region->add(file_offset, size)) {
            error = std::string("region accounting overflow: ") + name;
            break;
        }

        size_t padded = 0;
        if (!checked_pad(size, gguf_alignment, padded) ||
            !checked_add(expected_relative_offset, padded, expected_relative_offset)) {
            error = std::string("padded tensor extent overflow: ") + name;
            break;
        }
    }

    for (const std::string & required : required_types) {
        if (error.empty() && type_stats.find(required) == type_stats.end()) {
            error = "required tensor type not present: " + required;
        }
    }

    if (error.empty()) {
        size_t data_end = 0;
        if (!checked_add(data_offset, expected_relative_offset, data_end) ||
            data_end > file_size) {
            error = "padded data section extent is outside the file";
        }
    }

    if (error.empty()) {
        std::printf(
            "mmap_gguf_proof: OK path=%s file_bytes=%zu data_offset=%zu "
            "gguf_alignment=%zu cpu_alignment=%zu tensors=%" PRId64 " layers=%zu\n",
            path, file_size, data_offset, gguf_alignment, cpu_alignment,
            n_tensors, layer_stats.size());
        for (const auto & entry : type_stats) {
            std::printf("  type=%s tensors=%zu block_size=%" PRId64 " type_size=%zu\n",
                        entry.first.c_str(), entry.second.count,
                        entry.second.block_size, entry.second.type_size);
        }
        const auto print_region = [](const char * label, const RegionStats & stats) {
            if (stats.tensor_count == 0) return;
            std::printf("  region=%s tensors=%zu payload_bytes=%zu range=[%zu,%zu)\n",
                        label, stats.tensor_count, stats.payload_bytes,
                        stats.min_file_offset, stats.max_file_end);
        };
        print_region("token_embedding", token_embedding_stats);
        print_region("common", common_stats);
        for (const auto & entry : layer_stats) {
            const std::string label = "layer_" + std::to_string(entry.first);
            print_region(label.c_str(), entry.second);
        }
    } else {
        result = fail(error);
    }

    // Tensor metadata borrows the mapping. Break those borrows and release the
    // metadata before unmapping, then close the backing file last.
    for (ggml_tensor * tensor = ggml_get_first_tensor(meta);
         tensor != nullptr; tensor = ggml_get_next_tensor(meta, tensor)) {
        tensor->data = nullptr;
    }
    ggml_free(meta);
    gguf_free(gguf);
    if (munmap(mapping, file_size) != 0 && result == 0) {
        result = fail(std::string("munmap failed: ") + std::strerror(errno));
    }
    if (close(fd) != 0 && result == 0) {
        result = fail(std::string("close failed: ") + std::strerror(errno));
    }
    return result;
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <model.gguf> [--require-type <ggml-type>]...\n",
                     argv[0]);
        return 2;
    }
    std::set<std::string> required_types;
    for (int i = 2; i < argc; ++i) {
        if (std::strcmp(argv[i], "--require-type") != 0 || i + 1 >= argc) {
            std::fprintf(stderr, "mmap_gguf_proof: invalid argument: %s\n", argv[i]);
            return 2;
        }
        required_types.insert(argv[++i]);
    }
    return run(argv[1], required_types);
}
