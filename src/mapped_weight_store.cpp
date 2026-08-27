#include "mapped_weight_store.h"

#include "arch/model_arch.h"
#include "tokenizer/tokenizer.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <set>
#include <sstream>
#include <utility>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace nanoembed {

namespace mapped_weight_detail {

size_t checked_add(size_t lhs, size_t rhs, const std::string & subject) {
    if (rhs > std::numeric_limits<size_t>::max() - lhs) {
        throw MappedWeightError("size overflow while computing " + subject);
    }
    return lhs + rhs;
}

size_t checked_pad(size_t value, size_t alignment, const std::string & subject) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw MappedWeightError("invalid non-power-of-two alignment for " + subject);
    }
    const size_t mask = alignment - 1;
    return checked_add(value, mask, subject + " padding") & ~mask;
}

void validate_range(size_t file_size,
                    size_t data_offset,
                    size_t tensor_offset,
                    size_t nbytes,
                    const std::string & subject) {
    const size_t start = checked_add(data_offset, tensor_offset,
                                     subject + " absolute offset");
    const size_t end = checked_add(start, nbytes, subject + " absolute end");
    if (start > file_size || end > file_size) {
        std::ostringstream message;
        message << "tensor range is outside/truncated model file for " << subject
                << " (start=" << start << ", end=" << end
                << ", file_size=" << file_size << ")";
        throw MappedWeightError(message.str());
    }
}

} // namespace mapped_weight_detail

namespace {

#if !defined(_WIN32)
MappedFileIdentity identity_from_stat(const struct stat & st,
                                      const std::string & subject) {
    if (st.st_size <= 0) {
        throw MappedWeightError(subject + " is empty");
    }
    const uintmax_t file_size = static_cast<uintmax_t>(st.st_size);
    if (file_size > std::numeric_limits<size_t>::max()) {
        throw MappedWeightError(subject + " is too large for this process");
    }
    return MappedFileIdentity{
        static_cast<uint64_t>(st.st_dev),
        static_cast<uint64_t>(st.st_ino),
        static_cast<uint64_t>(st.st_size),
    };
}

bool same_identity(const MappedFileIdentity & lhs,
                   const MappedFileIdentity & rhs) noexcept {
    return lhs.device == rhs.device &&
           lhs.inode == rhs.inode &&
           lhs.size == rhs.size;
}

std::string system_error(const std::string & action) {
    return action + ": " + std::strerror(errno);
}
#endif

bool supported_mapped_type(ggml_type type) noexcept {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_K:
            return true;
        default:
            return false;
    }
}

size_t gguf_alignment(gguf_context * gguf) {
    const int64_t key = gguf_find_key(gguf, GGUF_KEY_GENERAL_ALIGNMENT);
    if (key < 0) return GGUF_DEFAULT_ALIGNMENT;
    if (gguf_get_kv_type(gguf, key) != GGUF_TYPE_UINT32) {
        throw MappedWeightError("general.alignment must be uint32");
    }
    const size_t result = gguf_get_val_u32(gguf, key);
    if (result == 0 || (result & (result - 1)) != 0) {
        throw MappedWeightError("general.alignment must be a non-zero power of two");
    }
    return result;
}

} // namespace

struct MappedWeightStore::Impl {
    struct Borrow {
        ggml_tensor * tensor = nullptr;
        std::string   name;
        size_t        absolute_offset = 0;
        size_t        nbytes = 0;
    };

    std::string        path;
    int                fd = -1;
    void *             mapping = nullptr;
    size_t             mapping_size = 0;
    MappedFileIdentity identity;
    gguf_context *     gguf = nullptr;
    ggml_context *     meta = nullptr;
    size_t             data_offset = 0;
    std::vector<Borrow> borrows;
    bool               bound = false;

    ~Impl() {
        // Borrow-before-owner order is intentional.  Metadata tensors never
        // own the mapping and must not retain pointers across ggml_free.
        for (Borrow & borrow : borrows) {
            if (borrow.tensor != nullptr) {
                borrow.tensor->data = nullptr;
            }
        }
        bound = false;
        if (meta != nullptr) {
            ggml_free(meta);
            meta = nullptr;
        }
        if (gguf != nullptr) {
            gguf_free(gguf);
            gguf = nullptr;
        }
#if !defined(_WIN32)
        if (mapping != nullptr && mapping != MAP_FAILED) {
            munmap(mapping, mapping_size);
            mapping = nullptr;
        }
        if (fd >= 0) {
            close(fd);
            fd = -1;
        }
#endif
    }

#if !defined(_WIN32)
    MappedFileIdentity stat_fd(const std::string & phase) const {
        struct stat st {};
        if (fstat(fd, &st) != 0) {
            throw MappedWeightError(system_error("fstat failed " + phase));
        }
        if (!S_ISREG(st.st_mode)) {
            throw MappedWeightError("mapped model must remain a regular file " + phase);
        }
        return identity_from_stat(st, "mapped model " + phase);
    }

    MappedFileIdentity stat_path(const std::string & phase) const {
        struct stat st {};
        if (stat(path.c_str(), &st) != 0) {
            throw MappedWeightError(system_error("stat failed " + phase));
        }
        if (!S_ISREG(st.st_mode)) {
            throw MappedWeightError("mapped model path must name a regular file " + phase);
        }
        return identity_from_stat(st, "mapped model path " + phase);
    }

    void verify_identity(const std::string & phase) const {
        const MappedFileIdentity current_fd = stat_fd(phase);
        const MappedFileIdentity current_path = stat_path(phase);
        if (!same_identity(identity, current_fd) ||
            !same_identity(identity, current_path)) {
            throw MappedWeightError(
                "mapped model FD/inode/size identity changed " + phase);
        }
    }
#endif

    void validate_tensor_metadata() {
        using mapped_weight_detail::checked_add;
        using mapped_weight_detail::checked_pad;
        using mapped_weight_detail::validate_range;

        if (gguf == nullptr || meta == nullptr || mapping == nullptr) {
            throw MappedWeightError("mapped store validation has incomplete owners");
        }

        const size_t file_alignment = gguf_alignment(gguf);
        const size_t cpu_alignment =
            ggml_backend_buft_get_alignment(ggml_backend_cpu_buffer_type());
        if (cpu_alignment == 0 || (cpu_alignment & (cpu_alignment - 1)) != 0) {
            throw MappedWeightError("CPU backend reported invalid tensor alignment");
        }
        if ((reinterpret_cast<uintptr_t>(mapping) % cpu_alignment) != 0) {
            throw MappedWeightError("mmap base address is not CPU-aligned");
        }

        data_offset = gguf_get_data_offset(gguf);
        if ((data_offset % file_alignment) != 0 ||
            (data_offset % cpu_alignment) != 0) {
            throw MappedWeightError(
                "GGUF data section is not aligned for file and CPU access");
        }

        const int64_t count = gguf_get_n_tensors(gguf);
        if (count <= 0) {
            throw MappedWeightError("mapped model has no tensors");
        }
        borrows.reserve(static_cast<size_t>(count));
        std::set<std::string> names;
        size_t expected_relative_offset = 0;

        for (int64_t index = 0; index < count; ++index) {
            const char * raw_name = gguf_get_tensor_name(gguf, index);
            if (raw_name == nullptr || raw_name[0] == '\0') {
                throw MappedWeightError("mapped tensor has an empty name");
            }
            const std::string name(raw_name);
            if (!names.insert(name).second) {
                throw MappedWeightError("duplicate mapped tensor name: " + name);
            }

            ggml_tensor * tensor = ggml_get_tensor(meta, name.c_str());
            if (tensor == nullptr) {
                throw MappedWeightError(
                    "GGUF tensor is missing from ggml metadata context: " + name);
            }
            if (tensor->data != nullptr || tensor->buffer != nullptr) {
                throw MappedWeightError(
                    "metadata-only mapped tensor unexpectedly owns data/buffer: " + name);
            }

            const ggml_type type = gguf_get_tensor_type(gguf, index);
            if (!supported_mapped_type(type)) {
                throw MappedWeightError(
                    "unsupported mapped tensor type for " + name + ": " +
                    ggml_type_name(type) +
                    " (B2 permits only F32/F16/Q8_0/Q4_0/Q4_K)");
            }
            if (tensor->type != type) {
                throw MappedWeightError("GGUF/ggml tensor type mismatch: " + name);
            }
            const int64_t block_size = ggml_blck_size(type);
            if (block_size <= 0 || tensor->ne[0] <= 0 ||
                tensor->ne[0] % block_size != 0) {
                throw MappedWeightError(
                    "tensor row is not divisible by its type block size: " + name);
            }

            const size_t nbytes = ggml_nbytes(tensor);
            if (nbytes == 0 || nbytes != gguf_get_tensor_size(gguf, index)) {
                throw MappedWeightError("GGUF/ggml tensor byte-size mismatch: " + name);
            }
            const size_t relative_offset = gguf_get_tensor_offset(gguf, index);
            if (relative_offset != expected_relative_offset ||
                (relative_offset % file_alignment) != 0) {
                throw MappedWeightError(
                    "non-canonical or unaligned GGUF tensor offset: " + name);
            }

            validate_range(mapping_size, data_offset, relative_offset, nbytes, name);
            const size_t absolute_offset = checked_add(
                data_offset, relative_offset, name + " absolute offset");
            if ((absolute_offset % cpu_alignment) != 0) {
                throw MappedWeightError(
                    "mapped tensor address is not CPU-aligned: " + name);
            }
            borrows.push_back(Borrow{tensor, name, absolute_offset, nbytes});
            expected_relative_offset = checked_add(
                expected_relative_offset,
                checked_pad(nbytes, file_alignment, name),
                name + " padded extent");
        }
    }
};

MappedWeightStore::MappedWeightStore(const std::string & path) {
#if defined(_WIN32)
    (void) path;
    throw MappedWeightError("read-only mapped weights require a POSIX platform");
#else
    auto candidate = std::make_unique<Impl>();
    candidate->path = path;
    candidate->fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (candidate->fd < 0) {
        throw MappedWeightError(system_error("failed to open mapped model '" + path + "'"));
    }

    candidate->identity = candidate->stat_fd("at open");
    const MappedFileIdentity path_identity = candidate->stat_path("at open");
    if (!same_identity(candidate->identity, path_identity)) {
        throw MappedWeightError("mapped model path and opened FD identities differ at open");
    }
    candidate->mapping_size = static_cast<size_t>(candidate->identity.size);
    candidate->mapping = mmap(nullptr, candidate->mapping_size, PROT_READ,
                              MAP_PRIVATE, candidate->fd, 0);
    if (candidate->mapping == MAP_FAILED) {
        candidate->mapping = nullptr;
        throw MappedWeightError(system_error("mmap(PROT_READ|MAP_PRIVATE) failed"));
    }

    const int parse_fd = dup(candidate->fd);
    if (parse_fd < 0) {
        throw MappedWeightError(system_error("dup failed for GGUF metadata parse"));
    }
    FILE * parse_file = fdopen(parse_fd, "rb");
    if (parse_file == nullptr) {
        const int saved_errno = errno;
        close(parse_fd);
        errno = saved_errno;
        throw MappedWeightError(system_error("fdopen failed for GGUF metadata parse"));
    }
    std::unique_ptr<FILE, int (*)(FILE *)> file_owner(parse_file, std::fclose);

    gguf_init_params params;
    params.no_alloc = true;
    params.ctx = &candidate->meta;
    candidate->gguf = gguf_init_from_file_ptr(parse_file, params);
    if (candidate->gguf == nullptr || candidate->meta == nullptr) {
        throw MappedWeightError("failed to parse mapped GGUF metadata: " + path);
    }

    candidate->verify_identity("after metadata parse");
    candidate->validate_tensor_metadata();
    candidate->verify_identity("after tensor validation");
    impl_ = std::move(candidate);
#endif
}

MappedWeightStore::~MappedWeightStore() = default;

gguf_context * MappedWeightStore::gguf() const noexcept { return impl_->gguf; }
ggml_context * MappedWeightStore::model_context() const noexcept { return impl_->meta; }
const MappedFileIdentity & MappedWeightStore::identity() const noexcept {
    return impl_->identity;
}
const std::string & MappedWeightStore::path() const noexcept { return impl_->path; }
size_t MappedWeightStore::mapped_size() const noexcept { return impl_->mapping_size; }
size_t MappedWeightStore::data_offset() const noexcept { return impl_->data_offset; }
int64_t MappedWeightStore::tensor_count() const noexcept {
    return static_cast<int64_t>(impl_->borrows.size());
}
bool MappedWeightStore::tensors_bound() const noexcept { return impl_->bound; }
const void * MappedWeightStore::mapping_base() const noexcept { return impl_->mapping; }

std::vector<MappedTensorInfo> MappedWeightStore::tensor_infos() const {
    std::vector<MappedTensorInfo> result;
    result.reserve(impl_->borrows.size());
    for (const Impl::Borrow & borrow : impl_->borrows) {
        result.push_back(MappedTensorInfo{
            borrow.name,
            borrow.absolute_offset,
            borrow.nbytes,
            borrow.tensor->nb[1],
            borrow.tensor->ne[1],
            borrow.tensor->ne[0],
            static_cast<int>(borrow.tensor->type),
        });
    }
    return result;
}

MappedTensorInfo MappedWeightStore::tensor_info(const std::string & name) const {
    for (const MappedTensorInfo & info : tensor_infos()) {
        if (info.name == name) return info;
    }
    throw MappedWeightError("mapped tensor is not classified/validated: " + name);
}

void MappedWeightStore::verify_identity() const {
#if defined(_WIN32)
    throw MappedWeightError("read-only mapped weights require a POSIX platform");
#else
    impl_->verify_identity("during explicit verification");
#endif
}

void MappedWeightStore::bind_validated_tensors() {
    if (impl_->bound) {
        throw MappedWeightError("mapped tensor data is already bound");
    }
    verify_identity();
    try {
        for (Impl::Borrow & borrow : impl_->borrows) {
            if (borrow.tensor->data != nullptr || borrow.tensor->buffer != nullptr) {
                throw MappedWeightError(
                    "mapped metadata tensor became owned before binding");
            }
            borrow.tensor->data =
                static_cast<unsigned char *>(impl_->mapping) + borrow.absolute_offset;
        }
        verify_identity();
        impl_->bound = true;
    } catch (...) {
        for (Impl::Borrow & borrow : impl_->borrows) {
            borrow.tensor->data = nullptr;
        }
        throw;
    }
}

MappedModelPreparation::MappedModelPreparation(const std::string & path)
    : store_(path) {
    // Both validators consume only metadata.  No mapped byte is borrowed until
    // they have completed, making pointer publication a single final phase.
    arch_ = create_model_arch(store_.gguf(), store_.model_context());
    tokenizer_ = create_tokenizer(store_.gguf());
    discard_consumed_tokenizer_metadata(store_.gguf());
    store_.bind_validated_tensors();
    arch_->bind_weights(store_.model_context());
}

MappedModelPreparation::~MappedModelPreparation() = default;

} // namespace nanoembed
