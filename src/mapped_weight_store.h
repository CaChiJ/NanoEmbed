// Internal B2 read-only GGUF weight ownership.
//
// This remains deliberately unwired from the public Embedder. B3's internal
// phase runner consumes MappedModelPreparation; B4 alone may make the public
// use_streaming=1 gate select it. Keeping the seam independently testable
// preserves the eager gguf_init_from_file(no_alloc=false) path.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

struct gguf_context;
struct ggml_context;

namespace nanoembed {

class ModelArch;
class Tokenizer;

class MappedWeightError : public std::runtime_error {
public:
    explicit MappedWeightError(const std::string & what)
        : std::runtime_error(what) {}
};

struct MappedFileIdentity {
    uint64_t device = 0;
    uint64_t inode  = 0;
    uint64_t size   = 0;
};

struct MappedTensorInfo {
    std::string name;
    size_t      absolute_offset = 0;
    size_t      nbytes = 0;
    size_t      row_stride = 0;
    int64_t     row_count = 0;
    int64_t     elements_per_row = 0;
    int         ggml_type = -1;
};

// Owns one immutable regular file, its private read-only whole-file mapping,
// and the GGUF/ggml metadata-only contexts describing its tensors.  Opening
// validates every tensor range without assigning a data pointer.  Only the
// friend preparation object may publish the borrows after architecture and
// tokenizer validation has succeeded.
//
// Model files are a trusted immutable input for the lifetime of this object.
// Replacing or resizing one is detected at preparation/explicit identity
// checks, but hostile concurrent writes or truncation can still cause stale
// bytes or SIGBUS after validation.  Defending that trust boundary is outside
// M4; callers must not mutate a live model file.
class MappedWeightStore final {
public:
    explicit MappedWeightStore(const std::string & path);
    ~MappedWeightStore();

    MappedWeightStore(const MappedWeightStore &) = delete;
    MappedWeightStore & operator=(const MappedWeightStore &) = delete;
    MappedWeightStore(MappedWeightStore &&) = delete;
    MappedWeightStore & operator=(MappedWeightStore &&) = delete;

    gguf_context * gguf() const noexcept;
    ggml_context * model_context() const noexcept;

    const MappedFileIdentity & identity() const noexcept;
    const std::string & path() const noexcept;
    size_t mapped_size() const noexcept;
    size_t data_offset() const noexcept;
    int64_t tensor_count() const noexcept;
    bool tensors_bound() const noexcept;
    const void * mapping_base() const noexcept;
    std::vector<MappedTensorInfo> tensor_infos() const;
    MappedTensorInfo tensor_info(const std::string & name) const;

    // Recheck the opened descriptor and pathname against the identity retained
    // at open.  B3 can use this at its mode-initialization boundary without
    // touching mapped bytes.
    void verify_identity() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;

    void bind_validated_tensors();
    friend class MappedModelPreparation;
};

// Enforces the B2 ordering in one object:
//   metadata/ranges -> architecture -> tokenizer -> data borrows -> arch bind.
// store_ is declared first, so reverse member destruction releases tokenizer
// and ModelArch (and all retained tensor pointers) before the mapping owner.
class MappedModelPreparation final {
public:
    explicit MappedModelPreparation(const std::string & path);
    ~MappedModelPreparation();

    MappedModelPreparation(const MappedModelPreparation &) = delete;
    MappedModelPreparation & operator=(const MappedModelPreparation &) = delete;
    MappedModelPreparation(MappedModelPreparation &&) = delete;
    MappedModelPreparation & operator=(MappedModelPreparation &&) = delete;

    const MappedWeightStore & store() const noexcept { return store_; }
    const ModelArch & arch() const noexcept { return *arch_; }
    const Tokenizer & tokenizer() const noexcept { return *tokenizer_; }

private:
    MappedWeightStore           store_;
    std::unique_ptr<ModelArch>  arch_;
    std::unique_ptr<Tokenizer>  tokenizer_;
};

namespace mapped_weight_detail {

// Pure checked arithmetic is exposed only in the internal header so negative
// tests can cover metadata-sized overflows without asking gguf to allocate a
// malicious multi-exabyte tensor first.
size_t checked_add(size_t lhs, size_t rhs, const std::string & subject);
size_t checked_pad(size_t value, size_t alignment, const std::string & subject);
void validate_range(size_t file_size,
                    size_t data_offset,
                    size_t tensor_offset,
                    size_t nbytes,
                    const std::string & subject);

} // namespace mapped_weight_detail
} // namespace nanoembed
