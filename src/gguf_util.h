// Architecture-agnostic GGUF reading helpers.
//
// These sit below the per-family scanners: every model family has to read
// metadata keys, look tensors up by name and assert their shapes, and the
// error text for "missing key" or "wrong shape" should not differ between
// families. Anything that mentions a specific architecture belongs one level
// up, in that family's scanner (gguf_scanner.h for bert, gemma3_arch.h for
// gemma3).
//
// Declarations only — the definitions live in gguf_util.cpp so that ggml.h
// and gguf.h stay out of every translation unit that just needs a TensorRef.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

struct gguf_context;
struct ggml_context;

namespace nanoembed {

// Domain exception thrown on any scan failure. Inherits std::runtime_error
// so it can be caught generically at the C ABI boundary.
class ScanError : public std::runtime_error {
public:
    explicit ScanError(const std::string & what) : std::runtime_error(what) {}
};

// Reference to one tensor in the GGUF file. The data itself is not loaded
// yet; inference code uses (gguf_index, offset, ne, ggml_type) to pull bytes
// via ggml/mmap.
struct TensorRef {
    int64_t gguf_index  = -1;            // -1 if slot unfilled
    int     ggml_type   = 0;             // ggml_type enum (F32/F16/Q8_0/...)
    int64_t ne[4]       = {0, 0, 0, 0};  // tensor shape (ggml is up to 4D)
    size_t  size_bytes  = 0;
    size_t  data_offset = 0;             // byte offset within tensor data blob

    bool valid() const noexcept { return gguf_index >= 0; }
};

namespace gguf_util {

// ---- Cleanup helpers ---------------------------------------------------

struct GgufDeleter {
    void operator()(gguf_context * p) const noexcept;
};
struct GgmlDeleter {
    void operator()(ggml_context * p) const noexcept;
};

using GgufPtr = std::unique_ptr<gguf_context, GgufDeleter>;
using GgmlPtr = std::unique_ptr<ggml_context, GgmlDeleter>;

[[noreturn]] void fail(const std::string & msg);

// ---- KV helpers --------------------------------------------------------

int64_t     require_kv(gguf_context * ctx, const char * key);
int         read_u32_as_int(gguf_context * ctx, const char * key);
std::string read_str(gguf_context * ctx, const char * key);

// Optional readers: return `fallback` when the key is absent or carries a
// type other than the expected one. Used for metadata a family can live
// without (an epsilon with a documented default, a flag that defaults off).
int   read_u32_or(gguf_context * ctx, const char * key, int fallback);
float read_f32_or(gguf_context * ctx, const char * key, float fallback);
bool  read_bool_or(gguf_context * ctx, const char * key, bool fallback);

// ---- Tensor helpers ----------------------------------------------------

TensorRef require_tensor(gguf_context *      gguf,
                         ggml_context *      meta,
                         const std::string & name);

// Same, but yields an unfilled TensorRef instead of throwing when the tensor
// is absent. Callers check TensorRef::valid().
TensorRef optional_tensor(gguf_context *      gguf,
                          ggml_context *      meta,
                          const std::string & name);

// llama.cpp's per-block naming convention, shared by every family that
// writes GGUF through it: "blk.<i>.<suffix>".
std::string layer_tensor_name(int layer_idx, const char * suffix);

void validate_shape_2d(const TensorRef &   r,
                       int64_t             expected_rows,
                       int64_t             expected_cols,
                       const std::string & name);

void validate_shape_1d(const TensorRef &   r,
                       int64_t             expected,
                       const std::string & name);

} // namespace gguf_util
} // namespace nanoembed
