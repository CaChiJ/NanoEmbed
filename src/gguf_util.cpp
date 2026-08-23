#include "gguf_util.h"

#include "ggml.h"
#include "gguf.h"

#include <sstream>

namespace nanoembed::gguf_util {

void GgufDeleter::operator()(gguf_context * p) const noexcept { if (p) gguf_free(p); }
void GgmlDeleter::operator()(ggml_context * p) const noexcept { if (p) ggml_free(p); }

void fail(const std::string & msg) {
    throw ScanError(msg);
}

// ---- KV helpers --------------------------------------------------------

int64_t require_kv(gguf_context * ctx, const char * key) {
    int64_t i = gguf_find_key(ctx, key);
    if (i < 0) fail(std::string("missing required metadata key: ") + key);
    return i;
}

int read_u32_as_int(gguf_context * ctx, const char * key) {
    int64_t i = require_kv(ctx, key);
    if (gguf_get_kv_type(ctx, i) != GGUF_TYPE_UINT32) {
        fail(std::string("metadata key has wrong type (expected u32): ") + key);
    }
    return static_cast<int>(gguf_get_val_u32(ctx, i));
}

std::string read_str(gguf_context * ctx, const char * key) {
    int64_t i = require_kv(ctx, key);
    if (gguf_get_kv_type(ctx, i) != GGUF_TYPE_STRING) {
        fail(std::string("metadata key has wrong type (expected string): ") + key);
    }
    return std::string(gguf_get_val_str(ctx, i));
}

int read_u32_or(gguf_context * ctx, const char * key, int fallback) {
    int64_t i = gguf_find_key(ctx, key);
    if (i < 0) return fallback;
    if (gguf_get_kv_type(ctx, i) != GGUF_TYPE_UINT32) return fallback;
    return static_cast<int>(gguf_get_val_u32(ctx, i));
}

float read_f32_or(gguf_context * ctx, const char * key, float fallback) {
    int64_t i = gguf_find_key(ctx, key);
    if (i < 0) return fallback;
    if (gguf_get_kv_type(ctx, i) != GGUF_TYPE_FLOAT32) return fallback;
    return gguf_get_val_f32(ctx, i);
}

bool read_bool_or(gguf_context * ctx, const char * key, bool fallback) {
    int64_t i = gguf_find_key(ctx, key);
    if (i < 0) return fallback;
    if (gguf_get_kv_type(ctx, i) != GGUF_TYPE_BOOL) return fallback;
    return gguf_get_val_bool(ctx, i);
}

// ---- Tensor helpers ----------------------------------------------------

namespace {

// Fill a TensorRef from a tensor known to exist in both contexts.
TensorRef make_ref(gguf_context * gguf, ggml_tensor * t, int64_t idx) {
    TensorRef r;
    r.gguf_index  = idx;
    r.ggml_type   = static_cast<int>(gguf_get_tensor_type(gguf, idx));
    r.size_bytes  = gguf_get_tensor_size(gguf, idx);
    r.data_offset = gguf_get_tensor_offset(gguf, idx);
    for (int d = 0; d < 4; ++d) {
        r.ne[d] = t->ne[d];
    }
    return r;
}

} // namespace

TensorRef require_tensor(gguf_context *      gguf,
                         ggml_context *      meta,
                         const std::string & name) {
    const int64_t idx = gguf_find_tensor(gguf, name.c_str());
    if (idx < 0) fail("missing required tensor: " + name);

    ggml_tensor * t = ggml_get_tensor(meta, name.c_str());
    if (t == nullptr) {
        fail("tensor present in gguf but not in ggml meta: " + name);
    }
    return make_ref(gguf, t, idx);
}

TensorRef optional_tensor(gguf_context *      gguf,
                          ggml_context *      meta,
                          const std::string & name) {
    const int64_t idx = gguf_find_tensor(gguf, name.c_str());
    if (idx < 0) return TensorRef{};

    ggml_tensor * t = ggml_get_tensor(meta, name.c_str());
    if (t == nullptr) {
        fail("tensor present in gguf but not in ggml meta: " + name);
    }
    return make_ref(gguf, t, idx);
}

std::string layer_tensor_name(int layer_idx, const char * suffix) {
    std::ostringstream s;
    s << "blk." << layer_idx << "." << suffix;
    return s.str();
}

void validate_shape_2d(const TensorRef &   r,
                       int64_t             expected_rows,
                       int64_t             expected_cols,
                       const std::string & name) {
    // ggml stores a 2D matrix W of shape [in, out] with ne = {in, out, 1, 1}.
    // We pass (rows, cols) in the matmul-natural order; convert here.
    if (r.ne[0] != expected_rows || r.ne[1] != expected_cols ||
        r.ne[2] != 1 || r.ne[3] != 1) {
        std::ostringstream s;
        s << "tensor shape mismatch for " << name
          << ": expected ne=[" << expected_rows << "," << expected_cols << ",1,1]"
          << " got ne=[" << r.ne[0] << "," << r.ne[1] << "," << r.ne[2] << "," << r.ne[3] << "]";
        fail(s.str());
    }
}

void validate_shape_1d(const TensorRef &   r,
                       int64_t             expected,
                       const std::string & name) {
    if (r.ne[0] != expected || r.ne[1] != 1 || r.ne[2] != 1 || r.ne[3] != 1) {
        std::ostringstream s;
        s << "tensor shape mismatch for " << name
          << ": expected ne=[" << expected << ",1,1,1]"
          << " got ne=[" << r.ne[0] << "," << r.ne[1] << "," << r.ne[2] << "," << r.ne[3] << "]";
        fail(s.str());
    }
}

} // namespace nanoembed::gguf_util
