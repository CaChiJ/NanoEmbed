// nanoembed-inspect: dump GGUF metadata.
//
// M1 scope: header summary (version, tensor count, KV count) plus a
// best-effort listing of metadata KV pairs and tensors. The structured
// BERT layer map lands in M2 alongside the GGUF scanner.

#include "ggml.h"
#include "gguf.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace {

void print_kv_value(const gguf_context * ctx, int64_t i) {
    const gguf_type t = gguf_get_kv_type(ctx, i);
    switch (t) {
        case GGUF_TYPE_UINT8:   std::printf("%u",   gguf_get_val_u8 (ctx, i)); break;
        case GGUF_TYPE_INT8:    std::printf("%d",   gguf_get_val_i8 (ctx, i)); break;
        case GGUF_TYPE_UINT16:  std::printf("%u",   gguf_get_val_u16(ctx, i)); break;
        case GGUF_TYPE_INT16:   std::printf("%d",   gguf_get_val_i16(ctx, i)); break;
        case GGUF_TYPE_UINT32:  std::printf("%u",   gguf_get_val_u32(ctx, i)); break;
        case GGUF_TYPE_INT32:   std::printf("%d",   gguf_get_val_i32(ctx, i)); break;
        case GGUF_TYPE_FLOAT32: std::printf("%g",   gguf_get_val_f32(ctx, i)); break;
        case GGUF_TYPE_UINT64:  std::printf("%" PRIu64, gguf_get_val_u64(ctx, i)); break;
        case GGUF_TYPE_INT64:   std::printf("%" PRId64, gguf_get_val_i64(ctx, i)); break;
        case GGUF_TYPE_FLOAT64: std::printf("%g",   gguf_get_val_f64(ctx, i)); break;
        case GGUF_TYPE_BOOL:    std::printf("%s",   gguf_get_val_bool(ctx, i) ? "true" : "false"); break;
        case GGUF_TYPE_STRING: {
            const char * s = gguf_get_val_str(ctx, i);
            // Truncate long strings (vocab merges, tokenizer config) for readability.
            const size_t maxlen = 80;
            const size_t len = std::strlen(s);
            if (len <= maxlen) {
                std::printf("\"%s\"", s);
            } else {
                std::printf("\"%.80s...\" (len=%zu)", s, len);
            }
            break;
        }
        case GGUF_TYPE_ARRAY: {
            const gguf_type at = gguf_get_arr_type(ctx, i);
            const size_t    n  = gguf_get_arr_n   (ctx, i);
            std::printf("[array of %s, n=%zu]", gguf_type_name(at), n);
            break;
        }
        default:
            std::printf("[unknown type %d]", static_cast<int>(t));
            break;
    }
}

void print_summary(const gguf_context * ctx, const char * path) {
    std::printf("file=%s\n", path);
    std::printf("version=%u\n",     gguf_get_version(ctx));
    std::printf("tensors=%" PRId64 "\n", gguf_get_n_tensors(ctx));
    std::printf("metadata_kv=%" PRId64 "\n", gguf_get_n_kv(ctx));
    std::printf("alignment=%zu\n",   gguf_get_alignment(ctx));
}

void print_metadata(const gguf_context * ctx) {
    const int64_t n_kv = gguf_get_n_kv(ctx);
    std::printf("\n# metadata\n");
    for (int64_t i = 0; i < n_kv; ++i) {
        std::printf("  %s (%s) = ",
                    gguf_get_key(ctx, i),
                    gguf_type_name(gguf_get_kv_type(ctx, i)));
        print_kv_value(ctx, i);
        std::printf("\n");
    }
}

void print_tensors(const gguf_context * ctx) {
    const int64_t n_tensors = gguf_get_n_tensors(ctx);
    std::printf("\n# tensors\n");
    for (int64_t i = 0; i < n_tensors; ++i) {
        std::printf("  [%4" PRId64 "] %-48s type=%-8s size=%zu\n",
                    i,
                    gguf_get_tensor_name(ctx, i),
                    ggml_type_name(gguf_get_tensor_type(ctx, i)),
                    gguf_get_tensor_size(ctx, i));
    }
}

} // namespace

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: %s <gguf-file>\n"
            "\n"
            "Prints GGUF header summary, metadata KV pairs, and tensor list.\n",
            argv[0]);
        return 2;
    }

    const char * path = argv[1];

    gguf_init_params params;
    params.no_alloc = true;   // header-only; do not read tensor data.
    params.ctx      = nullptr;

    gguf_context * ctx = gguf_init_from_file(path, params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "error: failed to open GGUF file: %s\n", path);
        return 1;
    }

    print_summary(ctx, path);
    print_metadata(ctx);
    print_tensors(ctx);

    gguf_free(ctx);
    return 0;
}
