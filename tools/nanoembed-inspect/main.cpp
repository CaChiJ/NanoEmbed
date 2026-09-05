// nanoembed-inspect: dump GGUF metadata and (when BERT) the structured manifest.
//
// Usage:
//   nanoembed-inspect <gguf-file> [--tensors] [--graph]
//
// Default output: header summary + metadata KV + BERT manifest (if applicable).
// --tensors also dumps the raw tensor table (long); --graph loads the model
// and reports the reserved activation buffer size.

#include "ggml.h"
#include "gguf.h"

#include "arch/model_arch.h"
#include "embedder.h"
#include "gguf_scanner.h"

#include <cinttypes>
#include <memory>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
    std::printf("version=%u\n",                 gguf_get_version(ctx));
    std::printf("tensors=%" PRId64 "\n",        gguf_get_n_tensors(ctx));
    std::printf("metadata_kv=%" PRId64 "\n",    gguf_get_n_kv(ctx));
    std::printf("alignment=%zu\n",              gguf_get_alignment(ctx));
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

const char * pool_name(nanoembed::PoolType p) {
    switch (p) {
        case nanoembed::PoolType::Cls:  return "cls";
        case nanoembed::PoolType::Last: return "last";
        default:                        return "mean";
    }
}

// Family-agnostic view: whatever create_model_arch() could make sense of.
// Works for any supported architecture and needs no tokenizer, so it reports
// a model whose tokenizer family is still unimplemented.
void print_arch(const nanoembed::ModelArch & arch) {
    const nanoembed::ArchParams & a = arch.params();
    std::printf("\n# architecture\n");
    std::printf("  name=%s layers=%d hidden=%d ffn=%d vocab=%d max_seq_len=%d\n",
                a.name.c_str(), a.n_layer, a.n_embed, a.n_ff, a.n_vocab, a.max_seq_len);
    std::printf("  heads=%d kv_heads=%d head_dim=%d%s\n",
                a.n_head, a.n_head_kv, a.head_dim,
                a.n_head_kv < a.n_head ? "  (grouped/multi-query)" : "");
    std::printf("  norm_eps=%g attention=%s",
                static_cast<double>(a.norm_eps), a.causal ? "causal" : "bidirectional");
    if (a.rope_freq_base > 0.0f) {
        std::printf(" rope_freq_base=%g", static_cast<double>(a.rope_freq_base));
    } else {
        std::printf(" positions=learned");
    }
    std::printf("\n  default pooling=%s\n", pool_name(arch.default_pooling()));

    const nanoembed::InputRequirements r = arch.inputs();
    std::printf("  graph inputs: token_ids%s%s%s%s\n",
                r.needs_learned_pos_ids ? " learned_pos_ids" : "",
                r.needs_rope_pos_ids    ? " rope_pos_ids"    : "",
                r.needs_type_ids        ? " type_ids"        : "",
                r.uses_kq_mask          ? " kq_mask(batch)"  : "");
}

void print_manifest(const nanoembed::ModelManifest & m) {
    const auto & a = m.arch;
    std::printf("\n# bert manifest\n");
    std::printf("  layers=%d hidden=%d heads=%d head_dim=%d ffn=%d\n",
                a.n_layer, a.n_embed, a.n_head,
                a.n_head > 0 ? a.n_embed / a.n_head : 0,
                a.n_ff);
    std::printf("  vocab=%d max_seq_len=%d ln_eps=%g\n",
                a.n_vocab, a.max_seq_len,
                static_cast<double>(a.layer_norm_eps));

    auto t_str = [](int t) { return ggml_type_name(static_cast<ggml_type>(t)); };
    std::printf("  embedding tensors:\n");
    std::printf("    token_embd        ne=[%" PRId64 ",%" PRId64 "] type=%s\n",
                m.tok_embed_w.ne[0], m.tok_embed_w.ne[1], t_str(m.tok_embed_w.ggml_type));
    std::printf("    position_embd     ne=[%" PRId64 ",%" PRId64 "] type=%s\n",
                m.pos_embed_w.ne[0], m.pos_embed_w.ne[1], t_str(m.pos_embed_w.ggml_type));
    std::printf("    token_types       ne=[%" PRId64 ",%" PRId64 "] type=%s\n",
                m.type_embed_w.ne[0], m.type_embed_w.ne[1], t_str(m.type_embed_w.ggml_type));

    std::printf("  per-layer tensor count: 16 (q/k/v/o w+b, 2 LN w+b, ffn_up/down w+b)\n");
    std::printf("  layer 0 spot-check:\n");
    if (!m.layers.empty()) {
        const auto & s = m.layers.front();
        std::printf("    blk.0.attn_q.weight  ne=[%" PRId64 ",%" PRId64 "] type=%s\n",
                    s.attn_q_w.ne[0], s.attn_q_w.ne[1], t_str(s.attn_q_w.ggml_type));
        std::printf("    blk.0.ffn_up.weight  ne=[%" PRId64 ",%" PRId64 "] type=%s\n",
                    s.ffn_up_w.ne[0], s.ffn_up_w.ne[1], t_str(s.ffn_up_w.ggml_type));
    }

    if (m.pooler_w.valid()) {
        std::printf("  pooler: present\n");
    }
}

// Costs a full weight load, so it is opt-in rather than part of the default
// dump: constructing an Embedder is what sizes the graph buffer.
void print_graph_budget(const char * path) {
    std::printf("\n# graph budget\n");
    try {
        const nanoembed::Embedder e(path);
        nanoembed::ComputeScratch scratch;
        e.reserve(scratch, 512, e.default_pooling(), /*normalize=*/true);
        const double mib = static_cast<double>(e.graph_buffer_size(scratch)) / (1024.0 * 1024.0);
        std::printf("  architecture: %s (model context length %d)\n",
                    e.architecture().c_str(), e.max_seq_len());
        std::printf("  activations reserved for seq_len=%d: %.2f MiB\n",
                    e.reserved_seq_len(scratch), mib);
    } catch (const std::exception & ex) {
        std::printf("  unavailable — %s\n", ex.what());
    }
}

void print_usage(const char * prog) {
    std::fprintf(stderr,
        "usage: %s <gguf-file> [--tensors] [--graph]\n"
        "\n"
        "Prints GGUF header summary, metadata KV pairs, and the BERT\n"
        "manifest (when applicable). Use --tensors to also dump the raw\n"
        "tensor table, and --graph to load the model and report the\n"
        "reserved activation buffer size.\n",
        prog);
}

} // namespace

int main(int argc, char ** argv) {
    bool         show_tensors = false;
    bool         show_graph   = false;
    const char * path         = nullptr;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--tensors") == 0) {
            show_tensors = true;
        } else if (std::strcmp(argv[i], "--graph") == 0) {
            show_graph = true;
        } else if (path == nullptr) {
            path = argv[i];
        } else {
            std::fprintf(stderr, "error: unexpected argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 2;
        }
    }

    if (path == nullptr) {
        print_usage(argv[0]);
        return 2;
    }

    gguf_init_params params;
    params.no_alloc = true;
    params.ctx      = nullptr;

    gguf_context * ctx = gguf_init_from_file(path, params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "error: failed to open GGUF file: %s\n", path);
        return 1;
    }

    print_summary(ctx, path);
    print_metadata(ctx);

    // Best-effort structured views, in addition to the raw dump. The first is
    // family-agnostic; the second adds BERT's per-tensor detail and is simply
    // absent for other architectures.
    try {
        const std::unique_ptr<nanoembed::ModelArch> arch =
            nanoembed::create_model_arch(path);
        print_arch(*arch);
    } catch (const nanoembed::ScanError & e) {
        std::printf("\n# architecture: scan failed — %s\n", e.what());
    }

    try {
        nanoembed::ScanResult scan = nanoembed::scan_gguf(path);
        print_manifest(scan.manifest());
    } catch (const nanoembed::ScanError &) {
        // Not a BERT file; the architecture section above already reported it.
    }

    if (show_tensors) {
        print_tensors(ctx);
    }

    if (show_graph) {
        print_graph_budget(path);
    }

    gguf_free(ctx);
    return 0;
}
