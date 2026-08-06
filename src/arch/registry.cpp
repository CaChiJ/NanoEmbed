#include "arch/model_arch.h"

#include "arch/bert_arch.h"
#include "gguf_scanner.h"

#include "gguf.h"

#include <memory>

namespace nanoembed {

namespace {

// Read general.architecture without committing to a family, so an unsupported
// model fails with its actual tag rather than a shape mismatch deep in a
// scanner that assumed the wrong layout.
std::string peek_architecture(const std::string & gguf_path) {
    ggml_context * meta = nullptr;
    gguf_init_params p;
    p.no_alloc = true;
    p.ctx      = &meta;

    gguf_context * g = gguf_init_from_file(gguf_path.c_str(), p);
    if (g == nullptr) {
        throw ScanError("failed to open GGUF file: " + gguf_path);
    }

    std::string arch;
    const int64_t k = gguf_find_key(g, "general.architecture");
    if (k >= 0 && gguf_get_kv_type(g, k) == GGUF_TYPE_STRING) {
        arch = gguf_get_val_str(g, k);
    }

    gguf_free(g);
    if (meta) ggml_free(meta);

    if (arch.empty()) {
        throw ScanError("GGUF has no general.architecture string: " + gguf_path);
    }
    return arch;
}

} // namespace

std::unique_ptr<ModelArch> create_model_arch(const std::string & gguf_path) {
    const std::string arch = peek_architecture(gguf_path);

    if (arch == "bert") {
        return std::make_unique<BertModelArch>(gguf_path);
    }

    // jina-embeddings-v5-text-nano ships as "eurobert": a Llama-style encoder
    // (rotary positions, RMSNorm, SwiGLU, unbiased projections) that shares no
    // tensor layout with BERT. Named explicitly so the error tells the user
    // what is missing rather than just refusing.
    if (arch == "eurobert") {
        throw ScanError(
            "architecture 'eurobert' is not implemented yet — "
            "planned as PLAN.md M3.6 (rotary/RMSNorm/SwiGLU + byte-level BPE)");
    }

    throw ScanError("unsupported architecture: '" + arch + "'");
}

} // namespace nanoembed
