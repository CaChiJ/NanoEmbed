// NanoEmbed C ABI implementations.
//
// M1: every function is a stub. nanoembed_context_default_params and
// nanoembed_last_error are implemented for real because they have no
// dependencies and downstream code (tests, the inspect tool, future
// callers) needs them to behave correctly from day one.
//
// M3+ replaces the stubs in load/embed/etc. with real implementations.

#include "nanoembed/nanoembed.h"

#include <cstring>

namespace {

// Thread-local fixed-size buffer keeps the C ABI free of std::string
// destruction subtleties. Plenty of room for a path + ggml error.
constexpr size_t kErrorBufferSize = 512;
thread_local char g_last_error[kErrorBufferSize] = {0};

void set_error(const char * msg) {
    if (msg == nullptr) {
        g_last_error[0] = '\0';
        return;
    }
    std::strncpy(g_last_error, msg, kErrorBufferSize - 1);
    g_last_error[kErrorBufferSize - 1] = '\0';
}

constexpr const char * kStubMsg =
    "not implemented in M1 stub — see PLAN.md milestone matrix";

} // namespace

extern "C" {

nanoembed_context_params nanoembed_context_default_params(void) {
    nanoembed_context_params p;
    p.n_threads     = 0;
    p.max_batch     = 64;
    p.max_seq_len   = 512;
    p.use_streaming = 0;
    p.pooling       = NANOEMBED_POOL_MEAN;
    p.normalize     = 1;
    return p;
}

const char * nanoembed_last_error(void) {
    return g_last_error;
}

nanoembed_model * nanoembed_load_model(const char * gguf_path) {
    (void) gguf_path;
    set_error(kStubMsg);
    return nullptr;
}

void nanoembed_free_model(nanoembed_model * model) {
    (void) model;
}

int nanoembed_n_embed(const nanoembed_model * model) {
    (void) model;
    set_error(kStubMsg);
    return NANOEMBED_ERR_NOT_IMPL;
}

int nanoembed_n_layer(const nanoembed_model * model) {
    (void) model;
    set_error(kStubMsg);
    return NANOEMBED_ERR_NOT_IMPL;
}

nanoembed_context * nanoembed_new_context(nanoembed_model * model,
                                          nanoembed_context_params params) {
    (void) model;
    (void) params;
    set_error(kStubMsg);
    return nullptr;
}

void nanoembed_free_context(nanoembed_context * ctx) {
    (void) ctx;
}

int nanoembed_embed(nanoembed_context * ctx,
                    const char *        text,
                    float *             out) {
    (void) ctx;
    (void) text;
    (void) out;
    set_error(kStubMsg);
    return NANOEMBED_ERR_NOT_IMPL;
}

int nanoembed_embed_batch(nanoembed_context *  ctx,
                          const char * const * texts,
                          int                  n_texts,
                          float *              out) {
    (void) ctx;
    (void) texts;
    (void) n_texts;
    (void) out;
    set_error(kStubMsg);
    return NANOEMBED_ERR_NOT_IMPL;
}

} // extern "C"
