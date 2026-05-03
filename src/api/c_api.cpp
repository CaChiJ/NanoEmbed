// NanoEmbed C ABI implementations.
//
// M3+: real implementations (M1 stubs replaced). nanoembed_load_model
// builds an Embedder under the opaque handle; embed delegates to it.

#include "nanoembed/nanoembed.h"

#include "embedder.h"

#include <cstring>
#include <exception>
#include <string>

// ---- Error handling ---------------------------------------------------------

namespace {

constexpr size_t   kErrorBufferSize = 512;
thread_local char  g_last_error[kErrorBufferSize] = {0};

void set_error(const char * msg) {
    if (msg == nullptr) {
        g_last_error[0] = '\0';
        return;
    }
    std::strncpy(g_last_error, msg, kErrorBufferSize - 1);
    g_last_error[kErrorBufferSize - 1] = '\0';
}

void clear_error() {
    g_last_error[0] = '\0';
}

void set_error_from_exception(const std::exception & e) {
    set_error(e.what());
}

void set_error_unknown() {
    set_error("unknown internal error");
}

nanoembed::EmbedderConfig from_params(const nanoembed_context_params & p) {
    nanoembed::EmbedderConfig c;
    c.n_threads   = p.n_threads;
    c.max_seq_len = p.max_seq_len;
    c.pooling     = (p.pooling == NANOEMBED_POOL_CLS)
                      ? nanoembed::PoolType::Cls
                      : nanoembed::PoolType::Mean;
    c.normalize   = (p.normalize != 0);
    return c;
}

} // namespace

// ---- Opaque handles ---------------------------------------------------------

struct nanoembed_model {
    explicit nanoembed_model(const std::string & path) : embedder(path) {}
    nanoembed::Embedder embedder;
};

struct nanoembed_context {
    nanoembed_model *         model = nullptr;
    nanoembed::EmbedderConfig cfg;
};

// ---- Public C API -----------------------------------------------------------

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
    if (gguf_path == nullptr) {
        set_error("gguf_path is null");
        return nullptr;
    }
    try {
        clear_error();
        return new nanoembed_model(std::string(gguf_path));
    } catch (const std::exception & e) {
        set_error_from_exception(e);
        return nullptr;
    } catch (...) {
        set_error_unknown();
        return nullptr;
    }
}

void nanoembed_free_model(nanoembed_model * model) {
    delete model;
}

int nanoembed_n_embed(const nanoembed_model * model) {
    if (model == nullptr) {
        set_error("model is null");
        return NANOEMBED_ERR_INVALID_ARG;
    }
    return model->embedder.n_embed();
}

int nanoembed_n_layer(const nanoembed_model * model) {
    if (model == nullptr) {
        set_error("model is null");
        return NANOEMBED_ERR_INVALID_ARG;
    }
    return model->embedder.n_layer();
}

nanoembed_context * nanoembed_new_context(nanoembed_model *         model,
                                          nanoembed_context_params  params) {
    if (model == nullptr) {
        set_error("model is null");
        return nullptr;
    }
    if (params.max_batch <= 0 || params.max_seq_len <= 0) {
        set_error("max_batch and max_seq_len must be positive");
        return nullptr;
    }
    try {
        auto * c   = new nanoembed_context;
        c->model   = model;
        c->cfg     = from_params(params);
        clear_error();
        return c;
    } catch (const std::exception & e) {
        set_error_from_exception(e);
        return nullptr;
    } catch (...) {
        set_error_unknown();
        return nullptr;
    }
}

void nanoembed_free_context(nanoembed_context * ctx) {
    delete ctx;
}

int nanoembed_embed(nanoembed_context * ctx,
                    const char *        text,
                    float *             out) {
    if (ctx == nullptr || text == nullptr || out == nullptr) {
        set_error("ctx / text / out must be non-null");
        return NANOEMBED_ERR_INVALID_ARG;
    }
    try {
        ctx->model->embedder.embed(std::string(text), ctx->cfg, out);
        clear_error();
        return NANOEMBED_OK;
    } catch (const std::exception & e) {
        set_error_from_exception(e);
        return NANOEMBED_ERR_INTERNAL;
    } catch (...) {
        set_error_unknown();
        return NANOEMBED_ERR_INTERNAL;
    }
}

int nanoembed_embed_batch(nanoembed_context *  ctx,
                          const char * const * texts,
                          int                  n_texts,
                          float *              out) {
    if (ctx == nullptr || texts == nullptr || out == nullptr) {
        set_error("ctx / texts / out must be non-null");
        return NANOEMBED_ERR_INVALID_ARG;
    }
    if (n_texts < 0) {
        set_error("n_texts must be non-negative");
        return NANOEMBED_ERR_INVALID_ARG;
    }
    try {
        const int H = ctx->model->embedder.n_embed();
        // M3 baseline: loop the single-input path. M5 replaces this with the
        // batched runner — same C ABI, real batching internally.
        for (int i = 0; i < n_texts; ++i) {
            if (texts[i] == nullptr) {
                set_error("null pointer in texts[]");
                return NANOEMBED_ERR_INVALID_ARG;
            }
            ctx->model->embedder.embed(std::string(texts[i]), ctx->cfg,
                                       out + static_cast<size_t>(i) * static_cast<size_t>(H));
        }
        clear_error();
        return NANOEMBED_OK;
    } catch (const std::exception & e) {
        set_error_from_exception(e);
        return NANOEMBED_ERR_INTERNAL;
    } catch (...) {
        set_error_unknown();
        return NANOEMBED_ERR_INTERNAL;
    }
}

} // extern "C"
