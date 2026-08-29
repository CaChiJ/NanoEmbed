/*
 * NanoEmbed — text embedding for edge devices, constrained to tens of MB RAM.
 *
 * This is the stable public C ABI. Signatures are frozen as of M1.
 * Implementations are filled in across M3 (in-memory), M4 (streaming),
 * and M5 (true batched). See PLAN.md for the milestone matrix.
 *
 * Conventions:
 *   - Functions returning int return 0 on success, a negative
 *     nanoembed_status error code on failure. Functions that produce a
 *     value (n_embed, n_layer) return the value when positive, the
 *     error code when negative.
 *   - Pointer-returning functions return NULL on failure;
 *     call nanoembed_last_error() for a thread-local error message.
 *   - Output buffers are caller-owned. Single embed: out length = n_embed.
 *     Batch embed: out length = n_texts * n_embed,
 *     with embedding i at out[i * n_embed .. (i+1) * n_embed).
 *   - free / destroy functions accept NULL safely (no-op).
 *   - All functions are thread-safe with respect to *different* model and
 *     context handles. A single nanoembed_context handle is NOT thread-safe;
 *     parallelize by creating multiple contexts (an mmap'd model is shared
 *     so this is cheap).
 */

#ifndef NANOEMBED_H
#define NANOEMBED_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- Versioning ----------------------------------------------------- */

#define NANOEMBED_VERSION_MAJOR 0
#define NANOEMBED_VERSION_MINOR 1
#define NANOEMBED_VERSION_PATCH 0

/* ---- Visibility / linkage ------------------------------------------- */

#if defined(_WIN32)
#  if defined(NANOEMBED_BUILD_SHARED)
#    define NANOEMBED_API __declspec(dllexport)
#  elif defined(NANOEMBED_USE_SHARED)
#    define NANOEMBED_API __declspec(dllimport)
#  else
#    define NANOEMBED_API
#  endif
#else
#  if defined(NANOEMBED_BUILD_SHARED) || defined(NANOEMBED_USE_SHARED)
#    define NANOEMBED_API __attribute__((visibility("default")))
#  else
#    define NANOEMBED_API
#  endif
#endif

/* ---- Status codes --------------------------------------------------- */

typedef enum {
    NANOEMBED_OK              =  0,
    NANOEMBED_ERR_INVALID_ARG = -1,
    NANOEMBED_ERR_FILE        = -2, /* file not found / unreadable */
    NANOEMBED_ERR_FORMAT      = -3, /* not a valid GGUF file */
    NANOEMBED_ERR_ARCH        = -4, /* unsupported model architecture */
    NANOEMBED_ERR_TENSOR      = -5, /* expected tensor missing or shape mismatch */
    NANOEMBED_ERR_TOKENIZE    = -6,
    NANOEMBED_ERR_OOM         = -7,
    NANOEMBED_ERR_NOT_IMPL    = -8, /* M1 stub or feature not yet enabled */
    NANOEMBED_ERR_INTERNAL    = -99
} nanoembed_status;

/* ---- Pooling -------------------------------------------------------- */

typedef enum {
    /* Whatever the loaded model was trained with. The default, because the
       right choice is a property of the model, not of the caller: pooling a
       last-token model with MEAN returns a confident, wrong vector and
       nothing in the output says so. */
    NANOEMBED_POOL_MODEL_DEFAULT = -1,
    NANOEMBED_POOL_MEAN          =  0, /* mean over non-pad tokens */
    NANOEMBED_POOL_CLS           =  1, /* first token (BERT's [CLS]) */
    NANOEMBED_POOL_LAST          =  2  /* last token (decoder-derived models) */
} nanoembed_pool_type;

/* ---- Opaque handles ------------------------------------------------- */

typedef struct nanoembed_model   nanoembed_model;
typedef struct nanoembed_context nanoembed_context;

/* ---- Context parameters --------------------------------------------- */

typedef struct {
    int                  n_threads;     /* 0 = auto (performance cores)       */
    int                  max_batch;     /* > 0; over-batch is auto-subdivided */
    int                  max_seq_len;   /* >= 2; longer inputs are truncated, */
                                        /* and clamped to the model's context */
    int                  use_streaming; /* 0=eager, 1=Linux mmap streaming    */
    nanoembed_pool_type  pooling;
    int                  normalize;     /* 0/1 — L2 normalize output          */
} nanoembed_context_params;

NANOEMBED_API nanoembed_context_params nanoembed_context_default_params(void);

/* ---- Model: load / free / metadata --------------------------------- */

NANOEMBED_API nanoembed_model * nanoembed_load_model(const char * gguf_path);
NANOEMBED_API void              nanoembed_free_model(nanoembed_model * model);

NANOEMBED_API int               nanoembed_n_embed(const nanoembed_model * model);
NANOEMBED_API int               nanoembed_n_layer(const nanoembed_model * model);

/* The model's own context length. Callers need it to choose max_seq_len:
   activation memory grows with its square, so a long-context model's full
   window can be far more than a machine has. */
NANOEMBED_API int               nanoembed_model_max_seq_len(const nanoembed_model * model);

/* Pooling the model was trained for. Never returns MODEL_DEFAULT. */
NANOEMBED_API nanoembed_pool_type
                                nanoembed_model_default_pooling(const nanoembed_model * model);

/* ---- Context: per-instance runtime state --------------------------- */

NANOEMBED_API nanoembed_context * nanoembed_new_context(
        nanoembed_model * model,
        nanoembed_context_params params);

NANOEMBED_API void                nanoembed_free_context(nanoembed_context * ctx);

/* ---- Inference ------------------------------------------------------ */

/* Single text. out must hold n_embed floats. */
NANOEMBED_API int nanoembed_embed(
        nanoembed_context * ctx,
        const char *        text,
        float *             out);

/* n_texts inputs. out must hold n_texts * n_embed floats. Inputs are
 * length-bucketed internally, split at max_batch, and returned in caller
 * order. Invalid arguments leave out untouched; execution failures poison
 * the entire output with NaN. */
NANOEMBED_API int nanoembed_embed_batch(
        nanoembed_context * ctx,
        const char * const * texts,
        int                 n_texts,
        float *             out);

/* ---- Diagnostics ---------------------------------------------------- */

/* Returns a thread-local message describing the most recent error.
 * The returned pointer is invalidated by any subsequent NanoEmbed call
 * on the same thread. Never returns NULL. */
NANOEMBED_API const char * nanoembed_last_error(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* NANOEMBED_H */
