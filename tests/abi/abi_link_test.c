/*
 * abi_link_test.c — pure-C consumer of the NanoEmbed public ABI.
 *
 * Purpose:
 *   1. Prove the public header compiles as C (extern "C" guards correct).
 *   2. Prove every public symbol resolves at link time.
 *   3. Prove documented invariants on invalid-arg paths:
 *      - default_params returns sane values
 *      - load_model on a missing file returns NULL with last_error set
 *      - free_model / free_context accept NULL safely
 *      - n_embed / n_layer / embed* on NULL return NANOEMBED_ERR_INVALID_ARG
 *
 * Scope grows over milestones (in M7 it becomes a true installed-package
 * consumer test); the link + invalid-arg checks stay.
 */

#include "nanoembed/nanoembed.h"

#include <stddef.h>
#include <stdio.h>

/* M4 changes behavior behind this frozen layout; it must not grow a mode
 * result field or reorder the pre-existing selector. */
_Static_assert(sizeof(nanoembed_context_params) == 24,
               "nanoembed_context_params ABI size changed");
_Static_assert(offsetof(nanoembed_context_params, n_threads) == 0, "ABI offset");
_Static_assert(offsetof(nanoembed_context_params, max_batch) == 4, "ABI offset");
_Static_assert(offsetof(nanoembed_context_params, max_seq_len) == 8, "ABI offset");
_Static_assert(offsetof(nanoembed_context_params, use_streaming) == 12, "ABI offset");
_Static_assert(offsetof(nanoembed_context_params, pooling) == 16, "ABI offset");
_Static_assert(offsetof(nanoembed_context_params, normalize) == 20, "ABI offset");
#include <string.h>

static int g_check_count = 0;

#define EXPECT(cond, msg)                                                       \
    do {                                                                        \
        ++g_check_count;                                                        \
        if (!(cond)) {                                                          \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__);     \
            return 1;                                                           \
        }                                                                       \
    } while (0)

int main(void) {
    /* default_params returns a fully populated struct. */
    nanoembed_context_params p = nanoembed_context_default_params();
    EXPECT(p.max_batch     >  0,                       "max_batch must be > 0");
    EXPECT(p.max_seq_len   >  0,                       "max_seq_len must be > 0");
    EXPECT(p.pooling       == NANOEMBED_POOL_MODEL_DEFAULT,
           "pooling defaults to the model's own");
    EXPECT(p.normalize     == 1,                       "normalize defaults on");
    EXPECT(p.use_streaming == 0,                       "streaming defaults off");

    /* load_model on a missing file returns NULL + sets last_error. */
    nanoembed_model * m = nanoembed_load_model("/nonexistent/path.gguf");
    EXPECT(m == NULL, "load_model on missing file must return NULL");
    const char * err = nanoembed_last_error();
    EXPECT(err != NULL && err[0] != '\0', "last_error must be non-empty after a failure");

    /* free_model accepts NULL safely (idempotent destructor pattern). */
    nanoembed_free_model(NULL);

    /* Metadata accessors on NULL handle return INVALID_ARG. */
    EXPECT(nanoembed_n_embed(NULL) == NANOEMBED_ERR_INVALID_ARG, "n_embed(NULL)");
    EXPECT(nanoembed_n_layer(NULL) == NANOEMBED_ERR_INVALID_ARG, "n_layer(NULL)");

    /* new_context with NULL model returns NULL. */
    nanoembed_context * ctx = nanoembed_new_context(NULL, p);
    EXPECT(ctx == NULL, "new_context(NULL) must return NULL");
    nanoembed_free_context(NULL);

    /* Inference on NULL ctx returns INVALID_ARG. */
    float scratch[4];
    EXPECT(nanoembed_embed(NULL, "hello", scratch) == NANOEMBED_ERR_INVALID_ARG,
           "embed(NULL ctx)");

    const char * texts[2] = {"a", "b"};
    EXPECT(nanoembed_embed_batch(NULL, texts, 2, scratch) == NANOEMBED_ERR_INVALID_ARG,
           "embed_batch(NULL ctx)");
    EXPECT(nanoembed_context_set_batch_layout(
               NULL, NANOEMBED_BATCH_LAYOUT_PADDED) == NANOEMBED_ERR_INVALID_ARG,
           "set_batch_layout(NULL ctx)");

    /* last_error remains valid after multiple calls. */
    err = nanoembed_last_error();
    EXPECT(err != NULL, "last_error never returns NULL");

    printf("ok: %d ABI checks passed\n", g_check_count);
    return 0;
}
