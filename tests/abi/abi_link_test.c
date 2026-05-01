/*
 * abi_link_test.c — pure-C consumer of the NanoEmbed public ABI.
 *
 * Purpose (M1):
 *   1. Prove the public header compiles as C (extern "C" guards correct).
 *   2. Prove every public symbol resolves at link time.
 *   3. Prove M1 stubs behave as documented:
 *      - default_params returns sane values
 *      - load_model / new_context return NULL with last_error set
 *      - embed* return NANOEMBED_ERR_NOT_IMPL
 *
 * This test will keep running across all milestones — its scope grows
 * (in M7 it becomes a true installed-package consumer test) but the
 * "all signatures link" check stays.
 */

#include "nanoembed/nanoembed.h"

#include <stdio.h>
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
    EXPECT(p.pooling       == NANOEMBED_POOL_MEAN,     "pooling defaults to MEAN");
    EXPECT(p.normalize     == 1,                       "normalize defaults on");
    EXPECT(p.use_streaming == 0,                       "streaming defaults off");

    /* load_model on a missing file: stub returns NULL + sets error. */
    nanoembed_model * m = nanoembed_load_model("/nonexistent/path.gguf");
    EXPECT(m == NULL, "load_model stub must return NULL");
    const char * err = nanoembed_last_error();
    EXPECT(err != NULL && err[0] != '\0', "last_error must be non-empty after a failure");

    /* free_model accepts NULL safely (idempotent destructor pattern). */
    nanoembed_free_model(NULL);

    /* Metadata accessors on NULL handle — stubs return error code. */
    EXPECT(nanoembed_n_embed(NULL) == NANOEMBED_ERR_NOT_IMPL, "n_embed stub");
    EXPECT(nanoembed_n_layer(NULL) == NANOEMBED_ERR_NOT_IMPL, "n_layer stub");

    /* new_context stub returns NULL. */
    nanoembed_context * ctx = nanoembed_new_context(NULL, p);
    EXPECT(ctx == NULL, "new_context stub must return NULL");
    nanoembed_free_context(NULL);

    /* Inference stubs return NANOEMBED_ERR_NOT_IMPL. */
    float scratch[4];
    EXPECT(nanoembed_embed(NULL, "hello", scratch) == NANOEMBED_ERR_NOT_IMPL,
           "embed stub");

    const char * texts[2] = {"a", "b"};
    EXPECT(nanoembed_embed_batch(NULL, texts, 2, scratch) == NANOEMBED_ERR_NOT_IMPL,
           "embed_batch stub");

    /* last_error remains valid after multiple calls. */
    err = nanoembed_last_error();
    EXPECT(err != NULL, "last_error never returns NULL");

    printf("ok: %d ABI checks passed (M1 stub behavior)\n", g_check_count);
    return 0;
}
