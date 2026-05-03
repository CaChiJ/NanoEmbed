// nanoembed-cli: read one line from stdin, embed it, print the vector.
//
// Usage:
//   echo "hello world" | nanoembed-cli model.gguf [--cls] [--no-normalize] [--threads N]
//
// Output: comma-separated f32 values on one line, "%.6g" precision, to stdout.
// Diagnostics go to stderr.

#include "nanoembed/nanoembed.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

void print_usage(const char * prog) {
    std::fprintf(stderr,
        "usage: %s <gguf-file> [--cls] [--no-normalize] [--threads N] [--max-seq-len N]\n"
        "\n"
        "Reads a single line from stdin and prints the embedding as comma-\n"
        "separated f32 values on one line.\n",
        prog);
}

} // namespace

int main(int argc, char ** argv) {
    const char * model_path  = nullptr;
    bool         use_cls     = false;
    bool         normalize   = true;
    int          n_threads   = 0;
    int          max_seq_len = 0;

    for (int i = 1; i < argc; ++i) {
        const char * a = argv[i];
        if (std::strcmp(a, "--cls") == 0) {
            use_cls = true;
        } else if (std::strcmp(a, "--no-normalize") == 0) {
            normalize = false;
        } else if (std::strcmp(a, "--threads") == 0 && i + 1 < argc) {
            n_threads = std::atoi(argv[++i]);
        } else if (std::strcmp(a, "--max-seq-len") == 0 && i + 1 < argc) {
            max_seq_len = std::atoi(argv[++i]);
        } else if (a[0] == '-') {
            std::fprintf(stderr, "error: unknown option: %s\n", a);
            print_usage(argv[0]);
            return 2;
        } else if (model_path == nullptr) {
            model_path = a;
        } else {
            std::fprintf(stderr, "error: unexpected positional arg: %s\n", a);
            print_usage(argv[0]);
            return 2;
        }
    }

    if (model_path == nullptr) {
        print_usage(argv[0]);
        return 2;
    }

    nanoembed_model * model = nanoembed_load_model(model_path);
    if (model == nullptr) {
        std::fprintf(stderr, "error: %s\n", nanoembed_last_error());
        return 1;
    }

    nanoembed_context_params params = nanoembed_context_default_params();
    params.pooling   = use_cls ? NANOEMBED_POOL_CLS : NANOEMBED_POOL_MEAN;
    params.normalize = normalize ? 1 : 0;
    if (n_threads   > 0) params.n_threads   = n_threads;
    if (max_seq_len > 0) params.max_seq_len = max_seq_len;

    nanoembed_context * ctx = nanoembed_new_context(model, params);
    if (ctx == nullptr) {
        std::fprintf(stderr, "error: %s\n", nanoembed_last_error());
        nanoembed_free_model(model);
        return 1;
    }

    std::string text;
    if (!std::getline(std::cin, text)) {
        std::fprintf(stderr, "error: no input on stdin\n");
        nanoembed_free_context(ctx);
        nanoembed_free_model(model);
        return 1;
    }

    const int H = nanoembed_n_embed(model);
    std::vector<float> out(static_cast<size_t>(H));
    const int rc = nanoembed_embed(ctx, text.c_str(), out.data());
    if (rc != NANOEMBED_OK) {
        std::fprintf(stderr, "error: embed failed (rc=%d): %s\n",
                     rc, nanoembed_last_error());
        nanoembed_free_context(ctx);
        nanoembed_free_model(model);
        return 1;
    }

    for (int i = 0; i < H; ++i) {
        std::printf("%.6g%c", out[i], (i + 1 < H) ? ',' : '\n');
    }

    nanoembed_free_context(ctx);
    nanoembed_free_model(model);
    return 0;
}
