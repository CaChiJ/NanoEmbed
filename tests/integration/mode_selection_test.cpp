// Public M4 execution-mode contract. The frozen ABI has no mode query:
// successful creation with an exact 0/1 request is the strict resolution
// evidence, because the library rejects unsupported and mixed selections.

#include "nanoembed/nanoembed.h"

#include <atomic>
#include <cerrno>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if !defined(_WIN32)
#include <unistd.h>
#endif

namespace {

std::atomic<int> failures{0};

void check(bool condition, const char * message) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s (last_error=%s)\n", message,
                     nanoembed_last_error());
        ++failures;
    }
}

void free_if(nanoembed_context * context);

#if defined(__linux__)
class StartBarrier {
public:
    explicit StartBarrier(int participants) : remaining_(participants) {}

    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        --remaining_;
        if (remaining_ == 0) {
            released_ = true;
            condition_.notify_all();
            return;
        }
        condition_.wait(lock, [&] { return released_; });
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    int remaining_;
    bool released_ = false;
};

bool close_vectors(const std::vector<float> & lhs,
                   const std::vector<float> & rhs) {
    if (lhs.size() != rhs.size()) return false;
    for (size_t i = 0; i < lhs.size(); ++i) {
        if (!std::isfinite(lhs[i]) || !std::isfinite(rhs[i]) ||
            std::abs(lhs[i] - rhs[i]) > 1e-5f) return false;
    }
    return true;
}

bool infer_finite(nanoembed_model * model,
                  nanoembed_context * context,
                  const char * text) {
    const int dimension = nanoembed_n_embed(model);
    if (dimension <= 0) return false;
    std::vector<float> output(static_cast<size_t>(dimension));
    if (nanoembed_embed(context, text, output.data()) != NANOEMBED_OK) return false;
    for (float value : output) {
        if (!std::isfinite(value)) return false;
    }
    return true;
}

void test_first_mode_lock_race(const char * path) {
    nanoembed_model * model = nanoembed_load_model(path);
    check(model != nullptr, "race test metadata-only model load succeeds");
    if (model == nullptr) return;

    nanoembed_context_params eager_params = nanoembed_context_default_params();
    eager_params.max_seq_len = 64;
    nanoembed_context_params streaming_params = eager_params;
    streaming_params.use_streaming = 1;

    StartBarrier barrier(2);
    nanoembed_context * eager = nullptr;
    nanoembed_context * streaming = nullptr;
    std::thread eager_thread([&] {
        barrier.arrive_and_wait();
        eager = nanoembed_new_context(model, eager_params);
    });
    std::thread streaming_thread([&] {
        barrier.arrive_and_wait();
        streaming = nanoembed_new_context(model, streaming_params);
    });
    eager_thread.join();
    streaming_thread.join();

    const bool eager_won = eager != nullptr;
    const bool streaming_won = streaming != nullptr;
    check(eager_won != streaming_won,
          "simultaneous first eager/streaming creators produce exactly one winner");

    nanoembed_context_params same_params = eager_won ? eager_params : streaming_params;
    nanoembed_context_params opposite_params = eager_won ? streaming_params : eager_params;
    nanoembed_context * same = nanoembed_new_context(model, same_params);
    nanoembed_context * opposite = nanoembed_new_context(model, opposite_params);
    check(same != nullptr, "same-mode context succeeds after the race winner locks mode");
    check(opposite == nullptr,
          "opposite-mode context fails after the race winner locks mode");

    nanoembed_context * winner = eager_won ? eager : streaming;
    if (winner != nullptr) {
        check(infer_finite(model, winner, "mode lock race inference"),
              "race-winning context performs a finite inference");
    }

    free_if(eager);
    free_if(streaming);
    free_if(same);
    free_if(opposite);
    nanoembed_free_model(model);
}
#endif

void free_if(nanoembed_context * context) {
    if (context != nullptr) nanoembed_free_context(context);
}

#if !defined(_WIN32)
void test_candidate_failure_recovery(const char * path) {
    char * resolved_path = ::realpath(path, nullptr);
    check(resolved_path != nullptr, "candidate recovery resolves the source model path");
    if (resolved_path == nullptr) return;
    const std::string target(resolved_path);
    std::free(resolved_path);

    const char * temp_root = std::getenv("TMPDIR");
    const std::string base = temp_root && *temp_root ? temp_root : "/tmp";
    std::string pattern = base + "/nanoembed-mode-selection-XXXXXX";
    std::vector<char> pattern_buffer(pattern.begin(), pattern.end());
    pattern_buffer.push_back('\0');
    char * directory_raw = ::mkdtemp(pattern_buffer.data());
    check(directory_raw != nullptr, "candidate recovery creates a private temp directory");
    if (directory_raw == nullptr) return;

    const std::string directory(directory_raw);
    const std::string link_path = directory + "/model.gguf";
    auto cleanup = [&] {
        (void) ::unlink(link_path.c_str());
        (void) ::rmdir(directory.c_str());
    };

    if (::symlink(target.c_str(), link_path.c_str()) != 0) {
        const int error_number = errno;
        std::fprintf(stderr,
                     "[mode_selection_test] SKIP candidate failure recovery: "
                     "symlink unavailable (%s)\n",
                     std::strerror(error_number));
        cleanup();
        return;
    }

    nanoembed_model * model = nanoembed_load_model(link_path.c_str());
    check(model != nullptr, "descriptor loads through the temporary symlink");
    if (model == nullptr) {
        cleanup();
        return;
    }

    const int unlink_result = ::unlink(link_path.c_str());
    check(unlink_result == 0,
          "temporary symlink is removed after descriptor load without touching source");
    if (unlink_result != 0) {
        nanoembed_free_model(model);
        cleanup();
        return;
    }

    nanoembed_context_params first = nanoembed_context_default_params();
    first.max_seq_len = 64;
    nanoembed_context * failed = nanoembed_new_context(model, first);
    check(failed == nullptr,
          "candidate open failure returns null after descriptor path disappears");
    free_if(failed);

    const int restore_result = ::symlink(target.c_str(), link_path.c_str());
    check(restore_result == 0, "temporary symlink is restored for context retry");
    if (restore_result != 0) {
        nanoembed_free_model(model);
        cleanup();
        return;
    }

    nanoembed_context_params retry = nanoembed_context_default_params();
    retry.max_seq_len = 64;
#if defined(__linux__)
    // The failed candidate requested eager. Succeeding with streaming proves
    // the failure did not publish even the requested mode.
    retry.use_streaming = 1;
#endif
    nanoembed_context * recovered = nanoembed_new_context(model, retry);
    check(recovered != nullptr,
          "restored descriptor path can retry after failed candidate construction");
    if (recovered != nullptr) {
#if defined(__linux__)
        check(infer_finite(model, recovered, "candidate failure recovery"),
              "recovered streaming context performs a finite inference");
#else
        const int dimension = nanoembed_n_embed(model);
        std::vector<float> output(static_cast<size_t>(dimension));
        check(dimension > 0 && nanoembed_embed(
                  recovered, "candidate failure recovery", output.data()) == NANOEMBED_OK,
              "recovered eager context performs an inference");
#endif
    }

    free_if(recovered);
    nanoembed_free_model(model);
    cleanup();
}
#endif

void run(const char * path) {
    nanoembed_model * query_model = nanoembed_load_model(path);
    check(query_model != nullptr, "metadata-only model load succeeds");
    if (query_model == nullptr) return;
    check(nanoembed_n_embed(query_model) > 0, "n_embed works before mode selection");
    check(nanoembed_n_layer(query_model) > 0, "n_layer works before mode selection");
    check(nanoembed_model_max_seq_len(query_model) > 0,
          "max_seq_len works before mode selection");
    check(nanoembed_model_default_pooling(query_model) != NANOEMBED_POOL_MODEL_DEFAULT,
          "default pooling works before mode selection");

    nanoembed_context_params invalid = nanoembed_context_default_params();
    invalid.use_streaming = 7;
    check(nanoembed_new_context(query_model, invalid) == nullptr,
          "selector outside 0/1 fails loudly");

    invalid = nanoembed_context_default_params();
    invalid.pooling = static_cast<nanoembed_pool_type>(99);
    check(nanoembed_new_context(query_model, invalid) == nullptr,
          "failed first context does not publish partial mode state");

#if !defined(__linux__)
    nanoembed_context_params stream = nanoembed_context_default_params();
    stream.use_streaming = 1;
    nanoembed_context * unsupported = nanoembed_new_context(query_model, stream);
    check(unsupported == nullptr, "streaming is explicitly unsupported outside Linux");
    check(std::strstr(nanoembed_last_error(), "unsupported outside Linux") != nullptr,
          "unsupported platform error is descriptive");
    free_if(unsupported);

    nanoembed_context * eager = nanoembed_new_context(
        query_model, nanoembed_context_default_params());
    check(eager != nullptr, "unsupported streaming attempt leaves model unlockable");
    free_if(eager);
    nanoembed_free_model(query_model);
    return;
#else
    nanoembed_context_params stream = nanoembed_context_default_params();
    stream.use_streaming = 1;
    stream.max_seq_len = 64;
    nanoembed_context * stream_a = nanoembed_new_context(query_model, stream);
    nanoembed_context * stream_b = nanoembed_new_context(query_model, stream);
    check(stream_a != nullptr && stream_b != nullptr,
          "same-mode streaming contexts share one locked model");

    nanoembed_context * mixed_eager = nanoembed_new_context(
        query_model, nanoembed_context_default_params());
    check(mixed_eager == nullptr, "eager context is rejected after streaming lock");
    check(std::strstr(nanoembed_last_error(), "locked to streaming") != nullptr,
          "mixed-mode rejection identifies the locked mode");

    if (stream_a != nullptr && stream_b != nullptr) {
        const int dimension = nanoembed_n_embed(query_model);
        std::vector<float> reference(static_cast<size_t>(dimension));
        std::vector<float> a(static_cast<size_t>(dimension));
        std::vector<float> b(static_cast<size_t>(dimension));
        check(nanoembed_embed(stream_a, "strict streaming mode", reference.data()) ==
                  NANOEMBED_OK,
              "streaming reference inference succeeds");
        std::thread ta([&] {
            for (int i = 0; i < 2; ++i) {
                if (nanoembed_embed(stream_a, "strict streaming mode", a.data()) !=
                    NANOEMBED_OK) ++failures;
            }
        });
        std::thread tb([&] {
            for (int i = 0; i < 2; ++i) {
                if (nanoembed_embed(stream_b, "strict streaming mode", b.data()) !=
                    NANOEMBED_OK) ++failures;
            }
        });
        ta.join();
        tb.join();
        check(close_vectors(reference, a) && close_vectors(reference, b),
              "distinct streaming contexts are deterministic under concurrency");
    }
    free_if(stream_a);
    free_if(stream_b);
    nanoembed_free_model(query_model);

    nanoembed_model * eager_model = nanoembed_load_model(path);
    check(eager_model != nullptr, "fresh model for eager lock");
    if (eager_model != nullptr) {
        nanoembed_context_params eager_params = nanoembed_context_default_params();
        eager_params.max_seq_len = 64;
        nanoembed_context * eager_a = nanoembed_new_context(eager_model, eager_params);
        nanoembed_context * eager_b = nanoembed_new_context(eager_model, eager_params);
        check(eager_a != nullptr && eager_b != nullptr,
              "same-mode eager contexts share one locked model");
        nanoembed_context * mixed_stream = nanoembed_new_context(eager_model, stream);
        check(mixed_stream == nullptr,
              "streaming context is rejected after eager lock");
        check(std::strstr(nanoembed_last_error(), "locked to eager") != nullptr,
              "reverse mixed-mode rejection identifies the locked mode");
        free_if(eager_a);
        free_if(eager_b);
        nanoembed_free_model(eager_model);
    }
#endif
}

} // namespace

int main() {
    const char * model = std::getenv("NANOEMBED_TEST_MODEL");
    if (model == nullptr || *model == '\0') {
        std::printf("mode_selection_test: SKIP: NANOEMBED_TEST_MODEL not set\n");
        return 0;
    }
    run(model);
#if defined(__linux__)
    test_first_mode_lock_race(model);
#endif
#if !defined(_WIN32)
    test_candidate_failure_recovery(model);
#endif
    std::printf("mode_selection_test: %s\n",
                failures.load() == 0 ? "ok" : "FAIL");
    return failures.load() == 0 ? 0 : 1;
}
