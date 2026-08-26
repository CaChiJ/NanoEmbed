// B2 ownership/validation seam.  With no arguments this uses only tiny
// temporary GGUFs for failure and cleanup coverage.  `--model` drives the
// same production preparation against a real BERT or Harrier variant without
// executing inference or issuing residency advice.

#include "mapped_weight_store.h"

#include "arch/model_arch.h"
#include "tokenizer/tokenizer.h"

#include "ggml.h"
#include "gguf.h"

#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <limits>
#include <set>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

int g_failures = 0;

#define EXPECT_TRUE(value)                                                         \
    do {                                                                           \
        if (!(value)) {                                                            \
            std::fprintf(stderr, "FAIL: %s (%s:%d)\n", #value, __FILE__, __LINE__); \
            ++g_failures;                                                          \
        }                                                                          \
    } while (0)

#define EXPECT_CONTAINS(value, needle)                                             \
    do {                                                                           \
        if ((value).find(needle) == std::string::npos) {                           \
            std::fprintf(stderr, "FAIL: '%s' lacks '%s' (%s:%d)\n",              \
                         (value).c_str(), needle, __FILE__, __LINE__);              \
            ++g_failures;                                                          \
        }                                                                          \
    } while (0)

class TempFile {
public:
    TempFile() {
        char pattern[] = "/tmp/nanoembed_mapped_store_XXXXXX";
        const int fd = mkstemp(pattern);
        if (fd < 0) {
            throw std::runtime_error("mkstemp failed");
        }
        close(fd);
        path_ = pattern;
    }
    ~TempFile() { if (!path_.empty()) std::remove(path_.c_str()); }
    const std::string & path() const noexcept { return path_; }

private:
    std::string path_;
};

size_t file_size(const std::string & path) {
    struct stat st {};
    if (stat(path.c_str(), &st) != 0 || st.st_size < 0) {
        throw std::runtime_error("stat failed");
    }
    return static_cast<size_t>(st.st_size);
}

size_t open_fd_count() {
    DIR * dir = opendir("/dev/fd");
    if (dir == nullptr) return 0;
    size_t count = 0;
    while (dirent * entry = readdir(dir)) {
        if (entry->d_name[0] != '.') ++count;
    }
    closedir(dir);
    return count;
}

void append_byte(const std::string & path) {
    const int fd = open(path.c_str(), O_WRONLY | O_APPEND);
    if (fd < 0) throw std::runtime_error("open append failed");
    const unsigned char byte = 0;
    if (write(fd, &byte, 1) != 1) {
        close(fd);
        throw std::runtime_error("append failed");
    }
    close(fd);
}

void write_bytes(const std::string & path, const void * data, size_t size) {
    const int fd = open(path.c_str(), O_WRONLY | O_TRUNC);
    if (fd < 0) throw std::runtime_error("open write failed");
    const ssize_t written = write(fd, data, size);
    close(fd);
    if (written < 0 || static_cast<size_t>(written) != size) {
        throw std::runtime_error("write failed");
    }
}

void write_synthetic_gguf(const std::string & path,
                          ggml_type          type = GGML_TYPE_F32,
                          uint32_t           alignment = GGUF_DEFAULT_ALIGNMENT) {
    ggml_init_params params;
    params.mem_size = 64 * 1024;
    params.mem_buffer = nullptr;
    params.no_alloc = false;
    ggml_context * tensor_ctx = ggml_init(params);
    if (tensor_ctx == nullptr) throw std::runtime_error("ggml_init failed");

    ggml_tensor * first = ggml_new_tensor_1d(tensor_ctx, type,
        type == GGML_TYPE_Q4_0 || type == GGML_TYPE_Q8_0 ? 32 : 16);
    ggml_set_name(first, "synthetic.first");
    if (first->data != nullptr) std::memset(first->data, 0x5a, ggml_nbytes(first));

    ggml_tensor * second = ggml_new_tensor_1d(tensor_ctx, GGML_TYPE_F32, 8);
    ggml_set_name(second, "synthetic.second");
    std::memset(second->data, 0xa5, ggml_nbytes(second));

    gguf_context * writer = gguf_init_empty();
    gguf_set_val_u32(writer, GGUF_KEY_GENERAL_ALIGNMENT, alignment);
    gguf_add_tensor(writer, first);
    gguf_add_tensor(writer, second);
    const bool ok = gguf_write_to_file(writer, path.c_str(), false);
    gguf_free(writer);
    ggml_free(tensor_ctx);
    if (!ok) throw std::runtime_error("gguf_write_to_file failed");
}

template <typename Function>
std::string exception_text(Function function) {
    try {
        function();
    } catch (const std::exception & error) {
        return error.what();
    }
    ++g_failures;
    return {};
}

void test_checked_arithmetic() {
    using namespace nanoembed::mapped_weight_detail;
    const size_t max = std::numeric_limits<size_t>::max();

    std::string message = exception_text([&] { (void) checked_add(max, 1, "test"); });
    EXPECT_CONTAINS(message, "overflow");

    message = exception_text([&] { (void) checked_pad(max, 32, "test"); });
    EXPECT_CONTAINS(message, "overflow");

    message = exception_text([&] { (void) checked_pad(4, 3, "test"); });
    EXPECT_CONTAINS(message, "alignment");

    message = exception_text([&] { validate_range(128, 96, 16, 32, "test"); });
    EXPECT_CONTAINS(message, "outside/truncated");

    message = exception_text([&] { validate_range(max, max - 4, 8, 1, "test"); });
    EXPECT_CONTAINS(message, "overflow");
}

void test_empty_and_corrupt() {
    TempFile empty;
    std::string message = exception_text([&] {
        nanoembed::MappedWeightStore store(empty.path());
    });
    EXPECT_CONTAINS(message, "empty");

    TempFile corrupt;
    static const char bad[] = "not a gguf file";
    write_bytes(corrupt.path(), bad, sizeof(bad));
    message = exception_text([&] {
        nanoembed::MappedWeightStore store(corrupt.path());
    });
    EXPECT_CONTAINS(message, "parse mapped GGUF");
}

void test_unbound_and_repeated_cleanup() {
    TempFile file;
    write_synthetic_gguf(file.path());
    const size_t before = open_fd_count();
    for (int iteration = 0; iteration < 64; ++iteration) {
        nanoembed::MappedWeightStore store(file.path());
        EXPECT_TRUE(store.tensor_count() == 2);
        EXPECT_TRUE(!store.tensors_bound());
        EXPECT_TRUE(store.mapped_size() == file_size(file.path()));
        for (int64_t index = 0; index < store.tensor_count(); ++index) {
            const char * name = gguf_get_tensor_name(store.gguf(), index);
            ggml_tensor * tensor = ggml_get_tensor(store.model_context(), name);
            EXPECT_TRUE(tensor != nullptr);
            EXPECT_TRUE(tensor->data == nullptr);
            EXPECT_TRUE(tensor->buffer == nullptr);
        }
    }
    const size_t after = open_fd_count();
    if (before != 0 && after != 0) EXPECT_TRUE(before == after);
}

void test_truncated_data_and_unsupported_type() {
    TempFile truncated;
    write_synthetic_gguf(truncated.path());
    size_t truncated_size = 0;
    {
        nanoembed::MappedWeightStore store(truncated.path());
        const int64_t last = store.tensor_count() - 1;
        truncated_size = store.data_offset() +
            gguf_get_tensor_offset(store.gguf(), last) +
            gguf_get_tensor_size(store.gguf(), last) - 1;
    }
    EXPECT_TRUE(truncate(truncated.path().c_str(), static_cast<off_t>(truncated_size)) == 0);
    std::string message = exception_text([&] {
        nanoembed::MappedWeightStore store(truncated.path());
    });
    EXPECT_CONTAINS(message, "outside/truncated");

    TempFile unsupported;
    write_synthetic_gguf(unsupported.path(), GGML_TYPE_I32);
    message = exception_text([&] {
        nanoembed::MappedWeightStore store(unsupported.path());
    });
    EXPECT_CONTAINS(message, "unsupported mapped tensor type");
    EXPECT_CONTAINS(message, "i32");
}

void test_identity_change() {
    TempFile file;
    write_synthetic_gguf(file.path());
    nanoembed::MappedWeightStore store(file.path());
    append_byte(file.path());
    const std::string message = exception_text([&] { store.verify_identity(); });
    EXPECT_CONTAINS(message, "identity changed");
}

void test_preparation_failure_cleanup() {
    // The generic store is structurally valid, but deliberately carries no
    // architecture/tokenizer metadata.  MappedModelPreparation must fail
    // before publishing data borrows and release its partial FD/mapping.
    TempFile file;
    write_synthetic_gguf(file.path());
    const size_t before = open_fd_count();
    const std::string message = exception_text([&] {
        nanoembed::MappedModelPreparation preparation(file.path());
    });
    EXPECT_CONTAINS(message, "general.architecture");
    const size_t after = open_fd_count();
    if (before != 0 && after != 0) EXPECT_TRUE(before == after);
}

ggml_type type_from_name(const std::string & name) {
    for (int value = 0; value < GGML_TYPE_COUNT; ++value) {
        const auto type = static_cast<ggml_type>(value);
        const char * candidate = ggml_type_name(type);
        if (candidate != nullptr && name == candidate) return type;
    }
    throw std::runtime_error("unknown type name: " + name);
}

void test_real_model(int argc, char ** argv) {
    if (argc < 4 || std::string(argv[1]) != "--model") {
        throw std::runtime_error(
            "usage: mapped_weight_store_test --model PATH ARCH [TYPE ...]");
    }
    const std::string path = argv[2];
    const std::string expected_arch = argv[3];
    std::set<ggml_type> required;
    for (int i = 4; i < argc; ++i) required.insert(type_from_name(argv[i]));

    nanoembed::MappedModelPreparation preparation(path);
    const auto & store = preparation.store();
    EXPECT_TRUE(store.tensors_bound());
    EXPECT_TRUE(store.tensor_count() > 0);
    EXPECT_TRUE(preparation.arch().params().name == expected_arch);
    EXPECT_TRUE(preparation.tokenizer().vocab_size() ==
                preparation.arch().params().n_vocab);
    EXPECT_TRUE(!preparation.tokenizer().encode("mapped preparation", 16).empty());
    store.verify_identity();

    std::set<ggml_type> observed;
    for (int64_t index = 0; index < store.tensor_count(); ++index) {
        const char * name = gguf_get_tensor_name(store.gguf(), index);
        ggml_tensor * tensor = ggml_get_tensor(store.model_context(), name);
        EXPECT_TRUE(tensor != nullptr);
        EXPECT_TRUE(tensor->data != nullptr);
        EXPECT_TRUE(tensor->buffer == nullptr);
        observed.insert(tensor->type);
    }
    for (ggml_type type : required) {
        EXPECT_TRUE(observed.count(type) == 1);
    }

    std::printf("mapped preparation: arch=%s tensors=%lld bytes=%zu\n",
                expected_arch.c_str(),
                static_cast<long long>(store.tensor_count()),
                store.mapped_size());
}

} // namespace

int main(int argc, char ** argv) {
    try {
        if (argc > 1) {
            test_real_model(argc, argv);
        } else {
            test_checked_arithmetic();
            test_empty_and_corrupt();
            test_unbound_and_repeated_cleanup();
            test_truncated_data_and_unsupported_type();
            test_identity_change();
            test_preparation_failure_cleanup();
        }
    } catch (const std::exception & error) {
        std::fprintf(stderr, "FAIL: unexpected exception: %s\n", error.what());
        ++g_failures;
    }
    std::printf("mapped_weight_store_test: %s\n",
                g_failures == 0 ? "ok" : "FAIL");
    return g_failures == 0 ? 0 : 1;
}
