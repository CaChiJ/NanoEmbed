#include "disk_merge_index.h"

#include "tokenizer.h"

#include "gguf.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory_resource>
#include <sstream>
#include <string_view>
#include <system_error>
#include <unordered_map>

#ifdef _WIN32
#  define NOMINMAX
#  include <windows.h>
#  include <io.h>
#  include <process.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

namespace nanoembed {

namespace {

constexpr size_t   kPageSize = 4096;
constexpr size_t   kPageHeaderSize = 16;
constexpr size_t   kRecordSize = 16;
constexpr uint32_t kRecordsPerPage = 255;
constexpr uint32_t kFormatVersion = 1;
constexpr uint32_t kEndianMarker = 0x01020304u;
constexpr char     kMagic[8] = {'N', 'E', 'B', 'P', 'E', 'I', '1', '\0'};
constexpr uint64_t kFenceOffset = kPageSize;

constexpr size_t kVersionOffset = 8;
constexpr size_t kEndianOffset = 12;
constexpr size_t kPageSizeOffset = 16;
constexpr size_t kRecordSizeOffset = 20;
constexpr size_t kRecordsPerPageOffset = 24;
constexpr size_t kRecordCountOffset = 32;
constexpr size_t kPageCountOffset = 40;
constexpr size_t kFenceOffsetOffset = 48;
constexpr size_t kFenceSizeOffset = 56;
constexpr size_t kDataOffsetOffset = 64;
constexpr size_t kSourceDigestOffset = 72;
constexpr size_t kPayloadDigestOffset = 104;

uint32_t read_le32(const uint8_t * p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

uint64_t read_le64(const uint8_t * p) {
    uint64_t result = 0;
    for (int i = 7; i >= 0; --i) result = (result << 8) | p[i];
    return result;
}

void write_le32(uint8_t * p, uint32_t value) {
    for (int i = 0; i < 4; ++i) {
        p[i] = static_cast<uint8_t>(value);
        value >>= 8;
    }
}

void write_le64(uint8_t * p, uint64_t value) {
    for (int i = 0; i < 8; ++i) {
        p[i] = static_cast<uint8_t>(value);
        value >>= 8;
    }
}

uint64_t align_page(uint64_t value) {
    if (value > std::numeric_limits<uint64_t>::max() - (kPageSize - 1)) {
        throw TokenizerError("BPE merge cache offset overflow");
    }
    return (value + kPageSize - 1) & ~(static_cast<uint64_t>(kPageSize) - 1);
}

uint64_t pair_key(int left, int right) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(left)) << 32) |
           static_cast<uint32_t>(right);
}

uint32_t crc32(const uint8_t * data, size_t size) {
    static const std::array<uint32_t, 256> table = [] {
        std::array<uint32_t, 256> values{};
        for (uint32_t i = 0; i < values.size(); ++i) {
            uint32_t value = i;
            for (int bit = 0; bit < 8; ++bit) {
                value = (value >> 1) ^ (0xedb88320u & (0u - (value & 1u)));
            }
            values[i] = value;
        }
        return values;
    }();
    uint32_t crc = 0xffffffffu;
    for (size_t i = 0; i < size; ++i) {
        crc = (crc >> 8) ^ table[(crc ^ data[i]) & 0xffu];
    }
    return ~crc;
}

std::string system_message(const std::string & operation, const std::filesystem::path & path) {
#ifdef _WIN32
    const DWORD code = GetLastError();
    return operation + " '" + path.string() + "' failed (Windows error " +
           std::to_string(static_cast<unsigned long>(code)) + ")";
#else
    return operation + " '" + path.string() + "' failed: " + std::strerror(errno);
#endif
}

std::filesystem::path cache_root() {
#ifdef _WIN32
    if (const wchar_t * override_dir = _wgetenv(L"NANOEMBED_CACHE_DIR")) {
        if (*override_dir != L'\0') return std::filesystem::path(override_dir);
    }
    if (const wchar_t * local = _wgetenv(L"LOCALAPPDATA")) {
        if (*local != L'\0') return std::filesystem::path(local) / "NanoEmbed" / "Cache";
    }
    throw TokenizerError(
        "cannot choose BPE merge cache directory: LOCALAPPDATA is unset; "
        "set NANOEMBED_CACHE_DIR");
#else
    if (const char * override_dir = std::getenv("NANOEMBED_CACHE_DIR")) {
        if (*override_dir != '\0') return std::filesystem::path(override_dir);
    }
#if defined(__APPLE__)
    if (const char * home = std::getenv("HOME")) {
        if (*home != '\0') return std::filesystem::path(home) / "Library" / "Caches" / "NanoEmbed";
    }
    throw TokenizerError(
        "cannot choose BPE merge cache directory: HOME is unset; set NANOEMBED_CACHE_DIR");
#else
    if (const char * xdg = std::getenv("XDG_CACHE_HOME")) {
        if (*xdg != '\0') return std::filesystem::path(xdg) / "nanoembed";
    }
    if (const char * home = std::getenv("HOME")) {
        if (*home != '\0') return std::filesystem::path(home) / ".cache" / "nanoembed";
    }
    throw TokenizerError(
        "cannot choose BPE merge cache directory: XDG_CACHE_HOME and HOME are unset; "
        "set NANOEMBED_CACHE_DIR");
#endif
#endif
}

void digest_u64(detail::Sha256 & hash, uint64_t value) {
    uint8_t bytes[8];
    write_le64(bytes, value);
    hash.update(bytes, sizeof(bytes));
}

detail::Sha256Digest tokenizer_digest(gguf_context * ctx, int64_t tk, int64_t mk) {
    detail::Sha256 hash;
    constexpr char kTokensTag[] = "nanoembed-bpe-tokens-v1";
    constexpr char kMergesTag[] = "nanoembed-bpe-merges-v1";
    hash.update(kTokensTag, sizeof(kTokensTag) - 1);
    const size_t token_count = gguf_get_arr_n(ctx, tk);
    digest_u64(hash, token_count);
    for (size_t i = 0; i < token_count; ++i) {
        const char * value = gguf_get_arr_str(ctx, tk, i);
        const size_t length = std::strlen(value);
        digest_u64(hash, length);
        hash.update(value, length);
    }
    hash.update(kMergesTag, sizeof(kMergesTag) - 1);
    const size_t merge_count = gguf_get_arr_n(ctx, mk);
    digest_u64(hash, merge_count);
    for (size_t i = 0; i < merge_count; ++i) {
        const char * value = gguf_get_arr_str(ctx, mk, i);
        const size_t length = std::strlen(value);
        digest_u64(hash, length);
        hash.update(value, length);
    }
    return hash.finish();
}

// monotonic_buffer_resource asks its upstream for a small number of growing
// chunks. Each chunk here is an independent OS mapping, so destroying the
// arena returns every page to the OS instead of leaving allocator arenas in
// the process RSS after a cold-cache build.
class OsMemoryResource final : public std::pmr::memory_resource {
    struct Allocation {
        void * base;
        size_t size;
    };

    void * do_allocate(size_t bytes, size_t alignment) override {
        if (bytes > std::numeric_limits<size_t>::max() - alignment - sizeof(Allocation)) {
            throw std::bad_alloc();
        }
        const size_t mapped = bytes + alignment + sizeof(Allocation);
#ifdef _WIN32
        void * base = VirtualAlloc(nullptr, mapped, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (base == nullptr) throw std::bad_alloc();
#else
        void * base = mmap(nullptr, mapped, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (base == MAP_FAILED) throw std::bad_alloc();
#endif
        const uintptr_t start = reinterpret_cast<uintptr_t>(base) + sizeof(Allocation);
        const uintptr_t aligned = (start + alignment - 1) & ~(static_cast<uintptr_t>(alignment) - 1);
        auto * metadata = reinterpret_cast<Allocation *>(aligned) - 1;
        metadata->base = base;
        metadata->size = mapped;
        return reinterpret_cast<void *>(aligned);
    }

    void do_deallocate(void * p, size_t, size_t) override {
        const Allocation allocation = *(reinterpret_cast<Allocation *>(p) - 1);
#ifdef _WIN32
        (void) VirtualFree(allocation.base, 0, MEM_RELEASE);
#else
        (void) munmap(allocation.base, allocation.size);
#endif
    }

    bool do_is_equal(const std::pmr::memory_resource & other) const noexcept override {
        return this == &other;
    }
};

struct RawRecord {
    uint64_t key;
    int32_t rank;
    int32_t merged;
};

void write_all(FILE * file, const uint8_t * data, size_t size,
               const std::filesystem::path & path) {
    while (size != 0) {
        const size_t written = std::fwrite(data, 1, size, file);
        if (written == 0) {
            throw TokenizerError("writing BPE merge cache '" + path.string() + "' failed");
        }
        data += written;
        size -= written;
    }
}

void durable_close(FILE *& file, const std::filesystem::path & path) {
    if (std::fflush(file) != 0) {
        throw TokenizerError(system_message("flush BPE merge cache", path));
    }
#ifdef _WIN32
    if (_commit(_fileno(file)) != 0) {
        throw TokenizerError("sync BPE merge cache '" + path.string() + "' failed");
    }
#else
    if (fsync(fileno(file)) != 0) {
        throw TokenizerError(system_message("sync BPE merge cache", path));
    }
#endif
    if (std::fclose(file) != 0) {
        file = nullptr;
        throw TokenizerError(system_message("close BPE merge cache", path));
    }
    file = nullptr;
}

std::filesystem::path unique_temp_path(const std::filesystem::path & target) {
    static std::atomic<uint64_t> serial{0};
#ifdef _WIN32
    const uint64_t pid = static_cast<uint64_t>(_getpid());
    return std::filesystem::path(target.wstring() + L".tmp." + std::to_wstring(pid) + L"." +
                                 std::to_wstring(serial.fetch_add(1, std::memory_order_relaxed)));
#else
    const uint64_t pid = static_cast<uint64_t>(getpid());
    return std::filesystem::path(target.string() + ".tmp." + std::to_string(pid) + "." +
                                 std::to_string(serial.fetch_add(1, std::memory_order_relaxed)));
#endif
}

#ifdef _WIN32
// Names the file a replacement displaces. It has to be distinct from the
// creation temp so a directory listing says which role a leftover file had.
std::filesystem::path displaced_path(const std::filesystem::path & target) {
    static std::atomic<uint64_t> serial{0};
    const uint64_t pid = static_cast<uint64_t>(_getpid());
    return std::filesystem::path(target.wstring() + L".old." + std::to_wstring(pid) + L"." +
                                 std::to_wstring(serial.fetch_add(1, std::memory_order_relaxed)));
}
#endif

void atomic_replace(const std::filesystem::path & temp, const std::filesystem::path & target) {
#ifdef _WIN32
    if (MoveFileExW(temp.c_str(), target.c_str(),
                    MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        return;
    }
    const DWORD replace_error = GetLastError();
    if (replace_error != ERROR_ACCESS_DENIED && replace_error != ERROR_SHARING_VIOLATION) {
        SetLastError(replace_error);
        throw TokenizerError(system_message("replace BPE merge cache", target));
    }

    // Windows cannot rename onto a name another handle still holds open. The
    // readers share delete, so the old file can be deleted, but the delete
    // stays pending until the last reader closes and the directory entry keeps
    // the name until then, which is what fails the replacement above. Move the
    // old file aside to a free name first, rename into the name it vacated,
    // and then delete the displaced file: readers that still hold it keep
    // reading the bytes they opened, and the file disappears when they close.
    const std::filesystem::path displaced = displaced_path(target);
    bool displaced_exists = true;
    if (!MoveFileExW(target.c_str(), displaced.c_str(), MOVEFILE_WRITE_THROUGH)) {
        const DWORD displace_error = GetLastError();
        if (displace_error != ERROR_FILE_NOT_FOUND && displace_error != ERROR_PATH_NOT_FOUND) {
            SetLastError(displace_error);
            throw TokenizerError(system_message("displace BPE merge cache", target));
        }
        // Someone else replaced or removed it in between; the name is free.
        displaced_exists = false;
    }
    if (!MoveFileExW(temp.c_str(), target.c_str(), MOVEFILE_WRITE_THROUGH)) {
        const DWORD move_error = GetLastError();
        // Leave a valid cache behind rather than a missing one.
        if (displaced_exists) {
            (void) MoveFileExW(displaced.c_str(), target.c_str(), MOVEFILE_WRITE_THROUGH);
        }
        SetLastError(move_error);
        throw TokenizerError(system_message("replace BPE merge cache", target));
    }
    if (displaced_exists) (void) DeleteFileW(displaced.c_str());
#else
    if (rename(temp.c_str(), target.c_str()) != 0) {
        throw TokenizerError(system_message("replace BPE merge cache", target));
    }
#endif
}

} // namespace

struct DiskMergeIndex::Impl {
#ifdef _WIN32
    HANDLE handle = INVALID_HANDLE_VALUE;
#else
    int fd = -1;
#endif
    uint64_t size = 0;

    ~Impl() {
#ifdef _WIN32
        if (handle != INVALID_HANDLE_VALUE) CloseHandle(handle);
#else
        if (fd >= 0) close(fd);
#endif
    }

    void read_at(uint64_t offset, void * output, size_t bytes,
                 const std::filesystem::path & path) const {
        if (offset > size || bytes > size - offset) {
            throw TokenizerError("short read from BPE merge cache '" + path.string() + "'");
        }
#ifdef _WIN32
        auto * p = static_cast<uint8_t *>(output);
        while (bytes != 0) {
            const DWORD chunk = static_cast<DWORD>(std::min<size_t>(bytes, MAXDWORD));
            OVERLAPPED overlapped{};
            overlapped.Offset = static_cast<DWORD>(offset);
            overlapped.OffsetHigh = static_cast<DWORD>(offset >> 32);
            overlapped.hEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
            if (overlapped.hEvent == nullptr) {
                throw TokenizerError(system_message("create BPE merge cache read event", path));
            }
            DWORD got = 0;
            const BOOL immediate = ReadFile(handle, p, chunk, &got, &overlapped);
            const DWORD read_error = immediate ? ERROR_SUCCESS : GetLastError();
            BOOL completed = immediate;
            if (!immediate && read_error == ERROR_IO_PENDING) {
                completed = GetOverlappedResult(handle, &overlapped, &got, TRUE);
            }
            CloseHandle(overlapped.hEvent);
            if (!completed || got != chunk) {
                throw TokenizerError(system_message("read BPE merge cache", path));
            }
            p += got;
            bytes -= got;
            offset += got;
        }
#else
        if (offset > static_cast<uint64_t>(std::numeric_limits<off_t>::max())) {
            throw TokenizerError("BPE merge cache offset exceeds platform range");
        }
        auto * p = static_cast<uint8_t *>(output);
        while (bytes != 0) {
            const size_t chunk = std::min<size_t>(
                bytes, static_cast<size_t>(std::numeric_limits<ssize_t>::max()));
            const ssize_t got = pread(fd, p, chunk, static_cast<off_t>(offset));
            if (got < 0 && errno == EINTR) continue;
            if (got <= 0) {
                throw TokenizerError(system_message("read BPE merge cache", path));
            }
            p += got;
            bytes -= static_cast<size_t>(got);
            offset += static_cast<size_t>(got);
        }
#endif
    }
};

DiskMergeIndex::DiskMergeIndex() = default;
DiskMergeIndex::~DiskMergeIndex() = default;
DiskMergeIndex::DiskMergeIndex(DiskMergeIndex &&) noexcept = default;
DiskMergeIndex & DiskMergeIndex::operator=(DiskMergeIndex &&) noexcept = default;

size_t DiskMergeIndex::fence_bytes() const noexcept {
    return fences_.size() * sizeof(Fence);
}

bool DiskMergeIndex::open_validated(const std::filesystem::path & path,
                                    const detail::Sha256Digest & source_digest,
                                    int64_t vocab_size,
                                    DiskMergeIndex & result,
                                    std::string & reason) {
    auto impl = std::make_unique<Impl>();
#ifdef _WIN32
    impl->handle = CreateFileW(path.c_str(), GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS | FILE_FLAG_OVERLAPPED, nullptr);
    if (impl->handle == INVALID_HANDLE_VALUE) {
        reason = system_message("open BPE merge cache", path);
        return false;
    }
    LARGE_INTEGER size{};
    if (!GetFileSizeEx(impl->handle, &size) || size.QuadPart < 0) {
        reason = system_message("size BPE merge cache", path);
        return false;
    }
    impl->size = static_cast<uint64_t>(size.QuadPart);
#else
    impl->fd = open(path.c_str(), O_RDONLY
#ifdef O_CLOEXEC
                    | O_CLOEXEC
#endif
    );
    if (impl->fd < 0) {
        reason = system_message("open BPE merge cache", path);
        return false;
    }
    struct stat st{};
    if (fstat(impl->fd, &st) != 0 || st.st_size < 0) {
        reason = system_message("size BPE merge cache", path);
        return false;
    }
    impl->size = static_cast<uint64_t>(st.st_size);
#endif

    try {
        if (impl->size < kPageSize) throw TokenizerError("cache is shorter than its header");
        std::array<uint8_t, kPageSize> header{};
        impl->read_at(0, header.data(), header.size(), path);
        if (std::memcmp(header.data(), kMagic, sizeof(kMagic)) != 0 ||
            read_le32(header.data() + kVersionOffset) != kFormatVersion ||
            read_le32(header.data() + kEndianOffset) != kEndianMarker ||
            read_le32(header.data() + kPageSizeOffset) != kPageSize ||
            read_le32(header.data() + kRecordSizeOffset) != kRecordSize ||
            read_le32(header.data() + kRecordsPerPageOffset) != kRecordsPerPage) {
            throw TokenizerError("cache header format does not match v1");
        }
        if (!std::equal(source_digest.begin(), source_digest.end(),
                        header.begin() + kSourceDigestOffset)) {
            throw TokenizerError("cache tokenizer digest does not match GGUF metadata");
        }

        const uint64_t record_count = read_le64(header.data() + kRecordCountOffset);
        const uint64_t page_count = read_le64(header.data() + kPageCountOffset);
        const uint64_t fence_offset = read_le64(header.data() + kFenceOffsetOffset);
        const uint64_t fence_size = read_le64(header.data() + kFenceSizeOffset);
        const uint64_t data_offset = read_le64(header.data() + kDataOffsetOffset);
        if (record_count == 0 || page_count == 0 ||
            page_count > impl->size / kPageSize ||
            page_count > std::numeric_limits<uint64_t>::max() / 16 ||
            record_count > std::numeric_limits<uint64_t>::max() - (kRecordsPerPage - 1) ||
            page_count != (record_count + kRecordsPerPage - 1) / kRecordsPerPage ||
            fence_offset != kFenceOffset || fence_size != page_count * 16 ||
            data_offset != align_page(fence_offset + fence_size) ||
            data_offset > impl->size || page_count > (impl->size - data_offset) / kPageSize ||
            data_offset + page_count * kPageSize != impl->size ||
            page_count > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / sizeof(Fence))) {
            throw TokenizerError("cache header contains invalid counts or offsets");
        }

        detail::Sha256 payload_hash;
        std::array<uint8_t, 64 * 1024> chunk{};
        for (uint64_t offset = kFenceOffset; offset < impl->size;) {
            const size_t amount = static_cast<size_t>(
                std::min<uint64_t>(chunk.size(), impl->size - offset));
            impl->read_at(offset, chunk.data(), amount, path);
            payload_hash.update(chunk.data(), amount);
            offset += amount;
        }
        const auto actual_payload = payload_hash.finish();
        if (!std::equal(actual_payload.begin(), actual_payload.end(),
                        header.begin() + kPayloadDigestOffset)) {
            throw TokenizerError("cache payload checksum mismatch");
        }

        std::vector<Fence> fences(static_cast<size_t>(page_count));
        std::array<uint8_t, 16> fence_bytes{};
        std::array<uint8_t, kPageSize> page{};
        uint64_t records_seen = 0;
        uint64_t previous_key = 0;
        bool have_previous = false;
        for (uint64_t i = 0; i < page_count; ++i) {
            impl->read_at(fence_offset + i * 16, fence_bytes.data(), fence_bytes.size(), path);
            const Fence fence{read_le64(fence_bytes.data()), read_le64(fence_bytes.data() + 8)};
            if (fence.offset != data_offset + i * kPageSize ||
                (i != 0 && fence.first_key <= fences[static_cast<size_t>(i - 1)].first_key)) {
                throw TokenizerError("cache fence table is not strictly ordered");
            }
            impl->read_at(fence.offset, page.data(), page.size(), path);
            const uint32_t count = read_le32(page.data());
            if (count == 0 || count > kRecordsPerPage ||
                read_le64(page.data() + 8) != fence.first_key ||
                read_le32(page.data() + 4) !=
                    crc32(page.data() + kPageHeaderSize, count * kRecordSize) ||
                records_seen + count > record_count) {
                throw TokenizerError("cache page header is invalid");
            }
            for (uint32_t j = 0; j < count; ++j) {
                const uint8_t * record = page.data() + kPageHeaderSize + j * kRecordSize;
                const uint64_t key = read_le64(record);
                if ((have_previous && key <= previous_key) ||
                    (j == 0 && key != fence.first_key)) {
                    throw TokenizerError("cache records are not strictly ordered");
                }
                // Reject the whole file here so it is regenerated, rather than
                // letting a bad ID surface at some later encode.
                const int64_t merged = static_cast<int64_t>(
                    static_cast<int32_t>(read_le32(record + 12)));
                if (merged < 0 || merged >= vocab_size) {
                    throw TokenizerError("cache record names a token outside the vocabulary");
                }
                previous_key = key;
                have_previous = true;
            }
            records_seen += count;
            fences[static_cast<size_t>(i)] = fence;
        }
        if (records_seen != record_count) {
            throw TokenizerError("cache page record counts do not match header");
        }

        result.impl_ = std::move(impl);
        result.fences_ = std::move(fences);
        result.cache_path_ = path;
        result.record_count_ = record_count;
        result.vocab_size_ = vocab_size;
        return true;
    } catch (const std::exception & e) {
        reason = e.what();
        return false;
    }
}

void DiskMergeIndex::create_cache(gguf_context * ctx,
                                  int64_t tk,
                                  int64_t mk,
                                  const detail::Sha256Digest & source_digest,
                                  const std::filesystem::path & path) {
    OsMemoryResource os_memory;
    std::pmr::monotonic_buffer_resource arena(&os_memory);
    std::pmr::unordered_map<std::string_view, int> ix(&arena);
    const size_t token_count = gguf_get_arr_n(ctx, tk);
    ix.reserve(token_count);
    for (size_t i = 0; i < token_count; ++i) {
        ix.emplace(gguf_get_arr_str(ctx, tk, i), static_cast<int>(i));
    }

    std::pmr::vector<RawRecord> records(&arena);
    const size_t merge_count = gguf_get_arr_n(ctx, mk);
    records.reserve(merge_count);
    std::string merged_piece;
    for (size_t i = 0; i < merge_count; ++i) {
        const std::string_view rule(gguf_get_arr_str(ctx, mk, i));
        const size_t separator = rule.find(' ');
        if (separator == std::string_view::npos) continue;
        const std::string_view left = rule.substr(0, separator);
        const std::string_view right = rule.substr(separator + 1);
        const auto left_it = ix.find(left);
        const auto right_it = ix.find(right);
        if (left_it == ix.end() || right_it == ix.end()) continue;
        merged_piece.clear();
        merged_piece.reserve(left.size() + right.size());
        merged_piece.append(left.data(), left.size());
        merged_piece.append(right.data(), right.size());
        const auto merged_it = ix.find(merged_piece);
        if (merged_it == ix.end()) continue;
        records.push_back(RawRecord{pair_key(left_it->second, right_it->second),
                                    static_cast<int32_t>(i),
                                    static_cast<int32_t>(merged_it->second)});
    }
    if (records.empty()) {
        throw TokenizerError(
            "tokenizer.ggml.merges yielded no usable rules — the file declares "
            "BPE but carries no merge table this vocab can resolve");
    }
    std::sort(records.begin(), records.end(), [](const RawRecord & a, const RawRecord & b) {
        return a.key < b.key || (a.key == b.key && a.rank < b.rank);
    });
    records.erase(std::unique(records.begin(), records.end(),
                              [](const RawRecord & a, const RawRecord & b) {
                                  return a.key == b.key;
                              }),
                  records.end());

    const uint64_t record_count = records.size();
    const uint64_t page_count = (record_count + kRecordsPerPage - 1) / kRecordsPerPage;
    const uint64_t fence_size = page_count * 16;
    const uint64_t data_offset = align_page(kFenceOffset + fence_size);
    const uint64_t file_size = data_offset + page_count * kPageSize;
    if (file_size > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        throw TokenizerError("BPE merge cache is too large for this platform");
    }

    std::pmr::vector<uint8_t> payload(&arena);
    payload.resize(static_cast<size_t>(file_size - kFenceOffset), 0);
    for (uint64_t page_index = 0; page_index < page_count; ++page_index) {
        const uint64_t begin = page_index * kRecordsPerPage;
        const uint32_t count = static_cast<uint32_t>(
            std::min<uint64_t>(kRecordsPerPage, record_count - begin));
        const uint64_t page_offset = data_offset + page_index * kPageSize;
        uint8_t * fence = payload.data() + page_index * 16;
        write_le64(fence, records[static_cast<size_t>(begin)].key);
        write_le64(fence + 8, page_offset);

        uint8_t * page = payload.data() + (page_offset - kFenceOffset);
        write_le32(page, count);
        write_le32(page + 4, 0);
        write_le64(page + 8, records[static_cast<size_t>(begin)].key);
        for (uint32_t j = 0; j < count; ++j) {
            const RawRecord & source = records[static_cast<size_t>(begin + j)];
            uint8_t * destination = page + kPageHeaderSize + j * kRecordSize;
            write_le64(destination, source.key);
            write_le32(destination + 8, static_cast<uint32_t>(source.rank));
            write_le32(destination + 12, static_cast<uint32_t>(source.merged));
        }
        write_le32(page + 4, crc32(page + kPageHeaderSize, count * kRecordSize));
    }

    const auto payload_digest = detail::sha256(payload.data(), payload.size());
    std::array<uint8_t, kPageSize> header{};
    std::memcpy(header.data(), kMagic, sizeof(kMagic));
    write_le32(header.data() + kVersionOffset, kFormatVersion);
    write_le32(header.data() + kEndianOffset, kEndianMarker);
    write_le32(header.data() + kPageSizeOffset, kPageSize);
    write_le32(header.data() + kRecordSizeOffset, kRecordSize);
    write_le32(header.data() + kRecordsPerPageOffset, kRecordsPerPage);
    write_le64(header.data() + kRecordCountOffset, record_count);
    write_le64(header.data() + kPageCountOffset, page_count);
    write_le64(header.data() + kFenceOffsetOffset, kFenceOffset);
    write_le64(header.data() + kFenceSizeOffset, fence_size);
    write_le64(header.data() + kDataOffsetOffset, data_offset);
    std::copy(source_digest.begin(), source_digest.end(), header.begin() + kSourceDigestOffset);
    std::copy(payload_digest.begin(), payload_digest.end(), header.begin() + kPayloadDigestOffset);

    const std::filesystem::path temp = unique_temp_path(path);
    FILE * file = nullptr;
    try {
#ifdef _WIN32
        file = _wfopen(temp.c_str(), L"wbx");
#else
        file = std::fopen(temp.c_str(), "wbx");
#endif
        if (file == nullptr) throw TokenizerError(system_message("create BPE merge cache", temp));
        write_all(file, header.data(), header.size(), temp);
        write_all(file, payload.data(), payload.size(), temp);
        durable_close(file, temp);
        atomic_replace(temp, path);
    } catch (...) {
        if (file != nullptr) std::fclose(file);
        std::error_code ignored;
        std::filesystem::remove(temp, ignored);
        throw;
    }
}

DiskMergeIndex DiskMergeIndex::from_gguf(gguf_context * ctx, int64_t tk, int64_t mk) {
    const detail::Sha256Digest source_digest = tokenizer_digest(ctx, tk, mk);
    const int64_t vocab_size = static_cast<int64_t>(gguf_get_arr_n(ctx, tk));
    const std::filesystem::path root = cache_root();
    std::error_code directory_error;
    std::filesystem::create_directories(root, directory_error);
    if (directory_error) {
        throw TokenizerError("cannot create BPE merge cache directory '" + root.string() +
                             "': " + directory_error.message());
    }
    const std::filesystem::path path =
        root / ("bpe-merges-v1-" + detail::sha256_hex(source_digest) + ".idx");

    DiskMergeIndex result;
    std::string reason;
    if (open_validated(path, source_digest, vocab_size, result, reason)) {
        result.cache_hit_ = true;
        return result;
    }

    try {
        create_cache(ctx, tk, mk, source_digest, path);
    } catch (...) {
        // Another process may have completed the same content-addressed file
        // while our atomic replacement was contending with it (notably on
        // Windows). Accept only a fully validated winner; otherwise preserve
        // the original creation error rather than hiding a real I/O failure.
        const std::exception_ptr creation_error = std::current_exception();
        reason.clear();
        if (open_validated(path, source_digest, vocab_size, result, reason)) {
            result.cache_hit_ = true;
            return result;
        }
        std::rethrow_exception(creation_error);
    }
    reason.clear();
    if (!open_validated(path, source_digest, vocab_size, result, reason)) {
        throw TokenizerError("generated BPE merge cache failed validation: " + reason);
    }
    result.cache_hit_ = false;
    return result;
}

bool DiskMergeIndex::find(uint64_t key, LookupScratch & scratch, Rule & result) const {
    if (impl_ == nullptr || fences_.empty() || key < fences_.front().first_key) return false;
    const auto upper = std::upper_bound(
        fences_.begin(), fences_.end(), key,
        [](uint64_t value, const Fence & fence) { return value < fence.first_key; });
    const size_t page_index = static_cast<size_t>((upper - fences_.begin()) - 1);

    LookupScratch::Slot * slot = nullptr;
    for (auto & candidate : scratch.slots) {
        if (candidate.page == page_index) {
            slot = &candidate;
            break;
        }
    }
    if (slot == nullptr) {
        slot = &scratch.slots[static_cast<size_t>(scratch.next_slot % scratch.slots.size())];
        ++scratch.next_slot;
        // Claim the slot only once its bytes have been validated below, so a
        // rejected page cannot be left advertising itself as cached and then
        // served unvalidated to a later lookup.
        slot->page = UINT64_MAX;
        impl_->read_at(fences_[page_index].offset, slot->bytes.data(), slot->bytes.size(), cache_path_);

        // Structural checks only. open_validated() verified this page's CRC and
        // the whole payload's SHA-256 at load; repeating the CRC per page read
        // measured at 96% of encode time (1,020 of 1,058 us), because crc32()
        // is a byte-at-a-time table lookup running at 376 MB/s and every encode
        // pushed 94 pages through it.
        //
        // What is given up is detection of a record edited in place after load
        // that still lands on this page and keeps its header intact. The two
        // checks below still catch truncation, a short read, and a page
        // replaced by a different one, which is what a swapped or rebuilt cache
        // file actually looks like. The bounds check also keeps `loaded` inside
        // the page for the search below.
        const uint32_t loaded = read_le32(slot->bytes.data());
        if (loaded == 0 || loaded > kRecordsPerPage ||
            read_le64(slot->bytes.data() + 8) != fences_[page_index].first_key) {
            throw TokenizerError("BPE merge cache page became invalid during encode: '" +
                                 cache_path_.string() + "'");
        }
        slot->page = page_index;
        ++scratch.page_reads;
    }

    const uint32_t count = read_le32(slot->bytes.data());
    uint32_t low = 0;
    uint32_t high = count;
    while (low < high) {
        const uint32_t mid = low + (high - low) / 2;
        const uint64_t candidate = read_le64(
            slot->bytes.data() + kPageHeaderSize + mid * kRecordSize);
        if (candidate < key) low = mid + 1;
        else high = mid;
    }
    if (low == count) return false;
    const uint8_t * record = slot->bytes.data() + kPageHeaderSize + low * kRecordSize;
    if (read_le64(record) != key) return false;
    result.rank = static_cast<int32_t>(read_le32(record + 8));
    result.merged = static_cast<int32_t>(read_le32(record + 12));
    // Checked again here, not only at load: the per-page CRC no longer runs on
    // each read, so a record edited after load reaches this point unnoticed.
    // Everything else it can corrupt yields a wrong-but-harmless token; an
    // out-of-range ID instead aborts the process inside ggml_get_rows, which no
    // caller can catch. Refusing it keeps the worst case an ordinary error.
    if (result.merged < 0 || static_cast<int64_t>(result.merged) >= vocab_size_) {
        throw TokenizerError("BPE merge cache names a token outside the vocabulary: '" +
                             cache_path_.string() + "'");
    }
    return true;
}

} // namespace nanoembed
