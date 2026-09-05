cmake_minimum_required(VERSION 3.18)

project(NanoEmbed
    VERSION 0.1.0
    DESCRIPTION "Text embeddings for edge devices, constrained to tens of MB RAM."
    LANGUAGES C CXX)

# ---- Language standards -----------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ---- Output layout ----------------------------------------------------------
if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif()
if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif()

# ---- Project-wide warning flags (applied per-target, not globally) ----------
# We want strict warnings on OUR code; third-party ggml is left alone.
set(NANOEMBED_WARNING_FLAGS
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall>
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wextra>
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wpedantic>
    $<$<C_COMPILER_ID:GNU,Clang,AppleClang>:-Wall>
    $<$<C_COMPILER_ID:GNU,Clang,AppleClang>:-Wextra>
    $<$<C_COMPILER_ID:GNU,Clang,AppleClang>:-Wpedantic>)

# ---- Vendored ggml ----------------------------------------------------------
# Disable ggml's tests/examples to keep our build lean.
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)

# ---- Core library -----------------------------------------------------------
add_library(nanoembed_core STATIC
    src/api/c_api.cpp
    src/batch.cpp
    src/embedder.cpp
    src/gguf_util.cpp
    src/gguf_scanner.cpp
    src/mapped_weight_store.cpp
    src/streaming_execution.cpp
    src/arch/registry.cpp
    src/arch/bert_arch.cpp
    src/arch/gemma3_arch.cpp
    src/tokenizer/registry.cpp
    src/tokenizer/wordpiece.cpp
    src/tokenizer/sha256.cpp
    src/tokenizer/disk_merge_index.cpp
    src/tokenizer/spm_bpe.cpp
    src/forward/embed_layer.cpp
    src/forward/attention.cpp
    src/forward/ffn.cpp
    src/forward/encoder_block.cpp
    src/forward/pool.cpp
    src/forward/rms_norm.cpp
    src/forward/rope.cpp
    src/forward/gated_ffn.cpp
    src/forward/gqa_attention.cpp)

target_include_directories(nanoembed_core
    PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(nanoembed_core
    PUBLIC ggml::ggml)

target_compile_options(nanoembed_core PRIVATE ${NANOEMBED_WARNING_FLAGS})

add_library(nanoembed::core ALIAS nanoembed_core)

# ---- Tools ------------------------------------------------------------------
add_executable(nanoembed-inspect
    tools/nanoembed-inspect/main.cpp)

# inspect is an internal dev tool, so it gets the private include dir
# of nanoembed_core to call the scanner. End-users go through include/.
target_include_directories(nanoembed-inspect
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(nanoembed-inspect PRIVATE nanoembed_core ggml::ggml)
target_compile_options(nanoembed-inspect PRIVATE ${NANOEMBED_WARNING_FLAGS})

add_executable(nanoembed-cli
    tools/nanoembed-cli/main.cpp)
target_link_libraries(nanoembed-cli PRIVATE nanoembed_core)
target_compile_options(nanoembed-cli PRIVATE ${NANOEMBED_WARNING_FLAGS})

# nanoembed-bench is Linux-only: its whole measurement model rests on
# /proc/<pid>/{status,statm,smaps_rollup,clear_refs}, which have no portable
# equivalent (macOS in particular offers no way to reset resident_size_max).
# The library and tests still build everywhere; only the bench tool is gated.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_package(Threads REQUIRED)

    add_executable(nanoembed-bench
        tools/nanoembed-bench/cache_control.cpp
        tools/nanoembed-bench/main.cpp
        tools/nanoembed-bench/metrics.cpp
        tools/nanoembed-bench/statistics.cpp)
    target_link_libraries(nanoembed-bench PRIVATE nanoembed_core Threads::Threads)
    target_compile_options(nanoembed-bench PRIVATE ${NANOEMBED_WARNING_FLAGS})

    # Narrow diagnostic for tokenizer metadata lifetime. Unlike the benchmark
    # worker, it deliberately reports construction phases in one process.
    add_executable(nanoembed-tokenizer-memory-probe
        tools/nanoembed-tokenizer-memory-probe.cpp)
    target_include_directories(nanoembed-tokenizer-memory-probe
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(nanoembed-tokenizer-memory-probe PRIVATE nanoembed_core)
    target_compile_options(nanoembed-tokenizer-memory-probe
        PRIVATE ${NANOEMBED_WARNING_FLAGS})
else()
    message(STATUS "nanoembed-bench: skipped (Linux-only)")
endif()

# ---- Tests ------------------------------------------------------------------
include(CTest)

if(BUILD_TESTING)
    find_package(Threads REQUIRED)

    # Pure benchmark-statistics coverage. This target is portable and does not
    # need Linux /proc or a model, even though nanoembed-bench itself does.
    add_executable(nanoembed_bench_statistics_test
        tests/unit/bench_statistics_test.cpp
        tools/nanoembed-bench/statistics.cpp)
    target_include_directories(nanoembed_bench_statistics_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tools/nanoembed-bench)
    target_compile_options(nanoembed_bench_statistics_test
        PRIVATE ${NANOEMBED_WARNING_FLAGS})
    add_test(NAME bench_statistics COMMAND nanoembed_bench_statistics_test)

    # /proc parsing and sample aggregation are pure even though the live
    # readers are Linux-only. Keep their coverage available on macOS too.
    add_executable(nanoembed_bench_metrics_test
        tests/unit/bench_metrics_test.cpp
        tools/nanoembed-bench/metrics.cpp)
    target_include_directories(nanoembed_bench_metrics_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tools/nanoembed-bench)
    target_compile_options(nanoembed_bench_metrics_test
        PRIVATE ${NANOEMBED_WARNING_FLAGS})
    add_test(NAME bench_metrics COMMAND nanoembed_bench_metrics_test)

    # Cache semantics are portable even though live mincore/fadvise operations
    # are Linux-only. This also guards explicit unsupported-platform behavior.
    add_executable(nanoembed_bench_cache_control_test
        tests/unit/bench_cache_control_test.cpp
        tools/nanoembed-bench/cache_control.cpp)
    target_include_directories(nanoembed_bench_cache_control_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tools/nanoembed-bench)
    target_compile_options(nanoembed_bench_cache_control_test
        PRIVATE ${NANOEMBED_WARNING_FLAGS})
    add_test(NAME bench_cache_control COMMAND nanoembed_bench_cache_control_test)

    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME bench_compare_compatibility
                 COMMAND ${Python3_EXECUTABLE} -m unittest
                         discover -s tests/unit -p bench_compare_test.py)
        set_tests_properties(bench_compare_compatibility PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

        add_test(NAME bench_corpus_selection
                 COMMAND ${Python3_EXECUTABLE} -m unittest
                         discover -s tests/unit -p bench_corpus_selection_test.py)
        set_tests_properties(bench_corpus_selection PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

        add_test(NAME bench_runner
                 COMMAND ${Python3_EXECUTABLE} -m unittest
                         discover -s tests/unit -p bench_runner_test.py)
        set_tests_properties(bench_runner PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

        add_test(NAME bench_fingerprint
                 COMMAND ${Python3_EXECUTABLE} -m unittest
                         discover -s tests/unit -p bench_fingerprint_test.py)
        set_tests_properties(bench_fingerprint PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

        add_test(NAME flores_accuracy_corpus
                 COMMAND ${Python3_EXECUTABLE} -m unittest
                         discover -s tests/unit -p prepare_flores_accuracy_corpus_test.py)
        set_tests_properties(flores_accuracy_corpus PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

        # Cross-feature runner contract: drives runner.py as a real CLI against
        # a short-lived executable fixture.  This is portable; live /proc
        # observation stays in the Linux-only native selftests below.
        add_test(NAME bench_harness_integration
                 COMMAND ${Python3_EXECUTABLE} -m unittest
                         discover -s tests/integration -p bench_harness_test.py)
        set_tests_properties(bench_harness_integration PROPERTIES
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            TIMEOUT 120)
    endif()

    # ABI link test: pure-C consumer that links the public ABI and exercises
    # the full surface. Catches accidental ABI breakage from M1 onward.
    add_executable(nanoembed_abi_link_test tests/abi/abi_link_test.c)
    target_link_libraries(nanoembed_abi_link_test PRIVATE nanoembed_core)
    target_compile_options(nanoembed_abi_link_test PRIVATE ${NANOEMBED_WARNING_FLAGS})
    add_test(NAME abi_link COMMAND nanoembed_abi_link_test)

    # GGUF scanner unit tests.
    add_executable(nanoembed_scanner_test tests/unit/scanner_test.cpp)
    target_include_directories(nanoembed_scanner_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(nanoembed_scanner_test PRIVATE nanoembed_core ggml::ggml)
    target_compile_options(nanoembed_scanner_test PRIVATE ${NANOEMBED_WARNING_FLAGS})
    add_test(NAME scanner COMMAND nanoembed_scanner_test)

    # B1 proof only: metadata tensors borrow a read-only whole-file mapping and
    # are checked against positioned reads. This does not change the production
    # loader or execute the B3 internal streaming path.
    if(UNIX)
        add_executable(nanoembed_mmap_gguf_proof_test
            tests/unit/mmap_gguf_proof_test.cpp)
        target_link_libraries(nanoembed_mmap_gguf_proof_test PRIVATE ggml::ggml)
        target_compile_options(nanoembed_mmap_gguf_proof_test
            PRIVATE ${NANOEMBED_WARNING_FLAGS})

        # B2 production ownership/validation seam.  The default test is fully
        # synthetic; four additional CTests below prepare local real models
        # when present, without inference or residency advice.
        add_executable(nanoembed_mapped_weight_store_test
            tests/unit/mapped_weight_store_test.cpp)
        target_include_directories(nanoembed_mapped_weight_store_test
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(nanoembed_mapped_weight_store_test
            PRIVATE nanoembed_core ggml::ggml)
        target_compile_options(nanoembed_mapped_weight_store_test
            PRIVATE ${NANOEMBED_WARNING_FLAGS})
        add_test(NAME mapped_weight_store
                 COMMAND nanoembed_mapped_weight_store_test)

        add_executable(nanoembed_streaming_ranges_test
            tests/unit/streaming_ranges_test.cpp)
        target_include_directories(nanoembed_streaming_ranges_test
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
        target_link_libraries(nanoembed_streaming_ranges_test
            PRIVATE nanoembed_core Threads::Threads)
        target_compile_options(nanoembed_streaming_ranges_test
            PRIVATE ${NANOEMBED_WARNING_FLAGS})
        add_test(NAME streaming_ranges COMMAND nanoembed_streaming_ranges_test)
    endif()

    # The partitioner's pure halves. No mmap, no model, no Linux -- the liveness
    # rule is platform-independent and there is no reason to skip it anywhere.
    add_executable(nanoembed_streaming_units_test
        tests/unit/streaming_units_test.cpp)
    target_include_directories(nanoembed_streaming_units_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(nanoembed_streaming_units_test
        PRIVATE nanoembed_core Threads::Threads)
    target_compile_options(nanoembed_streaming_units_test
        PRIVATE ${NANOEMBED_WARNING_FLAGS})
    add_test(NAME streaming_units COMMAND nanoembed_streaming_units_test)

    # WordPiece tokenizer parity test against HuggingFace fixture.
    add_executable(nanoembed_tokenizer_test tests/unit/tokenizer_test.cpp)
    target_include_directories(nanoembed_tokenizer_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(nanoembed_tokenizer_test
        PRIVATE nanoembed_core ggml::ggml Threads::Threads)
    target_compile_options(nanoembed_tokenizer_test PRIVATE ${NANOEMBED_WARNING_FLAGS})
    add_test(NAME tokenizer COMMAND nanoembed_tokenizer_test)

    # Forward graph builder parity test against HF activation fixtures.
    add_executable(nanoembed_forward_test tests/unit/forward_test.cpp)
    target_include_directories(nanoembed_forward_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(nanoembed_forward_test PRIVATE nanoembed_core ggml::ggml)
    target_compile_options(nanoembed_forward_test PRIVATE ${NANOEMBED_WARNING_FLAGS})
    add_test(NAME forward COMMAND nanoembed_forward_test)

    # End-to-end golden parity test vs sentence-transformers.
    add_executable(nanoembed_golden_test
        tests/integration/golden_test.cpp
        tests/integration/golden_accuracy.cpp)
    target_include_directories(nanoembed_golden_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration)
    target_link_libraries(nanoembed_golden_test PRIVATE nanoembed_core)
    target_compile_options(nanoembed_golden_test PRIVATE ${NANOEMBED_WARNING_FLAGS})
    add_test(NAME golden COMMAND nanoembed_golden_test)

    add_executable(nanoembed_golden_accuracy_test
        tests/unit/golden_accuracy_test.cpp
        tests/integration/golden_accuracy.cpp)
    target_include_directories(nanoembed_golden_accuracy_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests/integration)
    target_compile_options(nanoembed_golden_accuracy_test PRIVATE ${NANOEMBED_WARNING_FLAGS})
    add_test(NAME golden_accuracy COMMAND nanoembed_golden_accuracy_test)
    set_tests_properties(golden_accuracy PROPERTIES ENVIRONMENT
        "NANOEMBED_GOLDEN_PROVENANCE_TEST_FIXTURE=${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/golden/st-bge-small.bin")

    # Sequence-length / buffer-reuse regression (M3.5 graph allocator).
    add_executable(nanoembed_seq_len_test tests/integration/seq_len_test.cpp)
    target_link_libraries(nanoembed_seq_len_test PRIVATE nanoembed_core)
    target_compile_options(nanoembed_seq_len_test PRIVATE ${NANOEMBED_WARNING_FLAGS})
    add_test(NAME seq_len COMMAND nanoembed_seq_len_test)
    set_tests_properties(seq_len PROPERTIES TIMEOUT 600)

    # Per-layer parity for the gemma3 forward path.
    add_executable(nanoembed_gemma3_forward_test tests/unit/gemma3_forward_test.cpp)
    target_include_directories(nanoembed_gemma3_forward_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(nanoembed_gemma3_forward_test PRIVATE nanoembed_core ggml::ggml)
    target_compile_options(nanoembed_gemma3_forward_test PRIVATE ${NANOEMBED_WARNING_FLAGS})
    add_test(NAME gemma3_forward COMMAND nanoembed_gemma3_forward_test)
    set_tests_properties(gemma3_forward PROPERTIES TIMEOUT 600)

    # Sequence-length limits / clamping behaviour.
    add_executable(nanoembed_limits_test tests/integration/limits_test.cpp)
    target_link_libraries(nanoembed_limits_test PRIVATE nanoembed_core)
    target_compile_options(nanoembed_limits_test PRIVATE ${NANOEMBED_WARNING_FLAGS})
    add_test(NAME limits COMMAND nanoembed_limits_test)

    # The C ABI promises distinct contexts can run concurrently on one shared
    # model. This is the regression guard for context-local mutable scratch.
    add_executable(nanoembed_context_concurrency_test
        tests/integration/context_concurrency_test.cpp)
    target_link_libraries(nanoembed_context_concurrency_test
        PRIVATE nanoembed_core Threads::Threads)
    target_compile_options(nanoembed_context_concurrency_test
        PRIVATE ${NANOEMBED_WARNING_FLAGS})
    add_test(NAME context_concurrency COMMAND nanoembed_context_concurrency_test)
    set_tests_properties(context_concurrency PROPERTIES TIMEOUT 600)

    add_executable(nanoembed_batch_test tests/integration/batch_test.cpp)
    target_include_directories(nanoembed_batch_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(nanoembed_batch_test PRIVATE nanoembed_core)
    target_compile_options(nanoembed_batch_test PRIVATE ${NANOEMBED_WARNING_FLAGS})
    add_test(NAME batch COMMAND nanoembed_batch_test)
    set_tests_properties(batch PROPERTIES TIMEOUT 1800)

    add_executable(nanoembed_mode_selection_test
        tests/integration/mode_selection_test.cpp)
    target_link_libraries(nanoembed_mode_selection_test
        PRIVATE nanoembed_core Threads::Threads)
    target_compile_options(nanoembed_mode_selection_test
        PRIVATE ${NANOEMBED_WARNING_FLAGS})
    add_test(NAME mode_selection COMMAND nanoembed_mode_selection_test)
    set_tests_properties(mode_selection PROPERTIES TIMEOUT 1800)

    # B3 internal-only mapped phase runner. The executable is portable enough
    # to assert an explicit unsupported result, while real inference/advice is
    # registered and exercised only on Linux.
    add_executable(nanoembed_streaming_integration_test
        tests/integration/streaming_integration_test.cpp)
    target_include_directories(nanoembed_streaming_integration_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(nanoembed_streaming_integration_test
        PRIVATE nanoembed_core Threads::Threads)
    target_compile_options(nanoembed_streaming_integration_test
        PRIVATE ${NANOEMBED_WARNING_FLAGS})
    add_test(NAME streaming_integration COMMAND nanoembed_streaming_integration_test)
    set_tests_properties(streaming_integration PROPERTIES TIMEOUT 1800)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set_tests_properties(streaming_integration PROPERTIES
            SKIP_REGULAR_EXPRESSION "SKIP:" LABELS "linux-only")
    endif()

    # A context must not resize its gallocr buffer on the first embed. Guards
    # the sequence-length axis; see the test's header for why pooling and
    # normalization do not currently move the buffer size.
    add_executable(nanoembed_reserve_invariant_test
        tests/integration/reserve_invariant_test.cpp)
    target_include_directories(nanoembed_reserve_invariant_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(nanoembed_reserve_invariant_test
        PRIVATE nanoembed_core ggml::ggml)
    target_compile_options(nanoembed_reserve_invariant_test
        PRIVATE ${NANOEMBED_WARNING_FLAGS})
    add_test(NAME reserve_invariant COMMAND nanoembed_reserve_invariant_test)
    set_tests_properties(reserve_invariant PROPERTIES TIMEOUT 600)

    # Bench harness selftest. Unlike every other test here it needs no GGUF, so
    # it is the one bench-side check CI can actually run. It asserts that the
    # fork+exec'd worker's baseline is unaffected by memory the parent dirtied
    # (i.e. exec really did replace the address space) and that windowed RSS
    # growth matches a known allocation. The second invocation enables the
    # optional rollup sampler and verifies it sees a short-lived spike; the
    # first verifies the default path creates no detailed samples.
    if(TARGET nanoembed-bench)
        add_test(NAME bench_selftest
                 COMMAND nanoembed-bench --selftest --selftest-alloc-mb 64)
        add_test(NAME bench_selftest_memory_profile
                 COMMAND nanoembed-bench --selftest --selftest-alloc-mb 64
                         --memory-profile --memory-profile-interval-ms 10)
        set_tests_properties(bench_selftest bench_selftest_memory_profile
            PROPERTIES TIMEOUT 120)
    else()
        # These are runtime checks of Linux /proc semantics, not portable unit
        # tests.  Register explicit skips so a non-Linux CTest report cannot be
        # mistaken for evidence that profile-off avoided smaps_rollup live.
        foreach(_test bench_selftest bench_selftest_memory_profile)
            add_test(NAME ${_test}
                     COMMAND ${CMAKE_COMMAND} -E echo
                             "SKIP: ${_test} requires Linux /proc and nanoembed-bench")
            set_tests_properties(${_test} PROPERTIES
                SKIP_REGULAR_EXPRESSION "SKIP:"
                LABELS "linux-only")
        endforeach()
    endif()

    # Wire the local models + fixtures into env-gated tests. Two families and
    # the Harrier Q8_0 regression variant are exercised. Each file has its own
    # cache variable so it can live outside the tree.
    set(NANOEMBED_TEST_MODEL
        "${CMAKE_CURRENT_SOURCE_DIR}/models/bge-small-en-v1.5-f16.gguf"
        CACHE FILEPATH "BERT GGUF (bge-small-en-v1.5) used by the accuracy tests")
    set(NANOEMBED_TEST_MODEL_GEMMA3
        "${CMAKE_CURRENT_SOURCE_DIR}/models/harrier-270m.gguf"
        CACHE FILEPATH "gemma3 GGUF (harrier-oss-v1-270m) used by the accuracy tests")
    set(NANOEMBED_TEST_MODEL_GEMMA3_Q8
        "${CMAKE_CURRENT_SOURCE_DIR}/models/harrier-270m-q8_0.gguf"
        CACHE FILEPATH "Q8_0 gemma3 GGUF used by the quantized golden test")
    set(NANOEMBED_TEST_MODEL_GEMMA3_Q4
        "${CMAKE_CURRENT_SOURCE_DIR}/models/harrier-270m-q4_k.gguf"
        CACHE FILEPATH "Q4_K gemma3 GGUF used by report-only accuracy measurement")
    option(NANOEMBED_REQUIRE_MODEL
        "Fail configuration if an accuracy-test model is missing" OFF)

    set(_NE_FIXTURES "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")

    # A model that is present contributes its env var; one that is absent makes
    # the tests needing it skip, which is exactly what NANOEMBED_REQUIRE_MODEL
    # turns into a configure-time failure so CI cannot go green on skips.
    set(_NE_MODELS "")
    foreach(_var
            NANOEMBED_TEST_MODEL
            NANOEMBED_TEST_MODEL_GEMMA3
            NANOEMBED_TEST_MODEL_GEMMA3_Q8)
        if(EXISTS "${${_var}}")
            list(APPEND _NE_MODELS "${_var}=${${_var}}")
        elseif(NANOEMBED_REQUIRE_MODEL)
            message(FATAL_ERROR
                "NANOEMBED_REQUIRE_MODEL=ON but no model at ${${_var}} "
                "(set via ${_var}). The accuracy tests would silently skip.")
        else()
            message(STATUS
                "No model at ${${_var}} — tests needing ${_var} will skip.")
        endif()
    endforeach()
    # Q4 is a report-only measurement, not a new CI gate. Its absence must not
    # change NANOEMBED_REQUIRE_MODEL semantics for the historical accuracy set.
    if(EXISTS "${NANOEMBED_TEST_MODEL_GEMMA3_Q4}")
        list(APPEND _NE_MODELS
            "NANOEMBED_TEST_MODEL_GEMMA3_Q4=${NANOEMBED_TEST_MODEL_GEMMA3_Q4}")
    else()
        message(STATUS
            "No Q4 model at ${NANOEMBED_TEST_MODEL_GEMMA3_Q4} — report-only Q4 accuracy will skip.")
    endif()

    if(TARGET nanoembed_mmap_gguf_proof_test AND EXISTS "${NANOEMBED_TEST_MODEL}")
        add_test(NAME mmap_gguf_proof_bert_f16
                 COMMAND nanoembed_mmap_gguf_proof_test
                         "${NANOEMBED_TEST_MODEL}"
                         --require-type f16 --require-type f32)
    endif()
    if(TARGET nanoembed_mmap_gguf_proof_test AND EXISTS "${NANOEMBED_TEST_MODEL_GEMMA3}")
        add_test(NAME mmap_gguf_proof_harrier_f32
                 COMMAND nanoembed_mmap_gguf_proof_test
                         "${NANOEMBED_TEST_MODEL_GEMMA3}"
                         --require-type f32)
    endif()
    if(TARGET nanoembed_mmap_gguf_proof_test AND EXISTS "${NANOEMBED_TEST_MODEL_GEMMA3_Q8}")
        add_test(NAME mmap_gguf_proof_harrier_q8
                 COMMAND nanoembed_mmap_gguf_proof_test
                         "${NANOEMBED_TEST_MODEL_GEMMA3_Q8}"
                         --require-type q8_0 --require-type f32)
    endif()
    if(TARGET nanoembed_mmap_gguf_proof_test AND EXISTS "${NANOEMBED_TEST_MODEL_GEMMA3_Q4}")
        add_test(NAME mmap_gguf_proof_harrier_q4
                 COMMAND nanoembed_mmap_gguf_proof_test
                         "${NANOEMBED_TEST_MODEL_GEMMA3_Q4}"
                         --require-type q4_K --require-type q4_0
                         --require-type q8_0 --require-type f32)
    endif()

    if(TARGET nanoembed_mapped_weight_store_test AND EXISTS "${NANOEMBED_TEST_MODEL}")
        add_test(NAME mapped_prepare_bert_f16
                 COMMAND nanoembed_mapped_weight_store_test --model
                         "${NANOEMBED_TEST_MODEL}" bert f16 f32)
    endif()
    if(TARGET nanoembed_mapped_weight_store_test AND EXISTS "${NANOEMBED_TEST_MODEL_GEMMA3}")
        add_test(NAME mapped_prepare_harrier_f32
                 COMMAND nanoembed_mapped_weight_store_test --model
                         "${NANOEMBED_TEST_MODEL_GEMMA3}" gemma3 f32)
    endif()
    if(TARGET nanoembed_mapped_weight_store_test AND EXISTS "${NANOEMBED_TEST_MODEL_GEMMA3_Q8}")
        add_test(NAME mapped_prepare_harrier_q8
                 COMMAND nanoembed_mapped_weight_store_test --model
                         "${NANOEMBED_TEST_MODEL_GEMMA3_Q8}" gemma3 q8_0 f32)
    endif()
    if(TARGET nanoembed_mapped_weight_store_test AND EXISTS "${NANOEMBED_TEST_MODEL_GEMMA3_Q4}")
        add_test(NAME mapped_prepare_harrier_q4
                 COMMAND nanoembed_mapped_weight_store_test --model
                         "${NANOEMBED_TEST_MODEL_GEMMA3_Q4}" gemma3 q4_K q4_0 q8_0 f32)
    endif()

    # Model-only tests.
    foreach(_t scanner seq_len limits context_concurrency batch mode_selection reserve_invariant streaming_integration)
        set_tests_properties(${_t} PROPERTIES ENVIRONMENT "${_NE_MODELS}")
    endforeach()

    # Tests that also need a fixture. Built by appending to the model list so
    # an absent model leaves no stray empty entry behind.
    set(_NE_ENV ${_NE_MODELS})
    list(APPEND _NE_ENV
        "NANOEMBED_TOKENIZER_FIXTURE=${_NE_FIXTURES}/tokenizer/bge-small-eval.tsv"
        "NANOEMBED_TOKENIZER_FIXTURE_GEMMA3=${_NE_FIXTURES}/tokenizer/harrier-270m-eval.tsv")
    set_tests_properties(tokenizer PROPERTIES ENVIRONMENT "${_NE_ENV}")

    set(_NE_ENV ${_NE_MODELS})
    list(APPEND _NE_ENV
        "NANOEMBED_ACTIVATION_FIXTURES=${_NE_FIXTURES}/activations")
    set_tests_properties(forward PROPERTIES ENVIRONMENT "${_NE_ENV}")

    set(_NE_ENV ${_NE_MODELS})
    list(APPEND _NE_ENV
        "NANOEMBED_ACTIVATION_FIXTURES_GEMMA3=${_NE_FIXTURES}/activations_gemma3")
    set_tests_properties(gemma3_forward PROPERTIES ENVIRONMENT "${_NE_ENV}")

    set(_NE_ENV ${_NE_MODELS})
    list(APPEND _NE_ENV
        "NANOEMBED_GOLDEN_FIXTURE=${_NE_FIXTURES}/golden/st-bge-small.bin"
        "NANOEMBED_GOLDEN_FIXTURE_GEMMA3=${_NE_FIXTURES}/golden/st-harrier-270m.bin"
        "NANOEMBED_ACCURACY_JSON=${CMAKE_CURRENT_BINARY_DIR}/accuracy-report.json")
    set_tests_properties(golden PROPERTIES ENVIRONMENT "${_NE_ENV}")
    set_tests_properties(streaming_integration PROPERTIES ENVIRONMENT "${_NE_ENV}")
endif()
