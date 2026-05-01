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
    src/gguf_scanner.cpp
    src/tokenizer/wordpiece.cpp)

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

# ---- Tests ------------------------------------------------------------------
include(CTest)

if(BUILD_TESTING)
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

    # WordPiece tokenizer parity test against HuggingFace fixture.
    add_executable(nanoembed_tokenizer_test tests/unit/tokenizer_test.cpp)
    target_include_directories(nanoembed_tokenizer_test
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_link_libraries(nanoembed_tokenizer_test PRIVATE nanoembed_core ggml::ggml)
    target_compile_options(nanoembed_tokenizer_test PRIVATE ${NANOEMBED_WARNING_FLAGS})
    add_test(NAME tokenizer COMMAND nanoembed_tokenizer_test)

    # Wire the local model + fixture into env-gated tests.
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/models/bge-small-en-v1.5-f16.gguf)
        set(_NANOEMBED_MODEL "${CMAKE_CURRENT_SOURCE_DIR}/models/bge-small-en-v1.5-f16.gguf")
        set_tests_properties(scanner PROPERTIES
            ENVIRONMENT "NANOEMBED_TEST_MODEL=${_NANOEMBED_MODEL}")
        set_tests_properties(tokenizer PROPERTIES
            ENVIRONMENT
            "NANOEMBED_TEST_MODEL=${_NANOEMBED_MODEL};NANOEMBED_TOKENIZER_FIXTURE=${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/tokenizer/bge-small-eval.tsv")
    endif()
endif()
