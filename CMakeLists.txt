# BigMoeOnEdge — top-level build.
#
# The engine (`bmoe_core`) uses llama.cpp's public model, GGUF and eval-callback APIs for
# serial expert streaming. Optional CPU overlap uses the expert-ready callback from the synced
# llama.cpp fork. The serial path builds against stock upstream.
# See docs/seam.md § 4.
#
# llama.cpp is an optional dependency at configure time: when the submodule is absent
# only the pure-policy code (config validation) compiles, so the scaffold and a subset
# of CI stay green before the native dependency is fetched.
cmake_minimum_required(VERSION 3.21)

# The version is declared here and nowhere else in the build: the engine reports it (meitte-cli
# --version, and the run-parameter preamble of every metrics CSV), so a committed benchmark file
# says which engine produced it. Keep it in step with CHANGELOG.md; align the app's versionName
# when preparing a release.
project(bigmoeonedge VERSION 0.24.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Tests split along the same line as the build: core tests need only libmeitte; host tests (the
# CLI line protocol, the HTTP server, CLI/server flag parity) are added only with the frontends.
option(BMOE_BUILD_TESTS "Build byte-identity gates and unit tests" ON)
option(BMOE_BUILD_CLI   "Build the meitte-cli and meitte-server frontends" ON)
option(BMOE_BUILD_SHARED "Build libmeitte for C++ and C/FFI callers" OFF)
# Off by default: diagnostics, not part of the product. They need no llama.cpp, so they can be
# configured and cross-compiled on their own without building the engine.
option(BMOE_BUILD_TOOLS "Build standalone diagnostic tools (bmoe-iobench)" OFF)

# --- Optional native dependency: llama.cpp ----------------------------------------
# The streaming seam uses only the public C API (llama.h). We additionally link llama.cpp's
# `common` layer for chat-template rendering, reasoning parsing, and its template-delimited
# reasoning-budget sampler, so prompt formatting, thinking extraction, and budget end markers are
# template-driven per model instead of hardcoded. Unlike the streaming seam, `common` is NOT a
# stable API — it can change between upstream versions, so a submodule bump may need the chat glue
# in core/src/engine (session_*.cpp, chat_render.cpp, chat_parse.cpp) updated (the gates catch it
# at build time). That trade-off is deliberate; see docs/seam.md.
set(BMOE_HAVE_LLAMA OFF)
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/llama.cpp/CMakeLists.txt")
    set(BMOE_HAVE_LLAMA ON)
    # Upstream examples/tests/server stay off; build the common utils we link for chat.
    set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_COMMON   ON  CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
    # Build llama.cpp's multimodal projector library without its CLI/server tools.
    set(LLAMA_BUILD_MTMD     ON  CACHE BOOL "" FORCE)
    # Use upstream's FFmpeg-backed sampled-video helper.
    set(MTMD_VIDEO           ON  CACHE BOOL "" FORCE)
    set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)  # common's model-download path unused
    add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)
endif()

if(BMOE_BUILD_SHARED AND NOT BMOE_HAVE_LLAMA)
    message(FATAL_ERROR "BMOE_BUILD_SHARED requires the llama.cpp submodule")
endif()

add_subdirectory(core)

if(BMOE_BUILD_CLI AND BMOE_HAVE_LLAMA)
    add_subdirectory(cli)
endif()

if(BMOE_BUILD_TESTS AND BMOE_HAVE_LLAMA)
    enable_testing()
    add_subdirectory(tests)
endif()

if(BMOE_BUILD_TOOLS)
    add_subdirectory(tools)
endif()

message(STATUS "BigMoeOnEdge: llama.cpp present = ${BMOE_HAVE_LLAMA}")
