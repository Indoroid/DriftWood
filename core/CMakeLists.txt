# bmoe_core — the streaming engine.
#
# Pure-policy sources (config validation, the architecture registry, the n-gram draft
# matcher) build with no native dependency. The streaming adapters compile only when
# llama.cpp is present, because they include its public headers (llama.h, ggml.h, gguf.h).

set(_bmoe_library_type STATIC)
if(BMOE_BUILD_SHARED)
    set(_bmoe_library_type SHARED)
endif()
add_library(bmoe_core ${_bmoe_library_type}
    src/config.cpp
    src/moe/arch_registry.cpp
    src/engine/ngram_draft.cpp
)

target_include_directories(bmoe_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_features(bmoe_core PUBLIC cxx_std_17)
if(BMOE_BUILD_SHARED)
    set_target_properties(bmoe_core PROPERTIES OUTPUT_NAME meitte WINDOWS_EXPORT_ALL_SYMBOLS ON)
    target_compile_definitions(bmoe_core PRIVATE MEITTE_BUILDING_LIBRARY)
    target_compile_definitions(bmoe_core PUBLIC MEITTE_SHARED)
    include(GNUInstallDirs)
    install(TARGETS bmoe_core LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
                            RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
                            ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
    install(DIRECTORY include/bmoe DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()

# One source of truth for the version: the CMake project version reaches the code only here.
target_compile_definitions(bmoe_core PUBLIC BMOE_VERSION="${PROJECT_VERSION}")

if(BMOE_HAVE_LLAMA)
    target_sources(bmoe_core PRIVATE
        src/io/platform_io.cpp
        src/io/file_reader.cpp
        src/io/mapping_release.cpp
        src/moe/dense_weights.cpp
        src/moe/dense_stream.cpp
        src/moe/row_stream.cpp
        src/moe/gguf_offsets.cpp
        src/moe/expert_stream_source.cpp
        src/moe/router_hook.cpp
        src/engine/tensor_overrides.cpp
        src/engine/session.cpp
        src/engine/chat_parse.cpp
        src/engine/thinking_control.cpp
        src/engine/runtime.cpp
        src/engine/c_api.cpp
        src/multimodal/mtmd_runtime.cpp
        src/multimodal/media_io.cpp
        src/multimodal/audio_convert.cpp
        src/metrics/csv_metrics_sink.cpp
        src/metrics/route_trace_sink.cpp
        src/metrics/decode_trace_sink.cpp
    )
    # `llama` is the stable public API the streaming seam uses. `llama-common` (llama.cpp's
    # `common` utils) is linked only for chat-template rendering, reasoning parsing, and its
    # template-delimited reasoning-budget sampler; it is not a stable API. It propagates the
    # common/ and vendor/ includes.
    target_link_libraries(bmoe_core PUBLIC llama llama-common)
    # libmtmd is intentionally private: public bmoe headers expose only byte buffers/config.
    target_link_libraries(bmoe_core PRIVATE mtmd)
    target_compile_definitions(bmoe_core PUBLIC BMOE_HAVE_LLAMA=1)
    if(BMOE_BUILD_SHARED)
        if(APPLE)
            set_target_properties(bmoe_core PROPERTIES INSTALL_RPATH "@loader_path")
        elseif(UNIX)
            set_target_properties(bmoe_core PROPERTIES INSTALL_RPATH "$ORIGIN")
        endif()
        foreach(dependency llama llama-common mtmd ggml ggml-cpu ggml-base)
            # Only runtime libraries are part of the libmeitte package.
            set_property(TARGET ${dependency} PROPERTY PUBLIC_HEADER "")
            set_property(TARGET ${dependency} PROPERTY PRIVATE_HEADER "")
        endforeach()
        install(TARGETS llama llama-common mtmd ggml ggml-cpu ggml-base
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
                RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
    endif()
    find_package(Threads REQUIRED)
    target_link_libraries(bmoe_core PUBLIC Threads::Threads)

    if(ANDROID)
        # libandroid carries AHardwareBuffer_*, the reclaim-exempt allocation behind
        # `--dense-weights ahwb` (pio::pinned_alloc). No other platform has an equivalent.
        target_link_libraries(bmoe_core PUBLIC android)
    endif()

    # Intra-layer I/O–compute overlap needs the fork's expert-ready hook in ggml-cpu.h. Detect
    # it by inspecting the submodule header rather than the git remote, so any tree carrying the
    # hook (a rebase, a different fork) enables the feature. PUBLIC so cli/ and tests/ see it.
    set(_ggml_cpu_h "${CMAKE_SOURCE_DIR}/third_party/llama.cpp/ggml/include/ggml-cpu.h")
    if(EXISTS "${_ggml_cpu_h}")
        file(READ "${_ggml_cpu_h}" _ggml_cpu_src)
        if(_ggml_cpu_src MATCHES "ggml_cpu_set_expert_ready_hook")
            target_compile_definitions(bmoe_core PUBLIC BMOE_HAVE_EXPERT_READY_HOOK=1)
            message(STATUS "bmoe: expert-ready hook present (I/O–compute overlap available)")
        else()
            message(STATUS "bmoe: expert-ready hook absent (overlap unavailable)")
        endif()
        if(_ggml_cpu_src MATCHES "ggml_cpu_set_weight_ready_hook")
            target_compile_definitions(bmoe_core PUBLIC BMOE_HAVE_WEIGHT_READY_HOOK=1)
            message(STATUS "bmoe: dense weight-ready hook present")
        endif()
    else()
        message(STATUS "bmoe: expert-ready hook absent (overlap unavailable)")
    endif()
endif()
