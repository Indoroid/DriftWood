# Byte-identity gates. Tiny synthetic models are generated at test time from llama.cpp's
# vendored gguf Python package. One model covers each expert tensor layout.

add_executable(bmoe_moe_gates moe_gates.cpp)
target_link_libraries(bmoe_moe_gates PRIVATE bmoe_core)

add_executable(bmoe_dense_stream_test dense_stream_test.cpp)
target_link_libraries(bmoe_dense_stream_test PRIVATE bmoe_core)
target_include_directories(bmoe_dense_stream_test PRIVATE ${CMAKE_SOURCE_DIR}/core/src/moe)
add_test(NAME dense_stream COMMAND bmoe_dense_stream_test)

add_executable(bmoe_dense_gates dense_gates.cpp)
target_link_libraries(bmoe_dense_gates PRIVATE bmoe_core)

add_executable(bmoe_recipe_test recipe_test.cpp)
target_link_libraries(bmoe_recipe_test PRIVATE bmoe_core)
add_test(NAME moe_recipes COMMAND bmoe_recipe_test)

add_executable(bmoe_chat_parse_test chat_parse_test.cpp)
target_link_libraries(bmoe_chat_parse_test PRIVATE bmoe_core)
target_include_directories(bmoe_chat_parse_test PRIVATE ${CMAKE_SOURCE_DIR}/core/src/engine)
target_compile_definitions(bmoe_chat_parse_test PRIVATE
    BMOE_QWEN3_TEMPLATE_PATH="${CMAKE_SOURCE_DIR}/third_party/llama.cpp/models/templates/Qwen3.5-4B.jinja")
add_test(NAME chat_parse COMMAND bmoe_chat_parse_test)

add_executable(bmoe_think_control_test think_control_test.cpp)
target_link_libraries(bmoe_think_control_test PRIVATE bmoe_core)
target_include_directories(bmoe_think_control_test PRIVATE ${CMAKE_SOURCE_DIR}/core/src/engine)
set(_tmpl_dir "${CMAKE_SOURCE_DIR}/third_party/llama.cpp/models/templates")
target_compile_definitions(bmoe_think_control_test PRIVATE
    BMOE_TMPL_QWEN3="${_tmpl_dir}/Qwen3.5-4B.jinja"
    BMOE_TMPL_LFM25="${_tmpl_dir}/LFM2.5-8B-A1B.jinja"
    BMOE_TMPL_GPTOSS="${_tmpl_dir}/openai-gpt-oss-120b.jinja"
    BMOE_TMPL_LFM2="${_tmpl_dir}/LFM2-8B-A1B.jinja"
    BMOE_TMPL_LFM25_INSTRUCT="${_tmpl_dir}/LFM2.5-Instruct.jinja"
    BMOE_TMPL_GEMMA4="${_tmpl_dir}/google-gemma-4-31B-it.jinja")
add_test(NAME think_control COMMAND bmoe_think_control_test)

add_executable(bmoe_config_test config_test.cpp)
target_link_libraries(bmoe_config_test PRIVATE bmoe_core)
add_test(NAME config_validate COMMAND bmoe_config_test)

add_executable(bmoe_mapping_release_test mapping_release_test.cpp)
target_link_libraries(bmoe_mapping_release_test PRIVATE bmoe_core)
target_include_directories(bmoe_mapping_release_test PRIVATE ${CMAKE_SOURCE_DIR}/core/src/io)
add_test(NAME mapping_release COMMAND bmoe_mapping_release_test)

add_executable(bmoe_library_test library_test.cpp)
target_link_libraries(bmoe_library_test PRIVATE bmoe_core)
find_program(BMOE_TEST_FFMPEG_EXECUTABLE ffmpeg)
if(BMOE_TEST_FFMPEG_EXECUTABLE)
    target_compile_definitions(bmoe_library_test PRIVATE BMOE_TEST_FFMPEG=1)
endif()
add_test(NAME library_api COMMAND bmoe_library_test)
add_executable(bmoe_library_c_test library_c_test.c)
target_link_libraries(bmoe_library_c_test PRIVATE bmoe_core)
add_test(NAME library_c_api COMMAND bmoe_library_c_test)
add_executable(bmoe_library_abi_test library_abi_test.c)
target_link_libraries(bmoe_library_abi_test PRIVATE bmoe_core)
add_test(NAME library_abi COMMAND bmoe_library_abi_test)
add_executable(bmoe_context_test context_test.cpp)
target_link_libraries(bmoe_context_test PRIVATE bmoe_core)
add_executable(bmoe_session_transaction_test session_transaction_test.cpp)
target_link_libraries(bmoe_session_transaction_test PRIVATE bmoe_core)
target_include_directories(bmoe_session_transaction_test PRIVATE ${CMAKE_SOURCE_DIR}/core/src/engine)
add_executable(bmoe_multi_session_test multi_session_test.cpp)
target_link_libraries(bmoe_multi_session_test PRIVATE bmoe_core)
add_executable(bmoe_reasoning_budget_test reasoning_budget_test.cpp)
target_link_libraries(bmoe_reasoning_budget_test PRIVATE bmoe_core)
target_compile_definitions(bmoe_reasoning_budget_test PRIVATE BMOE_TMPL_QWEN3="${_tmpl_dir}/Qwen3.5-4B.jinja")

set(BMOE_TEST_MM_MODEL "" CACHE FILEPATH "Optional multimodal text model for the media trace fixture")
set(BMOE_TEST_MMPROJ "" CACHE FILEPATH "Optional projector for the media trace fixture")
set(BMOE_TEST_MEDIA "" CACHE FILEPATH "Optional input for the media trace fixture")
option(BMOE_TEST_MM_STREAM "Enable expert streaming in the media trace fixture" OFF)
if(BMOE_TEST_MM_MODEL AND BMOE_TEST_MMPROJ AND BMOE_TEST_MEDIA)
    add_executable(bmoe_media_trace_test media_trace_test.cpp)
    target_link_libraries(bmoe_media_trace_test PRIVATE bmoe_core)
    target_include_directories(bmoe_media_trace_test PRIVATE ${CMAKE_SOURCE_DIR}/core/src/multimodal)
    add_test(NAME media_prefill_trace
             COMMAND bmoe_media_trace_test ${BMOE_TEST_MM_MODEL} ${BMOE_TEST_MMPROJ}
                     ${BMOE_TEST_MEDIA} ${BMOE_TEST_MM_STREAM})
endif()

# Host tests: the frontends' protocols. They exist only when the frontends are built.
if(BMOE_BUILD_CLI)
    add_executable(bmoe_session_protocol_test session_protocol_test.cpp)
    target_link_libraries(bmoe_session_protocol_test PRIVATE bmoe_cli_protocol)
    add_test(NAME session_protocol COMMAND bmoe_session_protocol_test)
endif()

add_executable(bmoe_ngram_test ngram_test.cpp)
target_link_libraries(bmoe_ngram_test PRIVATE bmoe_core)
add_test(NAME ngram_draft COMMAND bmoe_ngram_test)

if(UNIX AND BMOE_BUILD_CLI)
    add_executable(bmoe_server_api_test server_api_test.cpp)
    target_link_libraries(bmoe_server_api_test PRIVATE bmoe_core)
    add_test(NAME server_api COMMAND bmoe_server_api_test)
endif()

# The interval-union stall clock. Injected timestamps, no wall clock, so overlap/nesting/
# separation and the open-interval snapshot rule are verified exactly (issue #98).
add_executable(bmoe_stall_union_test stall_union_test.cpp)
target_link_libraries(bmoe_stall_union_test PRIVATE bmoe_core)
target_include_directories(bmoe_stall_union_test PRIVATE ${CMAKE_SOURCE_DIR}/core/src/moe)
add_test(NAME stall_union COMMAND bmoe_stall_union_test)

# The CSV `# summary` trailer serialization. Drives the sink directly with a filled RunSummary —
# no model, no llama.cpp — so the emitted key names and units are pinned by a test rather than by
# whatever script last read them. The prefill keys must match BMOE_DONE's, token for token.
add_executable(bmoe_csv_summary_test csv_summary_test.cpp)
target_link_libraries(bmoe_csv_summary_test PRIVATE bmoe_core)
add_test(NAME csv_summary COMMAND bmoe_csv_summary_test)

find_program(PYTHON3 NAMES python3 python)
if(PYTHON3)
    if(BMOE_BUILD_SHARED)
        add_test(NAME ctypes_smoke
                 COMMAND ${PYTHON3} ${CMAKE_SOURCE_DIR}/examples/python/ctypes_smoke.py $<TARGET_FILE:bmoe_core>)
    endif()
    if(BMOE_BUILD_CLI)
        add_test(NAME server_cli_parity
                 COMMAND ${PYTHON3} ${CMAKE_SOURCE_DIR}/scripts/check-server-cli-parity.py)
    endif()

    set(_gguf_py "${CMAKE_SOURCE_DIR}/third_party/llama.cpp/gguf-py")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${_gguf_py}"
                ${PYTHON3} -c "import numpy, gguf"
        RESULT_VARIABLE _have_gguf_deps
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(_have_gguf_deps EQUAL 0)
        foreach(dtype "f16" "q4_0")
            set(_dense_model "${CMAKE_CURRENT_BINARY_DIR}/tiny-dense-${dtype}.gguf")
            add_test(NAME dense_make_tiny_model_${dtype}
                     COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${_gguf_py}"
                             ${PYTHON3} ${CMAKE_SOURCE_DIR}/scripts/make-tiny-moe.py
                             --arch qwen3dense --type ${dtype} --out ${_dense_model})
            set_tests_properties(dense_make_tiny_model_${dtype} PROPERTIES FIXTURES_SETUP TINY_DENSE_${dtype})
            add_test(NAME dense_gates_${dtype} COMMAND bmoe_dense_gates ${_dense_model})
            set_tests_properties(dense_gates_${dtype} PROPERTIES FIXTURES_REQUIRED TINY_DENSE_${dtype})
        endforeach()
        foreach(spec "qwen3moe" "gemma4")
            set(_model "${CMAKE_CURRENT_BINARY_DIR}/tiny-moe-${spec}.gguf")
            set(_fixture "TINY_MOE_MODEL_${spec}")

            add_test(
                NAME moe_make_tiny_model_${spec}
                COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${_gguf_py}"
                        ${PYTHON3} ${CMAKE_SOURCE_DIR}/scripts/make-tiny-moe.py
                        --arch ${spec} --out ${_model}
            )
            set_tests_properties(moe_make_tiny_model_${spec} PROPERTIES FIXTURES_SETUP ${_fixture})

            add_test(NAME moe_gates_${spec} COMMAND bmoe_moe_gates ${_model})
            set_tests_properties(moe_gates_${spec} PROPERTIES FIXTURES_REQUIRED ${_fixture})
            add_test(NAME context_${spec} COMMAND bmoe_context_test ${_model})
            set_tests_properties(context_${spec} PROPERTIES FIXTURES_REQUIRED ${_fixture})
            add_test(NAME session_transaction_${spec} COMMAND bmoe_session_transaction_test ${_model})
            set_tests_properties(session_transaction_${spec} PROPERTIES FIXTURES_REQUIRED ${_fixture})
            add_test(NAME multi_session_${spec} COMMAND bmoe_multi_session_test ${_model})
            set_tests_properties(multi_session_${spec} PROPERTIES FIXTURES_REQUIRED ${_fixture})
            add_test(NAME reasoning_budget_${spec} COMMAND bmoe_reasoning_budget_test ${_model})
            set_tests_properties(reasoning_budget_${spec} PROPERTIES FIXTURES_REQUIRED ${_fixture})
        endforeach()

        add_test(
            NAME moe_make_tiny_model_split
            COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${_gguf_py}"
                    ${PYTHON3} ${CMAKE_SOURCE_DIR}/scripts/make-tiny-moe.py --arch qwen3moe
                    --split-max-tensors 20 --out "${CMAKE_CURRENT_BINARY_DIR}/tiny-moe-split.gguf"
        )
        set_tests_properties(moe_make_tiny_model_split PROPERTIES FIXTURES_SETUP TINY_MOE_MODEL_split)

        add_test(NAME moe_gates_split
                 COMMAND bmoe_moe_gates "${CMAKE_CURRENT_BINARY_DIR}/tiny-moe-split-00001-of-00004.gguf")
        set_tests_properties(moe_gates_split PROPERTIES FIXTURES_REQUIRED TINY_MOE_MODEL_split)
    else()
        message(WARNING "numpy or vendored gguf-py unavailable: MoE byte-identity gates are disabled")
    endif()
else()
    message(WARNING "python3 not found: parity and MoE byte-identity gates are disabled")
endif()
