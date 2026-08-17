# Byte-identity gates. Tiny synthetic models are generated at test time from llama.cpp's
# vendored gguf Python package. One model covers each expert tensor layout.

add_executable(bmoe_moe_gates moe_gates.cpp)
target_link_libraries(bmoe_moe_gates PRIVATE bmoe_core)

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

add_executable(bmoe_ngram_test ngram_test.cpp)
target_link_libraries(bmoe_ngram_test PRIVATE bmoe_core)
add_test(NAME ngram_draft COMMAND bmoe_ngram_test)

if(UNIX)
    add_executable(bmoe_server_api_test server_api_test.cpp)
    target_link_libraries(bmoe_server_api_test PRIVATE bmoe_core)
    add_test(NAME server_api COMMAND bmoe_server_api_test)
endif()

find_program(PYTHON3 NAMES python3 python)
if(PYTHON3)
    add_test(NAME server_cli_parity
             COMMAND ${PYTHON3} ${CMAKE_SOURCE_DIR}/scripts/check-server-cli-parity.py)

    set(_gguf_py "${CMAKE_SOURCE_DIR}/third_party/llama.cpp/gguf-py")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E env "PYTHONPATH=${_gguf_py}"
                ${PYTHON3} -c "import numpy, gguf"
        RESULT_VARIABLE _have_gguf_deps
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(_have_gguf_deps EQUAL 0)
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
