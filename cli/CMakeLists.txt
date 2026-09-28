add_library(bmoe_cli_protocol STATIC session_protocol.cpp)
target_link_libraries(bmoe_cli_protocol PUBLIC bmoe_core)
target_include_directories(bmoe_cli_protocol PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

function(bmoe_portable_runtime_path target)
    if(APPLE)
        set(_build_rpath "@loader_path/../bin")
        set(_install_rpath "@loader_path/../lib")
    else()
        set(_build_rpath "$ORIGIN/../bin")
        set(_install_rpath "$ORIGIN/../lib")
    endif()
    set_target_properties(${target} PROPERTIES
                          BUILD_RPATH_USE_ORIGIN TRUE
                          BUILD_RPATH "${_build_rpath}"
                          INSTALL_RPATH "${_install_rpath}")
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux" OR ANDROID)
        target_link_options(${target} PRIVATE "LINKER:-z,origin")
    endif()
endfunction()

add_executable(meitte-cli main.cpp)
target_link_libraries(meitte-cli PRIVATE bmoe_core bmoe_cli_protocol)

if(UNIX)
    bmoe_portable_runtime_path(meitte-cli)
endif()

# The current HTTP transport uses POSIX sockets (Linux, macOS, Android/Termux). Keep the existing
# Windows CLI build green until the transport is moved onto llama.cpp's cross-platform httplib.
if(UNIX)
    add_executable(meitte-server server_main.cpp)
    target_link_libraries(meitte-server PRIVATE bmoe_core)

    # llama.cpp emits its shared libraries under build/bin while this executable lives under
    # build/cli. CMake's default build RPATH is an absolute path, which breaks as soon as a build
    # tree is moved or unpacked elsewhere. Keep both host builds and packaged build trees portable.
    bmoe_portable_runtime_path(meitte-server)
endif()
