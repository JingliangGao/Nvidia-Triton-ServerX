# Try to find libevent using pkg-config first
find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
    pkg_check_modules(LIBEVENT QUIET libevent)
    pkg_check_modules(LIBEVENT_PTHREADS QUIET libevent_pthreads)
endif()

# Try to find GTest
find_package(GTest QUIET)

# Try to find Protobuf
find_package(Protobuf REQUIRED)
if(PROTOBUF_FOUND)
    message(STATUS "Found Protobuf: ${PROTOBUF_VERSION}")
    file(GLOB PROTO_FILES "${CMAKE_CURRENT_SOURCE_DIR}/protobuf/*.proto")
    if(PROTO_FILES)
        message(STATUS "Found proto files: ${PROTO_FILES}")
        protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})
        message(STATUS "Generated Protobuf sources: ${PROTO_SRCS} and headers: ${PROTO_HDRS}")
        # 获取所有生成的 proto 头文件所在目录并去重
        set(PROTO_HDRS_DIRS "")
        foreach(_hdr ${PROTO_HDRS})
            get_filename_component(_dir ${_hdr} PATH)
            list(FIND PROTO_HDRS_DIRS ${_dir} _found)
            if(_found EQUAL -1)
                list(APPEND PROTO_HDRS_DIRS ${_dir})
            endif()
        endforeach()
        message(STATUS "Protobuf header dirs: ${PROTO_HDRS_DIRS}")
    endif()
else()
    message(FATAL_ERROR "Protobuf not found. Please install protobuf-compiler and libprotobuf-dev packages.")
endif()

add_library(
  kytensor-endpoint-library EXCLUDE_FROM_ALL
    src/log.cc
    src/kytensor_server.cc
    src/kytensor_codec.cc
    src/kytensor_common.cc
    src/infer_handler.cc
    src/kytensor_utils.cc
    src/kytensor_response.cc
    src/stream_infer_handler.cc
    ${PROTO_SRCS}
)

target_compile_features(kytensor-endpoint-library PRIVATE cxx_std_17)
target_compile_options(
kytensor-endpoint-library
PRIVATE
    -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -Wno-deprecated-declarations -Werror
)

set_target_properties(
  kytensor-endpoint-library
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(
    kytensor-endpoint-library
    PUBLIC
      triton-common-logging         # from repo-common
      triton-common-json            # from repo-common
      triton-core-serverapi         # from repo-core
      protobuf::libprotobuf
      ${LIBEVENT_LIBRARIES}
      -levent_pthreads
  )

target_include_directories(kytensor-endpoint-library PUBLIC
                            ${LIBEVENT_INCLUDE_DIRS}
                            ${CMAKE_CURRENT_BINARY_DIR}
                            ${PROTOBUF_INCLUDE_DIRS}
                            ${CMAKE_CURRENT_BINARY_DIR}/server/triton-server/kytensor_server)


# kytensor client library
add_library(
    kytensorclient SHARED
        src/log.cc
        src/kytensor_client.cc
        src/kytensor_codec.cc
        src/kytensor_response.cc
        src/kytensor_client_common.cc
        ${PROTO_SRCS}
)

target_compile_features(kytensorclient PRIVATE cxx_std_17)
target_compile_options(
    kytensorclient
    PRIVATE
        -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable -Wno-deprecated-declarations -Werror
)

set_target_properties(
    kytensorclient
    PROPERTIES
        POSITION_INDEPENDENT_CODE ON
)

target_link_libraries(
    kytensorclient
    PUBLIC
        triton-common-logging         # from repo-common
        triton-common-json            # from repo-common
        triton-core-serverapi         # from repo-core
        protobuf::libprotobuf
        -levent
        -levent_pthreads
)

target_include_directories(
    kytensorclient
    PRIVATE
        ${LIBEVENT_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR}
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)

# Create performance test executable
add_executable(performance_test tests/performance_test.cc)
target_include_directories(
    performance_test
    PRIVATE
        ${LIBEVENT_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR}
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
target_link_libraries(
    performance_test
    PUBLIC
        kytensorclient
)

# Create IsServerLive unit test executable
add_executable(kytensor_client_test tests/kytensor_client_test.cc)
target_include_directories(
    kytensor_client_test
    PRIVATE
        ${LIBEVENT_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR}
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
target_link_libraries(
    kytensor_client_test
    PUBLIC
        kytensorclient
        gtest
        gtest_main
        pthread
)

# Create Infer
add_executable(kytensor_infer_client tests/kytensor_infer_client.cc)
target_include_directories(
    kytensor_infer_client
    PRIVATE
        ${LIBEVENT_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR}
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
target_link_libraries(
    kytensor_infer_client
    PUBLIC
        kytensorclient
)

# Create Async Infer
add_executable(kytensor_async_infer_client tests/kytensor_async_infer_client.cc)
target_include_directories(
    kytensor_async_infer_client
    PRIVATE
        ${LIBEVENT_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR}
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
target_link_libraries(
    kytensor_async_infer_client
    PUBLIC
        kytensorclient
)

# Create thread_async_infer_client
add_executable(thread_async_infer_client tests/thread_async_infer_client.cc)
target_include_directories(
    thread_async_infer_client
    PRIVATE
        ${LIBEVENT_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR}
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
target_link_libraries(
    thread_async_infer_client
    PUBLIC
        kytensorclient
)

# Create kytensor_llm_async
add_executable(kytensor_llm_async tests/kytensor_llm_async.cc)
target_include_directories(
    kytensor_llm_async
    PRIVATE
        ${LIBEVENT_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR}
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
target_link_libraries(
    kytensor_llm_async
    PUBLIC
        kytensorclient
)

# Create kytensor_llm_stream
add_executable(kytensor_llm_stream tests/kytensor_llm_stream.cc)
target_include_directories(
    kytensor_llm_stream
    PRIVATE
        ${LIBEVENT_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR}
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
target_link_libraries(
    kytensor_llm_stream
    PUBLIC
        kytensorclient
)

# Create kytensor_llm_chat
add_executable(kytensor_llm_chat tests/kytensor_llm_chat.cc)
target_include_directories(
    kytensor_llm_chat
    PRIVATE
        ${LIBEVENT_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR}
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
target_link_libraries(
    kytensor_llm_chat
    PUBLIC
        kytensorclient
)

# Create kytensor_custom_repeat
add_executable(kytensor_custom_repeat tests/kytensor_custom_repeat.cc)
target_include_directories(
    kytensor_custom_repeat
    PRIVATE
        ${LIBEVENT_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR}
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
target_link_libraries(
    kytensor_custom_repeat
    PUBLIC
        kytensorclient
)

# Create kytensor-cli
add_executable(kytensor-cli tools/cli.cc)
target_include_directories(
    kytensor-cli
    PRIVATE
        ${LIBEVENT_INCLUDE_DIRS}
        ${CMAKE_CURRENT_BINARY_DIR}
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
target_link_libraries(
    kytensor-cli
    PUBLIC
        kytensorclient
)