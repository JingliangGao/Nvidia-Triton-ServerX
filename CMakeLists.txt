cmake_minimum_required(VERSION 3.16)

project(tritonserver LANGUAGES C CXX)

# Disable all RPATH/RUNPATH handling globally — no -Wl,-rpath in any link command.
set(CMAKE_SKIP_RPATH ON CACHE BOOL "Disable rpath" FORCE)
set(CMAKE_SKIP_BUILD_RPATH ON CACHE BOOL "Disable build rpath" FORCE)
set(CMAKE_SKIP_INSTALL_RPATH ON CACHE BOOL "Disable install rpath" FORCE)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH OFF CACHE BOOL "Disable link path rpath" FORCE)
set(CMAKE_INSTALL_RPATH "" CACHE STRING "Empty install rpath" FORCE)
set(CMAKE_BUILD_RPATH "" CACHE STRING "Empty build rpath" FORCE)

include(CMakeDependentOption)

# Use C++17 standard as Triton's minimum required.
set(TRITON_MIN_CXX_STANDARD 17 CACHE STRING "The minimum C++ standard which features are requested to build this target.")

set(TRITON_VERSION "2.49.0" CACHE STRING "The version of the Triton shared library" )
set(KYTENSOR_DIR "/usr/lib/kytensor" CACHE STRING "The version of the Triton shared library" )

option(TRITON_ENABLE_SERVER "Enable build triton server" ON)
option(TRITON_ENABLE_CLIENT "Enable build triton client" OFF)
option(TRITON_ENABLE_ONNX_BACKEND "Enable build triton onnx backend" OFF)
option(TRITON_ENABLE_LLAMACPP_BACKEND "Enable build triton llamacpp backend" ON)
option(TRITON_ENABLE_PYTHON_BACKEND "Enable build triton llamacpp backend" OFF)
option(TRITON_ENABLE_LOGGING "Include logging support in server" ON)
option(TRITON_ENABLE_STATS "Include statistics collections in server" ON)
option(TRITON_ENABLE_TRACING "Include tracing support in server" OFF)
option(TRITON_ENABLE_NVTX "Include NVTX support in server" OFF)
option(TRITON_ENABLE_GPU "Enable GPU support in server" ON)
option(TRITON_ENABLE_MALI_GPU "Enable Arm Mali GPU support in server" OFF)
option(TRITON_IGPU_BUILD "Enable options for iGPU compilation in sever" OFF)
set(TRITON_MIN_COMPUTE_CAPABILITY "6.0" CACHE STRING
    "The minimum CUDA compute capability supported by Triton" )
set(TRITON_EXTRA_LIB_PATHS "" CACHE PATH "Extra library paths for Triton Server build")

# Ensemble
option(TRITON_ENABLE_ENSEMBLE "Include ensemble support in server" OFF)

# Endpoints
option(TRITON_ENABLE_HTTP "Include HTTP API in server" ON)
option(TRITON_ENABLE_GRPC "Include GRPC API in server" ON)
option(TRITON_ENABLE_SAGEMAKER "Include AWS SageMaker API in server" OFF)
option(TRITON_ENABLE_VERTEX_AI "Include Vertex AI API in server" OFF)

# Metrics
option(TRITON_ENABLE_METRICS "Include metrics support in server" OFF)
option(TRITON_ENABLE_METRICS_GPU "Include GPU metrics support in server" OFF)
option(TRITON_ENABLE_METRICS_CPU "Include CPU metrics support in server" OFF)

# Cloud storage
option(TRITON_ENABLE_GCS "Include GCS Filesystem support in server" OFF)
option(TRITON_ENABLE_S3 "Include S3 Filesystem support in server" OFF)
option(TRITON_ENABLE_AZURE_STORAGE "Include Azure Storage Filesystem support in server" OFF)

# Need to know if TensorRT is available when building unit tests
option(TRITON_ENABLE_TENSORRT "Include TensorRT backend in server" OFF)

# ASAN
option(TRITON_ENABLE_ASAN "Build with address sanitizer" OFF)

# Repo tags
set(TRITON_REPO_ORGANIZATION "https://github.com/triton-inference-server" CACHE STRING "Git repository to pull from")
set(TRITON_THIRD_PARTY_REPO_TAG "main" CACHE STRING
    "Tag for triton-inference-server/third_party repo")
set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")

# Triton module source addr
set(TRITON_MODULE_SOURCE_DIR "${CMAKE_CURRENT_BINARY_DIR}/.." CACHE STRING "Addr for triton source")
message(STATUS "TRITON_MODULE_SOURCE_DIR ${TRITON_MODULE_SOURCE_DIR}")

# Third-party location
set(TRITON_THIRD_PARTY_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/third-party" CACHE STRING "Location of third-party build")
set(TRITON_THIRD_PARTY_SRC_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/third-party-src" CACHE STRING "Location of third-party source")

if(TRITON_ENABLE_METRICS AND NOT TRITON_ENABLE_STATS)
  message(FATAL_ERROR "TRITON_ENABLE_METRICS=ON requires TRITON_ENABLE_STATS=ON")
endif()

if(TRITON_ENABLE_TRACING AND NOT TRITON_ENABLE_STATS)
  message(FATAL_ERROR "TRITON_ENABLE_TRACING=ON requires TRITON_ENABLE_STATS=ON")
endif()

if (TRITON_ENABLE_METRICS_CPU AND NOT TRITON_ENABLE_METRICS)
  message(FATAL_ERROR "TRITON_ENABLE_METRICS_CPU=ON requires TRITON_ENABLE_METRICS=ON")
endif()

if (TRITON_ENABLE_METRICS_GPU AND NOT TRITON_ENABLE_METRICS)
  message(FATAL_ERROR "TRITON_ENABLE_METRICS_GPU=ON requires TRITON_ENABLE_METRICS=ON")
endif()

if (TRITON_ENABLE_METRICS_GPU AND NOT TRITON_ENABLE_GPU)
  message(FATAL_ERROR "TRITON_ENABLE_METRICS_GPU=ON requires TRITON_ENABLE_GPU=ON")
endif()

if(TRITON_ENABLE_ASAN AND TRITON_ENABLE_GPU)
  message(FATAL_ERROR "TRITON_ENABLE_ASAN=ON requires TRITON_ENABLE_GPU=OFF")
endif()

# onnx backend
set(TRITON_ONNXRUNTIME_INCLUDE_PATHS "" CACHE PATH "Paths to ONNXRuntime includes")
set(TRITON_ONNXRUNTIME_LIB_PATHS "" CACHE PATH "Paths to ONNXRuntime libraries")

# llamacpp backend
set(TRITON_LLAMACPP_INCLUDE_PATHS "" CACHE PATH "Paths to llamacpp includes")
set(TRITON_LLAMACPP_LIB_PATHS "" CACHE PATH "Paths to llamacpp libraries")

#
# Dependencies
#
include(FetchContent)

FetchContent_Declare(
  repo-core
  SOURCE_DIR ${TRITON_MODULE_SOURCE_DIR}/core
)
# FetchContent_Declare(
#   repo-third-party
#   SOURCE_DIR ${TRITON_MODULE_SOURCE_DIR}/third_party
# )

# Some libs are installed to ${TRITON_THIRD_PARTY_INSTALL_PREFIX}/{LIB}/lib64 instead
# of ${TRITON_THIRD_PARTY_INSTALL_PREFIX}/{LIB}/lib on Centos
set(LIB_DIR "lib")
if(LINUX)
  file(STRINGS "/etc/os-release" DISTRO_ID_LIKE REGEX "ID_LIKE")
  if(${DISTRO_ID_LIKE} MATCHES "rhel|centos")
    set (LIB_DIR "lib64")
  endif(${DISTRO_ID_LIKE} MATCHES "rhel|centos")
endif(LINUX)
set(TRITON_CORE_HEADERS_ONLY OFF)

FetchContent_MakeAvailable(repo-core)

###
#add_subdirectory(${TRITON_MODULE_SOURCE_DIR}/common)
add_subdirectory(${TRITON_MODULE_SOURCE_DIR}/backend)
#
# Triton server executable and examples
#

# Need to use ExternalProject for our builds so that we can get the
# correct dependencies between Triton executable and the
# ExternalProject dependencies (found in the third_party repo)
include(ExternalProject)

# If CMAKE_TOOLCHAIN_FILE is set, propagate that hint path to the external
# projects.
set(_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE "")
if (CMAKE_TOOLCHAIN_FILE)
  set(_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE "-DCMAKE_TOOLCHAIN_FILE:PATH=${CMAKE_TOOLCHAIN_FILE}")
endif()

# If VCPKG_TARGET_TRIPLET is set, propagate that hint path to the external
# projects.
set(_CMAKE_ARGS_VCPKG_TARGET_TRIPLET "")
if (VCPKG_TARGET_TRIPLET)
  set(_CMAKE_ARGS_VCPKG_TARGET_TRIPLET "-DVCPKG_TARGET_TRIPLET:STRING=${VCPKG_TARGET_TRIPLET}")
endif()

# If OPENSSL_ROOT_DIR is set, propagate that hint path to the external
# projects with OpenSSL dependency.
set(_CMAKE_ARGS_OPENSSL_ROOT_DIR "")
if (OPENSSL_ROOT_DIR)
  set(_CMAKE_ARGS_OPENSSL_ROOT_DIR "-DOPENSSL_ROOT_DIR:PATH=${OPENSSL_ROOT_DIR}")
endif()

# Location where protobuf-config.cmake will be installed varies by
# platform
if (WIN32)
  set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/protobuf/cmake")
else()
  set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/protobuf/${LIB_DIR}/cmake/protobuf")
endif()

# Triton with Opentelemetry is not supported on Windows
# FIXME: add location for Windows, when support is added
# JIRA DLIS-4786
if (WIN32)
  set(_FINDPACKAGE_OPENTELEMETRY_CONFIG_DIR "")
else()
  set(_FINDPACKAGE_OPENTELEMETRY_CONFIG_DIR "${TRITON_THIRD_PARTY_INSTALL_PREFIX}/opentelemetry-cpp/${LIB_DIR}/cmake/opentelemetry-cpp")
endif()

if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(TRITON_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/install)
else()
  set(TRITON_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
endif()

#set(PYTHONBACKEND_DEPENDS Boost)
#set(TRITON_DEPENDS triton-core protobuf googletest re2)
if(${TRITON_ENABLE_GCS})
  set(TRITON_DEPENDS ${TRITON_DEPENDS} google-cloud-cpp)
endif() # TRITON_ENABLE_GCS
if(${TRITON_ENABLE_S3})
  set(TRITON_DEPENDS ${TRITON_DEPENDS} aws-sdk-cpp)
endif() # TRITON_ENABLE_S3

# cover here      JingliangGao 2025/06/18
# if(${TRITON_ENABLE_HTTP} OR ${TRITON_ENABLE_METRICS} OR ${TRITON_ENABLE_SAGEMAKER} OR ${TRITON_ENABLE_VERTEX_AI})
#   set(TRITON_DEPENDS ${TRITON_DEPENDS} libevent libevhtp)
# endif() # TRITON_ENABLE_HTTP || TRITON_ENABLE_METRICS || TRITON_ENABLE_SAGEMAKER || TRITON_ENABLE_VERTEX_AI

if(${TRITON_ENABLE_GRPC})
#  set(TRITON_DEPENDS ${TRITON_DEPENDS} grpc)
endif() # TRITON_ENABLE_GRPC
if(NOT WIN32 AND ${TRITON_ENABLE_TRACING})
  set(TRITON_DEPENDS ${TRITON_DEPENDS} opentelemetry-cpp)
endif() # TRITON_ENABLE_TRACING

# server
if(${TRITON_ENABLE_SERVER})
ExternalProject_Add(triton-server
  PREFIX triton-server
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/server/src"
  BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/server/triton-server"
  CMAKE_CACHE_ARGS
    -DCMAKE_SKIP_RPATH:BOOL=ON
    -DCMAKE_SKIP_BUILD_RPATH:BOOL=ON
    -DCMAKE_SKIP_INSTALL_RPATH:BOOL=ON
    -DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=OFF
    -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
    ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
    ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
    ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
    -DBoost_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/boost
    -DGTEST_ROOT:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/googletest
    -DgRPC_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/grpc/lib/cmake/grpc
    -Dc-ares_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/c-ares/${LIB_DIR}/cmake/c-ares
    -Dre2_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/re2/${LIB_DIR}/cmake/re2
    -Dabsl_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/absl/${LIB_DIR}/cmake/absl
    -DCURL_DIR:STRING=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/curl/${LIB_DIR}/cmake/CURL
    -Dnlohmann_json_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/nlohmann_json/${LIB_DIR}/cmake/nlohmann_json
    #-DLibevent_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/libevent/lib/cmake/libevent
    #-Dlibevhtp_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/libevhtp/lib/cmake/libevhtp
    -Dstorage_client_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/google-cloud-cpp/${LIB_DIR}/cmake/storage_client
    -Dgoogle_cloud_cpp_common_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/google-cloud-cpp/${LIB_DIR}/cmake/google_cloud_cpp_common
    -DCrc32c_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/crc32c/${LIB_DIR}/cmake/Crc32c
    -DAWSSDK_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/aws-sdk-cpp/${LIB_DIR}/cmake/AWSSDK
    -Daws-cpp-sdk-core_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/aws-sdk-cpp/${LIB_DIR}/cmake/aws-cpp-sdk-core
    -Daws-cpp-sdk-s3_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/aws-sdk-cpp/${LIB_DIR}/cmake/aws-cpp-sdk-s3
    -Daws-c-event-stream_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/aws-sdk-cpp/${LIB_DIR}/aws-c-event-stream/cmake
    -Daws-c-common_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/aws-sdk-cpp/${LIB_DIR}/aws-c-common/cmake
    -Daws-checksums_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/aws-sdk-cpp/${LIB_DIR}/aws-checksums/cmake
    -Dopentelemetry-cpp_DIR:PATH=${_FINDPACKAGE_OPENTELEMETRY_CONFIG_DIR}
    -DTRITON_CORE_HEADERS_ONLY:BOOL=${TRITON_CORE_HEADERS_ONLY}
    -DTRITON_REPO_ORGANIZATION:STRING=${TRITON_REPO_ORGANIZATION}
    -DTRITON_MODULE_SOURCE_DIR:STRING=${TRITON_MODULE_SOURCE_DIR}
    -DTRITON_IGPU_BUILD:BOOL=${TRITON_IGPU_BUILD}
    -DTRITON_THIRD_PARTY_REPO_TAG:STRING=${TRITON_THIRD_PARTY_REPO_TAG}
    -DTRITON_COMMON_REPO_TAG:STRING=${TRITON_COMMON_REPO_TAG}
    -DTRITON_CORE_REPO_TAG:STRING=${TRITON_CORE_REPO_TAG}
    -DTRITON_BACKEND_REPO_TAG:STRING=${TRITON_BACKEND_REPO_TAG}
    -DTRITON_EXTRA_LIB_PATHS:PATH=${TRITON_EXTRA_LIB_PATHS}
    -DTRITON_ENABLE_ASAN:BOOL=${TRITON_ENABLE_ASAN}
    -DTRITON_ENABLE_NVTX:BOOL=${TRITON_ENABLE_NVTX}
    -DTRITON_ENABLE_TRACING:BOOL=${TRITON_ENABLE_TRACING}
    -DTRITON_ENABLE_LOGGING:BOOL=${TRITON_ENABLE_LOGGING}
    -DTRITON_ENABLE_STATS:BOOL=${TRITON_ENABLE_STATS}
    -DTRITON_ENABLE_GPU:BOOL=${TRITON_ENABLE_GPU}
    -DTRITON_ENABLE_MALI_GPU:BOOL=${TRITON_ENABLE_MALI_GPU}
    -DTRITON_ENABLE_HTTP:BOOL=${TRITON_ENABLE_HTTP}
    -DTRITON_ENABLE_SAGEMAKER:BOOL=${TRITON_ENABLE_SAGEMAKER}
    -DTRITON_ENABLE_VERTEX_AI:BOOL=${TRITON_ENABLE_VERTEX_AI}
    -DTRITON_ENABLE_GRPC:BOOL=${TRITON_ENABLE_GRPC}
    -DTRITON_MIN_COMPUTE_CAPABILITY:STRING=${TRITON_MIN_COMPUTE_CAPABILITY}
    -DTRITON_ENABLE_METRICS:BOOL=${TRITON_ENABLE_METRICS}
    -DTRITON_ENABLE_METRICS_GPU:BOOL=${TRITON_ENABLE_METRICS_GPU}
    -DTRITON_ENABLE_METRICS_CPU:BOOL=${TRITON_ENABLE_METRICS_CPU}
    -DTRITON_ENABLE_GCS:BOOL=${TRITON_ENABLE_GCS}
    -DTRITON_ENABLE_AZURE_STORAGE:BOOL=${TRITON_ENABLE_AZURE_STORAGE}
    -DTRITON_ENABLE_S3:BOOL=${TRITON_ENABLE_S3}
    -DTRITON_ENABLE_TENSORRT:BOOL=${TRITON_ENABLE_TENSORRT}
    -DTRITON_ENABLE_ENSEMBLE:BOOL=${TRITON_ENABLE_ENSEMBLE}
    -DTRITON_MIN_CXX_STANDARD:STRING=${TRITON_MIN_CXX_STANDARD}
    -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_INSTALL_PREFIX}/server
    -DTRITON_VERSION:STRING=${TRITON_VERSION}
    -DKYTENSOR_DIR:STRING=${KYTENSOR_DIR}
  INSTALL_COMMAND ""
  DEPENDS ${TRITON_DEPENDS}
)

INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/server/triton-server/kytensor DESTINATION ${CMAKE_INSTALL_BINDIR})
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/server/triton-server/_deps/repo-core-build/triton-core/libtritonserver.so DESTINATION ${CMAKE_INSTALL_LIBDIR})
INSTALL(FILES ${CMAKE_CURRENT_SOURCE_DIR}/server/kytensor.service DESTINATION /usr/lib/systemd/system/)
INSTALL(FILES ${CMAKE_CURRENT_SOURCE_DIR}/server/kytensor.conf DESTINATION /opt/system/conf/kytensor/)

INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/server/triton-server/kytensor_server/libkytensorclient.so DESTINATION ${CMAKE_INSTALL_LIBDIR})
INSTALL(FILES ${CMAKE_CURRENT_SOURCE_DIR}/server/src/kytensor_server/src/kytensor_client.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/triton/client/)
INSTALL(FILES ${CMAKE_CURRENT_SOURCE_DIR}/server/src/kytensor_server/src/kytensor_client_common.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/triton/client/)
INSTALL(FILES ${CMAKE_CURRENT_SOURCE_DIR}/server/src/kytensor_server/src/kytensor_client_common.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/triton/client/)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/server/triton-server/kytensor_server/kytensor_service.pb.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/triton/client/)
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/server/triton-server/kytensor_server/model_config.pb.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/triton/client/)
endif() # TRITON_ENABLE_SERVER

# clent
if(${TRITON_ENABLE_CLIENT})

  #
  # Options
  #
  option(TRITON_ENABLE_CC_HTTP "Build C++ HTTP client libraries" OFF)
  option(TRITON_ENABLE_CC_GRPC "Build C++ GRPC client libraries" OFF)
  option(TRITON_ENABLE_PYTHON_HTTP "Enable Python HTTP client libraries" OFF)
  option(TRITON_ENABLE_PYTHON_GRPC "Enable Python GRPC client libraries" OFF)
  option(TRITON_ENABLE_JAVA_HTTP "Enable JAVA HTTP client libraries" OFF)
  option(TRITON_ENABLE_EXAMPLES "Include examples in build" OFF)
  option(TRITON_ENABLE_TESTS "Include tests in build" OFF)
  option(TRITON_ENABLE_ZLIB "Include ZLIB library in build" ON)

  # Package Perf Analyzer with the Python Client -- Intended to be used by PA repo
  option(TRITON_PACKAGE_PERF_ANALYZER "Include Perf Analyzer in pip wheel" OFF)


  if(NOT TRITON_ENABLE_PYTHON_GRPC)
    set(TRITON_COMMON_ENABLE_PROTOBUF_PYTHON OFF)
  endif()

  if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(TRITON_INSTALL_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/cc-clients/install)
  else()
    set(TRITON_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
  endif()

  if(TRITON_ENABLE_CC_HTTP OR TRITON_ENABLE_CC_GRPC)

    set(_cc_client_depends re2)
    if(${TRITON_ENABLE_CC_HTTP})
      set(_cc_client_depends ${_cc_client_depends} curl)
    endif() # TRITON_ENABLE_CC_HTTP
    if(${TRITON_ENABLE_CC_GRPC})
      set(_cc_client_depends ${_cc_client_depends} grpc protobuf)
    endif() # TRITON_ENABLE_CC_GRPC

    ExternalProject_Add(cc-clients
      PREFIX cc-clients
      SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/client/src/c++"
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/client/cc-clients"
      CMAKE_CACHE_ARGS
        ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
        ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
        ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
        -DCURL_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/curl/${LIB_DIR}/cmake/CURL
        -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
        -DgRPC_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/grpc/lib/cmake/grpc
        -Dabsl_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/absl/${LIB_DIR}/cmake/absl
        -Dre2_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/re2/${LIB_DIR}/cmake/re2
        -Dc-ares_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/c-ares/${LIB_DIR}/cmake/c-ares
        -DGTEST_ROOT:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/googletest
        -DTRITON_MODULE_SOURCE_DIR:STRING=${TRITON_MODULE_SOURCE_DIR}
        -DTRITON_REPO_ORGANIZATION:STRING=${TRITON_REPO_ORGANIZATION}
        -DTRITON_COMMON_REPO_TAG:STRING=${TRITON_COMMON_REPO_TAG}
        -DTRITON_CORE_REPO_TAG:STRING=${TRITON_CORE_REPO_TAG}
        -DTRITON_ENABLE_CC_HTTP:BOOL=${TRITON_ENABLE_CC_HTTP}
        -DTRITON_ENABLE_CC_GRPC:BOOL=${TRITON_ENABLE_CC_GRPC}
        -DTRITON_ENABLE_EXAMPLES:BOOL=${TRITON_ENABLE_EXAMPLES}
        -DTRITON_ENABLE_TESTS:BOOL=${TRITON_ENABLE_TESTS}
        -DTRITON_ENABLE_GPU:BOOL=${TRITON_ENABLE_GPU}
        -DTRITON_ENABLE_ZLIB:BOOL=${TRITON_ENABLE_ZLIB}
        -DTRITON_MIN_CXX_STANDARD:STRING=${TRITON_MIN_CXX_STANDARD}
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_INSTALL_PREFIX}/cc-clients
	-DKYTENSOR_DIR:STRING=${KYTENSOR_DIR}
      INSTALL_COMMAND ""
      #DEPENDS ${_cc_client_depends}
    )

    INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/client/cc-clients/library/libhttpclient.so DESTINATION ${CMAKE_INSTALL_LIBDIR})
    INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/client/cc-clients/library/libgrpcclient.so DESTINATION ${CMAKE_INSTALL_LIBDIR})
    INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/server/triton-server/kytensor_server/libkytensorclient.so DESTINATION ${CMAKE_INSTALL_LIBDIR})
    INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/client/cc-clients/library/libhttpclient_static.a DESTINATION ${CMAKE_INSTALL_LIBDIR})
    INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/client/cc-clients/library/libgrpcclient_static.a DESTINATION ${CMAKE_INSTALL_LIBDIR})
    INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/client/cc-clients/library/TritonClientConfig.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TritonClient)
    INSTALL(FILES ${CMAKE_CURRENT_SOURCE_DIR}/client/TritonClientTargets.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/TritonClient)
    INSTALL(FILES ${CMAKE_CURRENT_SOURCE_DIR}/client/src/c++/library/share_tensor.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/triton/client/)
    INSTALL(FILES ${CMAKE_CURRENT_SOURCE_DIR}/client/src/c++/library/grpc_client.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/triton/client/)
    INSTALL(FILES ${CMAKE_CURRENT_SOURCE_DIR}/client/src/c++/library/http_client.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/triton/client/)
    INSTALL(FILES ${CMAKE_CURRENT_SOURCE_DIR}/client/src/c++/library/shm_utils.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/triton/client/)
    INSTALL(FILES ${CMAKE_CURRENT_SOURCE_DIR}/client/src/c++/library/common.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/triton/client/)
    INSTALL(FILES ${CMAKE_CURRENT_SOURCE_DIR}/client/src/c++/library/ipc.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/triton/client/)
    INSTALL(FILES ${CMAKE_CURRENT_SOURCE_DIR}//server/src/kytensor_server/src/kytensor_client.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/triton/client/)
    INSTALL(FILES ${CMAKE_CURRENT_SOURCE_DIR}//server/src/kytensor_server/src/kytensor_client_common.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/triton/client/)
    INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/client/cc-clients/_deps/repo-common-build/protobuf/grpc_service.grpc.pb.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/triton/client/)
    INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/client/cc-clients/_deps/repo-common-build/protobuf/grpc_service.pb.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/triton/client/)
    INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/client/cc-clients/_deps/repo-common-build/protobuf/model_config.pb.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/triton/client/)

    INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/client/cc-clients/tests/cc_client_test DESTINATION /opt/appdata/kylin-ai/libtriton-client-unittest/)
    INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/client/cc-clients/tests/client_timeout_test DESTINATION /opt/appdata/kylin-ai/libtriton-client-unittest/)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/client/unittest_models/ DESTINATION /opt/appdata/kylin-ai/libtriton-client-unittest//unittest_models/)
    install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/client/src/c++/tests/README.md DESTINATION /opt/appdata/kylin-ai/libtriton-client-unittest/)
  endif() # TRITON_ENABLE_CC_HTTP OR TRITON_ENABLE_CC_GRPC

  if(TRITON_ENABLE_PYTHON_HTTP OR TRITON_ENABLE_PYTHON_GRPC)

    ExternalProject_Add(python-clients
      PREFIX python-clients
      SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/client/src/python"
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/client/python-clients"
      CMAKE_CACHE_ARGS
        ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
        ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
        ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
        -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR}
        -DgRPC_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/grpc/lib/cmake/grpc
        -Dabsl_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/absl/${LIB_DIR}/cmake/absl
        -Dc-ares_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/c-ares/${LIB_DIR}/cmake/c-ares
        -Dre2_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/re2/${LIB_DIR}/cmake/re2
        -DTRITON_MODULE_SOURCE_DIR:STRING=${TRITON_MODULE_SOURCE_DIR}
        -DTRITON_REPO_ORGANIZATION:STRING=${TRITON_REPO_ORGANIZATION}
        -DTRITON_COMMON_REPO_TAG:STRING=${TRITON_COMMON_REPO_TAG}
        -DTRITON_CORE_REPO_TAG:STRING=${TRITON_CORE_REPO_TAG}
        -DTRITON_VERSION:STRING=${TRITON_VERSION}
        -DTRITON_ENABLE_PYTHON_HTTP:BOOL=${TRITON_ENABLE_PYTHON_HTTP}
        -DTRITON_ENABLE_PYTHON_GRPC:BOOL=${TRITON_ENABLE_PYTHON_GRPC}
        -DTRITON_ENABLE_EXAMPLES:BOOL=${TRITON_ENABLE_EXAMPLES}
        -DTRITON_ENABLE_TESTS:BOOL=${TRITON_ENABLE_TESTS}
        -DTRITON_PACKAGE_PERF_ANALYZER:BOOL=${TRITON_PACKAGE_PERF_ANALYZER}
        -DTRITON_ENABLE_GPU:BOOL=${TRITON_ENABLE_GPU}
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_INSTALL_PREFIX}/python-clients
    )
  endif() # TRITON_ENABLE_PYTHON_HTTP OR TRITON_ENABLE_PYTHON_GRPC

  if(TRITON_ENABLE_JAVA_HTTP)
    ExternalProject_Add(java-clients
      PREFIX java-clients
      SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/client/src/java"
      BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/client/java-clients"
      CMAKE_CACHE_ARGS
        ${_CMAKE_ARGS_OPENSSL_ROOT_DIR}
        ${_CMAKE_ARGS_CMAKE_TOOLCHAIN_FILE}
        ${_CMAKE_ARGS_VCPKG_TARGET_TRIPLET}
        -DTRITON_REPO_ORGANIZATION:STRING=${TRITON_REPO_ORGANIZATION}
        -DTRITON_VERSION:STRING=${TRITON_VERSION}
        -DTRITON_ENABLE_JAVA_HTTP:BOOL=${TRITON_ENABLE_JAVA_HTTP}
        -DTRITON_ENABLE_EXAMPLES:BOOL=${TRITON_ENABLE_EXAMPLES}
        -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
        -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_INSTALL_PREFIX}/java-clients
      INSTALL_COMMAND ""
    )

  endif() # TRITON_ENABLE_JAVA_HTTP

endif() # TRITON_ENABLE_CLIENT

# onnx backend
if(${TRITON_ENABLE_ONNX_BACKEND})
  ExternalProject_Add(onnxruntimebackend
    PREFIX onnxruntimebackend
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/framework_backends/onnxruntime_backend"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_backend"
    CMAKE_CACHE_ARGS
      -DTRITON_ONNXRUNTIME_INCLUDE_PATHS:PATH=${TRITON_ONNXRUNTIME_INCLUDE_PATHS}
      -DTRITON_ONNXRUNTIME_LIB_PATHS:PATH=${TRITON_ONNXRUNTIME_LIB_PATHS}
      -DTRITON_MODULE_SOURCE_DIR:STRING=${TRITON_MODULE_SOURCE_DIR}
      -DTRITON_MIN_CXX_STANDARD:STRING=${TRITON_MIN_CXX_STANDARD}
      -DTRITON_ENABLE_GPU:BOOL=${TRITON_ENABLE_GPU}
      -DTRITON_ENABLE_NVTX:BOOL=${TRITON_ENABLE_NVTX}
      -DTRITON_ENABLE_STATS:BOOL=${TRITON_ENABLE_STATS}
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_INSTALL_PREFIX}/backends
      -DTRITON_VERSION:STRING=${TRITON_VERSION}
    INSTALL_COMMAND ""
  )

  if(CMAKE_INSTALL_PREFIX STREQUAL "/usr")
	  INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_backend/libtriton_onnxruntime.so DESTINATION ${KYTENSOR_DIR}/backends/onnxruntime)
  else()
    INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_backend/libtriton_onnxruntime.so DESTINATION ${CMAKE_INSTALL_LIBDIR}/backends/onnxruntime)
  endif()
endif() # TRITON_ENABLE_ONNX_BACKEND

# llamacpp backend
if(${TRITON_ENABLE_LLAMACPP_BACKEND})
  ExternalProject_Add(llamacppbackend
    PREFIX llamacppbackend
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/framework_backends/llamacpp_backend"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/llamacpp_backend"
    CMAKE_CACHE_ARGS
      -DTRITON_LLAMACPP_INCLUDE_PATHS:PATH=${TRITON_LLAMACPP_INCLUDE_PATHS}
      -DTRITON_LLAMACPP_LIB_PATHS:PATH=${TRITON_LLAMACPP_LIB_PATHS}
      -DTRITON_ENABLE_METRICS:BOOL=${TRITON_ENABLE_METRICS}
      -DTRITON_MODULE_SOURCE_DIR:STRING=${TRITON_MODULE_SOURCE_DIR}
      -DTRITON_MIN_CXX_STANDARD:STRING=${TRITON_MIN_CXX_STANDARD}
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_INSTALL_PREFIX}/backends
      -DTRITON_VERSION:STRING=${TRITON_VERSION}
    INSTALL_COMMAND ""
  #  DEPENDS libllama
  )

  if(CMAKE_INSTALL_PREFIX STREQUAL "/usr")
	  INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/llamacpp_backend/libtriton_llamacpp.so DESTINATION ${KYTENSOR_DIR}/backends/llamacpp)
  else()
    INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/llamacpp_backend/libtriton_llamacpp.so DESTINATION ${CMAKE_INSTALL_LIBDIR}/backends/llamacpp)
  endif()
endif() # TRITON_ENABLE_LLAMACPP_BACKEND


# python backend
if(${TRITON_ENABLE_PYTHON_BACKEND})
  ExternalProject_Add(pythonbackend
    PREFIX pythonbackend
    SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/framework_backends/python_backend"
    BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/python_backend"
    CMAKE_CACHE_ARGS
      -DCMAKE_SKIP_RPATH:BOOL=ON
      -DCMAKE_SKIP_BUILD_RPATH:BOOL=ON
      -DCMAKE_SKIP_INSTALL_RPATH:BOOL=ON
      -DCMAKE_INSTALL_RPATH_USE_LINK_PATH:BOOL=OFF
      -DBoost_DIR:PATH=${TRITON_THIRD_PARTY_INSTALL_PREFIX}/boost
      -DTRITON_MODULE_SOURCE_DIR:STRING=${TRITON_MODULE_SOURCE_DIR}
      -DTRITON_MIN_CXX_STANDARD:STRING=${TRITON_MIN_CXX_STANDARD}
      -DTRITON_ENABLE_GPU:BOOL=${TRITON_ENABLE_GPU}
      -DTRITON_ENABLE_NVTX:BOOL=${TRITON_ENABLE_NVTX}
      -DTRITON_ENABLE_STATS:BOOL=${TRITON_ENABLE_STATS}
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX:PATH=${TRITON_INSTALL_PREFIX}/backends
      -DTRITON_VERSION:STRING=${TRITON_VERSION}
      -DKYTENSOR_DIR:STRING=${KYTENSOR_DIR}
    INSTALL_COMMAND ""
    DEPENDS ${PYTHONBACKEND_DEPENDS}
)


  if(CMAKE_INSTALL_PREFIX STREQUAL "/usr")
    INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/python_backend/libtriton_python.so DESTINATION ${KYTENSOR_DIR}/backends/python)
    INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/python_backend/kytensor-stub DESTINATION ${KYTENSOR_DIR}/backends/python)
    INSTALL(FILES ${TRITON_MODULE_SOURCE_DIR}/framework_backends/python_backend/src/resources/triton_python_backend_utils.py DESTINATION ${KYTENSOR_DIR}/backends/python)
  else()
    INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/python_backend/libtriton_python.so DESTINATION ${CMAKE_INSTALL_LIBDIR}/backends/python)
    INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/python_backend/kytensor-stub DESTINATION ${CMAKE_INSTALL_LIBDIR}/backends/python)
    INSTALL(FILES ${TRITON_MODULE_SOURCE_DIR}/framework_backends/python_backend/src/resources/triton_python_backend_utils.py DESTINATION ${CMAKE_INSTALL_LIBDIR}/backends/python)
  endif()
endif() # TRITON_ENABLE_PYTHON_BACKEND
