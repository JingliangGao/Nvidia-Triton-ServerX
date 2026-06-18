cmake_minimum_required(VERSION 3.16)

project(tritonllamacppbackend LANGUAGES C CXX)

#
# Options
#
# Must include options required for this project as well as any
# projects included in this one by FetchContent.
#
# GPU support is disabled by default because recommended backend
# doesn't use GPUs.
#
option(TRITON_ENABLE_GPU "Enable GPU support in backend" OFF)
option(TRITON_ENABLE_STATS "Include statistics collections in backend" ON)
option(TRITON_ENABLE_METRICS "Include metrics support in server" OFF)

set(TRITON_LLAMACPP_INCLUDE_PATHS "" CACHE PATH "Paths to llamacpp includes")
set(TRITON_LLAMACPP_LIB_PATHS "" CACHE PATH "Paths to llamacpp libraries")
set(TRT_VERSION "" CACHE STRING "TRT version for this build.")

set(TRITON_REPO_ORGANIZATION "https://github.com/triton-inference-server" CACHE STRING "Git repository to pull from")
set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo")

#
# Setting C++ min standard
#
set(TRITON_MIN_CXX_STANDARD 11 CACHE STRING "The minimum C++ standard whose features are requested to build this target.")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

#
# Dependencies
#
# FetchContent requires us to include the transitive closure of all
# repos that we depend on so that we can override the tags.
#
include(FetchContent)

FetchContent_Declare(
  repo-common
  SOURCE_DIR ${TRITON_MODULE_SOURCE_DIR}/common
)
FetchContent_Declare(
  repo-core
  SOURCE_DIR ${TRITON_MODULE_SOURCE_DIR}/core
)
FetchContent_Declare(
  repo-backend
  SOURCE_DIR ${TRITON_MODULE_SOURCE_DIR}/backend
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

message(STATUS "TRITON_LLAMACPP_LIB_PATHS ${TRITON_LLAMACPP_LIB_PATHS}")
if(TRITON_LLAMACPP_LIB_PATHS STREQUAL "")
  find_package(llamacpp CONFIG REQUIRED)
  message(STATUS "Using llama ${LLAMA_VERSION}")
  message(STATUS "${LLAMA_INCLUDE_DIR}")
  message(STATUS "${LLAMA_LIBRARY}")
  find_library(LLAMACPP_LIBRARY NAMES llama PATHS ${llama_LIBRARY})
  find_library(LLAMACPP_COMMON_LIBRARY NAMES common PATHS ${llama_LIBRARY})

else()
  find_library(LLAMACPP_LIBRARY NAMES llama PATHS ${TRITON_LLAMACPP_LIB_PATHS})
  find_library(LLAMACPP_GGML_LIBRARY NAMES ggml PATHS ${TRITON_LLAMACPP_LIB_PATHS})
  find_library(LLAMACPP_COMMON_LIBRARY NAMES common PATHS ${TRITON_LLAMACPP_LIB_PATHS})
endif()

#
# The backend must be built into a shared library. Use an ldscript to
# hide all symbols except for the TRITONBACKEND API.
#
configure_file(src/libtriton_llamacpp.ldscript libtriton_llamacpp.ldscript COPYONLY)

set(LLAMACPP_BACKEND_SOURCE
    src/libllamacpp.cc
    src/llamacpp_engine.cc
    src/llamacpp_queue.cc
    src/llamacpp_response.cc
    src/llamacpp_task.cc
    src/llamacpp_worker.cc
    src/model_instance_state.cc
    src/model_state.cc
    src/utils.cc
)

if(TRITON_ENABLE_METRICS)
  list(APPEND LLAMACPP_BACKEND_SOURCE src/llamacpp_metrics.cc)
endif()

add_library(
  triton-llamacpp-backend SHARED
  ${LLAMACPP_BACKEND_SOURCE}
)

add_library(
  TritonLlamaCppBackend::triton-llamacpp-backend ALIAS triton-llamacpp-backend
)

target_include_directories(
  triton-llamacpp-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TRITON_LLAMACPP_INCLUDE_PATHS}
)

target_compile_features(triton-llamacpp-backend PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})
target_compile_options(
  triton-llamacpp-backend PRIVATE
  $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -Wno-type-limits -Werror>
  $<$<CXX_COMPILER_ID:MSVC>:/Wall /D_WIN32_WINNT=0x0A00 /EHsc /Zc:preprocessor>
)

if(WIN32)
  set_target_properties(
    triton-llamacpp-backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_llamacpp
  )
else()
  set_target_properties(
    triton-llamacpp-backend PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME triton_llamacpp
    LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_llamacpp.ldscript
    LINK_FLAGS "-Wl,--version-script libtriton_llamacpp.ldscript"
  )
endif()

target_link_libraries(
  triton-llamacpp-backend
  PRIVATE
    triton-core-serverstub   # from repo-core
    triton-backend-utils     # from repo-backend
    ${LLAMACPP_LIBRARY}
    ${LLAMACPP_GGML_LIBRARY}
    ${LLAMACPP_COMMON_LIBRARY}
)

option(USE_GCOVR "Use gcovr for coverage" OFF)
if(USE_GCOVR)
    set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} -fprofile-arcs -ftest-coverage")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lgcov --coverage")
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    find_package(Catch2 3 REQUIRED)
    find_package(TritonClient REQUIRED)

    add_executable(llm_unit_test tests/llm_unit_test.cc)
    target_link_libraries(llm_unit_test PRIVATE httpclient grpcclient pthread Catch2::Catch2WithMain)

    # 添加测试
  enable_testing()
  add_test(NAME LLMTest COMMAND llm_unit_test)
endif()

# 覆盖率目标
if(USE_GCOVR)
add_custom_target(coverage
     COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/coverage-report
     #COMMAND llm_unit_test
     COMMAND gcovr -r ${CMAKE_SOURCE_DIR} --xml -o ${CMAKE_BINARY_DIR}/coverage-report/coverage.xml
     COMMAND gcovr -r ${CMAKE_SOURCE_DIR} --html --html-details -o ${CMAKE_BINARY_DIR}/coverage-report/coverage.html
     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
     COMMENT "Generating coverage report"
)
endif()

#
# Install
#
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/libtriton_llamacpp.so DESTINATION /usr/lib/kytensor/backends/llamacpp)
