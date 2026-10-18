# thermal-surrogate-cpp -- top-level build
#
# Targets
#   thermal_surrogate   static library: the public API in include/thermal_surrogate/
#   ts_demo             command-line tool (S4.1: `--info` prints runtime version and model IO)
#   ts_tests            GoogleTest binary, registered with CTest
#
# Build (macOS arm64 / Linux x64):
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#   ctest --test-dir build --output-on-failure
cmake_minimum_required(VERSION 3.24)
project(thermal_surrogate VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(TS_BUILD_TESTS "Build the GoogleTest suite" ON)
option(TS_WARNINGS_AS_ERRORS "Treat warnings as errors" ON)

include(cmake/onnxruntime.cmake)
include(cmake/json.cmake)

# ---- library ------------------------------------------------------------------
add_library(thermal_surrogate STATIC
  src/status.cpp
  src/version.cpp
  src/model_info.cpp
  src/npy.cpp
  src/predictor.cpp
  src/validator.cpp
  src/dispatcher.cpp)
target_include_directories(thermal_surrogate PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
# ORT is a private implementation detail: callers of the library include only
# our headers, which mention no ORT type. It is still exported transitively
# for linking because the static library needs the shared runtime at link time.
target_link_libraries(thermal_surrogate PRIVATE onnxruntime::onnxruntime nlohmann_json::nlohmann_json)
target_compile_definitions(thermal_surrogate PRIVATE
  TS_VERSION_STRING="${PROJECT_VERSION}"
  TS_ORT_VERSION_STRING="${TS_ORT_VERSION}")
target_compile_options(thermal_surrogate PRIVATE
  -Wall -Wextra -Wpedantic -Wshadow -Wconversion
  $<$<BOOL:${TS_WARNINGS_AS_ERRORS}>:-Werror>)

# Paths the demo and tests use to find the model and fixtures without
# copying 20 MB into the build tree.
set(TS_MODEL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/models/model.onnx")
set(TS_FIXTURE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")
set(TS_CONTRACT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/contract")
set(TS_MODELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/models")

# ---- demo ---------------------------------------------------------------------
add_executable(ts_demo apps/ts_demo.cpp)
target_link_libraries(ts_demo PRIVATE thermal_surrogate onnxruntime::onnxruntime nlohmann_json::nlohmann_json)
target_compile_definitions(ts_demo PRIVATE TS_DEFAULT_MODEL_PATH="${TS_MODEL_PATH}" TS_FIXTURE_DIR="${TS_FIXTURE_DIR}")
# The runtime is a shared library inside build/_deps; tell the loader where.
set_target_properties(ts_demo PROPERTIES BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}")

# ---- tests --------------------------------------------------------------------
if(TS_BUILD_TESTS)
  include(cmake/googletest.cmake)
  enable_testing()
  add_executable(ts_tests tests/test_skeleton.cpp tests/test_predictor.cpp tests/test_validator.cpp tests/test_dispatcher.cpp)
  target_link_libraries(ts_tests PRIVATE thermal_surrogate onnxruntime::onnxruntime nlohmann_json::nlohmann_json GTest::gtest_main)
  target_compile_definitions(ts_tests PRIVATE
    TS_MODEL_PATH="${TS_MODEL_PATH}"
    TS_FIXTURE_DIR="${TS_FIXTURE_DIR}"
    TS_CONTRACT_DIR="${TS_CONTRACT_DIR}"
    TS_MODELS_DIR="${TS_MODELS_DIR}")
  set_target_properties(ts_tests PROPERTIES BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}")
  include(GoogleTest)
  gtest_discover_tests(ts_tests)
endif()

# ---- benchmarks -----------------------------------------------------------------
# Separate executables so a CI build can skip them; numbers go to docs/results/.
option(TS_BUILD_BENCH "Build the benchmark executables" ON)
if(TS_BUILD_BENCH)
  add_executable(bench_latency bench/latency.cpp)
  target_link_libraries(bench_latency PRIVATE thermal_surrogate onnxruntime::onnxruntime nlohmann_json::nlohmann_json)
  target_include_directories(bench_latency PRIVATE bench)
  target_compile_definitions(bench_latency PRIVATE TS_MODEL_PATH="${TS_MODEL_PATH}" TS_FIXTURE_DIR="${TS_FIXTURE_DIR}")
  set_target_properties(bench_latency PROPERTIES BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}")

  add_executable(bench_throughput bench/throughput.cpp)
  target_link_libraries(bench_throughput PRIVATE thermal_surrogate onnxruntime::onnxruntime nlohmann_json::nlohmann_json)
  target_include_directories(bench_throughput PRIVATE bench)
  target_compile_definitions(bench_throughput PRIVATE TS_MODELS_DIR="${TS_MODELS_DIR}" TS_FIXTURE_DIR="${TS_FIXTURE_DIR}")
  set_target_properties(bench_throughput PROPERTIES BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}")

  add_executable(bench_dynamic_size bench/dynamic_size.cpp)
  target_link_libraries(bench_dynamic_size PRIVATE thermal_surrogate onnxruntime::onnxruntime)
  target_compile_definitions(bench_dynamic_size PRIVATE TS_MODELS_DIR="${TS_MODELS_DIR}" TS_FIXTURE_DIR="${TS_FIXTURE_DIR}")
  set_target_properties(bench_dynamic_size PROPERTIES BUILD_RPATH "${ONNXRUNTIME_LIB_DIR}")
endif()
