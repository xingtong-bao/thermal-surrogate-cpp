# cmake/onnxruntime.cmake -- ONNX Runtime as a prebuilt, hash-pinned dependency
#
# Why prebuilt: building ORT from source takes ~1 h and needs its own toolchain;
# Microsoft publishes signed binaries per platform. We pin the version (the same
# one the Python export used, see models/metadata.json) and the SHA-256 of the
# archive, so a build on another machine either gets byte-identical bits or
# fails loudly. The archive lands in build/_deps (gitignored); nothing is
# installed system-wide.
#
# Result: an IMPORTED target `onnxruntime::onnxruntime` with include dirs and
# the shared library, plus ONNXRUNTIME_LIB_DIR for RPATH handling.
include(FetchContent)

set(TS_ORT_VERSION "1.23.2" CACHE STRING "ONNX Runtime version (must match models/metadata.json)")

if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
  set(_ort_pkg "onnxruntime-osx-arm64-${TS_ORT_VERSION}")
  set(_ort_sha "b4d513ab2b26f088c66891dbbc1408166708773d7cc4163de7bdca0e9bbb7856")
  set(_ort_lib "libonnxruntime.${TS_ORT_VERSION}.dylib")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux" AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  set(_ort_pkg "onnxruntime-linux-x64-${TS_ORT_VERSION}")
  set(_ort_sha "1fa4dcaef22f6f7d5cd81b28c2800414350c10116f5fdd46a2160082551c5f9b")
  set(_ort_lib "libonnxruntime.so.${TS_ORT_VERSION}")
else()
  message(FATAL_ERROR "No prebuilt ONNX Runtime configured for ${CMAKE_SYSTEM_NAME}/${CMAKE_SYSTEM_PROCESSOR}; "
                      "add a branch here or set ONNXRUNTIME_ROOT manually.")
endif()

FetchContent_Declare(onnxruntime_prebuilt
  URL      "https://github.com/microsoft/onnxruntime/releases/download/v${TS_ORT_VERSION}/${_ort_pkg}.tgz"
  URL_HASH "SHA256=${_ort_sha}"
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
FetchContent_MakeAvailable(onnxruntime_prebuilt)

set(ONNXRUNTIME_ROOT "${onnxruntime_prebuilt_SOURCE_DIR}")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_ROOT}/lib")

add_library(onnxruntime::onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime::onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ONNXRUNTIME_LIB_DIR}/${_ort_lib}"
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include")
message(STATUS "ONNX Runtime ${TS_ORT_VERSION} (prebuilt): ${ONNXRUNTIME_ROOT}")
