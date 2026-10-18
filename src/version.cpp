// src/version.cpp -- library and runtime version strings
//
// TS_VERSION_STRING comes from CMake's project() version; TS_ORT_VERSION_STRING
// is the pinned ONNX Runtime version from cmake/onnxruntime.cmake. The runtime
// also reports its version at run time (Ort::GetVersionString); ts_demo --info
// prints both so a mismatch between "built against" and "loaded" is visible.
#include "thermal_surrogate/types.h"

namespace thermal_surrogate {

Version version() { return Version{TS_VERSION_STRING, TS_ORT_VERSION_STRING}; }

}  // namespace thermal_surrogate
