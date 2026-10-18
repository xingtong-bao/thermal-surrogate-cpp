// thermal_surrogate/model_info.h -- "hello session": what does this ONNX file say about itself?
//
// S4.1 deliverable. Opens the model once, reports the runtime version and
// the input/output names, shapes and element types. Used by `ts_demo --info`
// and by the first test; the real Predictor arrives in S4.2.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace thermal_surrogate {

struct TensorInfo {
  std::string name;
  std::vector<std::int64_t> shape;   // -1 for a dynamic dimension
  std::string element_type;          // e.g. "float32"
};

struct ModelInfo {
  std::string onnxruntime_version;
  std::string path;
  std::vector<TensorInfo> inputs;
  std::vector<TensorInfo> outputs;
};

// Throws std::runtime_error if the file cannot be opened or is not a model.
ModelInfo inspect_model(const std::string& model_path);

}  // namespace thermal_surrogate
