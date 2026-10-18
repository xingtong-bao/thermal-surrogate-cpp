// src/model_info.cpp -- open a model with ONNX Runtime and describe its IO
//
// This is the only S4.1 file that touches ONNX Runtime. It is deliberately
// small: create an Env, open a Session on the file, walk the inputs and
// outputs. Everything the C++ API returns here is RAII-owned (Ort::Env,
// Ort::Session, Ort::AllocatedStringPtr), so there is nothing to free.
#include "thermal_surrogate/model_info.h"

#include <onnxruntime_cxx_api.h>

#include <stdexcept>

namespace thermal_surrogate {
namespace {

// ORT reports element types as an enum; the contract speaks numpy dtypes.
const char* dtype_name(ONNXTensorElementDataType t) {
  switch (t) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return "float32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: return "float64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return "int64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return "int32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "float16";
    default: return "other";
  }
}

TensorInfo describe(const char* name, const Ort::TypeInfo& type_info) {
  TensorInfo info;
  info.name = name;
  auto tensor = type_info.GetTensorTypeAndShapeInfo();
  info.shape = tensor.GetShape();          // dynamic dims come back as -1
  info.element_type = dtype_name(tensor.GetElementType());
  return info;
}

}  // namespace

ModelInfo inspect_model(const std::string& model_path) {
  ModelInfo info;
  info.path = model_path;
  info.onnxruntime_version = Ort::GetVersionString();
  try {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "thermal_surrogate");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);        // inspection only; no need to spin up a pool
    Ort::Session session(env, model_path.c_str(), options);
    Ort::AllocatorWithDefaultOptions allocator;

    const std::size_t n_in = session.GetInputCount();
    for (std::size_t i = 0; i < n_in; ++i) {
      auto name = session.GetInputNameAllocated(i, allocator);
      info.inputs.push_back(describe(name.get(), session.GetInputTypeInfo(i)));
    }
    const std::size_t n_out = session.GetOutputCount();
    for (std::size_t i = 0; i < n_out; ++i) {
      auto name = session.GetOutputNameAllocated(i, allocator);
      info.outputs.push_back(describe(name.get(), session.GetOutputTypeInfo(i)));
    }
  } catch (const Ort::Exception& e) {
    // Translate the runtime's exception type so callers only ever see std::.
    throw std::runtime_error(std::string("ONNX Runtime: ") + e.what());
  }
  return info;
}

}  // namespace thermal_surrogate
