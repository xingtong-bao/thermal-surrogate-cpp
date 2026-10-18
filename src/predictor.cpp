// src/predictor.cpp -- Predictor implementation (the only file besides model_info.cpp that sees ONNX Runtime)
#include "thermal_surrogate/predictor.h"

#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace thermal_surrogate {

struct Predictor::Impl {
  // Declaration order is destruction order reversed: the session must die
  // before the env, so env is declared first.
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "thermal_surrogate"};
  Ort::SessionOptions options;
  std::unique_ptr<Ort::Session> session;
  Ort::MemoryInfo memory_info{Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)};
  std::string model_path;
  std::string model_version;
  std::string input_name;
  std::string output_name;
  std::array<int64_t, 5> input_dims{1, kX, kY, kZ, kP};     // (B,X,Y,Z,P) as the graph wants it
};

namespace {

// Compare the metadata's declared IO with what the session actually exposes.
// A mismatch means the .onnx and the .json come from different exports.
void check_io(Ort::Session& session, const nlohmann::json& io, std::string& input_name, std::string& output_name) {
  Ort::AllocatorWithDefaultOptions alloc;
  if (session.GetInputCount() != 1 || session.GetOutputCount() != 1)
    throw std::runtime_error("model must have exactly one input and one output");
  input_name = session.GetInputNameAllocated(0, alloc).get();
  output_name = session.GetOutputNameAllocated(0, alloc).get();
  if (input_name != io["input"]["name"].get<std::string>() ||
      output_name != io["output"]["name"].get<std::string>())
    throw std::runtime_error("model IO names do not match metadata.json");
  const auto in_shape = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
  const auto want = io["input"]["shape"].get<std::vector<int64_t>>();
  if (in_shape != want) throw std::runtime_error("model input shape does not match metadata.json");
  if (want.size() != 5 || want[1] != static_cast<int64_t>(kX) || want[2] != static_cast<int64_t>(kY) ||
      want[3] != static_cast<int64_t>(kZ) || want[4] != static_cast<int64_t>(kP))
    throw std::runtime_error("model input shape does not match the contract geometry");
}

}  // namespace

Predictor::Predictor(const std::string& model_path, const std::string& metadata_path, PredictorOptions opts)
    : impl_(std::make_unique<Impl>()) {
  // 1. metadata first: cheap, and it tells us what the model must look like.
  std::ifstream mf(metadata_path);
  if (!mf) throw std::runtime_error("cannot open metadata: " + metadata_path);
  nlohmann::json meta;
  try {
    mf >> meta;
  } catch (const std::exception& e) {
    throw std::runtime_error(std::string("metadata is not valid JSON: ") + e.what());
  }
  impl_->model_version = meta.at("model_version").get<std::string>();
  impl_->model_path = model_path;
  // Size check against the recorded export: catches a truncated download or
  // a swapped file without hashing 20 MB on every construction.
  {
    std::ifstream f(model_path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot open model: " + model_path);
    const auto size = static_cast<std::uint64_t>(f.tellg());
    const auto files = meta.at("files");
    const auto base = model_path.substr(model_path.find_last_of("/\\") + 1);
    if (files.contains(base) && files[base].at("size_bytes").get<std::uint64_t>() != size)
      throw std::runtime_error("model file size differs from metadata.json (different export?)");
  }

  // 2. the session.
  try {
    if (opts.intra_op_threads > 0) impl_->options.SetIntraOpNumThreads(opts.intra_op_threads);
    impl_->options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    impl_->session = std::make_unique<Ort::Session>(impl_->env, model_path.c_str(), impl_->options);
    check_io(*impl_->session, meta.at("io"), impl_->input_name, impl_->output_name);
  } catch (const Ort::Exception& e) {
    throw std::runtime_error(std::string("ONNX Runtime: ") + e.what());
  }
}

Predictor::~Predictor() = default;
Predictor::Predictor(Predictor&&) noexcept = default;
Predictor& Predictor::operator=(Predictor&&) noexcept = default;

const std::string& Predictor::model_version() const { return impl_->model_version; }
const std::string& Predictor::model_path() const { return impl_->model_path; }

void Predictor::predict(const Input& in, Output& out) const noexcept {
  out = Output{};                                   // no stale fields from a previous call
  out.model_version = impl_->model_version;
  out.source = Source::Surrogate;

  // Minimal checks the runtime needs; the contract validator (S4.3) does the rest.
  if (in.data == nullptr) {
    out.status = {StatusCode::InvalidInput, "input data pointer is null"};
    return;
  }
  if (in.shape != kInputShape) {
    out.status = {StatusCode::ShapeMismatch, "input shape is not (64, 64, 1, 7)"};
    return;
  }
  for (std::size_t i = 0; i < kInputSize; ++i) {
    if (!std::isfinite(in.data[i])) {
      out.status = {StatusCode::InvalidInput, "input contains NaN or Inf at flat index " + std::to_string(i)};
      return;
    }
  }

  try {
    const auto t0 = std::chrono::steady_clock::now();
    // Zero-copy input: the tensor wraps the caller's buffer; ORT reads it in place.
    // const_cast is safe because ORT never writes to an input tensor.
    Ort::Value input = Ort::Value::CreateTensor<float>(
        impl_->memory_info, const_cast<float*>(in.data), kInputSize,
        impl_->input_dims.data(), impl_->input_dims.size());
    const char* in_names[] = {impl_->input_name.c_str()};
    const char* out_names[] = {impl_->output_name.c_str()};
    auto results = impl_->session->Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
    const float* y = results[0].GetTensorData<float>();
    const auto n = results[0].GetTensorTypeAndShapeInfo().GetElementCount();
    if (n != kCells) {
      out.status = {StatusCode::InferenceFailed, "unexpected output size " + std::to_string(n)};
      return;
    }
    out.field.assign(y, y + kCells);
    const auto t1 = std::chrono::steady_clock::now();
    out.latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  } catch (const Ort::Exception& e) {
    out.status = {StatusCode::InferenceFailed, std::string("ONNX Runtime: ") + e.what()};
    return;
  } catch (const std::exception& e) {
    out.status = {StatusCode::InferenceFailed, e.what()};
    return;
  }

  // Hotspot: contract O2 says tmax == max(field); the index is informative only.
  std::size_t best = 0;
  for (std::size_t i = 1; i < kCells; ++i)
    if (out.field[i] > out.field[best]) best = i;
  out.tmax = out.field[best];
  out.tmax_index = {best / (kY * kZ), (best / kZ) % kY};     // flat index -> (x, y) for Z = 1
  out.status = {StatusCode::Ok, ""};
}

}  // namespace thermal_surrogate
