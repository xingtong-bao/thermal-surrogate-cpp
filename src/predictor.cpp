// src/predictor.cpp -- Predictor implementation (the only file besides model_info.cpp that sees ONNX Runtime)
#include "thermal_surrogate/predictor.h"

#include <onnxruntime_cxx_api.h>

#include <chrono>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <limits>
#include <stdexcept>
#include <vector>

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
  std::array<int64_t, 4> output_dims{1, kX, kY, kZ};
  bool batch_capable = false;                               // batch axis is dynamic in the graph

  // Shared by predict / predict_into / predict_batch: run `n` samples from
  // `in` into the caller's `out` buffer, both bound zero-copy. Throws Ort::Exception.
  void run(const float* in, std::size_t n, float* out) {
    std::array<int64_t, 5> idims = input_dims;
    std::array<int64_t, 4> odims = output_dims;
    idims[0] = static_cast<int64_t>(n);
    odims[0] = static_cast<int64_t>(n);
    Ort::Value input = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(in), n * kInputSize,
                                                       idims.data(), idims.size());
    Ort::Value output = Ort::Value::CreateTensor<float>(memory_info, out, n * kCells, odims.data(), odims.size());
    const char* in_names[] = {input_name.c_str()};
    const char* out_names[] = {output_name.c_str()};
    // Passing a pre-created output Value makes ORT write the result in place
    // instead of allocating its own buffer: that is the zero-copy output path.
    session->Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, &output, 1);
  }
};

namespace {

// Compare the metadata's declared IO with what the session actually exposes.
// A mismatch means the .onnx and the .json come from different exports.
// Returns true when the batch axis is dynamic (-1) in the graph.
bool check_io(Ort::Session& session, const nlohmann::json& io, std::string& input_name, std::string& output_name) {
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
  if (in_shape.size() != want.size()) throw std::runtime_error("model input rank does not match metadata.json");
  // A dynamic axis (-1) in the graph is accepted wherever the metadata has a
  // concrete size; every concrete axis must match exactly.
  for (std::size_t i = 0; i < want.size(); ++i)
    if (in_shape[i] != -1 && in_shape[i] != want[i])
      throw std::runtime_error("model input shape does not match metadata.json");
  if (want.size() != 5 || want[1] != static_cast<int64_t>(kX) || want[2] != static_cast<int64_t>(kY) ||
      want[3] != static_cast<int64_t>(kZ) || want[4] != static_cast<int64_t>(kP))
    throw std::runtime_error("model input shape does not match the contract geometry");
  if (in_shape[3] != static_cast<int64_t>(kZ) || in_shape[4] != static_cast<int64_t>(kP))
    throw std::runtime_error("model Z/P axes must be fixed to the contract values");
  return in_shape[0] == -1;
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
    impl_->batch_capable = check_io(*impl_->session, meta.at("io"), impl_->input_name, impl_->output_name);
  } catch (const Ort::Exception& e) {
    throw std::runtime_error(std::string("ONNX Runtime: ") + e.what());
  }
}

Predictor::~Predictor() = default;
Predictor::Predictor(Predictor&&) noexcept = default;
Predictor& Predictor::operator=(Predictor&&) noexcept = default;

const std::string& Predictor::model_version() const { return impl_->model_version; }
const std::string& Predictor::model_path() const { return impl_->model_path; }

bool Predictor::batch_capable() const { return impl_->batch_capable; }

namespace {

// Checks shared by the single-sample entry points. Returns false and sets
// out.status when the input must not reach the runtime.
bool precheck(const Input& in, Output& out) {
  if (in.data == nullptr) {
    out.status = {StatusCode::InvalidInput, "input data pointer is null"};
    return false;
  }
  if (in.shape != kInputShape) {
    out.status = {StatusCode::ShapeMismatch, "input shape is not (64, 64, 1, 7)"};
    return false;
  }
  for (std::size_t i = 0; i < kInputSize; ++i) {
    if (!std::isfinite(in.data[i])) {
      out.status = {StatusCode::InvalidInput, "input contains NaN or Inf at flat index " + std::to_string(i)};
      return false;
    }
  }
  return true;
}

// Hotspot: contract O2 says tmax == max(field); the index is informative only.
void fill_hotspot(const float* field, Output& out) {
  std::size_t best = 0;
  for (std::size_t i = 1; i < kCells; ++i)
    if (field[i] > field[best]) best = i;
  out.tmax = field[best];
  out.tmax_index = {best / (kY * kZ), (best / kZ) % kY};     // flat index -> (x, y) for Z = 1
}

}  // namespace

void Predictor::predict_into(const Input& in, float* field, Output& out) const noexcept {
  out = Output{};                                   // no stale fields from a previous call
  out.model_version = impl_->model_version;
  out.source = Source::Surrogate;
  if (field == nullptr) { out.status = {StatusCode::InvalidInput, "output buffer is null"}; return; }
  if (!precheck(in, out)) return;
  try {
    const auto t0 = std::chrono::steady_clock::now();
    impl_->run(in.data, 1, field);
    out.latency_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  } catch (const Ort::Exception& e) {
    out.status = {StatusCode::InferenceFailed, std::string("ONNX Runtime: ") + e.what()};
    return;
  } catch (const std::exception& e) {
    out.status = {StatusCode::InferenceFailed, e.what()};
    return;
  }
  fill_hotspot(field, out);
  out.status = {StatusCode::Ok, ""};
}

void Predictor::predict(const Input& in, Output& out) const noexcept {
  // The owning variant: allocate the field, then run the zero-copy path into it.
  std::vector<float> field;
  try {
    field.resize(kCells);
  } catch (...) {
    out = Output{};
    out.status = {StatusCode::InferenceFailed, "out of memory"};
    return;
  }
  predict_into(in, field.data(), out);
  if (out.status.ok()) out.field = std::move(field);
}

Status Predictor::predict_batch(const float* inputs, std::size_t n, float* fields) const noexcept {
  if (inputs == nullptr || fields == nullptr) return {StatusCode::InvalidInput, "null buffer"};
  if (n == 0) return {StatusCode::InvalidInput, "empty batch"};
  if (!impl_->batch_capable && n != 1)
    return {StatusCode::ShapeMismatch, "loaded model has a fixed batch of 1; use models/model_dynamic.onnx"};
  for (std::size_t i = 0; i < n * kInputSize; ++i)
    if (!std::isfinite(inputs[i])) return {StatusCode::InvalidInput, "NaN or Inf at flat index " + std::to_string(i)};
  try {
    impl_->run(inputs, n, fields);
  } catch (const Ort::Exception& e) {
    return {StatusCode::InferenceFailed, std::string("ONNX Runtime: ") + e.what()};
  } catch (const std::exception& e) {
    return {StatusCode::InferenceFailed, e.what()};
  }
  return {StatusCode::Ok, ""};
}

}  // namespace thermal_surrogate
