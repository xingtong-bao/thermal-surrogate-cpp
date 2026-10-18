// thermal_surrogate/predictor.h -- the surrogate path: one ONNX session, RAII, no exceptions after construction
//
// Lifetime: a Predictor owns one Ort::Env/Session/MemoryInfo (hidden behind
// a pimpl so this header has no ONNX Runtime include). Construct it once per
// model (expensive: ~40 ms to load 20 MB and build the kernels), keep it for
// the life of the document/tool, let the destructor release everything.
//
// Threading: predict() is const and may be called from several threads at
// once; ONNX Runtime's Session::Run is thread-safe and every call uses its
// own Output. Construction and destruction are not concurrent-safe with
// predict() on the same object.
//
// Errors: the constructor throws std::runtime_error (missing file, metadata
// mismatch, runtime failure). predict() never throws: every failure is
// reported in Output::status (contract section 6).
#pragma once

#include <memory>
#include <string>

#include "thermal_surrogate/types.h"

namespace thermal_surrogate {

struct PredictorOptions {
  // 0 = let ONNX Runtime pick (all cores). The latency benchmark (S4.4)
  // fixes this explicitly so numbers are reproducible.
  int intra_op_threads = 0;
};

class Predictor {
 public:
  Predictor(const std::string& model_path, const std::string& metadata_path,
            PredictorOptions options = {});
  ~Predictor();
  Predictor(Predictor&&) noexcept;
  Predictor& operator=(Predictor&&) noexcept;
  Predictor(const Predictor&) = delete;
  Predictor& operator=(const Predictor&) = delete;

  // Runs the model on one contract tensor. Checks only what the runtime needs
  // (non-null pointer, exact shape, finite values); the full validator with
  // I4/I5 lives in the dispatcher (S4.3). On success fills field, tmax,
  // tmax_index, latency_ms, source = Surrogate, model_version, status = Ok.
  void predict(const Input& in, Output& out) const noexcept;

  // Zero-copy variant: the model writes straight into `field` (kCells floats
  // owned by the caller) and nothing is allocated per call. Fills the scalar
  // fields of `out` but leaves out.field empty. Same checks as predict().
  void predict_into(const Input& in, float* field, Output& out) const noexcept;

  // Batch: `inputs` is n contiguous contract tensors (n * kInputSize floats),
  // `fields` receives n * kCells floats. Requires a batch-capable model
  // (models/model_dynamic.onnx); on the fixed-shape model returns
  // ShapeMismatch. Per-sample results are numerically the batched kernel's:
  // may differ from predict() by float32 accumulation order (see S3.2).
  Status predict_batch(const float* inputs, std::size_t n, float* fields) const noexcept;

  bool batch_capable() const;                 // true when the loaded graph has a dynamic batch axis

  const std::string& model_version() const;   // models/metadata.json -> model_version
  const std::string& model_path() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace thermal_surrogate
