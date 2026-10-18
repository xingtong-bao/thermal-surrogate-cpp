// tests/test_predictor.cpp -- S4.2: the C++ predictor reproduces the Python ONNX Runtime output
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <string>

#include "thermal_surrogate/npy.h"
#include "thermal_surrogate/predictor.h"

using namespace thermal_surrogate;

namespace {
const std::string kParityDir = std::string(TS_FIXTURE_DIR) + "/parity/";
const std::string kMetadata = std::string(TS_MODEL_PATH).substr(0, std::string(TS_MODEL_PATH).rfind('/')) + "/metadata.json";

std::string pair_file(const char* tag, int k) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%s_%02d.npy", tag, k);
  return kParityDir + buf;
}
}  // namespace

TEST(Predictor, ConstructsAndReportsVersion) {
  Predictor p(TS_MODEL_PATH, kMetadata);
  EXPECT_EQ(p.model_version(), "level4-unet-1.0.0");
}

TEST(Predictor, MissingFilesThrow) {
  EXPECT_THROW(Predictor("/nonexistent.onnx", kMetadata), std::runtime_error);
  EXPECT_THROW(Predictor(TS_MODEL_PATH, "/nonexistent.json"), std::runtime_error);
}

TEST(Predictor, RejectsBadInputWithoutThrowing) {
  Predictor p(TS_MODEL_PATH, kMetadata);
  Output out;
  p.predict(Input{nullptr, kInputShape}, out);
  EXPECT_EQ(out.status.code, StatusCode::InvalidInput);

  auto a = npy::load_f32(pair_file("input", 0));
  p.predict(Input{a.data.data(), {64, 64, 1, 6}}, out);
  EXPECT_EQ(out.status.code, StatusCode::ShapeMismatch);

  a.data[123] = std::nanf("");
  p.predict(Input{a.data.data(), kInputShape}, out);
  EXPECT_EQ(out.status.code, StatusCode::InvalidInput);
  EXPECT_TRUE(out.field.empty());
}

TEST(Predictor, ParityWithPythonOnnxRuntime) {
  // Gate from contract section 4: ORT(Python) <-> ORT(C++) max |diff| < 1e-4 K.
  // The PyTorch reference is checked too, with its own gate (5e-3 K).
  Predictor p(TS_MODEL_PATH, kMetadata);
  double worst_ort = 0.0, worst_torch = 0.0;
  for (int k = 0; k < 10; ++k) {
    const auto x = npy::load_f32(pair_file("input", k));
    const auto ref = npy::load_f32(pair_file("field", k));
    const auto torch = npy::load_f32(pair_file("torch", k));
    ASSERT_EQ(x.size(), kInputSize);
    ASSERT_EQ(ref.size(), kCells);
    Output out;
    p.predict(Input{x.data.data(), kInputShape}, out);
    ASSERT_TRUE(out.status.ok()) << out.status.message;
    ASSERT_EQ(out.field.size(), kCells);
    for (std::size_t i = 0; i < kCells; ++i) {
      worst_ort = std::max(worst_ort, static_cast<double>(std::fabs(out.field[i] - ref.data[i])));
      worst_torch = std::max(worst_torch, static_cast<double>(std::fabs(out.field[i] - torch.data[i])));
    }
    // O2: tmax is the max of the field, and the index points at it.
    float m = out.field[0];
    for (float v : out.field) m = std::max(m, v);
    EXPECT_EQ(out.tmax, m);
    EXPECT_EQ(out.field[out.tmax_index[0] * kY + out.tmax_index[1]], out.tmax);
    EXPECT_EQ(out.source, Source::Surrogate);
    EXPECT_GT(out.latency_ms, 0.0);
  }
  EXPECT_LT(worst_ort, 1e-4) << "C++ vs Python ONNX Runtime";
  EXPECT_LT(worst_torch, 5e-3) << "C++ vs PyTorch batch-1";
}
