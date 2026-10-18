// tests/test_accuracy_gate.cpp -- S6.1: the contract's accuracy gates, enforced by the build
//
// Runs the surrogate on the committed 50-sample level4 test subset
// (tests/fixtures/eval50, fixed global indices in manifest.json) and asserts
// the three gates of contract section 4. The gates are read from
// docs/results/baseline_unet_level4_per_sample.json (where S2.2 wrote them),
// so a regenerated baseline moves the test automatically and a hand-edited
// gate is impossible.
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>

#include "thermal_surrogate/npy.h"
#include "thermal_surrogate/predictor.h"
#include "thermal_surrogate/types.h"

using namespace thermal_surrogate;

namespace {
double percentile(std::vector<double> v, double p) {
  std::sort(v.begin(), v.end());
  return v[std::min(v.size() - 1, static_cast<std::size_t>(p * static_cast<double>(v.size())))];
}
}  // namespace

TEST(AccuracyGate, Level4SubsetMeetsContract) {
  nlohmann::json gates_doc;
  std::ifstream(std::string(TS_REPO_DIR) + "/docs/results/baseline_unet_level4_per_sample.json") >> gates_doc;
  const auto gates = gates_doc["gates"];
  const double gate_mae = gates["field_mae_K"], gate_p95 = gates["worst5pct_per_sample_mae_K"], gate_tmax = gates["tmax_abs_err_median_K"];

  const std::string dir = std::string(TS_FIXTURE_DIR) + "/eval50/";
  const auto x = npy::load_f32(dir + "level4_inputs.npy");
  const auto y = npy::load_f32(dir + "level4_truth.npy");
  ASSERT_EQ(x.shape[0], 50u);
  const std::string models = std::string(TS_MODEL_PATH).substr(0, std::string(TS_MODEL_PATH).rfind('/'));
  Predictor pred(TS_MODEL_PATH, models + "/metadata.json");

  std::vector<double> mae, tmax_err;
  for (std::size_t i = 0; i < 50; ++i) {
    Output o;
    pred.predict(Input{x.data.data() + i * kInputSize, kInputShape}, o);
    ASSERT_TRUE(o.status.ok()) << o.status.message;
    const float* t = y.data.data() + i * kCells;
    double acc = 0, tmax = t[0];
    for (std::size_t c = 0; c < kCells; ++c) { acc += std::fabs(static_cast<double>(o.field[c]) - t[c]); tmax = std::max(tmax, static_cast<double>(t[c])); }
    mae.push_back(acc / static_cast<double>(kCells));
    tmax_err.push_back(std::fabs(o.tmax - tmax));
  }
  double mean = 0;
  for (double m : mae) mean += m;
  mean /= 50.0;
  RecordProperty("mae_K", mean);
  EXPECT_LT(mean, gate_mae) << "field MAE gate";
  EXPECT_LT(percentile(mae, 0.95), gate_p95) << "worst-5% per-sample MAE gate";
  EXPECT_LT(percentile(tmax_err, 0.50), gate_tmax) << "Tmax median error gate";
}

TEST(AccuracyGate, StatusSpellingsMatchSchema) {
  // contract/schema.json is the single source of the status vocabulary.
  nlohmann::json schema;
  std::ifstream(std::string(TS_CONTRACT_DIR) + "/schema.json") >> schema;
  const auto codes = schema["$defs"]["status"]["properties"]["code"]["enum"].get<std::vector<std::string>>();
  for (int c = 0; c <= static_cast<int>(StatusCode::Timeout); ++c)
    EXPECT_NE(std::find(codes.begin(), codes.end(), to_string(static_cast<StatusCode>(c))), codes.end()) << "code " << c;
  EXPECT_EQ(codes.size(), static_cast<std::size_t>(StatusCode::Timeout) + 1);
}
