// tests/test_conformal.cpp -- S8.2: the 90 % bounds hold on the held-out subset
//
// Split conformal guarantees coverage >= 0.9 in expectation over calibration
// sets; on a 50-sample test subset the empirical coverage fluctuates, so the
// test asserts >= 0.85 (a 5 % slack) for both scores and that bounds are
// absent on reference answers.
#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include "thermal_surrogate/dispatcher.h"
#include "thermal_surrogate/fd_solver.h"
#include "thermal_surrogate/npy.h"

using namespace thermal_surrogate;

TEST(Conformal, TableLoadsAndIsMonotone) {
  ConformalTable t(std::string(TS_CONTRACT_DIR) + "/conformal.json");
  ASSERT_TRUE(t.loaded());
  float m1, t1, m2, t2;
  t.bounds(20.0, m1, t1);
  t.bounds(150.0, m2, t2);
  EXPECT_GT(m1, 0.0f);
  EXPECT_GE(m2, m1);                       // hotter designs get a wider bound (S2b.2)
  EXPECT_THROW(ConformalTable("/nonexistent.json"), std::runtime_error);
}

TEST(Conformal, CoverageOnEval50) {
  const std::string models = TS_MODELS_DIR, contract = TS_CONTRACT_DIR;
  Validator validator(contract + "/stats.json", models + "/config_bank.npy");
  Predictor pred(models + "/model.onnx", models + "/metadata.json");
  std::vector<std::unique_ptr<ISolverBackend>> chain;
  chain.push_back(std::make_unique<SurrogateBackend>(pred));
  chain.push_back(std::make_unique<FdSolverBackend>(load_fd_params(models + "/fd_solver_params.json")));
  Dispatcher d(validator, std::move(chain));
  d.set_conformal(ConformalTable(contract + "/conformal.json"));

  const std::string dir = std::string(TS_FIXTURE_DIR) + "/eval50/";
  const auto x = npy::load_f32(dir + "level4_inputs.npy");
  const auto y = npy::load_f32(dir + "level4_truth.npy");
  int covered_mae = 0, covered_tmax = 0;
  for (std::size_t i = 0; i < 50; ++i) {
    Output o;
    d.solve(Input{x.data.data() + i * kInputSize, kInputShape}, o);
    ASSERT_TRUE(o.status.ok());
    ASSERT_EQ(o.source, Source::Surrogate);
    ASSERT_GT(o.mae_bound_90_K, 0.0f);
    const float* t = y.data.data() + i * kCells;
    double acc = 0, tmax = t[0];
    for (std::size_t c = 0; c < kCells; ++c) { acc += std::fabs(static_cast<double>(o.field[c]) - t[c]); tmax = std::max(tmax, static_cast<double>(t[c])); }
    if (acc / static_cast<double>(kCells) <= o.mae_bound_90_K) ++covered_mae;
    if (std::fabs(o.tmax - tmax) <= o.tmax_err_bound_90_K) ++covered_tmax;
  }
  RecordProperty("coverage_mae", covered_mae / 50.0);
  RecordProperty("coverage_tmax", covered_tmax / 50.0);
  EXPECT_GE(covered_mae, 43);              // >= 0.85 of 50
  EXPECT_GE(covered_tmax, 43);

  // A reference answer carries no bound.
  const auto x5 = npy::load_f32(std::string(TS_FIXTURE_DIR) + "/sample_input_level5.npy");
  Output o5;
  d.solve(Input{x5.data.data(), kInputShape}, o5);
  EXPECT_EQ(o5.source, Source::ReferenceInProcess);
  EXPECT_EQ(o5.mae_bound_90_K, 0.0f);
}
