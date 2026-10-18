// tests/test_power_whatif.cpp -- S8.6: power what-ifs by physics have gain exactly 1
#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include "thermal_surrogate/dispatcher.h"
#include "thermal_surrogate/fd_solver.h"
#include "thermal_surrogate/npy.h"

using namespace thermal_surrogate;

namespace {
struct PowerWhatIf : ::testing::Test {
  std::string models = TS_MODELS_DIR, contract = TS_CONTRACT_DIR;
  Validator validator{contract + "/stats.json", models + "/config_bank.npy"};
  Predictor pred{models + "/model.onnx", models + "/metadata.json"};
  std::unique_ptr<Dispatcher> d;
  void SetUp() override {
    std::vector<std::unique_ptr<ISolverBackend>> chain;
    chain.push_back(std::make_unique<SurrogateBackend>(pred));
    chain.push_back(std::make_unique<FdSolverBackend>(load_fd_params(models + "/fd_solver_params.json")));
    d = std::make_unique<Dispatcher>(validator, std::move(chain));
    d->set_conformal(ConformalTable(contract + "/conformal.json"));
  }
};
}  // namespace

TEST_F(PowerWhatIf, ScaledRiseIsExactlyLinear) {
  const auto x = npy::load_f32(std::string(TS_FIXTURE_DIR) + "/sample_input.npy");
  Output base, up;
  d->solve(Input{x.data.data(), kInputShape}, base);
  d->solve_power_scaled(Input{x.data.data(), kInputShape}, 1.1, up);
  ASSERT_TRUE(base.status.ok() && up.status.ok());
  const float amb = x.data[4];
  double worst = 0;
  for (std::size_t i = 0; i < kCells; ++i)
    worst = std::max(worst, std::fabs((up.field[i] - amb) - 1.1 * (base.field[i] - amb)));
  EXPECT_LT(worst, 1e-3);                                   // float32 rounding only
  // Gain = measured hotspot change / physics expectation = 1.
  const double gain = (up.tmax - base.tmax) / (0.1 * (base.tmax - amb));
  EXPECT_NEAR(gain, 1.0, 1e-4);
  EXPECT_EQ(up.tmax_index, base.tmax_index);
  EXPECT_NEAR(up.mae_bound_90_K, 1.1f * base.mae_bound_90_K, 1e-5);
  EXPECT_EQ(up.source, Source::Surrogate);
}

TEST_F(PowerWhatIf, RejectsBadFactorAndKeepsOodBehaviour) {
  const auto x = npy::load_f32(std::string(TS_FIXTURE_DIR) + "/sample_input.npy");
  Output o;
  d->solve_power_scaled(Input{x.data.data(), kInputShape}, 0.0, o);
  EXPECT_EQ(o.status.code, StatusCode::InvalidInput);
  d->solve_power_scaled(Input{x.data.data(), kInputShape}, -1.0, o);
  EXPECT_EQ(o.status.code, StatusCode::InvalidInput);
  const auto x5 = npy::load_f32(std::string(TS_FIXTURE_DIR) + "/sample_input_level5.npy");
  d->solve_power_scaled(Input{x5.data.data(), kInputShape}, 1.2, o);
  EXPECT_TRUE(o.oob_flag);                                  // judged on the base design
  EXPECT_EQ(o.source, Source::ReferenceInProcess);
  EXPECT_EQ(o.mae_bound_90_K, 0.0f);
}
