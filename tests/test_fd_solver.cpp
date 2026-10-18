// tests/test_fd_solver.cpp -- S5.2: the in-process reference converges, is physical, and plugs into the dispatcher
#include <gtest/gtest.h>

#include <cmath>
#include <memory>

#include "thermal_surrogate/dispatcher.h"
#include "thermal_surrogate/fd_solver.h"
#include "thermal_surrogate/npy.h"

using namespace thermal_surrogate;

namespace {
const std::string kFix = std::string(TS_FIXTURE_DIR) + "/";
const std::string kParams = std::string(TS_MODELS_DIR) + "/fd_solver_params.json";
}  // namespace

TEST(FdSolver, ConvergesOnTheFixtureAndRespectsPhysics) {
  const auto p = load_fd_params(kParams);
  const auto x = npy::load_f32(kFix + "sample_input.npy");
  std::vector<float> T(kCells);
  FdSolveStats st;
  ASSERT_TRUE(fd_solve(x.data.data(), p, T.data(), &st).ok());
  EXPECT_TRUE(st.converged);
  const float amb = x.data[4];
  float mn = T[0], mx = T[0];
  for (float v : T) { mn = std::min(mn, v); mx = std::max(mx, v); }
  EXPECT_GT(mn, amb);                        // every cell is above ambient (heat only enters)
  EXPECT_LT(mx, amb + 400.0f);               // and nowhere absurd
  // Hottest cell lies on a powered chiplet.
  std::size_t best = 0;
  for (std::size_t i = 1; i < kCells; ++i) if (T[i] > T[best]) best = i;
  EXPECT_GT(x.data[best * kP + 0], 0.0f);
}

TEST(FdSolver, AmbientShiftIsExact) {
  // The model is linear in the ambient: shifting it by 10 K shifts the field by 10 K.
  const auto p = load_fd_params(kParams);
  auto x = npy::load_f32(kFix + "sample_input.npy");
  std::vector<float> a(kCells), b(kCells);
  ASSERT_TRUE(fd_solve(x.data.data(), p, a.data()).ok());
  for (std::size_t i = 0; i < kCells; ++i) x.data[i * kP + 4] += 10.0f;
  ASSERT_TRUE(fd_solve(x.data.data(), p, b.data()).ok());
  double worst = 0;
  for (std::size_t i = 0; i < kCells; ++i) worst = std::max(worst, std::fabs(static_cast<double>(b[i] - a[i]) - 10.0));
  EXPECT_LT(worst, 1e-2);
}

TEST(FdSolver, PowerScalesTheRise) {
  // Unlike the surrogate (S2b.1 gain 0.1), the physics reference must have gain 1:
  // scaling all chiplet powers by 1.1 scales (T - T_sink) by 1.1.
  const auto p = load_fd_params(kParams);
  auto x = npy::load_f32(kFix + "sample_input.npy");
  std::vector<float> a(kCells), b(kCells);
  ASSERT_TRUE(fd_solve(x.data.data(), p, a.data()).ok());
  const double amb = x.data[4], r = x.data[6];
  double ptot = 0;
  for (std::size_t i = 0; i < kCells; ++i) x.data[i * kP + 0] *= 1.1f;
  ASSERT_TRUE(fd_solve(x.data.data(), p, b.data()).ok());
  (void)ptot; (void)r;
  double worst = 0;
  for (std::size_t i = 0; i < kCells; ++i) worst = std::max(worst, std::fabs((b[i] - amb) / (a[i] - amb) - 1.1));
  EXPECT_LT(worst, 2e-2);
}

TEST(FdSolver, ServesAsFallbackForTheUnseenLayout) {
  Validator validator(std::string(TS_CONTRACT_DIR) + "/stats.json", std::string(TS_MODELS_DIR) + "/config_bank.npy");
  Predictor pred(TS_MODEL_PATH, std::string(TS_MODELS_DIR) + "/metadata.json");
  std::vector<std::unique_ptr<ISolverBackend>> v;
  v.push_back(std::make_unique<SurrogateBackend>(pred));
  v.push_back(std::make_unique<FdSolverBackend>(load_fd_params(kParams)));
  Dispatcher d(validator, std::move(v));
  const auto x5 = npy::load_f32(kFix + "sample_input_level5.npy");
  Output out;
  d.solve(Input{x5.data.data(), kInputShape}, out);
  ASSERT_TRUE(out.status.ok()) << out.status.message;
  EXPECT_EQ(out.source, Source::ReferenceInProcess);
  EXPECT_TRUE(out.oob_flag);
  EXPECT_EQ(out.field.size(), kCells);
}
