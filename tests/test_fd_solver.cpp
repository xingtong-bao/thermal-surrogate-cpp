// tests/test_fd_solver.cpp -- S5.2: the in-process reference converges, is physical, and plugs into the dispatcher;
//                           S9.1: the parallel red-black sweeps give the same bits as the serial ones
#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

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

// ---- S9.1: parallel red-black sweeps -------------------------------------------

namespace {

// A small synthetic grid for the core: uniform conductivity, one hot block.
SorGrid synthetic_grid(std::size_t n) {
  SorGrid g;
  g.nx = g.ny = n;
  const double h = 16e-3 / static_cast<double>(n);
  g.area = h * h;
  g.kx.assign(n * n, 0.0); g.ky.assign(n * n, 0.0); g.q.assign(n * n, 0.0);
  for (std::size_t x = 0; x < n; ++x)
    for (std::size_t y = 0; y < n; ++y) {
      if (x + 1 < n) g.kx[x * n + y] = 0.15;
      if (y + 1 < n) g.ky[x * n + y] = 0.15;
      if (x >= n / 4 && x < n / 2 && y >= n / 4 && y < n / 2) g.q[x * n + y] = 20.0 / (static_cast<double>(n * n / 16) * g.area);
    }
  g.g_sink = 1000.0 * g.area;
  g.t_sink = 320.0;
  return g;
}

// The pre-S9.1 sweep (every cell visited, other colour skipped), kept here as
// the numerical reference for the restructured row loop.
double cell_order_sweep(const SorGrid& g, double omega, std::vector<double>& Tv) {
  const std::size_t nx = g.nx, ny = g.ny;
  double max_upd = 0.0;
  for (int colour = 0; colour < 2; ++colour)
    for (std::size_t i = 0; i < nx * ny; ++i) {
      const std::size_t x = i / ny, y = i % ny;
      if (static_cast<int>((x + y) & 1u) != colour) continue;
      double diag = g.g_sink, rhs = g.q[i] * g.area + g.g_sink * g.t_sink;
      if (x + 1 < nx) { diag += g.kx[i]; rhs += g.kx[i] * Tv[i + ny]; }
      if (x > 0) { diag += g.kx[i - ny]; rhs += g.kx[i - ny] * Tv[i - ny]; }
      if (y + 1 < ny) { diag += g.ky[i]; rhs += g.ky[i] * Tv[i + 1]; }
      if (y > 0) { diag += g.ky[i - 1]; rhs += g.ky[i - 1] * Tv[i - 1]; }
      const double upd = omega * (rhs / diag - Tv[i]);
      Tv[i] += upd;
      max_upd = std::max(max_upd, std::fabs(upd));
    }
  return max_upd;
}

}  // namespace

TEST(FdSolver, RowSweepMatchesTheOriginalCellOrderSweepBitForBit) {
  // Restructuring the loop (rows, parity start) must not change a single bit:
  // within a colour, cells are independent, so visiting order is irrelevant.
  const auto g = synthetic_grid(64);
  std::vector<double> a(g.cells(), 321.0), b(g.cells(), 321.0);
  for (int s = 0; s < 20; ++s) {
    const double ua = cell_order_sweep(g, 1.8, a);
    const double ub = sor_sweep(g, 1.8, 1, b);
    ASSERT_EQ(ua, ub) << "sweep " << s;
  }
  EXPECT_EQ(0, std::memcmp(a.data(), b.data(), a.size() * sizeof(double)));
}

TEST(FdSolver, ParallelSweepsAreBitIdenticalToSerial) {
  // Red-black ordering: every neighbour has the other colour, so threads never
  // read what another thread writes in the same half-sweep. Same iteration
  // count and the same bits for 1, 2, 4, 8 threads (serially if no OpenMP).
  for (std::size_t n : {64u, 256u}) {
    const auto g = synthetic_grid(n);
    std::vector<double> ref(g.cells(), g.t_sink + 1.0);
    const FdSolveStats rs = sor_solve(g, 1.8, 1e-4, 100000, 1, ref);
    ASSERT_TRUE(rs.converged);
    for (int nt : {2, 4, 8}) {
      std::vector<double> T(g.cells(), g.t_sink + 1.0);
      const FdSolveStats st = sor_solve(g, 1.8, 1e-4, 100000, nt, T);
      EXPECT_EQ(st.iterations, rs.iterations) << n << "^2, " << nt << " threads";
      EXPECT_EQ(0, std::memcmp(T.data(), ref.data(), T.size() * sizeof(double))) << n << "^2, " << nt << " threads";
    }
  }
}

TEST(FdSolver, ThreadsParameterDoesNotChangeTheContractAnswer) {
  // The production entry point with threads = 1 and threads = 4 on the real fixture.
  auto p = load_fd_params(kParams);
  const auto x = npy::load_f32(kFix + "sample_input.npy");
  std::vector<float> a(kCells), b(kCells);
  FdSolveStats sa, sb;
  p.threads = 1;
  ASSERT_TRUE(fd_solve(x.data.data(), p, a.data(), &sa).ok());
  p.threads = 4;
  ASSERT_TRUE(fd_solve(x.data.data(), p, b.data(), &sb).ok());
  EXPECT_EQ(sa.iterations, sb.iterations);
  EXPECT_EQ(0, std::memcmp(a.data(), b.data(), a.size() * sizeof(float)));
}
