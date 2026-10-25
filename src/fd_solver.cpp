// src/fd_solver.cpp -- see fd_solver.h
#include "thermal_surrogate/fd_solver.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace thermal_surrogate {
namespace {

inline double ch(const float* d, std::size_t i, Channel c) { return d[i * kP + static_cast<std::size_t>(c)]; }

// Cell spacing in metres from the coordinate channels (max positive step).
double cell_size_m(const float* d, Channel c) {
  double best = 0.0;
  for (std::size_t x = 0; x < kX; ++x)
    for (std::size_t y = 0; y < kY; ++y) {
      const std::size_t i = x * kY + y;
      if (x + 1 < kX) best = std::max(best, ch(d, i + kY, c) - ch(d, i, c));
      if (y + 1 < kY) best = std::max(best, ch(d, i + 1, c) - ch(d, i, c));
    }
  return best * 1e-3;                                   // mm -> m
}

}  // namespace

// ---- red-black SOR core (grid-generic, S9.1) --------------------------------------
//
// update_cell: the SOR relaxation of one cell from its four neighbours. Every
// neighbour of cell (x, y) has the other colour, so within one half-sweep no
// cell reads a value written by another cell of the same half-sweep. That is
// what makes the row loop below safe to split over threads and what makes the
// result independent of the thread count (same operands, same operations, in
// the same order *per cell*; only the max-reduction order changes, and max is
// exact).
namespace {

inline double update_cell(const SorGrid& g, double omega, std::size_t x, std::size_t y, double* T) {
  const std::size_t ny = g.ny, i = x * ny + y;
  double diag = g.g_sink, rhs = g.q[i] * g.area + g.g_sink * g.t_sink;
  if (x + 1 < g.nx) { diag += g.kx[i]; rhs += g.kx[i] * T[i + ny]; }
  if (x > 0) { diag += g.kx[i - ny]; rhs += g.kx[i - ny] * T[i - ny]; }
  if (y + 1 < ny) { diag += g.ky[i]; rhs += g.ky[i] * T[i + 1]; }
  if (y > 0) { diag += g.ky[i - 1]; rhs += g.ky[i - 1] * T[i - 1]; }
  const double upd = omega * (rhs / diag - T[i]);
  T[i] += upd;
  return std::fabs(upd);
}

// One row of one colour: cells (x, y) with (x + y) % 2 == colour.
inline double update_row(const SorGrid& g, double omega, int colour, std::size_t x, double* T) {
  double m = 0.0;
  for (std::size_t y = (x + static_cast<std::size_t>(colour)) & 1u; y < g.ny; y += 2) m = std::max(m, update_cell(g, omega, x, y, T));
  return m;
}

int resolve_threads(int threads) {
#ifdef _OPENMP
  if (threads == 0) return omp_get_max_threads();
  return threads < 1 ? 1 : threads;
#else
  (void)threads;
  return 1;
#endif
}

double half_sweep(const SorGrid& g, double omega, int colour, int threads, double* T) {
  double max_upd = 0.0;
  const std::size_t nx = g.nx;
#ifdef _OPENMP
  if (threads > 1) {
    // Static schedule: rows cost the same, so contiguous blocks of rows per
    // thread keep each thread on its own stretch of memory.
#pragma omp parallel for schedule(static) num_threads(threads) reduction(max : max_upd)
    for (std::size_t x = 0; x < nx; ++x) max_upd = std::max(max_upd, update_row(g, omega, colour, x, T));
    return max_upd;
  }
#endif
  for (std::size_t x = 0; x < nx; ++x) max_upd = std::max(max_upd, update_row(g, omega, colour, x, T));
  return max_upd;
}

}  // namespace

bool fd_solver_has_openmp() noexcept {
#ifdef _OPENMP
  return true;
#else
  return false;
#endif
}

double sor_sweep(const SorGrid& g, double omega, int threads, std::vector<double>& T) noexcept {
  const int nt = resolve_threads(threads);
  const double red = half_sweep(g, omega, 0, nt, T.data());
  const double black = half_sweep(g, omega, 1, nt, T.data());
  return std::max(red, black);
}

FdSolveStats sor_solve(const SorGrid& g, double omega, double tol, int max_iter, int threads, std::vector<double>& T,
                       SorSchedule sched) noexcept {
  FdSolveStats st;
  const int nt = resolve_threads(threads);
  double max_upd = 0.0;
  int it = 0;
#ifdef _OPENMP
  if (nt > 1) {
    // One parallel region for the whole iteration and exactly two barriers per
    // sweep: the implicit one after the red loop, an explicit one after the
    // black loop (which runs `nowait` so each thread can publish its max first).
    //
    // History, numbers in docs/results/sor_scaling_notes.md:
    //   v1: a `parallel for` per half-sweep -> two fork/joins per sweep;
    //   v2: one region, `single` resetting a shared max + explicit barrier
    //       -> four synchronisation points per sweep, measurably slower than v1;
    //   v3 (this): no shared reduction variable. Each thread keeps its own max
    //       in a cache-line-sized slot; after the barrier every thread reads all
    //       slots and decides alone. All threads see the same values, so they
    //       all stop on the same sweep.
    // Slot reuse needs no extra barrier: a thread rewrites its slot in sweep
    // k+1 only after passing the red barrier of k+1, and every thread passes
    // that barrier only after it finished reading the slots of sweep k.
    struct alignas(128) Slot { double v; };
    std::vector<Slot> slots(static_cast<std::size_t>(nt), Slot{0.0});
    double* Tp = T.data();
    const std::size_t nx = g.nx;
    const bool dyn = sched == SorSchedule::Dynamic;
    int done_it = max_iter;
#pragma omp parallel num_threads(nt) shared(slots, done_it)
    {
      const std::size_t tid = static_cast<std::size_t>(omp_get_thread_num());
      for (int k = 0; k < max_iter; ++k) {
        double m = 0.0;
        if (dyn) {
          // Heterogeneous cores (performance + efficiency): small chunks let
          // fast cores take more rows instead of waiting for the slow ones.
#pragma omp for schedule(dynamic, 4)
          for (std::size_t x = 0; x < nx; ++x) m = std::max(m, update_row(g, omega, 0, x, Tp));
#pragma omp for schedule(dynamic, 4) nowait
          for (std::size_t x = 0; x < nx; ++x) m = std::max(m, update_row(g, omega, 1, x, Tp));
        } else {
          // Static: equal contiguous row blocks, each thread on its own memory.
#pragma omp for schedule(static)
          for (std::size_t x = 0; x < nx; ++x) m = std::max(m, update_row(g, omega, 0, x, Tp));
#pragma omp for schedule(static) nowait
          for (std::size_t x = 0; x < nx; ++x) m = std::max(m, update_row(g, omega, 1, x, Tp));
        }
        slots[tid].v = m;
#pragma omp barrier
        double sweep_max = 0.0;
        for (const Slot& sl : slots) sweep_max = std::max(sweep_max, sl.v);
        if (sweep_max < tol) {
          if (tid == 0) done_it = k;
          break;   // every thread computed the same sweep_max, so all break here
        }
      }
    }
    for (const Slot& sl : slots) max_upd = std::max(max_upd, sl.v);
    it = done_it;
    st.iterations = it;
    st.final_update_K = max_upd;
    st.converged = max_upd < tol;
    return st;
  }
#else
  (void)sched;
#endif
  for (; it < max_iter; ++it) {
    max_upd = sor_sweep(g, omega, 1, T);
    if (max_upd < tol) break;
  }
  st.iterations = it;
  st.final_update_K = max_upd;
  st.converged = max_upd < tol;
  return st;
}

Status fd_solve(const float* d, const FdSolverParams& p, float* T, FdSolveStats* st) noexcept {
  try {
    const double dx = cell_size_m(d, Channel::GridX), dy = cell_size_m(d, Channel::GridY);
    if (!(dx > 0) || !(dy > 0)) return {StatusCode::ReferenceFailed, "cannot infer cell size from grid_x/grid_y"};
    SorGrid g;
    g.nx = kX; g.ny = kY;
    g.area = dx * dy;
    const double t_amb = ch(d, 0, Channel::AmbientK);
    const double r_conv = ch(d, 0, Channel::RConvec_K_per_W);

    // Power density per cell: the channel stores the chiplet's total power on
    // every cell of its footprint, so divide by the footprint area. Footprint =
    // connected component of equal power (same rule as the validator).
    g.q.assign(kCells, 0.0);
    std::vector<int> label(kCells, -1);
    std::vector<std::size_t> stack, comp;
    double p_total = 0.0;
    for (std::size_t s = 0; s < kCells; ++s) {
      const double val = ch(d, s, Channel::ChipletPower);
      if (val <= 0.0 || label[s] >= 0) continue;
      comp.clear(); stack.push_back(s); label[s] = 1;
      while (!stack.empty()) {
        const std::size_t i = stack.back(); stack.pop_back(); comp.push_back(i);
        const std::size_t x = i / kY, y = i % kY;
        const std::size_t nb[4] = {x + 1 < kX ? i + kY : i, x > 0 ? i - kY : i, y + 1 < kY ? i + 1 : i, y > 0 ? i - 1 : i};
        for (std::size_t j : nb)
          if (label[j] < 0 && ch(d, j, Channel::ChipletPower) == val) { label[j] = 1; stack.push_back(j); }
      }
      const double density = val / (static_cast<double>(comp.size()) * g.area);   // W/m^2
      for (std::size_t i : comp) g.q[i] = density;
      p_total += val;
    }
    g.t_sink = t_amb + p_total * (p.alpha * r_conv + p.beta);   // lumped sink term

    // Conductances between neighbours: G = k_harm * t * (edge length / distance).
    // East/west neighbours differ in x (distance dx, edge dy); north/south in y.
    g.kx.assign(kCells, 0.0); g.ky.assign(kCells, 0.0);
    for (std::size_t i = 0; i < kCells; ++i) {
      const std::size_t x = i / kY, y = i % kY;
      const double ki = ch(d, i, Channel::LocalThermalK);
      if (x + 1 < kX) { const double kj = ch(d, i + kY, Channel::LocalThermalK); g.kx[i] = 2.0 * ki * kj / (ki + kj) * p.thickness_m * dy / dx; }
      if (y + 1 < kY) { const double kj = ch(d, i + 1, Channel::LocalThermalK); g.ky[i] = 2.0 * ki * kj / (ki + kj) * p.thickness_m * dx / dy; }
    }
    g.g_sink = p.g_v_w_m2k * g.area;             // W/K per cell into the sink

    std::vector<double> Tv(kCells, g.t_sink + 1.0);
    const FdSolveStats stats = sor_solve(g, p.omega, p.tol, p.max_iter, p.threads, Tv,
                                         p.dynamic_schedule ? SorSchedule::Dynamic : SorSchedule::Static);
    for (std::size_t i = 0; i < kCells; ++i) {
      if (!std::isfinite(Tv[i])) return {StatusCode::ReferenceFailed, "solver diverged (non-finite temperature)"};
      T[i] = static_cast<float>(Tv[i]);
    }
    if (st) *st = stats;
    if (!stats.converged) return {StatusCode::ReferenceFailed, "solver did not converge in " + std::to_string(p.max_iter) + " sweeps"};
    return {StatusCode::Ok, ""};
  } catch (const std::exception& e) {
    return {StatusCode::ReferenceFailed, e.what()};
  }
}

FdSolverParams load_fd_params(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open fd solver params: " + path);
  nlohmann::json j;
  f >> j;
  FdSolverParams p;
  p.thickness_m = j.at("thickness_m").get<double>();
  p.g_v_w_m2k = j.at("g_v_w_m2k").get<double>();
  p.alpha = j.at("alpha").get<double>();
  p.beta = j.at("beta").get<double>();
  p.omega = j.value("omega", p.omega);
  p.tol = j.value("tol_K", p.tol);
  p.max_iter = j.value("max_iter", p.max_iter);
  p.threads = j.value("threads", p.threads);
  p.dynamic_schedule = j.value("dynamic_schedule", p.dynamic_schedule);
  return p;
}

void FdSolverBackend::solve(const Input& in, Output& out) noexcept {
  out = Output{};
  out.source = Source::ReferenceInProcess;
  if (in.data == nullptr || in.shape != kInputShape) { out.status = {StatusCode::InvalidInput, "bad input"}; return; }
  try {
    out.field.resize(kCells);
  } catch (...) { out.status = {StatusCode::ReferenceFailed, "out of memory"}; return; }
  const auto t0 = std::chrono::steady_clock::now();
  FdSolveStats st;
  out.status = fd_solve(in.data, params_, out.field.data(), &st);
  out.latency_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  if (!out.status.ok()) { out.field.clear(); return; }
  std::size_t best = 0;
  for (std::size_t i = 1; i < kCells; ++i) if (out.field[i] > out.field[best]) best = i;
  out.tmax = out.field[best];
  out.tmax_index = {best / kY, best % kY};
}

}  // namespace thermal_surrogate
