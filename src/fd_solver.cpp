// src/fd_solver.cpp -- see fd_solver.h
#include "thermal_surrogate/fd_solver.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

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

Status fd_solve(const float* d, const FdSolverParams& p, float* T, FdSolveStats* st) noexcept {
  try {
    const double dx = cell_size_m(d, Channel::GridX), dy = cell_size_m(d, Channel::GridY);
    if (!(dx > 0) || !(dy > 0)) return {StatusCode::ReferenceFailed, "cannot infer cell size from grid_x/grid_y"};
    const double area = dx * dy;
    const double t_amb = ch(d, 0, Channel::AmbientK);
    const double r_conv = ch(d, 0, Channel::RConvec_K_per_W);

    // Power density per cell: the channel stores the chiplet's total power on
    // every cell of its footprint, so divide by the footprint area. Footprint =
    // connected component of equal power (same rule as the validator).
    std::vector<double> q(kCells, 0.0);
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
      const double density = val / (static_cast<double>(comp.size()) * area);   // W/m^2
      for (std::size_t i : comp) q[i] = density;
      p_total += val;
    }
    const double t_sink = t_amb + p_total * (p.alpha * r_conv + p.beta);   // lumped sink term

    // Conductances between neighbours: G = k_harm * t * (edge length / distance).
    // East/west neighbours differ in x (distance dx, edge dy); north/south in y.
    std::vector<double> kx(kCells, 0.0), ky(kCells, 0.0);
    for (std::size_t i = 0; i < kCells; ++i) {
      const std::size_t x = i / kY, y = i % kY;
      const double ki = ch(d, i, Channel::LocalThermalK);
      if (x + 1 < kX) { const double kj = ch(d, i + kY, Channel::LocalThermalK); kx[i] = 2.0 * ki * kj / (ki + kj) * p.thickness_m * dy / dx; }
      if (y + 1 < kY) { const double kj = ch(d, i + 1, Channel::LocalThermalK); ky[i] = 2.0 * ki * kj / (ki + kj) * p.thickness_m * dx / dy; }
    }
    const double g_sink = p.g_v_w_m2k * area;             // W/K per cell into the sink

    // SOR, red-black so the sweep is order-independent and vectorisable.
    std::vector<double> Tv(kCells, t_sink + 1.0);
    int it = 0;
    double max_upd = 0.0;
    for (; it < p.max_iter; ++it) {
      max_upd = 0.0;
      for (int colour = 0; colour < 2; ++colour) {
        for (std::size_t i = 0; i < kCells; ++i) {
          const std::size_t x = i / kY, y = i % kY;
          if (static_cast<int>((x + y) & 1u) != colour) continue;
          double diag = g_sink, rhs = q[i] * area + g_sink * t_sink;
          if (x + 1 < kX) { diag += kx[i]; rhs += kx[i] * Tv[i + kY]; }
          if (x > 0) { diag += kx[i - kY]; rhs += kx[i - kY] * Tv[i - kY]; }
          if (y + 1 < kY) { diag += ky[i]; rhs += ky[i] * Tv[i + 1]; }
          if (y > 0) { diag += ky[i - 1]; rhs += ky[i - 1] * Tv[i - 1]; }
          const double t_new = rhs / diag;
          const double upd = p.omega * (t_new - Tv[i]);
          Tv[i] += upd;
          max_upd = std::max(max_upd, std::fabs(upd));
        }
      }
      if (max_upd < p.tol) break;
    }
    for (std::size_t i = 0; i < kCells; ++i) {
      if (!std::isfinite(Tv[i])) return {StatusCode::ReferenceFailed, "solver diverged (non-finite temperature)"};
      T[i] = static_cast<float>(Tv[i]);
    }
    if (st) { st->iterations = it; st->final_update_K = max_upd; st->converged = max_upd < p.tol; }
    if (max_upd >= p.tol) return {StatusCode::ReferenceFailed, "solver did not converge in " + std::to_string(p.max_iter) + " sweeps"};
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
