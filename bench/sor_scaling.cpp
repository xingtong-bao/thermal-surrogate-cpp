// bench/sor_scaling.cpp -- S9.1: thread scaling of the red-black SOR solver
//
// Question answered: how much faster does the in-process reference solver get
// with 2 / 4 / 8 OpenMP threads, and why not N times?
//
// Protocol
//   * synthetic chip at n x n cells for n in {64, 128, 256, 512}: a 16 mm die
//     (cell size shrinks with n, so the physics is the same and only the grid
//     gets finer), uniform conductivity, four powered blocks, lumped sink;
//   * the SAME sor_sweep() that fd_solve() uses in production (src/fd_solver.cpp);
//   * a fixed number of sweeps is timed (not "until converged"), so every
//     thread count does exactly the same arithmetic; convergence is checked
//     separately and must give the same iteration count for every thread
//     count, and the fields must be bit-identical;
//   * `repeat` timed runs per (size, threads), median and min reported;
//   * speedup = t(1 thread) / t(N threads); efficiency = speedup / N;
//   * static and dynamic row scheduling both measured (performance + efficiency
//     cores make the static split wait for the slowest core);
//   * the pre-S9.1 loop ("legacy": scan every cell twice, skip by parity) is
//     timed as well so the gain from restructuring the loop is not confused
//     with the gain from threads.
//
//   * one parallel region per solve (sor_solve) vs two per sweep (sor_sweep)
//     timed side by side;
//
//   bench_sor_scaling [--sizes 64,128,256,512] [--threads 1,2,4,8] [--sweeps 200]
//                     [--repeat 11] [--converge 1] [--out docs/results/sor_scaling]
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "machine_info.h"
#include "thermal_surrogate/fd_solver.h"

namespace ts = thermal_surrogate;

namespace {

std::vector<int> parse_list(const std::string& s) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) out.push_back(std::stoi(tok));
  return out;
}

// A chip that looks like the dataset's: 16 mm square die, k = 150 W/m/K
// (silicon-ish), t = 1 mm effective thickness, g_v = 1000 W/m^2/K into a
// 320 K sink, four 20 W blocks. Same construction as fd_solve() but on n x n.
ts::SorGrid make_chip(std::size_t n) {
  ts::SorGrid g;
  g.nx = g.ny = n;
  const double die_m = 16e-3, h = die_m / static_cast<double>(n);
  g.area = h * h;
  const double k = 150.0, t = 1e-3;
  g.kx.assign(n * n, 0.0);
  g.ky.assign(n * n, 0.0);
  g.q.assign(n * n, 0.0);
  for (std::size_t x = 0; x < n; ++x)
    for (std::size_t y = 0; y < n; ++y) {
      const std::size_t i = x * n + y;
      if (x + 1 < n) g.kx[i] = k * t;        // k_harm * t * (h / h)
      if (y + 1 < n) g.ky[i] = k * t;
    }
  // four powered blocks, each 20 W over a quarter-die-sized footprint
  const double block_w = 20.0;
  const std::size_t b = n / 4, gap = n / 8;
  const std::size_t origins[4][2] = {{gap, gap}, {n - gap - b, gap}, {gap, n - gap - b}, {n - gap - b, n - gap - b}};
  for (const auto& o : origins) {
    const double density = block_w / (static_cast<double>(b * b) * g.area);
    for (std::size_t x = o[0]; x < o[0] + b; ++x)
      for (std::size_t y = o[1]; y < o[1] + b; ++y) g.q[x * n + y] = density;
  }
  g.g_sink = 1000.0 * g.area;
  g.t_sink = 320.0;
  return g;
}

// The loop as it was before S9.1 (src/fd_solver.cpp at tag v0.3.1): for each
// colour, visit every cell and skip the other colour. Same arithmetic per cell.
double legacy_sweep(const ts::SorGrid& g, double omega, std::vector<double>& Tv) {
  const std::size_t nx = g.nx, ny = g.ny, cells = nx * ny;
  double max_upd = 0.0;
  for (int colour = 0; colour < 2; ++colour) {
    for (std::size_t i = 0; i < cells; ++i) {
      const std::size_t x = i / ny, y = i % ny;
      if (static_cast<int>((x + y) & 1u) != colour) continue;
      double diag = g.g_sink, rhs = g.q[i] * g.area + g.g_sink * g.t_sink;
      if (x + 1 < nx) { diag += g.kx[i]; rhs += g.kx[i] * Tv[i + ny]; }
      if (x > 0) { diag += g.kx[i - ny]; rhs += g.kx[i - ny] * Tv[i - ny]; }
      if (y + 1 < ny) { diag += g.ky[i]; rhs += g.ky[i] * Tv[i + 1]; }
      if (y > 0) { diag += g.ky[i - 1]; rhs += g.ky[i - 1] * Tv[i - 1]; }
      const double t_new = rhs / diag;
      const double upd = omega * (t_new - Tv[i]);
      Tv[i] += upd;
      max_upd = std::max(max_upd, std::fabs(upd));
    }
  }
  return max_upd;
}

double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

}  // namespace

int main(int argc, char** argv) {
  std::vector<int> sizes{64, 128, 256, 512}, threads{1, 2, 4, 8};
  int sweeps = 200, repeat = 11, converge = 1;
  std::string out = "sor_scaling";
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string k = argv[i], v = argv[i + 1];
    if (k == "--sizes") sizes = parse_list(v);
    else if (k == "--threads") threads = parse_list(v);
    else if (k == "--sweeps") sweeps = std::stoi(v);
    else if (k == "--repeat") repeat = std::stoi(v);
    else if (k == "--converge") converge = std::stoi(v);   // 0: skip the solve-to-tolerance runs (quick A/B)
    else if (k == "--out") out = v;
    else { std::cerr << "unknown option " << k << "\n"; return 2; }
  }
  const double omega = 1.8;
  if (!ts::fd_solver_has_openmp()) std::cerr << "warning: built without OpenMP, threads > 1 run serially\n";

  nlohmann::json j;
  j["machine"] = bench::machine_info();
  j["openmp"] = ts::fd_solver_has_openmp();
  j["sweeps_timed"] = sweeps;
  j["repeat"] = repeat;
  j["omega"] = omega;
  std::ofstream csv(out + ".csv");
  csv << "grid,cells,threads,schedule,sweeps,ms_median,ms_min,legacy_ms_median,region_per_halfsweep_ms_median,speedup_vs_1thread,efficiency,"
         "converge_iters,converged_ms_median,bit_identical_to_1thread\n";
  std::cout << "grid     threads schedule   ms(" << sweeps << " sweeps)   legacy ms   2regions/sweep   speedup   eff    converge iters   identical\n";
  // static for every thread count, dynamic for the multi-thread ones (the only case where it can matter)
  std::vector<std::pair<int, ts::SorSchedule>> configs;
  for (int nt : threads) configs.emplace_back(nt, ts::SorSchedule::Static);
  for (int nt : threads) if (nt > 1) configs.emplace_back(nt, ts::SorSchedule::Dynamic);

  for (int n : sizes) {
    const auto g = make_chip(static_cast<std::size_t>(n));
    const std::vector<double> T0(g.cells(), g.t_sink + 1.0);
    // Reference: 1-thread solution to convergence, for the bit-identity check.
    std::vector<double> ref = T0;
    const ts::FdSolveStats ref_st = converge ? ts::sor_solve(g, omega, 1e-4, 200000, 1, ref) : ts::FdSolveStats{};
    // Legacy loop timing (serial only).
    std::vector<double> legacy_ms;
    for (int r = 0; r < repeat; ++r) {
      std::vector<double> T = T0;
      const auto t0 = std::chrono::steady_clock::now();
      for (int s = 0; s < sweeps; ++s) legacy_sweep(g, omega, T);
      legacy_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    const double legacy_med = median(legacy_ms);
    double t1 = 0.0;
    for (const auto& cfg : configs) {
      const int nt = cfg.first;
      const ts::SorSchedule sched = cfg.second;
      const char* sched_name = sched == ts::SorSchedule::Dynamic ? "dynamic" : "static";
      std::vector<double> ms;
      for (int r = 0; r < repeat; ++r) {
        std::vector<double> T = T0;
        const auto t0 = std::chrono::steady_clock::now();
        ts::sor_solve(g, omega, 0.0, sweeps, nt, T, sched);   // tol 0: exactly `sweeps` sweeps, one parallel region
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
      }
      const double med = median(ms), mn = *std::min_element(ms.begin(), ms.end());
      if (nt == 1 && sched == ts::SorSchedule::Static) t1 = med;
      // Same sweeps through sor_sweep(): two parallel regions per sweep instead of
      // one region per solve. Static schedule only (sor_sweep has no schedule knob).
      double per_sweep_med = 0.0;
      if (nt > 1 && sched == ts::SorSchedule::Static) {
        std::vector<double> ms2;
        for (int r = 0; r < repeat; ++r) {
          std::vector<double> T = T0;
          const auto t0 = std::chrono::steady_clock::now();
          for (int sw = 0; sw < sweeps; ++sw) ts::sor_sweep(g, omega, nt, T);
          ms2.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
        }
        per_sweep_med = median(ms2);
      }
      // Convergence with this thread count: same iterations, bit-identical field?
      std::vector<double> T = T0;
      const auto c0 = std::chrono::steady_clock::now();
      const ts::FdSolveStats st = converge ? ts::sor_solve(g, omega, 1e-4, 200000, nt, T, sched) : ref_st;
      const double conv_ms = converge ? std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count() : 0.0;
      const bool identical = !converge || (st.iterations == ref_st.iterations && std::memcmp(T.data(), ref.data(), T.size() * sizeof(double)) == 0);
      const double speedup = t1 / med, eff = speedup / static_cast<double>(nt);
      csv << n << "," << g.cells() << "," << nt << "," << sched_name << "," << sweeps << "," << med << "," << mn << "," << legacy_med << ","
          << per_sweep_med << "," << speedup << "," << eff << "," << st.iterations << "," << conv_ms << "," << (identical ? 1 : 0) << "\n";
      std::printf("%4d^2   %4d %-8s %10.2f      %9.2f   %10.2f   %6.2fx  %5.2f   %7d (%.0f ms)   %s\n", n, nt, sched_name, med, legacy_med,
                  per_sweep_med, speedup, eff, st.iterations, conv_ms, identical ? "yes" : "NO");
      nlohmann::json row;
      row["grid"] = n; row["threads"] = nt; row["schedule"] = sched_name; row["ms_median"] = med; row["ms_min"] = mn; row["legacy_ms_median"] = legacy_med;
      row["region_per_halfsweep_ms_median"] = per_sweep_med; row["speedup"] = speedup; row["efficiency"] = eff; row["converge_iters"] = st.iterations; row["converged_ms"] = conv_ms;
      row["bit_identical_to_1thread"] = identical;
      j["rows"].push_back(row);
    }
  }
  std::ofstream(out + ".json") << j.dump(2) << "\n";
  std::cout << "wrote " << out << ".csv / .json\n";
  return 0;
}
