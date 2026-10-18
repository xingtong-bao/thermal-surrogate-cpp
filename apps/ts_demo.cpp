// apps/ts_demo.cpp -- command-line front end
//
//   ts_demo --info   [model.onnx]                       runtime version and model IO (S4.1)
//   ts_demo --parity [fixture_dir] [out.json]           run the 10 reference pairs, write parity_cpp.json (S4.2)
//   ts_demo --fd-calibrate <inputs.npy> <truth.npy> <params.json>   fit the FD solver's thickness on training samples (S5.2)
//   ts_demo --fd-validate <inputs.npy> <truth.npy> <n> <params.json> <out.json>   MAE vs truth + runtime (S5.2)
//   ts_demo --compare <label> <inputs.npy> <truth.npy> <n> <out.csv> [external cmd...]
//        every sample through surrogate, FD reference, external reference (if given) and the
//        dispatcher; per-sample MAE / Tmax error / latency / source -> CSV (S5.3, S5.5)
//
// Keeping the demo a thin shell over the library is deliberate: anything
// worth doing twice goes into the library, not into main().
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>
#include <cstring>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

#include "thermal_surrogate/dispatcher.h"
#include "thermal_surrogate/external_backend.h"
#include "thermal_surrogate/fd_solver.h"
#include "thermal_surrogate/model_info.h"
#include "thermal_surrogate/npy.h"
#include "thermal_surrogate/predictor.h"
#include "thermal_surrogate/types.h"

namespace ts = thermal_surrogate;

namespace {

void print_tensors(const char* heading, const std::vector<ts::TensorInfo>& tensors) {
  std::cout << heading << "\n";
  for (const auto& t : tensors) {
    std::cout << "  " << t.name << "  " << t.element_type << "  (";
    for (std::size_t i = 0; i < t.shape.size(); ++i) {
      if (i) std::cout << ", ";
      if (t.shape[i] < 0) std::cout << "dyn"; else std::cout << t.shape[i];
    }
    std::cout << ")\n";
  }
}

int cmd_info(const std::string& model) {
  const auto v = ts::version();
  std::cout << "thermal_surrogate " << v.library << " (built against ONNX Runtime " << v.onnxruntime << ")\n";
  const auto info = ts::inspect_model(model);
  std::cout << "ONNX Runtime loaded: " << info.onnxruntime_version << "\nmodel: " << info.path << "\n";
  print_tensors("inputs:", info.inputs);
  print_tensors("outputs:", info.outputs);
  return 0;
}

// Runs every pair in <dir>/manifest.json through the C++ predictor and writes
// a JSON summary comparable with docs/results/parity_python.json.
int cmd_parity(const std::string& model, const std::string& dir, const std::string& out_path) {
  const std::string meta = model.substr(0, model.rfind('/')) + "/metadata.json";
  ts::Predictor pred(model, meta);
  nlohmann::json manifest;
  std::ifstream(dir + "/manifest.json") >> manifest;

  nlohmann::json pairs = nlohmann::json::array();
  double max_ort = 0, max_torch = 0, sum_ort = 0, latency_sum = 0;
  std::size_t n_pixels = 0;
  for (const auto& p : manifest["pairs"]) {
    const auto x = ts::npy::load_f32(dir + "/" + p["input"]["file"].get<std::string>());
    const auto ref = ts::npy::load_f32(dir + "/" + p["field"]["file"].get<std::string>());
    const auto torch = ts::npy::load_f32(dir + "/" + p["torch"]["file"].get<std::string>());
    ts::Output out;
    pred.predict(ts::Input{x.data.data(), ts::kInputShape}, out);
    if (!out.status.ok()) throw std::runtime_error("predict failed: " + out.status.message);
    double m_ort = 0, m_torch = 0, s_ort = 0;
    for (std::size_t i = 0; i < ts::kCells; ++i) {
      const double d = std::fabs(static_cast<double>(out.field[i]) - ref.data[i]);
      m_ort = std::max(m_ort, d); s_ort += d;
      m_torch = std::max(m_torch, std::fabs(static_cast<double>(out.field[i]) - torch.data[i]));
    }
    pairs.push_back({{"pair", p["pair"]}, {"max_abs_diff_vs_python_ort_K", m_ort},
                     {"max_abs_diff_vs_torch_K", m_torch}, {"tmax_K", out.tmax},
                     {"tmax_index", {out.tmax_index[0], out.tmax_index[1]}}, {"latency_ms", out.latency_ms}});
    max_ort = std::max(max_ort, m_ort); max_torch = std::max(max_torch, m_torch);
    sum_ort += s_ort; n_pixels += ts::kCells; latency_sum += out.latency_ms;
    std::cout << "pair " << p["pair"] << ": vs Python ORT " << m_ort << " K, vs torch " << m_torch
              << " K, " << out.latency_ms << " ms\n";
  }
  const double gate = manifest.value("gate_cpp_vs_python_ort_K", 1e-4);
  nlohmann::json summary = {
      {"n_pairs", pairs.size()}, {"model_version", pred.model_version()},
      {"onnxruntime_cpp", ts::version().onnxruntime}, {"library_version", ts::version().library},
      {"max_abs_diff_vs_python_ort_K", max_ort}, {"mean_abs_diff_vs_python_ort_K", sum_ort / static_cast<double>(n_pixels)},
      {"max_abs_diff_vs_torch_b1_K", max_torch}, {"gate_K", gate}, {"passes_gate", max_ort < gate},
      {"mean_latency_ms_unwarmed", latency_sum / static_cast<double>(pairs.size())},
      {"note", "latency here includes the first (cold) call; see S4.4 for the benchmark"},
      {"pairs", pairs}};
  std::ofstream(out_path) << summary.dump(2) << "\n";
  std::cout << "max |diff| vs Python ORT = " << max_ort << " K (gate " << gate << ") -> "
            << (max_ort < gate ? "PASS" : "FAIL") << "\n-> " << out_path << "\n";
  return max_ort < gate ? 0 : 1;
}

// ---- S5.2: calibrate and validate the finite-difference reference ----------------
struct SetView {
  ts::npy::Array x, y;
  std::size_t n() const { return x.shape[0]; }
  const float* in(std::size_t i) const { return x.data.data() + i * ts::kInputSize; }
  const float* truth(std::size_t i) const { return y.data.data() + i * ts::kCells; }
};

SetView load_set(const std::string& inputs, const std::string& truth, std::size_t limit) {
  SetView s{ts::npy::load_f32(inputs), ts::npy::load_f32(truth)};
  if (s.x.shape.size() != 5 || s.y.shape.size() != 4 || s.x.shape[0] != s.y.shape[0])
    throw std::runtime_error("eval set must be (n,64,64,1,7) + (n,64,64,1)");
  if (limit && limit < s.x.shape[0]) { s.x.shape[0] = limit; s.y.shape[0] = limit; }
  return s;
}

// Mean absolute error of the solver over a set for one parameter choice; NaN if any solve fails.
double set_mae(const SetView& s, const ts::FdSolverParams& p, double* tmax_err = nullptr, double* ms_per_solve = nullptr) {
  std::vector<float> field(ts::kCells);
  double sum = 0, tsum = 0;
  const auto t0 = std::chrono::steady_clock::now();
  for (std::size_t i = 0; i < s.n(); ++i) {
    if (!ts::fd_solve(s.in(i), p, field.data()).ok()) return std::nan("");
    double acc = 0, fmax = field[0], tmax = s.truth(i)[0];
    for (std::size_t c = 0; c < ts::kCells; ++c) {
      acc += std::fabs(static_cast<double>(field[c]) - s.truth(i)[c]);
      fmax = std::max(fmax, static_cast<double>(field[c])); tmax = std::max(tmax, static_cast<double>(s.truth(i)[c]));
    }
    sum += acc / static_cast<double>(ts::kCells); tsum += std::fabs(fmax - tmax);
  }
  if (tmax_err) *tmax_err = tsum / static_cast<double>(s.n());
  if (ms_per_solve) *ms_per_solve = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / static_cast<double>(s.n());
  return sum / static_cast<double>(s.n());
}

// Per-sample lumped quantities the calibration regresses on.
struct Lumped { double P, r, amb, truth_min, truth_mean; };
Lumped lumped(const SetView& s, std::size_t i) {
  const float* x = s.in(i);
  std::vector<int> seen(ts::kCells, 0);
  double P = 0;
  // Total power = sum over chiplets (connected components of equal power), as in the validator.
  for (std::size_t c = 0; c < ts::kCells; ++c) {
    const float v = x[c * ts::kP];
    if (v <= 0 || seen[c]) continue;
    std::vector<std::size_t> st{c}; seen[c] = 1;
    while (!st.empty()) {
      const std::size_t k = st.back(); st.pop_back();
      const std::size_t xx = k / ts::kY, yy = k % ts::kY;
      const std::size_t nb[4] = {xx + 1 < ts::kX ? k + ts::kY : k, xx > 0 ? k - ts::kY : k, yy + 1 < ts::kY ? k + 1 : k, yy > 0 ? k - 1 : k};
      for (std::size_t j : nb) if (!seen[j] && x[j * ts::kP] == v) { seen[j] = 1; st.push_back(j); }
    }
    P += v;
  }
  double mn = s.truth(i)[0], mean = 0;
  for (std::size_t c = 0; c < ts::kCells; ++c) { mn = std::min(mn, static_cast<double>(s.truth(i)[c])); mean += s.truth(i)[c]; }
  return {P, x[6], x[4], mn, mean / static_cast<double>(ts::kCells)};
}

// Least squares y ~ a * u + b * v (2x2 normal equations).
std::pair<double, double> fit2(const std::vector<double>& u, const std::vector<double>& v, const std::vector<double>& y) {
  double suu = 0, suv = 0, svv = 0, suy = 0, svy = 0;
  for (std::size_t i = 0; i < y.size(); ++i) { suu += u[i] * u[i]; suv += u[i] * v[i]; svv += v[i] * v[i]; suy += u[i] * y[i]; svy += v[i] * y[i]; }
  const double det = suu * svv - suv * suv;
  return {(suy * svv - svy * suv) / det, (suu * svy - suv * suy) / det};
}

int cmd_fd_calibrate(const std::string& inputs, const std::string& truth, const std::string& out_path) {
  // Three steps, mirroring the numpy study that designed the model:
  //   1. alpha, beta from the minimum rise:   min(T) - T_amb ~ P * (alpha r + beta)
  //   2. grid search t, g_v for the local term with alpha, beta fixed
  //   3. refit alpha, beta on the mean residual with the local term in place.
  const SetView s = load_set(inputs, truth, 0);
  std::vector<Lumped> L;
  for (std::size_t i = 0; i < s.n(); ++i) L.push_back(lumped(s, i));
  std::vector<double> u, v, y;
  for (const auto& l : L) { u.push_back(l.P * l.r); v.push_back(l.P); y.push_back(l.truth_min - l.amb); }
  auto [alpha, beta] = fit2(u, v, y);
  std::cout << "step 1: min rise = P*(" << alpha << " r + " << beta << ")\n";

  ts::FdSolverParams best; best.alpha = alpha; best.beta = beta;
  double best_mae = std::numeric_limits<double>::infinity();
  nlohmann::json grid = nlohmann::json::array();
  auto try_point = [&](double t, double gv, bool refine) {
    ts::FdSolverParams p = best; p.thickness_m = t; p.g_v_w_m2k = gv;
    const double mae = set_mae(s, p);
    grid.push_back({{"thickness_m", t}, {"g_v_w_m2k", gv}, {"mae_K", mae}, {"refine", refine}});
    if (mae < best_mae) { best_mae = mae; best.thickness_m = t; best.g_v_w_m2k = gv; }
  };
  for (double lt = -4.0; lt <= -1.0 + 1e-9; lt += 0.5)
    for (double lg = 2.0; lg <= 4.5 + 1e-9; lg += 0.5) try_point(std::pow(10.0, lt), std::pow(10.0, lg), false);
  const double lt0 = std::log10(best.thickness_m), lg0 = std::log10(best.g_v_w_m2k);
  for (double lt = lt0 - 0.4; lt <= lt0 + 0.4 + 1e-9; lt += 0.2)
    for (double lg = lg0 - 0.4; lg <= lg0 + 0.4 + 1e-9; lg += 0.2) try_point(std::pow(10.0, lt), std::pow(10.0, lg), true);
  std::cout << "step 2: t " << best.thickness_m << " m, g_v " << best.g_v_w_m2k << " W/m^2K, MAE " << best_mae << " K\n";

  // step 3: residual of the mean against the local term.
  {
    ts::FdSolverParams p0 = best; p0.alpha = 0; p0.beta = 0;        // local term only
    std::vector<float> field(ts::kCells);
    std::vector<double> yy;
    for (std::size_t i = 0; i < s.n(); ++i) {
      ts::fd_solve(s.in(i), p0, field.data());
      double m = 0; for (float f : field) m += f;
      yy.push_back(L[i].truth_mean - (m / static_cast<double>(ts::kCells)));   // = P(alpha r + beta) since local-only field already includes T_amb
    }
    auto [a2, b2] = fit2(u, v, yy);
    best.alpha = a2; best.beta = b2;
    best_mae = set_mae(s, best);
  }
  double tmax_err = 0, ms = 0;
  set_mae(s, best, &tmax_err, &ms);
  nlohmann::json doc = {{"model", "lumped sink P*(alpha r + beta) + 2-D local conduction with vertical loss g_v; see include/thermal_surrogate/fd_solver.h"},
                        {"calibrated_on", {{"inputs", inputs.substr(inputs.find_last_of('/') + 1)}, {"n", s.n()}, {"split", "level4 train"}}},
                        {"thickness_m", best.thickness_m}, {"g_v_w_m2k", best.g_v_w_m2k}, {"alpha", best.alpha}, {"beta", best.beta},
                        {"omega", best.omega}, {"tol_K", best.tol}, {"max_iter", best.max_iter},
                        {"train_mae_K", best_mae}, {"train_tmax_abs_err_K", tmax_err}, {"ms_per_solve", ms}, {"search_grid", grid}};
  std::ofstream(out_path) << doc.dump(2) << "\n";
  std::cout << "step 3: alpha " << best.alpha << " beta " << best.beta << "; train MAE " << best_mae << " K, Tmax err " << tmax_err
            << " K, " << ms << " ms/solve\n-> " << out_path << "\n";
  return 0;
}

int cmd_fd_validate(const std::string& inputs, const std::string& truth, std::size_t n, const std::string& params, const std::string& out_path) {
  const SetView s = load_set(inputs, truth, n);
  const auto p = ts::load_fd_params(params);
  nlohmann::json per = nlohmann::json::array();
  std::vector<float> field(ts::kCells);
  double sum = 0, tsum = 0, worst = 0, ms_sum = 0;
  int iters_sum = 0;
  for (std::size_t i = 0; i < s.n(); ++i) {
    ts::FdSolveStats st;
    const auto t0 = std::chrono::steady_clock::now();
    const auto status = ts::fd_solve(s.in(i), p, field.data(), &st);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!status.ok()) throw std::runtime_error("solve " + std::to_string(i) + " failed: " + status.message);
    double acc = 0, mx = 0, fmax = field[0], tmax = s.truth(i)[0];
    for (std::size_t c = 0; c < ts::kCells; ++c) {
      const double d = std::fabs(static_cast<double>(field[c]) - s.truth(i)[c]);
      acc += d; mx = std::max(mx, d);
      fmax = std::max(fmax, static_cast<double>(field[c])); tmax = std::max(tmax, static_cast<double>(s.truth(i)[c]));
    }
    per.push_back({{"i", i}, {"mae_K", acc / static_cast<double>(ts::kCells)}, {"max_abs_err_K", mx}, {"tmax_abs_err_K", std::fabs(fmax - tmax)},
                   {"tmax_true_K", tmax}, {"tmax_fd_K", fmax}, {"iterations", st.iterations}, {"ms", ms}});
    sum += acc / static_cast<double>(ts::kCells); tsum += std::fabs(fmax - tmax); worst = std::max(worst, mx); ms_sum += ms; iters_sum += st.iterations;
  }
  nlohmann::json doc = {{"params", params.substr(params.find_last_of('/') + 1)}, {"thickness_m", p.thickness_m}, {"g_v_w_m2k", p.g_v_w_m2k}, {"alpha", p.alpha}, {"beta", p.beta},
                        {"n", s.n()}, {"set", inputs.substr(inputs.find_last_of('/') + 1)},
                        {"mae_K", sum / static_cast<double>(s.n())}, {"tmax_abs_err_mean_K", tsum / static_cast<double>(s.n())},
                        {"max_abs_err_K", worst}, {"ms_per_solve_mean", ms_sum / static_cast<double>(s.n())},
                        {"iterations_mean", static_cast<double>(iters_sum) / static_cast<double>(s.n())}, {"per_sample", per}};
  std::ofstream(out_path) << doc.dump(2) << "\n";
  std::cout << "n=" << s.n() << " MAE " << doc["mae_K"] << " K, Tmax err " << doc["tmax_abs_err_mean_K"] << " K, max "
            << worst << " K, " << doc["ms_per_solve_mean"] << " ms/solve, " << doc["iterations_mean"] << " sweeps\n-> " << out_path << "\n";
  return 0;
}

// ---- S5.3 / S5.5: run a set through every backend and the dispatcher ---------------
int cmd_compare(const std::string& label, const std::string& inputs, const std::string& truth, std::size_t n,
                const std::string& out_csv, std::vector<std::string> external_cmd) {
  const SetView s = load_set(inputs, truth, n);
  const std::string models = TS_MODELS_DIR, contract = TS_CONTRACT_DIR;
  ts::Validator validator(contract + "/stats.json", models + "/config_bank.npy");
  ts::Predictor pred(models + "/model.onnx", models + "/metadata.json");
  ts::SurrogateBackend sur(pred);
  ts::FdSolverBackend fd(ts::load_fd_params(models + "/fd_solver_params.json"));
  ts::ExternalBackendOptions eo;
  eo.command = external_cmd;
  eo.timeout_s = 120.0;
  ts::ExternalReferenceBackend ext(eo);

  // The dispatcher as the contract prescribes: surrogate -> in-process -> external.
  std::vector<std::unique_ptr<ts::ISolverBackend>> chain;
  chain.push_back(std::make_unique<ts::SurrogateBackend>(pred));
  if (!external_cmd.empty()) chain.push_back(std::make_unique<ts::ExternalReferenceBackend>(eo));
  chain.push_back(std::make_unique<ts::FdSolverBackend>(ts::load_fd_params(models + "/fd_solver_params.json")));
  ts::Dispatcher dispatcher(validator, std::move(chain));

  std::ofstream csv(out_csv);
  csv << "set,i,backend,status,source,oob_flag,mae_K,tmax_abs_err_K,max_abs_err_K,latency_ms\n";
  auto score = [&](std::size_t i, const ts::Output& o, double& mae, double& terr, double& mx) {
    mae = terr = mx = std::nan("");
    if (!o.status.ok()) return;
    double acc = 0, fmax = o.field[0], tmax = s.truth(i)[0];
    mx = 0;
    for (std::size_t c = 0; c < ts::kCells; ++c) {
      const double d = std::fabs(static_cast<double>(o.field[c]) - s.truth(i)[c]);
      acc += d; mx = std::max(mx, d);
      fmax = std::max(fmax, static_cast<double>(o.field[c])); tmax = std::max(tmax, static_cast<double>(s.truth(i)[c]));
    }
    mae = acc / static_cast<double>(ts::kCells); terr = std::fabs(fmax - tmax);
  };
  auto row = [&](std::size_t i, const char* backend, const ts::Output& o) {
    double mae, terr, mx;
    score(i, o, mae, terr, mx);
    csv << label << "," << i << "," << backend << "," << ts::to_string(o.status.code) << "," << ts::to_string(o.source) << ","
        << (o.oob_flag ? 1 : 0) << "," << mae << "," << terr << "," << mx << "," << o.latency_ms << "\n";
  };
  for (std::size_t i = 0; i < s.n(); ++i) {
    const ts::Input in{s.in(i), ts::kInputShape};
    ts::Output o;
    sur.solve(in, o); row(i, "surrogate", o);
    fd.solve(in, o); row(i, "fd_solver", o);
    if (!external_cmd.empty()) { ext.solve(in, o); row(i, "external", o); }
    dispatcher.solve(in, o); row(i, "dispatcher", o);
    std::cout << label << " " << i << ": dispatcher -> " << ts::to_string(o.source) << " (" << ts::to_string(o.status.code) << ")\n";
  }
  std::cout << "-> " << out_csv << "\n";
  return 0;
}

int usage() {
  std::cerr << "usage: ts_demo --info [model.onnx]\n"
               "       ts_demo --parity [fixture_dir] [out.json]\n"
               "       ts_demo --fd-calibrate <inputs.npy> <truth.npy> <params.json>\n"
               "       ts_demo --fd-validate <inputs.npy> <truth.npy> <n> <params.json> <out.json>\n"
               "       ts_demo --compare <label> <inputs.npy> <truth.npy> <n> <out.csv> [external cmd...]\n";
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) return usage();
  try {
    if (std::strcmp(argv[1], "--info") == 0)
      return cmd_info(argc >= 3 ? argv[2] : TS_DEFAULT_MODEL_PATH);
    if (std::strcmp(argv[1], "--parity") == 0)
      return cmd_parity(TS_DEFAULT_MODEL_PATH, argc >= 3 ? argv[2] : std::string(TS_FIXTURE_DIR) + "/parity",
                        argc >= 4 ? argv[3] : "parity_cpp.json");
    if (std::strcmp(argv[1], "--fd-calibrate") == 0 && argc >= 5) return cmd_fd_calibrate(argv[2], argv[3], argv[4]);
    if (std::strcmp(argv[1], "--compare") == 0 && argc >= 7) {
      std::vector<std::string> ext(argv + 7, argv + argc);
      return cmd_compare(argv[2], argv[3], argv[4], static_cast<std::size_t>(std::stoul(argv[5])), argv[6], ext);
    }
    if (std::strcmp(argv[1], "--fd-validate") == 0 && argc >= 7)
      return cmd_fd_validate(argv[2], argv[3], static_cast<std::size_t>(std::stoul(argv[4])), argv[5], argv[6]);
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return usage();
}
