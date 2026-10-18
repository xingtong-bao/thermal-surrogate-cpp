// apps/ts_demo.cpp -- command-line front end
//
//   ts_demo --info   [model.onnx]                       runtime version and model IO (S4.1)
//   ts_demo --parity [fixture_dir] [out.json]           run the 10 reference pairs, write parity_cpp.json (S4.2)
//
// Keeping the demo a thin shell over the library is deliberate: anything
// worth doing twice goes into the library, not into main().
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

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

int usage() {
  std::cerr << "usage: ts_demo --info [model.onnx]\n"
               "       ts_demo --parity [fixture_dir] [out.json]\n";
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
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return usage();
}
