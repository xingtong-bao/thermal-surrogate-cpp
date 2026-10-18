// bench/latency.cpp -- S4.4: single-prediction latency of the surrogate path
//
// Protocol
//   * one Predictor per thread setting; the same in-support fixture every call
//     (the network's cost does not depend on the values, only on the shape);
//   * `warmup` calls discarded: the first call pays for kernel selection and
//     memory-arena growth and is not representative of steady state;
//   * `n` timed calls, wall clock around predict() only (includes the input
//     tensor wrap, Run, and the copy into Output::field; excludes validation);
//   * p50 / p95 / p99 / mean / min reported in milliseconds;
//   * intra-op thread count fixed per run so the number is reproducible; the
//     default (0 = ORT picks, usually all cores) is reported as well because
//     that is what an unconfigured caller gets.
//
//   bench_latency [--n 200] [--warmup 20] [--threads 1,4,0] [--out docs/results/latency_single.json]
#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include "machine_info.h"
#include "thermal_surrogate/npy.h"
#include "thermal_surrogate/predictor.h"

namespace ts = thermal_surrogate;

namespace {

struct Stats {
  double p50, p95, p99, mean, min, max;
};

Stats summarize(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  auto pct = [&](double p) { return v[std::min(v.size() - 1, static_cast<std::size_t>(p * static_cast<double>(v.size())))]; };
  return {pct(0.50), pct(0.95), pct(0.99), std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size()), v.front(), v.back()};
}

std::vector<int> parse_list(const std::string& s) {
  std::vector<int> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) out.push_back(std::stoi(tok));
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  int n = 200, warmup = 20;
  std::vector<int> threads{1, 4, 0};
  std::string out_path = "latency_single.json";
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--n")) n = std::stoi(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--warmup")) warmup = std::stoi(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--threads")) threads = parse_list(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--out")) out_path = argv[i + 1];
  }
  const std::string model = TS_MODEL_PATH;
  const std::string meta = model.substr(0, model.rfind('/')) + "/metadata.json";
  const auto x = ts::npy::load_f32(std::string(TS_FIXTURE_DIR) + "/sample_input.npy");
  const ts::Input in{x.data.data(), ts::kInputShape};

  nlohmann::json results = nlohmann::json::array();
  for (int t : threads) {
    ts::PredictorOptions opt;
    opt.intra_op_threads = t;
    const auto t_load0 = std::chrono::steady_clock::now();
    ts::Predictor pred(model, meta, opt);
    const double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_load0).count();
    ts::Output out;
    double first_call_ms = 0;
    for (int i = 0; i < warmup; ++i) {
      const auto a = std::chrono::steady_clock::now();
      pred.predict(in, out);
      if (i == 0) first_call_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
    }
    std::vector<double> ms;
    ms.reserve(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
      const auto a = std::chrono::steady_clock::now();
      pred.predict(in, out);
      ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
      if (!out.status.ok()) { std::cerr << "predict failed: " << out.status.message << "\n"; return 1; }
    }
    const auto s = summarize(ms);
    std::cout << "intra_op_threads=" << t << (t == 0 ? " (ORT default)" : "") << ": p50 " << s.p50 << " ms, p95 " << s.p95
              << " ms, p99 " << s.p99 << " ms, mean " << s.mean << " ms, min " << s.min << " ms; load " << load_ms
              << " ms, first call " << first_call_ms << " ms\n";
    results.push_back({{"intra_op_threads", t}, {"n", n}, {"warmup", warmup}, {"p50_ms", s.p50}, {"p95_ms", s.p95},
                       {"p99_ms", s.p99}, {"mean_ms", s.mean}, {"min_ms", s.min}, {"max_ms", s.max},
                       {"session_load_ms", load_ms}, {"first_call_ms", first_call_ms}});
  }
  nlohmann::json doc = {{"benchmark", "single prediction latency, surrogate path, one 64x64x1x7 sample"},
                        {"model_version", "level4-unet-1.0.0"}, {"onnxruntime", ts::version().onnxruntime},
                        {"library_version", ts::version().library}, {"machine", bench::machine_info()},
                        {"timing_scope", "Predictor::predict() wall time: input tensor wrap + Run + copy into Output::field; validation excluded"},
                        {"runs", results}};
  std::ofstream(out_path) << doc.dump(2) << "\n";
  std::cout << "-> " << out_path << "\n";
  return 0;
}
