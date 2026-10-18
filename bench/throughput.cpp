// bench/throughput.cpp -- S4.5: batch size, memory layout and output copy, measured
//
// Three questions, each answered by a timed loop over the same fixture:
//   1. batch size:   samples/s for batch 1, 8, 32, 64 through predict_batch()
//                    on models/model_dynamic.onnx (contract layout, dynamic B)
//   2. layout:       the same batches through models/model_nchw_dynamic.onnx,
//                    whose input is already (B,P,X,Y) so the graph has no
//                    Transpose node; the caller pays the transpose instead.
//                    Measured with a raw ORT session (this is a study, not an
//                    API), input transposed once outside the timed region.
//   3. output copy:  predict() (allocates a std::vector and copies) versus
//                    predict_into() (ORT writes into the caller's buffer).
// Thread count is fixed (default 4 = the M4's performance cores) so the
// numbers are comparable with docs/results/latency_single.json.
//
//   bench_throughput [--threads 4] [--repeat 20] [--out docs/results/throughput.json]
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <string>
#include <vector>

#include "machine_info.h"
#include "thermal_surrogate/npy.h"
#include "thermal_surrogate/predictor.h"

namespace ts = thermal_surrogate;
using clk = std::chrono::steady_clock;

namespace {

double ms_since(clk::time_point a) { return std::chrono::duration<double, std::milli>(clk::now() - a).count(); }

// Median of `repeat` timings of fn(), after one untimed warm-up.
template <class F>
double median_ms(int repeat, F fn) {
  fn();
  std::vector<double> t;
  for (int i = 0; i < repeat; ++i) {
    const auto a = clk::now();
    fn();
    t.push_back(ms_since(a));
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

}  // namespace

int main(int argc, char** argv) {
  int threads = 4, repeat = 20;
  std::string out_path = "throughput.json";
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--threads")) threads = std::stoi(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--repeat")) repeat = std::stoi(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--out")) out_path = argv[i + 1];
  }
  const std::string models = TS_MODELS_DIR;
  const auto fx = ts::npy::load_f32(std::string(TS_FIXTURE_DIR) + "/sample_input.npy");
  const std::vector<int> batches{1, 8, 32, 64};
  const int max_b = batches.back();

  // Inputs: the fixture repeated max_b times, in both layouts.
  std::vector<float> in_nhwc(static_cast<std::size_t>(max_b) * ts::kInputSize);
  for (int b = 0; b < max_b; ++b) std::copy(fx.data.begin(), fx.data.end(), in_nhwc.begin() + b * static_cast<long>(ts::kInputSize));
  std::vector<float> in_nchw(in_nhwc.size());
  for (int b = 0; b < max_b; ++b)
    for (std::size_t x = 0; x < ts::kX; ++x)
      for (std::size_t y = 0; y < ts::kY; ++y)
        for (std::size_t p = 0; p < ts::kP; ++p)
          in_nchw[(static_cast<std::size_t>(b) * ts::kP + p) * ts::kCells + x * ts::kY + y] =
              in_nhwc[static_cast<std::size_t>(b) * ts::kInputSize + (x * ts::kY + y) * ts::kP + p];
  std::vector<float> out(static_cast<std::size_t>(max_b) * ts::kCells);

  ts::PredictorOptions opt;
  opt.intra_op_threads = threads;
  ts::Predictor dyn(models + "/model_dynamic.onnx", models + "/metadata.json", opt);
  ts::Predictor fixed(models + "/model.onnx", models + "/metadata.json", opt);

  // Raw session on the NCHW graph (study only).
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "bench");
  Ort::SessionOptions so;
  so.SetIntraOpNumThreads(threads);
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  Ort::Session nchw(env, (models + "/model_nchw_dynamic.onnx").c_str(), so);
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto run_nchw = [&](int b) {
    const std::array<int64_t, 4> idims{b, static_cast<int64_t>(ts::kP), static_cast<int64_t>(ts::kX), static_cast<int64_t>(ts::kY)};
    const std::array<int64_t, 4> odims{b, 1, static_cast<int64_t>(ts::kX), static_cast<int64_t>(ts::kY)};
    Ort::Value in = Ort::Value::CreateTensor<float>(mem, in_nchw.data(), static_cast<std::size_t>(b) * ts::kInputSize, idims.data(), 4);
    Ort::Value ov = Ort::Value::CreateTensor<float>(mem, out.data(), static_cast<std::size_t>(b) * ts::kCells, odims.data(), 4);
    const char* in_names[] = {"input"};
    const char* out_names[] = {"field"};
    nchw.Run(Ort::RunOptions{nullptr}, in_names, &in, 1, out_names, &ov, 1);
  };

  nlohmann::json rows = nlohmann::json::array();
  auto record = [&](const std::string& path, const std::string& layout, int b, double ms, const std::string& output_mode) {
    const double sps = 1000.0 * b / ms;
    std::cout << path << "  " << layout << "  batch " << b << "  " << output_mode << ": " << ms << " ms  -> " << sps << " samples/s\n";
    rows.push_back({{"path", path}, {"layout", layout}, {"batch", b}, {"output", output_mode}, {"median_ms_per_call", ms},
                    {"ms_per_sample", ms / b}, {"samples_per_s", sps}});
  };

  // 1 + 2: batch size, both layouts.
  for (int b : batches) {
    record("predict_batch(model_dynamic)", "contract (B,X,Y,Z,P)", b,
           median_ms(repeat, [&] { auto s = dyn.predict_batch(in_nhwc.data(), static_cast<std::size_t>(b), out.data()); if (!s.ok()) throw std::runtime_error(s.message); }),
           "zero-copy");
    record("raw session(model_nchw_dynamic)", "NCHW (B,P,X,Y)", b, median_ms(repeat, [&] { run_nchw(b); }), "zero-copy");
  }
  // 3: output copy vs zero-copy on the fixed-shape production model, batch 1.
  const ts::Input in1{in_nhwc.data(), ts::kInputShape};
  ts::Output o;
  record("predict(model)", "contract (B,X,Y,Z,P)", 1, median_ms(repeat, [&] { fixed.predict(in1, o); }), "copy into std::vector");
  record("predict_into(model)", "contract (B,X,Y,Z,P)", 1, median_ms(repeat, [&] { fixed.predict_into(in1, out.data(), o); }), "zero-copy");

  // Correctness guard: the batched result must equal the single result per sample (gate 1e-4 K).
  std::vector<float> single(ts::kCells);
  fixed.predict_into(in1, single.data(), o);
  dyn.predict_batch(in_nhwc.data(), 64, out.data());
  double worst = 0;
  for (int b = 0; b < 64; ++b)
    for (std::size_t i = 0; i < ts::kCells; ++i)
      worst = std::max(worst, static_cast<double>(std::fabs(out[static_cast<std::size_t>(b) * ts::kCells + i] - single[i])));
  run_nchw(64);
  double worst_nchw = 0;
  for (int b = 0; b < 64; ++b)
    for (std::size_t i = 0; i < ts::kCells; ++i)
      worst_nchw = std::max(worst_nchw, static_cast<double>(std::fabs(out[static_cast<std::size_t>(b) * ts::kCells + i] - single[i])));
  std::cout << "batch-64 vs single max |diff|: contract " << worst << " K, NCHW " << worst_nchw << " K\n";

  nlohmann::json doc = {{"benchmark", "throughput vs batch size, layout and output copy"},
                        {"intra_op_threads", threads}, {"repeat", repeat}, {"timing", "median of repeat calls after one warm-up"},
                        {"onnxruntime", ts::version().onnxruntime}, {"machine", bench::machine_info()},
                        {"batch64_vs_single_max_abs_diff_K", {{"contract", worst}, {"nchw", worst_nchw}}}, {"rows", rows}};
  std::ofstream(out_path) << doc.dump(2) << "\n";
  std::cout << "-> " << out_path << "\n";
  return 0;
}
