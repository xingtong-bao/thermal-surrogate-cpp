// bench/dynamic_size.cpp -- S4.6: what happens when the grid is larger than 64x64?
//
// The contract fixes 64x64, but the question "what about a million-cell
// mesh?" is the first thing an integrator asks. models/model_dynamic.onnx has
// dynamic X and Y (multiples of 8), so the same weights can be run on larger
// synthetic inputs. This bench records, per size: median latency, memory
// growth (peak RSS before/after), and the output for a sanity check (finite,
// plausible range). The design note in README draws the conclusions.
//
// Synthetic input: the 64x64 fixture tiled; the network is fully
// convolutional, so the cost depends on the cell count only.
//
//   bench_dynamic_size [--threads 4] [--repeat 5] [--sizes 64,128,256] [--out docs/results/dynamic_size.csv]
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/resource.h>
#include <vector>

#include "thermal_surrogate/npy.h"
#include "thermal_surrogate/types.h"

namespace ts = thermal_surrogate;
using clk = std::chrono::steady_clock;

namespace {
double peak_rss_mb() {
  rusage ru{};
  getrusage(RUSAGE_SELF, &ru);
#if defined(__APPLE__)
  return static_cast<double>(ru.ru_maxrss) / (1024.0 * 1024.0);   // bytes on macOS
#else
  return static_cast<double>(ru.ru_maxrss) / 1024.0;              // kilobytes on Linux
#endif
}
}  // namespace

int main(int argc, char** argv) {
  int threads = 4, repeat = 5;
  std::vector<int> sizes{64, 128, 256};
  std::string out_path = "dynamic_size.csv";
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--threads")) threads = std::stoi(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--repeat")) repeat = std::stoi(argv[i + 1]);
    else if (!std::strcmp(argv[i], "--out")) out_path = argv[i + 1];
    else if (!std::strcmp(argv[i], "--sizes")) {
      sizes.clear();
      std::stringstream ss(argv[i + 1]);
      std::string tok;
      while (std::getline(ss, tok, ',')) sizes.push_back(std::stoi(tok));
    }
  }
  const auto fx = ts::npy::load_f32(std::string(TS_FIXTURE_DIR) + "/sample_input.npy");
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "bench");
  Ort::SessionOptions so;
  so.SetIntraOpNumThreads(threads);
  so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  Ort::Session session(env, (std::string(TS_MODELS_DIR) + "/model_dynamic.onnx").c_str(), so);
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  std::ofstream csv(out_path);
  csv << "size,cells,threads,median_ms,min_ms,ms_per_1k_cells,peak_rss_mb_before,peak_rss_mb_after,field_min_K,field_max_K,finite\n";
  for (int n : sizes) {
    const auto N = static_cast<std::size_t>(n);
    const std::size_t cells = N * N;
    // Tile the 64x64 fixture; grid_x/grid_y keep their local values, which is
    // fine for a cost measurement (the network does not know about geometry).
    std::vector<float> in(cells * ts::kP), out(cells);
    for (std::size_t x = 0; x < N; ++x)
      for (std::size_t y = 0; y < N; ++y)
        std::memcpy(&in[(x * N + y) * ts::kP], &fx.data[((x % 64) * 64 + (y % 64)) * ts::kP], ts::kP * sizeof(float));
    const std::array<int64_t, 5> idims{1, n, n, 1, static_cast<int64_t>(ts::kP)};
    const std::array<int64_t, 4> odims{1, n, n, 1};
    const char* in_names[] = {"input"};
    const char* out_names[] = {"field"};
    auto run = [&] {
      Ort::Value iv = Ort::Value::CreateTensor<float>(mem, in.data(), in.size(), idims.data(), 5);
      Ort::Value ov = Ort::Value::CreateTensor<float>(mem, out.data(), out.size(), odims.data(), 4);
      session.Run(Ort::RunOptions{nullptr}, in_names, &iv, 1, out_names, &ov, 1);
    };
    const double rss0 = peak_rss_mb();
    run();                                                   // warm-up (also grows the arena)
    std::vector<double> t;
    for (int i = 0; i < repeat; ++i) {
      const auto a = clk::now();
      run();
      t.push_back(std::chrono::duration<double, std::milli>(clk::now() - a).count());
    }
    std::sort(t.begin(), t.end());
    const double rss1 = peak_rss_mb();
    float mn = out[0], mx = out[0];
    bool finite = true;
    for (float v : out) { mn = std::min(mn, v); mx = std::max(mx, v); finite = finite && std::isfinite(v); }
    std::cout << n << "x" << n << " (" << cells << " cells): median " << t[t.size() / 2] << " ms, min " << t.front()
              << " ms, " << t[t.size() / 2] / (static_cast<double>(cells) / 1000.0) << " ms per 1k cells, peak RSS "
              << rss0 << " -> " << rss1 << " MB, field [" << mn << ", " << mx << "] K\n";
    csv << n << "," << cells << "," << threads << "," << t[t.size() / 2] << "," << t.front() << ","
        << t[t.size() / 2] / (static_cast<double>(cells) / 1000.0) << "," << rss0 << "," << rss1 << "," << mn << "," << mx
        << "," << (finite ? "true" : "false") << "\n";
  }
  std::cout << "-> " << out_path << "\n";
  return 0;
}
