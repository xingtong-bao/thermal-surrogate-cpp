// tests/test_concurrency.cpp -- S6.2: one Predictor, many threads, bit-identical outputs
//
// 8 threads x 100 predictions each on a single Predictor (one ONNX session),
// alternating between the ten parity inputs. Every result must equal the
// single-threaded result for that input bit for bit: ONNX Runtime's
// Session::Run is documented thread-safe, and this test (also run under
// ThreadSanitizer via -DTS_SANITIZE=thread) is our evidence. The per-run
// timing summary is written by the demo, not here; a test asserts, it does
// not benchmark.
#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <thread>
#include <vector>

#include "thermal_surrogate/npy.h"
#include "thermal_surrogate/predictor.h"

using namespace thermal_surrogate;

TEST(Concurrency, EightThreadsBitIdentical) {
  const std::string models = std::string(TS_MODEL_PATH).substr(0, std::string(TS_MODEL_PATH).rfind('/'));
  PredictorOptions opt;
  opt.intra_op_threads = 1;                         // many callers, one worker each: the realistic server shape
  Predictor pred(TS_MODEL_PATH, models + "/metadata.json", opt);

  std::vector<npy::Array> inputs;
  std::vector<std::vector<float>> expected;
  for (int k = 0; k < 10; ++k) {
    char num[8];
    std::snprintf(num, sizeof num, "%02d", k);
    inputs.push_back(npy::load_f32(std::string(TS_FIXTURE_DIR) + "/parity/input_" + num + ".npy"));
    Output o;
    pred.predict(Input{inputs.back().data.data(), kInputShape}, o);
    ASSERT_TRUE(o.status.ok());
    expected.push_back(o.field);
  }

  constexpr int kThreads = 8, kPerThread = 100;
  std::atomic<int> mismatches{0}, failures{0};
  std::vector<std::thread> pool;
  for (int t = 0; t < kThreads; ++t) {
    pool.emplace_back([&, t] {
      std::vector<float> buf(kCells);
      for (int i = 0; i < kPerThread; ++i) {
        const int k = (t * 7 + i) % 10;
        Output o;
        pred.predict_into(Input{inputs[static_cast<std::size_t>(k)].data.data(), kInputShape}, buf.data(), o);
        if (!o.status.ok()) { ++failures; continue; }
        if (std::memcmp(buf.data(), expected[static_cast<std::size_t>(k)].data(), kCells * sizeof(float)) != 0) ++mismatches;
      }
    });
  }
  for (auto& th : pool) th.join();
  EXPECT_EQ(failures.load(), 0);
  EXPECT_EQ(mismatches.load(), 0);
}
