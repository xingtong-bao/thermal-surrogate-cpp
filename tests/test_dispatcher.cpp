// tests/test_dispatcher.cpp -- S5.1: fallback rules F1-F4 with scripted backends
#include <gtest/gtest.h>

#include <memory>

#include "thermal_surrogate/dispatcher.h"
#include "thermal_surrogate/npy.h"

using namespace thermal_surrogate;

namespace {
const std::string kFix = std::string(TS_FIXTURE_DIR) + "/";
const std::string kStats = std::string(TS_CONTRACT_DIR) + "/stats.json";
const std::string kBank = std::string(TS_MODELS_DIR) + "/config_bank.npy";
const std::string kMeta = std::string(TS_MODELS_DIR) + "/metadata.json";

// A backend whose behaviour is scripted: succeed with a constant field, or fail with a code.
class FakeBackend final : public ISolverBackend {
 public:
  FakeBackend(Source s, std::string name, StatusCode code, float fill = 300.0f)
      : source_(s), name_(std::move(name)), code_(code), fill_(fill) {}
  Source source() const override { return source_; }
  std::string name() const override { return name_; }
  void solve(const Input&, Output& out) noexcept override {
    ++calls;
    out = Output{};
    if (code_ != StatusCode::Ok) { out.status = {code_, name_ + " scripted failure"}; return; }
    out.field.assign(kCells, fill_);
    out.tmax = fill_;
    out.status = {StatusCode::Ok, ""};
  }
  int calls = 0;

 private:
  Source source_;
  std::string name_;
  StatusCode code_;
  float fill_;
};

struct Fixture : ::testing::Test {
  Validator validator{kStats, kBank};
  npy::Array in4 = npy::load_f32(kFix + "sample_input.npy");
  npy::Array in5 = npy::load_f32(kFix + "sample_input_level5.npy");
};
}  // namespace

TEST_F(Fixture, InSupportGoesToSurrogate) {
  auto s = std::make_unique<FakeBackend>(Source::Surrogate, "sur", StatusCode::Ok, 350.f);
  auto r = std::make_unique<FakeBackend>(Source::ReferenceInProcess, "ref", StatusCode::Ok, 400.f);
  auto* sp = s.get(); auto* rp = r.get();
  std::vector<std::unique_ptr<ISolverBackend>> v; v.push_back(std::move(s)); v.push_back(std::move(r));
  Dispatcher d(validator, std::move(v));
  Output out; Dispatcher::Trace tr;
  d.solve(Input{in4.data.data(), kInputShape}, out, tr);
  EXPECT_TRUE(out.status.ok());
  EXPECT_EQ(out.source, Source::Surrogate);
  EXPECT_FALSE(out.oob_flag);
  EXPECT_EQ(sp->calls, 1); EXPECT_EQ(rp->calls, 0);
  EXPECT_EQ(tr.tried, (std::vector<std::string>{"sur"}));
}

TEST_F(Fixture, OutOfDistributionSkipsSurrogate) {
  // Rule F1: the level5 fixture must never reach the surrogate.
  auto s = std::make_unique<FakeBackend>(Source::Surrogate, "sur", StatusCode::Ok, 350.f);
  auto r = std::make_unique<FakeBackend>(Source::ReferenceInProcess, "ref", StatusCode::Ok, 400.f);
  auto* sp = s.get();
  std::vector<std::unique_ptr<ISolverBackend>> v; v.push_back(std::move(s)); v.push_back(std::move(r));
  Dispatcher d(validator, std::move(v));
  Output out; Dispatcher::Trace tr;
  d.solve(Input{in5.data.data(), kInputShape}, out, tr);
  EXPECT_TRUE(out.status.ok());
  EXPECT_EQ(out.source, Source::ReferenceInProcess);
  EXPECT_TRUE(out.oob_flag);
  EXPECT_FALSE(out.oob_channels.empty());
  EXPECT_EQ(out.oob_channels.back(), "layout");
  EXPECT_EQ(sp->calls, 0);
  EXPECT_TRUE(out.model_version.empty());
}

TEST_F(Fixture, InferenceFailureFallsThrough) {
  // Rule F3: a failing surrogate is skipped; the external reference answers.
  std::vector<std::unique_ptr<ISolverBackend>> v;
  v.push_back(std::make_unique<FakeBackend>(Source::Surrogate, "sur", StatusCode::InferenceFailed));
  v.push_back(std::make_unique<FakeBackend>(Source::ReferenceInProcess, "ref", StatusCode::ReferenceFailed));
  v.push_back(std::make_unique<FakeBackend>(Source::ReferenceExternal, "ext", StatusCode::Ok, 420.f));
  Dispatcher d(validator, std::move(v));
  Output out; Dispatcher::Trace tr;
  d.solve(Input{in4.data.data(), kInputShape}, out, tr);
  EXPECT_TRUE(out.status.ok());
  EXPECT_EQ(out.source, Source::ReferenceExternal);
  EXPECT_EQ(tr.tried, (std::vector<std::string>{"sur", "ref", "ext"}));
  EXPECT_EQ(tr.failures.size(), 2u);
}

TEST_F(Fixture, AllBackendsFailReportsLastFailure) {
  // Rule F4.
  std::vector<std::unique_ptr<ISolverBackend>> v;
  v.push_back(std::make_unique<FakeBackend>(Source::Surrogate, "sur", StatusCode::InferenceFailed));
  v.push_back(std::make_unique<FakeBackend>(Source::ReferenceExternal, "ext", StatusCode::Timeout));
  Dispatcher d(validator, std::move(v));
  Output out;
  d.solve(Input{in4.data.data(), kInputShape}, out);
  EXPECT_EQ(out.status.code, StatusCode::Timeout);
  EXPECT_EQ(out.source, Source::None);
  EXPECT_TRUE(out.field.empty());
}

TEST_F(Fixture, HardRejectRunsNothing) {
  auto s = std::make_unique<FakeBackend>(Source::Surrogate, "sur", StatusCode::Ok);
  auto* sp = s.get();
  std::vector<std::unique_ptr<ISolverBackend>> v; v.push_back(std::move(s));
  Dispatcher d(validator, std::move(v));
  Output out;
  d.solve(Input{in4.data.data(), {64, 64, 1, 3}}, out);
  EXPECT_EQ(out.status.code, StatusCode::ChannelMismatch);
  EXPECT_EQ(sp->calls, 0);
}

TEST_F(Fixture, RealSurrogateBackendEndToEnd) {
  Predictor pred(TS_MODEL_PATH, kMeta);
  std::vector<std::unique_ptr<ISolverBackend>> v;
  v.push_back(std::make_unique<SurrogateBackend>(pred));
  Dispatcher d(validator, std::move(v));
  Output out;
  d.solve(Input{in4.data.data(), kInputShape}, out);
  ASSERT_TRUE(out.status.ok()) << out.status.message;
  EXPECT_EQ(out.source, Source::Surrogate);
  EXPECT_EQ(out.model_version, "level4-unet-1.0.0");
  EXPECT_EQ(out.field.size(), kCells);
  // The level5 fixture has no reference backend here: OOD with nothing to fall back to.
  d.solve(Input{in5.data.data(), kInputShape}, out);
  EXPECT_FALSE(out.status.ok());
  EXPECT_TRUE(out.oob_flag);
}
