// tests/test_validator.cpp -- S4.3: every status code, and the two fixtures end where the contract says
#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>

#include "thermal_surrogate/npy.h"
#include "thermal_surrogate/validator.h"

using namespace thermal_surrogate;

namespace {
const std::string kFix = std::string(TS_FIXTURE_DIR) + "/";
const std::string kStats = std::string(TS_CONTRACT_DIR) + "/stats.json";
const std::string kBank = std::string(TS_MODELS_DIR) + "/config_bank.npy";

nlohmann::json expected() {
  nlohmann::json j;
  std::ifstream(kFix + "validator_expected.json") >> j;
  return j;
}
}  // namespace

TEST(Validator, LoadsContractParameters) {
  Validator v(kStats, kBank);
  EXPECT_EQ(v.bank_size(), 10800u);
  EXPECT_DOUBLE_EQ(v.oob_scale(), expected()["oob_scale"].get<double>());
  EXPECT_DOUBLE_EQ(v.novelty_threshold(), expected()["threshold"].get<double>());
  EXPECT_THROW(Validator("/nonexistent.json", kBank), std::runtime_error);
  EXPECT_THROW(Validator(kStats, "/nonexistent.npy"), std::runtime_error);
}

TEST(Validator, FeaturesMatchPython) {
  // The C++ feature extraction must reproduce tools/layout_novelty.py exactly.
  const auto e = expected();
  for (const auto& name : {"sample_input", "sample_input_level5"}) {
    const auto x = npy::load_f32(kFix + name + ".npy");
    const auto f = Validator::compute_features(x.data.data()).as_array();
    const auto want = e["fixtures"][name]["features"].get<std::vector<double>>();
    for (std::size_t i = 0; i < 6; ++i)
      EXPECT_NEAR(f[i], want[i], 1e-6 * std::max(1.0, std::fabs(want[i]))) << name << " feature " << i;
  }
}

TEST(Validator, InSupportFixtureIsOk) {
  Validator v(kStats, kBank);
  const auto x = npy::load_f32(kFix + "sample_input.npy");
  const auto r = v.validate(Input{x.data.data(), kInputShape});
  EXPECT_EQ(r.status.code, StatusCode::Ok) << r.status.message;
  EXPECT_TRUE(r.oob_channels.empty());
  EXPECT_NEAR(r.novelty_score, expected()["fixtures"]["sample_input"]["novelty_score"].get<double>(), 1e-6);
}

TEST(Validator, UnseenLayoutFixtureIsOutOfDistribution) {
  // level5 sample: out of range in grid_x, grid_y, r_convec AND a novel configuration.
  Validator v(kStats, kBank);
  const auto e = expected()["fixtures"]["sample_input_level5"];
  const auto x = npy::load_f32(kFix + "sample_input_level5.npy");
  const auto r = v.validate(Input{x.data.data(), kInputShape});
  EXPECT_EQ(r.status.code, StatusCode::OutOfDistribution);
  EXPECT_TRUE(r.oob());
  std::vector<std::string> want = e["oob_channels"].get<std::vector<std::string>>();
  want.push_back("layout");
  EXPECT_EQ(r.oob_channels, want);
  EXPECT_NEAR(r.novelty_score, e["novelty_score"].get<double>(), 1e-6);
  EXPECT_GT(r.novelty_score, v.novelty_threshold());
  const auto d = e["oob_distance_per_channel"].get<std::vector<double>>();
  for (std::size_t c = 0; c < kP; ++c) EXPECT_NEAR(r.oob_distance[c], d[c], 1e-9) << "channel " << c;
}

TEST(Validator, HardRejectsInContractOrder) {
  Validator v(kStats, kBank);
  auto x = npy::load_f32(kFix + "sample_input.npy");
  EXPECT_EQ(v.validate(Input{nullptr, kInputShape}).status.code, StatusCode::InvalidInput);
  EXPECT_EQ(v.validate(Input{x.data.data(), {32, 64, 1, 7}}).status.code, StatusCode::ShapeMismatch);
  EXPECT_EQ(v.validate(Input{x.data.data(), {64, 64, 1, 6}}).status.code, StatusCode::ChannelMismatch);
  x.data[7 * 100 + 4] = INFINITY;
  EXPECT_EQ(v.validate(Input{x.data.data(), kInputShape}).status.code, StatusCode::InvalidInput);
}

TEST(Validator, RangeViolationNamesTheChannel) {
  // Push ambient_K 1 K above its training max on an otherwise in-support sample.
  Validator v(kStats, kBank);
  auto x = npy::load_f32(kFix + "sample_input.npy");
  for (std::size_t i = 0; i < kCells; ++i) x.data[i * kP + 4] = 328.141f + 1.0f;
  const auto r = v.validate(Input{x.data.data(), kInputShape});
  EXPECT_EQ(r.status.code, StatusCode::OutOfDistribution);
  ASSERT_FALSE(r.oob_channels.empty());
  EXPECT_EQ(r.oob_channels[0], "ambient_K");
  EXPECT_GT(r.oob_distance[4], 0.0);
}
