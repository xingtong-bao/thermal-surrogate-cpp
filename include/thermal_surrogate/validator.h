// thermal_surrogate/validator.h -- contract rules I1..I5 in one place
//
// The Validator is the gatekeeper in front of every backend. It answers one
// question: "may the surrogate be trusted with this input?" It never runs
// the model. Rules (contract section 2):
//   I1  float32, no NaN/Inf                        -> InvalidInput
//   I2  shape exactly (64, 64, 1, 7)               -> ShapeMismatch
//   I3  channel count == model input_channels      -> ChannelMismatch
//   I4  every channel within [min - m, max + m], m = oob_scale * range
//                                                  -> OutOfDistribution (fallback)
//   I5  configuration-novelty score <= threshold   -> OutOfDistribution (fallback)
// I4/I5 parameters come from contract/stats.json; the I5 bank of training
// configuration vectors from models/config_bank.npy. Both are loaded once.
#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "thermal_surrogate/types.h"

namespace thermal_surrogate {

// The six numbers rule I5 looks at (contract section 2, stats.json -> layout_novelty.features).
struct ConfigFeatures {
  double chip_x_mm = 0, chip_y_mm = 0, n_chiplets = 0, total_power_w = 0,
         max_power_density_w_mm2 = 0, mean_chiplet_area_mm2 = 0;
  std::array<double, 6> as_array() const {
    return {chip_x_mm, chip_y_mm, n_chiplets, total_power_w, max_power_density_w_mm2, mean_chiplet_area_mm2};
  }
};

// Everything the validator found out, whether or not it rejected the input.
struct ValidationReport {
  Status status;                                    // Ok or the first failing rule's code
  std::vector<std::string> oob_channels;            // I4 violators by name, plus "layout" for I5
  std::array<double, kP> oob_distance{};            // per channel, fraction of training range (0 = inside)
  ConfigFeatures features;                          // what I5 computed
  double novelty_score = 0.0;                       // distance to nearest training configuration
  bool oob() const { return status.code == StatusCode::OutOfDistribution; }
};

class Validator {
 public:
  // Throws std::runtime_error if a file is missing or inconsistent with the contract.
  Validator(const std::string& stats_json_path, const std::string& config_bank_npy_path);

  // Never throws. Runs I1..I5 in order and stops at the first hard failure;
  // I4 and I5 are both evaluated (so the report lists every reason) and
  // yield OutOfDistribution together.
  ValidationReport validate(const Input& in) const noexcept;

  // Exposed for tests and for the Python bridge.
  static ConfigFeatures compute_features(const float* data);   // data: (64,64,1,7) C order
  double oob_scale() const { return oob_scale_; }
  double novelty_threshold() const { return threshold_; }
  std::size_t bank_size() const { return bank_.size() / 6; }

 private:
  std::array<double, kP> min_{}, max_{};
  double oob_scale_ = 0.0;
  std::array<double, 6> mean_{}, std_{};
  double threshold_ = 0.0;
  std::vector<float> bank_;                         // (n, 6) training configuration vectors, raw units
  std::vector<double> bank_z_;                      // same, standardized, for the search
};

}  // namespace thermal_surrogate
