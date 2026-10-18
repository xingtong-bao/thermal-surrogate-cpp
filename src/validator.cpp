// src/validator.cpp -- see validator.h. The feature extraction mirrors
// tools/layout_novelty.py::config_features line by line; the Python script
// tools/export_config_bank.py writes the expected values for the fixtures
// and tests/test_validator.cpp asserts the C++ numbers match them.
#include "thermal_surrogate/validator.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <stdexcept>

#include "thermal_surrogate/npy.h"

namespace thermal_surrogate {
namespace {

inline float at(const float* d, std::size_t x, std::size_t y, Channel c) {
  // (X, Y, Z=1, P) C order: index = ((x * kY + y) * kZ + 0) * kP + p
  return d[(x * kY + y) * kP + static_cast<std::size_t>(c)];
}

// Largest positive step of a coordinate map along either axis. The map varies
// along exactly one axis (contract section 2), so max over both is that step.
double spacing(const float* d, Channel c) {
  double best = 0.0;
  for (std::size_t x = 0; x < kX; ++x)
    for (std::size_t y = 0; y < kY; ++y) {
      if (x + 1 < kX) best = std::max(best, static_cast<double>(at(d, x + 1, y, c) - at(d, x, y, c)));
      if (y + 1 < kY) best = std::max(best, static_cast<double>(at(d, x, y + 1, c) - at(d, x, y, c)));
    }
  return best;
}

}  // namespace

ConfigFeatures Validator::compute_features(const float* d) {
  ConfigFeatures f;
  // Chip extent and cell area from the coordinate channels.
  double gx_max = 0, gy_max = 0;
  for (std::size_t i = 0; i < kCells; ++i) {
    gx_max = std::max(gx_max, static_cast<double>(d[i * kP + 1]));
    gy_max = std::max(gy_max, static_cast<double>(d[i * kP + 2]));
  }
  f.chip_x_mm = gx_max;
  f.chip_y_mm = gy_max;
  const double cell_mm2 = spacing(d, Channel::GridX) * spacing(d, Channel::GridY);

  // Connected components (4-neighbourhood) of cells sharing the same non-zero
  // power value: one component = one chiplet. Flood fill with an explicit stack.
  std::vector<int> label(kCells, -1);
  std::vector<std::size_t> stack;
  int n_comp = 0;
  double total = 0, max_density = 0, area_sum = 0;
  for (std::size_t start = 0; start < kCells; ++start) {
    const float val = d[start * kP + 0];
    if (val <= 0.0f || label[start] >= 0) continue;
    std::size_t cells = 0;
    stack.push_back(start);
    label[start] = n_comp;
    while (!stack.empty()) {
      const std::size_t i = stack.back();
      stack.pop_back();
      ++cells;
      const std::size_t x = i / kY, y = i % kY;
      const std::size_t nb[4][2] = {{x + 1, y}, {x > 0 ? x - 1 : kX, y}, {x, y + 1}, {x, y > 0 ? y - 1 : kY}};
      for (const auto& n : nb) {
        if (n[0] >= kX || n[1] >= kY) continue;
        const std::size_t j = n[0] * kY + n[1];
        if (label[j] < 0 && d[j * kP + 0] == val) {
          label[j] = n_comp;
          stack.push_back(j);
        }
      }
    }
    const double area = static_cast<double>(cells) * cell_mm2;
    total += static_cast<double>(val);               // one 'val' per chiplet, not per cell
    max_density = std::max(max_density, static_cast<double>(val) / area);
    area_sum += area;
    ++n_comp;
  }
  f.n_chiplets = n_comp;
  f.total_power_w = total;
  f.max_power_density_w_mm2 = max_density;
  f.mean_chiplet_area_mm2 = n_comp ? area_sum / n_comp : 0.0;
  return f;
}

Validator::Validator(const std::string& stats_path, const std::string& bank_path) {
  std::ifstream sf(stats_path);
  if (!sf) throw std::runtime_error("cannot open stats: " + stats_path);
  nlohmann::json s;
  sf >> s;
  const auto& ch = s.at("channels");
  if (ch.size() != kP) throw std::runtime_error("stats.json channel count != contract P");
  for (std::size_t i = 0; i < kP; ++i) {
    if (ch[i].at("name").get<std::string>() != kChannelNames[i])
      throw std::runtime_error("stats.json channel order differs from the contract");
    min_[i] = ch[i].at("min").get<double>();
    max_[i] = ch[i].at("max").get<double>();
  }
  if (s.at("oob_scale").is_null()) throw std::runtime_error("stats.json oob_scale is not decided");
  oob_scale_ = s.at("oob_scale").get<double>();
  const auto& ln = s.at("layout_novelty");
  for (std::size_t i = 0; i < 6; ++i) {
    mean_[i] = ln.at("train_mean")[i].get<double>();
    std_[i] = ln.at("train_std")[i].get<double>();
  }
  threshold_ = ln.at("threshold").get<double>();

  auto bank = npy::load_f32(bank_path);
  if (bank.shape.size() != 2 || bank.shape[1] != 6) throw std::runtime_error("config bank must be (n, 6)");
  bank_ = std::move(bank.data);
  bank_z_.resize(bank_.size());
  for (std::size_t r = 0; r < bank_.size() / 6; ++r)
    for (std::size_t c = 0; c < 6; ++c)
      bank_z_[r * 6 + c] = (static_cast<double>(bank_[r * 6 + c]) - mean_[c]) / std_[c];
}

ValidationReport Validator::validate(const Input& in) const noexcept {
  ValidationReport r;
  // I1 / I2 / I3 -- hard rejects, in contract order.
  if (in.data == nullptr) { r.status = {StatusCode::InvalidInput, "input data pointer is null"}; return r; }
  if (in.shape[0] != kX || in.shape[1] != kY || in.shape[2] != kZ) {
    r.status = {StatusCode::ShapeMismatch, "input shape is not (64, 64, 1, P)"}; return r;
  }
  if (in.shape[3] != kP) {
    r.status = {StatusCode::ChannelMismatch, "input has " + std::to_string(in.shape[3]) + " channels, model expects " + std::to_string(kP)};
    return r;
  }
  for (std::size_t i = 0; i < kInputSize; ++i)
    if (!std::isfinite(in.data[i])) { r.status = {StatusCode::InvalidInput, "NaN or Inf at flat index " + std::to_string(i)}; return r; }

  // I4 -- per-channel range with margin oob_scale * range.
  std::array<double, kP> lo{}, hi{};
  for (std::size_t c = 0; c < kP; ++c) { lo[c] = in.data[c]; hi[c] = in.data[c]; }
  for (std::size_t i = 1; i < kCells; ++i)
    for (std::size_t c = 0; c < kP; ++c) {
      const double v = in.data[i * kP + c];
      lo[c] = std::min(lo[c], v); hi[c] = std::max(hi[c], v);
    }
  for (std::size_t c = 0; c < kP; ++c) {
    const double range = max_[c] - min_[c];
    const double margin = oob_scale_ * range;
    const double d = std::max({0.0, (min_[c] - margin - lo[c]) / range, (hi[c] - max_[c] - margin) / range});
    r.oob_distance[c] = d;
    if (d > 0.0) r.oob_channels.emplace_back(kChannelNames[c]);
  }

  // I5 -- nearest training configuration in standardized feature space.
  r.features = compute_features(in.data);
  const auto f = r.features.as_array();
  std::array<double, 6> z{};
  for (std::size_t c = 0; c < 6; ++c) z[c] = (f[c] - mean_[c]) / std_[c];
  double best = std::numeric_limits<double>::infinity();
  for (std::size_t row = 0; row < bank_z_.size() / 6; ++row) {
    double d2 = 0.0;
    for (std::size_t c = 0; c < 6; ++c) { const double t = z[c] - bank_z_[row * 6 + c]; d2 += t * t; }
    best = std::min(best, d2);
  }
  r.novelty_score = std::sqrt(best);
  if (r.novelty_score > threshold_) r.oob_channels.emplace_back("layout");

  if (!r.oob_channels.empty()) {
    std::string why = "out of distribution:";
    for (const auto& n : r.oob_channels) why += " " + n;
    r.status = {StatusCode::OutOfDistribution, why};
  } else {
    r.status = {StatusCode::Ok, ""};
  }
  return r;
}

}  // namespace thermal_surrogate
