// src/dispatcher.cpp -- see dispatcher.h
#include "thermal_surrogate/dispatcher.h"

#include <chrono>
#include <cmath>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace thermal_surrogate {

ConformalTable::ConformalTable(const std::string& path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("cannot open conformal table: " + path);
  nlohmann::json j;
  f >> j;
  model_version_ = j.value("model_version", "");
  for (const auto& b : j.at("rise_bins")) {
    Bin bin;
    bin.lo = b.at("rise_lo_K").is_null() ? -std::numeric_limits<double>::infinity() : b.at("rise_lo_K").get<double>();
    bin.hi = b.at("rise_hi_K").is_null() ? std::numeric_limits<double>::infinity() : b.at("rise_hi_K").get<double>();
    bin.mae_q90 = b.at("mae_q90_K").get<double>();
    bin.tmax_q90 = b.at("tmax_err_q90_K").get<double>();
    bins_.push_back(bin);
  }
  if (bins_.empty()) throw std::runtime_error("conformal table has no bins");
}

void ConformalTable::bounds(double rise_K, float& mae_bound, float& tmax_bound) const {
  for (const auto& b : bins_)
    if (rise_K >= b.lo && rise_K < b.hi) {
      mae_bound = static_cast<float>(b.mae_q90);
      tmax_bound = static_cast<float>(b.tmax_q90);
      return;
    }
  mae_bound = static_cast<float>(bins_.back().mae_q90);     // numerically out of every bin (NaN): widest bin
  tmax_bound = static_cast<float>(bins_.back().tmax_q90);
}

Dispatcher::Dispatcher(const Validator& validator, std::vector<std::unique_ptr<ISolverBackend>> backends)
    : validator_(validator), backends_(std::move(backends)) {}

void Dispatcher::solve(const Input& in, Output& out) const noexcept {
  Trace trace;
  solve(in, out, trace);
}

void Dispatcher::solve(const Input& in, Output& out, Trace& trace) const noexcept {
  const auto t0 = std::chrono::steady_clock::now();
  trace = Trace{};
  trace.report = validator_.validate(in);
  const auto& rep = trace.report;

  // 1. Hard rejects: nothing runs, the caller gets the rule that failed.
  if (!rep.status.ok() && !rep.oob()) {
    out = Output{};
    out.status = rep.status;
    return;
  }

  // 2./3. Trial order: skip surrogates when out of distribution.
  Status last{StatusCode::ModelNotLoaded, "no backend available"};
  for (const auto& b : backends_) {
    if (rep.oob() && b->source() == Source::Surrogate) continue;        // rule F1
    trace.tried.push_back(b->name());
    Output attempt;
    b->solve(in, attempt);
    if (attempt.status.ok()) {
      out = std::move(attempt);
      out.source = b->source();
      break;
    }
    trace.failures.push_back(b->name() + ": " + attempt.status.message);   // rule F3
    last = attempt.status;
  }
  if (trace.tried.empty() || !out.status.ok() || out.source == Source::None) {
    // Nothing succeeded (rule F4): report the last failure, keep the audit fields.
    out = Output{};
    out.status = last;
  }

  // Error bounds: only for surrogate answers on in-support inputs (the
  // conformal guarantee needs exchangeability with the calibration set).
  if (out.status.ok() && out.source == Source::Surrogate && conformal_.loaded() && !out.field.empty()) {
    const double ambient = in.data[static_cast<std::size_t>(Channel::AmbientK)];   // constant per sample
    conformal_.bounds(static_cast<double>(out.tmax) - ambient, out.mae_bound_90_K, out.tmax_err_bound_90_K);
  }

  // Audit fields are filled whichever way it went.
  out.oob_flag = rep.oob();
  out.oob_channels = rep.oob_channels;
  if (!out.status.ok() || out.source != Source::Surrogate) out.model_version.clear();
  out.latency_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void Dispatcher::solve_power_scaled(const Input& base, double factor, Output& out) const noexcept {
  if (!(factor > 0.0) || !std::isfinite(factor)) {
    out = Output{};
    out.status = {StatusCode::InvalidInput, "power factor must be a positive finite number"};
    return;
  }
  solve(base, out);
  if (!out.status.ok()) return;
  const double ambient = base.data[static_cast<std::size_t>(Channel::AmbientK)];
  const auto f = static_cast<float>(factor);
  const auto amb = static_cast<float>(ambient);
  for (auto& v : out.field) v = amb + f * (v - amb);         // exact linear response
  out.tmax = amb + f * (out.tmax - amb);                      // argmax is unchanged by a positive scale
  out.mae_bound_90_K *= f;                                    // bounds are on the rise, so they scale too
  out.tmax_err_bound_90_K *= f;
}

}  // namespace thermal_surrogate
