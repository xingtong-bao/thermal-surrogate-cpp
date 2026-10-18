// src/dispatcher.cpp -- see dispatcher.h
#include "thermal_surrogate/dispatcher.h"

#include <chrono>

namespace thermal_surrogate {

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

  // Audit fields are filled whichever way it went.
  out.oob_flag = rep.oob();
  out.oob_channels = rep.oob_channels;
  if (!out.status.ok() || out.source != Source::Surrogate) out.model_version.clear();
  out.latency_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

}  // namespace thermal_surrogate
