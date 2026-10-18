// thermal_surrogate/dispatcher.h -- the integration layer: validate, choose a backend, fall back
//
// One call does what contract section 5 prescribes:
//   1. Validator: I1-I3 hard reject -> return the status, nothing runs.
//   2. I4/I5 out of distribution -> the surrogate is skipped (rule F1),
//      reference backends are tried in order.
//   3. Otherwise backends are tried in the order given: surrogate first,
//      then in-process reference, then external reference (rule F2).
//   4. A backend that reports a failure (InferenceFailed, ReferenceFailed,
//      Timeout) is skipped and the next one is tried (rule F3); the last
//      failure is reported if nothing succeeds (rule F4).
// The Output always says which backend answered (`source`) and whether the
// input was out of distribution (`oob_flag`, `oob_channels`), so a caller can
// log and audit every decision.
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "thermal_surrogate/backend.h"
#include "thermal_surrogate/predictor.h"
#include "thermal_surrogate/validator.h"

namespace thermal_surrogate {

// ISolverBackend over a Predictor. Owns nothing but a reference: the
// Predictor's lifetime is the integrator's business (one per document).
class SurrogateBackend final : public ISolverBackend {
 public:
  explicit SurrogateBackend(const Predictor& predictor) : predictor_(predictor) {}
  Source source() const override { return Source::Surrogate; }
  std::string name() const override { return "surrogate:" + predictor_.model_version(); }
  void solve(const Input& in, Output& out) noexcept override { predictor_.predict(in, out); }

 private:
  const Predictor& predictor_;
};

class Dispatcher {
 public:
  // `backends` in trial order. The Validator is shared (read-only, thread-safe).
  Dispatcher(const Validator& validator, std::vector<std::unique_ptr<ISolverBackend>> backends);

  // Never throws. See the header comment for the decision sequence.
  void solve(const Input& in, Output& out) const noexcept;

  // What the last decision was about, for tests and logs: the backends tried, in order.
  struct Trace {
    ValidationReport report;
    std::vector<std::string> tried;       // backend names in the order they ran
    std::vector<std::string> failures;    // "<name>: <status message>" for each skipped backend
  };
  void solve(const Input& in, Output& out, Trace& trace) const noexcept;

  std::size_t backend_count() const { return backends_.size(); }

 private:
  const Validator& validator_;
  std::vector<std::unique_ptr<ISolverBackend>> backends_;
};

}  // namespace thermal_surrogate
