// thermal_surrogate/backend.h -- the interface every solver implements
//
// A backend turns one Input into one Output. The dispatcher (S5.1) owns a
// list of them and tries them in contract order: surrogate, in-process
// reference, external reference. Keeping the interface this small is what
// lets a 2-D finite-difference solver, a HotSpot subprocess and an ONNX
// session be interchangeable in tests.
#pragma once

#include <string>

#include "thermal_surrogate/types.h"

namespace thermal_surrogate {

class ISolverBackend {
 public:
  virtual ~ISolverBackend() = default;

  // Stable identifier used in logs and in Output::source.
  virtual Source source() const = 0;
  virtual std::string name() const = 0;

  // Must never throw: every failure is reported through out.status.
  // `in` has already passed validation (I1-I3) when the dispatcher calls
  // this; I4/I5 are the dispatcher's business, not the backend's.
  virtual void solve(const Input& in, Output& out) noexcept = 0;
};

}  // namespace thermal_surrogate
