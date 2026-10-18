// thermal_surrogate/external_backend.h -- reference solver as a separate process
//
// Protocol (deliberately trivial so any solver can be wrapped):
//   1. the backend writes the input tensor to <tmpdir>/input.npy (64,64,1,7) float32;
//   2. it runs  <command> <tmpdir>/input.npy <tmpdir>/output.npy  with a wall-clock budget;
//   3. it reads <tmpdir>/output.npy (64,64,1) float32 in Kelvin.
// Exit code != 0, a missing/invalid output, or the budget expiring map to
// ReferenceFailed / Timeout. The process is killed on timeout so a hung
// solver never hangs the caller.
//
// tools/reference_cli.py implements this protocol around HotSpot's grid
// model (S5.3); tests use a stub script so the suite needs no HotSpot.
#pragma once

#include <string>
#include <vector>

#include "thermal_surrogate/backend.h"

namespace thermal_surrogate {

struct ExternalBackendOptions {
  std::vector<std::string> command;   // argv prefix, e.g. {"python3", "tools/reference_cli.py"}
  double timeout_s = 60.0;            // contract: configurable budget (section 8)
  std::string workdir;                // where temp dirs are made; empty = system temp
  bool keep_files = false;            // leave the temp dir for debugging
};

class ExternalReferenceBackend final : public ISolverBackend {
 public:
  explicit ExternalReferenceBackend(ExternalBackendOptions o) : opt_(std::move(o)) {}
  Source source() const override { return Source::ReferenceExternal; }
  std::string name() const override { return "external:" + (opt_.command.empty() ? std::string("?") : opt_.command.back()); }
  void solve(const Input& in, Output& out) noexcept override;

 private:
  ExternalBackendOptions opt_;
};

}  // namespace thermal_surrogate
