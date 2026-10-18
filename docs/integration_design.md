# Integration design (one page)

How `thermal_surrogate` is meant to sit inside a host application (an EDA
tool, a design script, a service). Every rule below is implemented in the
code named next to it.

## Ownership

- **Input is a non-owning view.** `Input{data, shape}` points at the caller's
  float32 buffer; the library never copies, resizes or frees it
  (`include/thermal_surrogate/types.h`). The ONNX tensor wraps the same memory
  (`src/predictor.cpp`, `CreateTensor` over `in.data`).
- **Output is the caller's choice.** `predict()` returns an owning
  `std::vector<float>`; `predict_into()` writes into a buffer the caller
  provides and allocates nothing per call. Batches use caller buffers only.
- **Model files are read once, at construction.** Nothing is re-read or
  cached globally; two engines for two models are two objects.

## Lifetime

- One `Predictor` (one ONNX session) per model per document/tool instance;
  construction costs 8–30 ms, prediction 20–30 ms, so sessions are kept, not
  recreated. Destruction releases the runtime; destroy from the thread that
  created it, after the last prediction has returned.
- `Dispatcher` and `SurrogateBackend` hold references, not ownership, to the
  `Validator` and `Predictor`: the host decides who outlives whom and must
  keep them alive while the dispatcher exists (`ts_engine` in
  `src/c_api.cpp` shows the one safe ordering).

## Threading

- `Predictor::predict*` and `Validator::validate` are `const` and safe to
  call concurrently on one object (ONNX Runtime's `Session::Run` is
  thread-safe; every call has its own `Output`). Verified in S6.2 under
  ThreadSanitizer with 8 threads × 100 predictions, outputs bit-identical.
- No globals, no static mutable state. Thread count is per session
  (`PredictorOptions::intra_op_threads`); the benchmark fixes it to 4 on the
  M4 for reproducibility, and batching across samples does not help on CPU
  (S4.5), so concurrency should come from calling threads, not from batches.

## Error propagation

- Constructors throw `std::runtime_error` (missing file, metadata mismatch,
  runtime failure) because a half-built object is worse than no object.
- After construction nothing throws: `predict`, `validate`, `solve` are
  `noexcept` and report through `Status{code, message}` using the nine
  contract codes (`contract/contract.md` §6, spelled identically in
  `schema.json`, `types.h` and the C ABI).
- Fallback is explicit and auditable: `Output::source`, `oob_flag`,
  `oob_channels` and `Dispatcher::Trace` say which backend answered and why
  the others did not.

## ABI

- `include/thermal_surrogate/ts_c_api.h` is a C header (compiled by `cc` in
  `examples/c_caller.c`): opaque `ts_engine*`, integer status codes, caller
  buffers, fixed-size strings in `ts_result`. No exceptions, no `std::`,
  no ONNX Runtime types cross it. This is what a plugin loader, a Python
  `ctypes` user or another compiler's build links against; the C++ headers
  are for in-tree integrations.
- The C++ public headers include no ONNX Runtime header; the runtime is a
  private implementation detail behind a pimpl, so upgrading it does not
  rebuild callers.

## Versioning

- `models/metadata.json → model_version` (bumped whenever the checkpoint or
  the graph changes) is checked at construction and echoed in every
  surrogate `Output`; `schema_version` ties the model to the contract.
- The library reports its own version and the ONNX Runtime it was built
  against (`version()`), and the test suite asserts the loaded runtime is
  that version. Incompatible files fail at construction, never silently.
- The reference solver's calibration (`models/fd_solver_params.json`)
  records the training subset it was fitted on; refitting is a versioned
  change, not a tweak.
