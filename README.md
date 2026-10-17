# thermal-surrogate-cpp

Contract-driven C++ packaging of a pre-trained 3D-IC thermal surrogate model:
interface contract → ONNX export → C++ inference library (ONNX Runtime) →
fallback-aware solver integration layer → numerical-parity and accuracy gates → CI.

The surrogate and data come from the public
[IC-ThermBench](https://github.com/Day333/ThermalBench) benchmark (code MIT, data CC BY 4.0).
No model is trained here; the work is the production wrapping around a released model.

> Status: work in progress. Every number that will appear in this README is produced by a
> committed script and stored under `docs/results/`.

## Layout

```
contract/     interface contract (contract.md), input/output schema (schema.json), training-range stats (stats.json)
tools/        Python tooling: asset fetching, data loading, baseline evaluation, ONNX export, parity checks, failure analysis
models/       model metadata (committed); exported ONNX files are downloaded/exported locally and git-ignored
include/      public C++ headers (C++ API and C ABI)
src/          library sources: predictor, validation, backends, dispatcher, reference FD solver
tests/        GoogleTest suites and small parity fixtures
bench/        latency / throughput / memory-layout benchmarks
examples/     C++ demo, minimal C caller
docs/         design notes, failure analysis, measured results
```

## License

MIT for the code in this repository. Benchmark data and checkpoints retain their upstream
licenses (see the IC-ThermBench repository).
