# thermal-surrogate-cpp

[![ci](https://github.com/xingtong-bao/thermal-surrogate-cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/xingtong-bao/thermal-surrogate-cpp/actions/workflows/ci.yml)
![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)
![ONNX Runtime 1.23.2](https://img.shields.io/badge/ONNX%20Runtime-1.23.2-lightgrey)
![License MIT](https://img.shields.io/badge/license-MIT-green)

A C++ inference library that wraps a published neural thermal surrogate
(the IC-ThermBench level4 U-Net) for early-stage chip floorplanning, and
tells you **when not to trust it**. The surrogate predicts a 64×64
steady-state temperature field in ~22 ms; the library validates every input
against a written contract, detects inputs the model has never seen, and
falls back to a physics reference solver instead of returning a confident
wrong answer. Every number below was measured in this repository and links
to the file that holds it.

![architecture](docs/figures/architecture.png)

## What it does, in numbers

| | seen chips (level4 test) | unseen chips (level5) |
|---|---|---|
| surrogate alone, MAE vs ground truth | **1.05 K** (50 samples), 1.83 K (full 3 000) | 14.8 K |
| library with fallback (dispatcher) | **1.05 K**, 22 ms, 0 % fallback | **3.36 K**, 2.0 s, 100 % fallback to HotSpot |
| what catches the unseen chips | rule I5 (configuration novelty): rejects 100 % of unseen chips, 1 % of seen | |

Source: [backend_comparison.csv](docs/results/backend_comparison.csv),
[fallback_curve.json](docs/results/fallback_curve.json),
[layout_novelty.json](docs/results/layout_novelty.json).

![results](docs/figures/results_summary.png)

## Three findings worth knowing before you use any thermal surrogate

1. **The model learned ten chips, not heat conduction.** Scaling every
   chiplet's power by 10 % should move the hotspot by 6.3 K; the surrogate
   moves it by 0.6 K (gain 0.1, the same at ±2 %). The ambient temperature,
   by contrast, is reproduced exactly (gain 1.0 even 50 % outside the
   training range). [sensitivity_notes.md](docs/results/sensitivity_notes.md)
2. **A per-channel range check cannot see a new layout.** 80 % of the unseen
   chips lie inside every channel's training range and still carry 14 K of
   error (gate 2.0 K). A six-number configuration vector (chip size, chiplet
   count, total power, max density, mean chiplet area) separates seen from
   unseen chips perfectly (AUROC 1.0). [extrapolation_notes.md](docs/results/extrapolation_notes.md)
3. **Batching does not help on CPU, and layout does not matter.** Per-sample
   cost is flat from batch 8 upward; NCHW and the contract layout are within
   0.3 %; zero-copy output is a 1.7 % effect. Get concurrency from threads,
   not batches. [throughput_notes.md](docs/results/throughput_notes.md)

## Quick start

```bash
# prerequisites: cmake >= 3.24, ninja, a C++17 compiler, python3 (fixtures), internet (ONNX Runtime + models)
git clone https://github.com/xingtong-bao/thermal-surrogate-cpp && cd thermal-surrogate-cpp
gh release download models-v1 --dir models          # model.onnx + model_dynamic.onnx (20 MB each)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure        # 34 tests, ~20 s
./build/ts_demo --info                            # runtime version + model IO
./build/c_caller models contract tests/fixtures/sample_input.npy   # C ABI example
```

Without `gh`: download the two files from the
[models-v1 release](https://github.com/xingtong-bao/thermal-surrogate-cpp/releases/tag/models-v1)
and check them against the SHA-256 in [models/metadata.json](models/metadata.json),
or regenerate them with [models/README.md](models/README.md).

## Using the library

```cpp
#include "thermal_surrogate/dispatcher.h"
#include "thermal_surrogate/fd_solver.h"
using namespace thermal_surrogate;

Validator validator("contract/stats.json", "models/config_bank.npy");
Predictor predictor("models/model.onnx", "models/metadata.json");
std::vector<std::unique_ptr<ISolverBackend>> chain;
chain.push_back(std::make_unique<SurrogateBackend>(predictor));
chain.push_back(std::make_unique<FdSolverBackend>(load_fd_params("models/fd_solver_params.json")));
Dispatcher dispatcher(validator, std::move(chain));

Output out;                                   // never throws after construction
dispatcher.solve(Input{data /* 64*64*1*7 floats */, kInputShape}, out);
if (out.status.ok()) use(out.field, out.tmax, out.source, out.oob_flag);
```

Input is a non-owning view of your buffer; output is yours (`predict_into`
writes into a buffer you own). The C ABI (`ts_c_api.h`, opaque handle,
integer codes) is in [examples/c_caller.c](examples/c_caller.c). Rules for
ownership, lifetime, threading, errors and versioning:
[docs/integration_design.md](docs/integration_design.md).

## The contract

[contract/contract.md](contract/contract.md) is the human-readable interface
contract; [contract/schema.json](contract/schema.json) is its machine-readable
form (JSON Schema 2020-12, used by the Python checker and mirrored by the C++
enums); [contract/stats.json](contract/stats.json) holds every measured
constant the library reads (training ranges, normalization, the novelty
threshold). Every `TBD` the contract started with was filled by a later
measurement, never by hand; the parity gate's three revisions are kept in
§4 as a record.

| rule | what | on violation |
|---|---|---|
| I1–I3 | float32, finite, shape (64,64,1,7), 7 channels | hard reject (`InvalidInput`, `ShapeMismatch`, `ChannelMismatch`) |
| I4 | every channel inside its training range (`oob_scale` = 0) | `OutOfDistribution` → fallback |
| I5 | configuration vector within 0.091 std-units of a training chip | `OutOfDistribution` → fallback |
| F1–F4 | surrogate → external reference → in-process reference; skip failures; report the last | `source` says who answered |

## Results

**Accuracy** ([baseline_notes.md](docs/results/baseline_notes.md),
[parity_notes.md](docs/results/parity_notes.md))

| metric | value | file |
|---|---|---|
| baseline reproduction, level4 test (3 000): MAE / RMSE / Tmax err | 1.8293 / 2.5510 / 1.1855 K (authors: 1.8291 / 2.5508 / 1.1851) | [baseline_unet_level4.json](docs/results/baseline_unet_level4.json) |
| contract gates (baseline × 1.1): MAE / worst-5 % / Tmax median | 2.012 / 3.792 / 0.952 K, enforced by `AccuracyGate.*` | [baseline_unet_level4_per_sample.json](docs/results/baseline_unet_level4_per_sample.json) |
| PyTorch (batch 1) ↔ ONNX Runtime, 100 samples | max 8.2e-4 K, mean 3.5e-6 K, 88.5 % pixels bit-identical | [parity_python.json](docs/results/parity_python.json) |
| ONNX Runtime Python ↔ C++, 10 pairs | 0 K (bit-identical) | [parity_cpp.json](docs/results/parity_cpp.json) |
| hotspot *location*: argmax matches truth | 3.7 % of samples (informative only, no gate) | [baseline_unet_level4_per_sample.json](docs/results/baseline_unet_level4_per_sample.json) |

**Latency and throughput** (Apple M4, 4 P + 6 E cores, 16 GB, ONNX Runtime 1.23.2 CPU)

| measurement | value | file |
|---|---|---|
| single prediction, 1 / 4 / 8 threads, p50 | 79.9 / 30.3 / 27.2 ms (p95 80.8 / 33.0 / 34.1) | [latency_single.json](docs/results/latency_single.json) |
| session load / first call | 8–30 ms / 21–80 ms | same |
| batch 1 / 8 / 32 / 64, samples per second | 48.8 / 33.8 / 34.3 / 34.6 (no gain from batching) | [throughput.json](docs/results/throughput.json) |
| larger grids: 128² / 256² / 512² | 117 ms / 468 ms / 1.87 s, 7.14 ms per 1 k cells, 620 MB at 512² | [dynamic_size.csv](docs/results/dynamic_size.csv) |
| 8 threads × 100 predictions on one session | bit-identical, ThreadSanitizer clean | [concurrency.json](docs/results/concurrency.json) |

**Reference solvers** (both are reduced models of the generator; the dataset's
package configuration is not public)

| backend | seen chips MAE | unseen chips MAE | latency | file |
|---|---|---|---|---|
| in-process 2-D finite differences, 4 calibrated scalars | 12.8 K | 17.8 K | 34–41 ms | [fd_solver_validation.json](docs/results/fd_solver_validation.json) |
| HotSpot grid model via subprocess | 13.4 K | 3.4 K | 2–2.7 s | [hotspot_validation.json](docs/results/hotspot_validation.json) |

**Failure analysis**: [docs/failure_analysis.md](docs/failure_analysis.md)
(one page: trusted region, untrusted region, physical reason), with
[error_buckets_notes.md](docs/results/error_buckets_notes.md) and
[backend_comparison_notes.md](docs/results/backend_comparison_notes.md).

## Beyond 64×64

The exported graph is fully convolutional; `model_dynamic.onnx` runs any
grid whose sides are multiples of 8. Cost is linear at 7.14 ms per thousand
cells and ~1.6 MB per thousand cells on this CPU, so a million-cell mesh
would take ~7.5 s and ~2.4 GB in one call. The design for serving such a
mesh — 256-cell tiles with a 32-cell halo (receptive field ≈ 44 cells),
output streamed into the caller's buffer, a GPU execution provider past
~100 k cells where batching does pay, validation and fallback per tile — is
in [dynamic_size_notes.md](docs/results/dynamic_size_notes.md). It is a cost
measurement, not a validated prediction: the surrogate was trained on 64×64
chips and does not extrapolate physics.

## Known limitations

- The surrogate answers "what-if" questions about power magnitude with a
  gain of 0.1; such queries are not in its trusted region and no input
  check can detect them. Use the reference path for them.
- Rule I5 is, on this benchmark, a whitelist of the ten chip configurations
  the model has seen. That is the honest scope of the model, not a general
  novelty detector.
- Neither reference solver reproduces the dataset's generator; HotSpot is
  within 3.4 K on the unseen chips and 13 K off on the seen ones, the FD
  solver 11–18 K off on both. They are physics-consistent fallbacks, not
  oracles.
- Hotspot location is not reproduced (3.7 % exact); `tmax_index` is
  informative only.
- CPU execution provider only; macOS arm64 and Linux x64 prebuilt runtimes
  are wired in `cmake/onnxruntime.cmake`.

## Reproduce every number

| step | command | writes |
|---|---|---|
| fetch data + checkpoint (HTTP Range extraction, 220 MB instead of 46.8 GB) | `python tools/fetch_assets.py` | `docs/results/data_manifest.json` |
| baseline reproduction | `tools/run_baseline.sh --device cpu`; `tools/baseline_per_sample.py` | `baseline_*.json` |
| training statistics, fixtures | `python tools/export_stats.py --bench ../external/ThermalBench` | `contract/stats.json`, fixtures |
| sensitivity, error buckets, extrapolation, novelty, fallback curve | `tools/sensitivity.py`, `error_buckets.py`, `extrapolation.py`, `layout_novelty.py`, `fallback_curve.py` | `docs/results/*` |
| ONNX export + parity | `python tools/export_onnx.py`; `python tools/parity_check.py` | `models/*.onnx`, `metadata.json`, `parity_python.json`, `tests/fixtures/parity/` |
| C++ parity, latency, throughput, sizes | `ts_demo --parity`, `bench_latency`, `bench_throughput`, `bench_dynamic_size` | `parity_cpp.json`, `latency_single.json`, `throughput.json`, `dynamic_size.csv` |
| reference solvers | `ts_demo --fd-calibrate` / `--fd-validate`; `ts_demo --compare ... python tools/reference_cli.py` | `fd_solver_*.json`, `backend_comparison*.csv` |
| concurrency under TSan | `cmake -B build-tsan -DTS_SANITIZE=thread && ./build-tsan/ts_tests --gtest_filter='Concurrency.*'` | `concurrency.json` |

Python tools need the IC-ThermBench checkout next to this repository
(`../external/ThermalBench`) and its Python environment (3.10, torch 2.0.1,
onnxruntime 1.23.2); the C++ side needs only CMake and internet access for
the pinned ONNX Runtime package.

## Tests

`ctest` runs 34 tests: `Skeleton.*` (runtime and model IO match the
contract), `Predictor.*` (construction, rejection without exceptions, parity
with the Python runtime, zero-copy and batch paths), `Validator.*` (each
status code, both fixtures, feature extraction equals Python),
`Fixture.*` (dispatcher rules F1–F4 with scripted backends),
`FdSolver.*` (convergence, exact ambient shift, power gain 1, fallback),
`ExternalBackend.*` (protocol, exit codes, timeout kill),
`Concurrency.*`, `AccuracyGate.*` (contract gates on the committed
50-sample subset; status spellings equal the schema enum).

## Layout

```
contract/        contract.md, schema.json, stats.json      the interface and its measured constants
include/, src/   Predictor, Validator, Dispatcher, FD solver, external backend, C ABI
apps/ts_demo     --info --parity --fd-calibrate --fd-validate --compare
bench/           latency, throughput, dynamic size
tests/           GoogleTest suite + fixtures (parity pairs, eval50 subset, validator expectations)
tools/           Python: asset fetch, baseline, statistics, analyses, export, parity, HotSpot wrapper
models/          metadata.json, config_bank.npy, fd_solver_params.json (ONNX files via release)
docs/results/    every measured number, with notes; docs/failure_analysis.md; docs/integration_design.md
```

## License and attribution

Code: MIT ([LICENSE](LICENSE)). Model weights and data: IC-ThermBench
(CC BY 4.0), used unmodified; the exported ONNX files are derivatives of the
released level4 U-Net checkpoint. HotSpot is used as an external tool under
its own license. ONNX Runtime is fetched as Microsoft's prebuilt package.
