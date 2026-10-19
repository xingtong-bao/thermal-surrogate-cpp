# thermal-surrogate-cpp

[![ci](https://github.com/xingtong-bao/thermal-surrogate-cpp/actions/workflows/ci.yml/badge.svg)](https://github.com/xingtong-bao/thermal-surrogate-cpp/actions/workflows/ci.yml)
![C++17](https://img.shields.io/badge/C%2B%2B-17-blue)
![ONNX Runtime 1.23.2](https://img.shields.io/badge/ONNX%20Runtime-1.23.2-lightgrey)
![License MIT](https://img.shields.io/badge/license-MIT-green)
[中文](README_CH.md)

A C++17 inference library for neural chip-thermal surrogates, built around
the published IC-ThermBench level4 U-Net (7 input maps → 64×64 steady-state
temperature field). The library validates inputs against a written
contract, detects out-of-distribution designs, attaches a calibrated error
bound to every prediction, falls back to a physics solver when the model
cannot be trusted, and ships with the tooling to update, gate and release a
new model version. Every number in this file is measured in this
repository and links to the file that holds it.

![architecture](docs/figures/architecture.png)

## Features

- **Contract-driven validation** — shape, dtype, per-channel training range
  and a configuration-novelty check (rules I1–I5), with status codes
  mirrored in C++, JSON Schema and the C ABI. [contract/](contract/)
- **Fallback dispatcher** — surrogate → external solver (HotSpot subprocess
  with timeout) → in-process finite-difference solver; every answer reports
  which backend produced it. [docs/integration_design.md](docs/integration_design.md)
- **Per-prediction error bounds** — split-conformal 90 % bounds on field
  MAE and hotspot error, conditioned on the predicted temperature rise.
- **Physics-exact what-ifs** — whole-chip power scaling answered by the
  linearity of conduction, not by the model.
- **Model lifecycle** — scripts to fine-tune on solver answers or augmented
  data, export to ONNX, check parity, recalibrate bounds and run the release
  gates on the full test split.
- **Model-agnostic** — U-Net, U-FNO (spectral layers exported as matrix
  products) and an ensemble load through the same API.
- **Bindings** — C ABI with an opaque handle (`ts_c_api.h`), pybind11 module.
- **Engineering** — zero-copy input, caller-owned output, thread-safe
  prediction (ThreadSanitizer-checked), hash-pinned dependencies, 38 tests,
  CI on Linux.

## Results

Measured on Apple M4 (4 P + 6 E cores, 16 GB), ONNX Runtime 1.23.2 CPU.
"Seen chips" = the level4 test split (10 chip families in training);
"unseen chips" = the level5 set (5 families never trained on).

**Surrogate alone**

| metric | value | file |
|---|---|---|
| MAE, seen chips, 50-sample committed subset / full 3 000 | 1.05 K / 1.83 K (authors: 1.83 K) | [baseline_notes.md](docs/results/baseline_notes.md) |
| MAE, unseen chips (50) | 14.8 K | [backend_comparison.csv](docs/results/backend_comparison.csv) |
| contract gates (baseline × 1.1): MAE / worst-5 % / hotspot median | 2.012 / 3.792 / 0.952 K | [contract.md §4](contract/contract.md) |
| latency p50, 1 / 4 / 8 threads | 79.9 / 30.3 / 27.2 ms | [latency_single.json](docs/results/latency_single.json) |
| throughput, batch 1 / 8 / 32 / 64 | 48.8 / 33.8 / 34.3 / 34.6 samples/s | [throughput.json](docs/results/throughput.json) |
| larger grids 128² / 256² / 512² | 117 ms / 468 ms / 1.87 s (7.14 ms per 1 k cells) | [dynamic_size.csv](docs/results/dynamic_size.csv) |
| PyTorch ↔ ONNX Runtime (100 samples) / ONNX Runtime Python ↔ C++ (10) | max 8.2e-4 K / 0 K | [parity_notes.md](docs/results/parity_notes.md) |

**Guards and fallback**

| metric | value | file |
|---|---|---|
| novelty guard (I5): unseen chips rejected / seen chips falsely rejected | 100 % / 1 % | [layout_novelty.json](docs/results/layout_novelty.json) |
| per-channel range check (I4) alone: unseen chips rejected | 20 % | [extrapolation_notes.md](docs/results/extrapolation_notes.md) |
| conformal 90 % MAE bound (by predicted rise) / held-out coverage | 1.6–3.7 K / 1.00 | [conformal_notes.md](docs/results/conformal_notes.md) |
| dispatcher with HotSpot behind it, unseen chips MAE | 3.4 K (surrogate alone 14.8 K) | [backend_comparison_notes.md](docs/results/backend_comparison_notes.md) |
| blended MAE on a 50/50 seen/unseen stream | 2.2 K (surrogate alone 7.9 K) | [fallback_curve.json](docs/results/fallback_curve.json) |
| reference solvers: FD (in-process) / HotSpot (subprocess), seen chips MAE, latency | 12.8 K, 34 ms / 13.4 K, 2.7 s | [fd_solver_validation.json](docs/results/fd_solver_validation.json), [hotspot_validation.json](docs/results/hotspot_validation.json) |
| 8 threads × 100 predictions on one session | bit-identical, TSan clean | [concurrency.json](docs/results/concurrency.json) |

**Model variants through the same library**

| model | seen chips MAE / hotspot median | unseen chips MAE | latency p50 (4 threads) | file |
|---|---|---|---|---|
| U-Net (shipped) | 1.05 / 0.60 K | 14.8 K | 30 ms | [models/metadata.json](models/metadata.json) |
| U-FNO | 0.60 / 0.27 K | 24.6 K | 760 ms | [model_swap_notes.md](docs/results/model_swap_notes.md) |
| U-Net + U-FNO ensemble | 0.69 / 0.27 K | 18.4 K | 745 ms | [ensemble_spread.json](docs/results/ensemble_spread.json) |

**Model updates and release gates** (full 3 000-sample split; "pass" = all three contract gates)

| candidate | purpose | unseen chips MAE | seen chips MAE | hotspot median | gates | file |
|---|---|---|---|---|---|---|
| v1.0 (shipped) | — | 15.8 K | 1.83 K | 0.87 K | pass | [baseline_unet_level4.json](docs/results/baseline_unet_level4.json) |
| v1.1, 50 solver answers per unseen chip | adapt to new chips | 2.6 K | 5.8 K | — | fail (forgetting) | [lifecycle_notes.md](docs/results/lifecycle_notes.md) |
| v1.1 + replay of 250 training samples | same, without forgetting | 2.8 K | 1.45 K | 1.19 K | fail (hotspot) | same |
| v1.4, power augmentation ±15 % | power sensitivity 0.1 → 0.85 | 13.4 K | 2.07 K | 1.09 K | fail (MAE, hotspot) | [power_aug_notes.md](docs/results/power_aug_notes.md) |

**Characterisation of the shipped model** — power-sensitivity gain 0.1–0.3
of the solver's response (HotSpot-verified), cooling-sensitivity slope 2×
the solver's, hotspot location exact in 3.7 % of samples, error growing with
power density and temperature rise: [docs/failure_analysis.md](docs/failure_analysis.md),
[sensitivity_notes.md](docs/results/sensitivity_notes.md),
[error_buckets_notes.md](docs/results/error_buckets_notes.md),
[exploration_notes.md](docs/results/exploration_notes.md).

![results](docs/figures/results_summary.png)

## Quick start

```bash
# prerequisites: cmake >= 3.24, ninja, a C++17 compiler, python3 (fixtures), internet (ONNX Runtime + models)
git clone https://github.com/xingtong-bao/thermal-surrogate-cpp && cd thermal-surrogate-cpp
gh release download models-v1 --dir models          # model.onnx + model_dynamic.onnx (20 MB each)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure        # 38 tests, ~20 s
./build/ts_demo --info                            # runtime version + model IO
./build/c_caller models contract tests/fixtures/sample_input.npy   # C ABI example
```

Without `gh`: download the two files from the
[models-v1 release](https://github.com/xingtong-bao/thermal-surrogate-cpp/releases/tag/models-v1)
and verify them against [models/metadata.json](models/metadata.json), or
regenerate them following [models/README.md](models/README.md).

## Usage

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
dispatcher.set_conformal(ConformalTable("contract/conformal.json"));   // optional error bounds

Output out;                                   // never throws after construction
dispatcher.solve(Input{data /* 64*64*1*7 floats */, kInputShape}, out);
if (out.status.ok()) use(out.field, out.tmax, out.mae_bound_90_K, out.source, out.oob_flag);
dispatcher.solve_power_scaled(Input{data, kInputShape}, 1.1, out);   // all chiplet powers +10 %
```

Input is a non-owning view of the caller's buffer; `predict_into` writes
the field into a caller-owned buffer; `predict_batch` runs several samples
through the dynamic-batch graph. C ABI: [examples/c_caller.c](examples/c_caller.c).
Python: [python/test_binding.py](python/test_binding.py). Ownership,
lifetime, threading, error and versioning rules:
[docs/integration_design.md](docs/integration_design.md).

## Contract

[contract/contract.md](contract/contract.md) defines the interface;
[contract/schema.json](contract/schema.json) is its JSON Schema form;
[contract/stats.json](contract/stats.json) and
[contract/conformal.json](contract/conformal.json) hold the measured
constants the library reads.

| rule | condition | on violation |
|---|---|---|
| I1–I3 | float32, finite, shape (64,64,1,7), 7 channels | `InvalidInput` / `ShapeMismatch` / `ChannelMismatch` (hard reject) |
| I4 | every channel within its training range | `OutOfDistribution` → fallback |
| I5 | configuration vector within 0.091 std-units of a training chip | `OutOfDistribution` → fallback |
| F1–F4 | backends tried in order; failures skipped; last failure reported | `source` names the answering backend |
| §4 | accuracy gates on the full test split; the committed 50-sample subset is a smoke test | release blocked |
| §9 | whole-chip power scaling computed from the base prediction | — |

## Architecture

```
Input (64,64,1,7)
   │
   ▼
Validator ── I1–I3 reject ──► Status
   │ I4 / I5
   ▼
Dispatcher ──► SurrogateBackend (ONNX Runtime)        ~30 ms
           ──► ExternalReferenceBackend (HotSpot)     ~2 s, on OOD or failure
           ──► FdSolverBackend (2-D SOR)              ~35 ms, last resort
   │
   ▼
Output: field, tmax, tmax_index, error bounds, source, oob_flag, latency, status
```

Public headers contain no ONNX Runtime types; the runtime is a private
implementation detail behind a pimpl. Dependencies (ONNX Runtime, GoogleTest,
nlohmann/json, pybind11) are fetched by CMake with pinned versions and, for
the runtime package, pinned SHA-256.

## Repository layout

```
contract/        contract.md, schema.json, stats.json, conformal.json
include/, src/   Predictor, Validator, Dispatcher, ConformalTable, FD solver, external backend, C ABI
apps/ts_demo     --info --parity --fd-calibrate --fd-validate --compare
bench/           latency, throughput, dynamic grid size
tests/           GoogleTest suite + fixtures (parity pairs, 50-sample subset, validator expectations)
tools/           Python: asset fetch, baseline, statistics, analyses, ONNX export, parity, HotSpot wrapper, lifecycle
python/          pybind11 module + check script
models/          metadata.json, config_bank.npy, fd_solver_params.json (ONNX files via GitHub release)
models_*/        candidate versions (fine-tuned, augmented, U-FNO, ensemble) with their own metadata, bank and bounds
docs/results/    all measured numbers with notes; docs/failure_analysis.md; docs/integration_design.md
```

## Reproducing the numbers

| step | command | output |
|---|---|---|
| data and checkpoints (HTTP Range extraction from the public archives) | `python tools/fetch_assets.py [--checkpoint-members level4_UNet/,level4_UFNO/]` | `docs/results/data_manifest.json` |
| baseline | `tools/run_baseline.sh --device cpu`; `tools/baseline_per_sample.py` | `baseline_*.json` |
| statistics, fixtures | `python tools/export_stats.py --bench ../external/ThermalBench` | `contract/stats.json`, `tests/fixtures/` |
| model characterisation | `tools/sensitivity.py`, `error_buckets.py`, `extrapolation.py`, `layout_novelty.py`, `fallback_curve.py`, `power_gain_verification.py`, `exploration_demo.py` | `docs/results/*` |
| ONNX export, parity | `tools/export_onnx.py`, `parity_check.py`, `export_ufno_onnx.py`, `export_ensemble_onnx.py` | `models*/`, `parity_python.json`, `tests/fixtures/parity/` |
| C++ parity and benchmarks | `ts_demo --parity`, `bench_latency`, `bench_throughput`, `bench_dynamic_size` | `parity_cpp.json`, `latency_*.json`, `throughput.json`, `dynamic_size.csv` |
| reference solvers, backend comparison | `ts_demo --fd-calibrate` / `--fd-validate`; `ts_demo --compare ... python tools/reference_cli.py` | `fd_solver_*.json`, `backend_comparison*.csv` |
| error bounds | `python tools/conformal.py` | `contract/conformal.json` |
| model updates | `finetune_fewshot.py` + `lifecycle_v11.py`; `finetune_power_aug.py` + `lifecycle_poweraug.py` | `lifecycle_*.json`, `power_aug_*.json`, `models_v1.*/` |
| concurrency under TSan | `cmake -B build-tsan -DTS_SANITIZE=thread && ./build-tsan/ts_tests --gtest_filter='Concurrency.*'` | `concurrency.json` |

The Python tools expect the IC-ThermBench checkout at `../external/ThermalBench`
with its environment (Python 3.10, torch 2.0.1, onnxruntime 1.23.2). The
C++ build needs only CMake, a compiler and network access for the pinned
ONNX Runtime package.

## Tests

`ctest` runs 38 tests: `Skeleton.*` (runtime version, model IO),
`Predictor.*` (construction, rejection paths, parity, zero-copy, batch),
`Validator.*` (each status code, both fixtures, feature extraction),
`Fixture.*` (dispatcher rules F1–F4), `FdSolver.*`, `ExternalBackend.*`
(protocol, exit codes, timeout), `Concurrency.*`, `AccuracyGate.*`
(contract gates, schema spellings), `Conformal.*`, `PowerWhatIf.*`.

## Known limitations

- Per-chiplet power changes: the shipped model's sensitivity is 0.1 of the
  physical response; the augmented candidate reaches 0.85 but is held by
  the release gate. Whole-chip scaling is exact (§9).
- Rule I5 is, on this benchmark, a whitelist of the ten trained chip
  configurations; the ensemble disagreement is an independent signal.
- Neither reference solver reproduces the dataset's generator (HotSpot
  13 K off on seen chips, 3.4 K on unseen; FD 11–18 K).
- Hotspot location is not reproduced (3.7 % exact); `tmax_index` is
  informative only.
- CPU execution provider only (macOS arm64, Linux x64).

## License and attribution

Code: MIT ([LICENSE](LICENSE)). Model weights and data: IC-ThermBench
(CC BY 4.0), used unmodified; the exported ONNX files derive from the
released level4 U-Net and U-FNO checkpoints. HotSpot is used as an external
tool under its own license. ONNX Runtime is Microsoft's prebuilt package.
