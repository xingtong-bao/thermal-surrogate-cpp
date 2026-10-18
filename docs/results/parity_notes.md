# S3.2 — PyTorch ↔ ONNX Runtime parity

Script: `tools/parity_check.py`; numbers: `parity_python.json`; C++ reference
pairs: `tests/fixtures/parity/` (10 pairs, 1.48 MB, manifest with SHA-256).
100 level4 test samples (every 30th, so all ten cases and the whole
temperature range), compared in Kelvin end to end: the ONNX graph carries the
normalization, so the comparison includes it.

## What "PyTorch" means here

The first run compared ONNX Runtime with the benchmark's own `evaluate()`
path and found a 1.05e-2 K outlier (344 ulp). Diagnosis showed the outlier
was not in ONNX Runtime: the benchmark evaluates in batches of 20, and
PyTorch's batched CPU convolution kernel accumulates in a different order
from the batch-1 kernel. Three references were therefore computed:

| max abs diff over 100 samples (K) | torch fp32, batch 1 | torch fp32, batch 20 | torch fp64 |
|---|---|---|---|
| ONNX Runtime (Python, batch 1) | 8.2e-4 | 1.05e-2 | 8.5e-4 |
| torch fp32, batch 1 | — | 9.8e-3 | 3.4e-4 |
| torch fp32, batch 20 | | — | 9.7e-3 |

Reading: ONNX Runtime is as close to the float64 "true" output of the network
as PyTorch's own single-sample fp32 path, and ~10× closer than the batched
path the benchmark uses for its published metrics. Mean |ORT − torch_b1| is
3.5e-6 K; 88.5 % of pixels are bit-identical; the tail (p99 = 1 ulp = 3.05e-5 K, max
8e-4 K) sits on a few pixels where a ReLU or max-pool decision flips under a
1-ulp perturbation and the difference propagates. None of this is visible at
the accuracy level: MAE vs ground truth is 1.1150 K for both paths on these
samples.

## Gates adopted (contract §4)

- PyTorch (batch 1) ↔ ONNX Runtime: max < 5e-3 K and mean < 1e-4 K. A wiring
  bug (wrong axis order, float64 input, stale normalization constant) shows
  up as ≥ 0.1 K, so the gate has a 20× margin to bugs and a 6× margin to the
  measured noise floor.
- ONNX Runtime Python ↔ C++: max < 1e-4 K. Same runtime, same file; anything
  larger is a buffer bug in the C++ code.

Why not keep tightening the reference instead: a float64 export would remove
the noise but double memory and time for no accuracy gain (1e-3 K against a
1.8 K model error), and the production runtime is fp32 anyway.

## Reference pairs for S4.2

`input_XX.npy` (64,64,1,7) float32 C-order, `field_XX.npy` = ONNX Runtime
Python output, `torch_XX.npy` = PyTorch batch-1 output. Pairs span Tmax from
the coolest to the hottest of the 100 samples. The C++ test loads a pair, runs
`model.onnx`, and asserts max |diff| < 1e-4 K against `field_XX`.
