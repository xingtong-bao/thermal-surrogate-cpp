# Interface Contract — thermal surrogate v1

This document is the single source of truth for what the C++ library accepts, what it
returns, when it is allowed to answer from the surrogate, and when it must fall back to a
reference solver. Every rule is written as an assertion that the code checks. Placeholders
marked `TBD(...)` are filled from measured data, never by hand.

## 1. Use case and non-goals

- Use case: fast steady-state temperature-field prediction for design-of-experiments (DOE)
  pre-screening of 2.5D chiplet layouts on a fixed 64 × 64 grid.
- Non-goal: sign-off. Surrogate output is never the final thermal result; it ranks and
  filters candidates that are then verified by the reference solver.
- The model is the released IC-ThermBench U-Net trained on the S4 (`level4`) split. The
  library does not train or fine-tune.

## 2. Input

One sample is a dense float32 tensor of shape `(X, Y, Z, P)` with `X = Y = 64`, `Z = 1`,
`P = 7`. Batches are `(B, X, Y, Z, P)`. Memory layout handed to the library is C-contiguous
in that index order (P varies fastest). Channel order is fixed:

| idx | channel            | unit                     | training range        |
|----:|--------------------|--------------------------|-----------------------|
| 0   | `chiplet_power`    | W, total power of the chiplet covering the cell, 0 outside chiplets (inferred) | `[0, 300]` |
| 1   | `grid_x`           | mm, physical cell coordinate, uniform spacing per sample (inferred) | `[0, 59]` |
| 2   | `grid_y`           | mm, physical cell coordinate, uniform spacing per sample (inferred) | `[0, 61]` |
| 3   | `local_thermal_k`  | W·m⁻¹·K⁻¹                | `[0.200029, 100]` |
| 4   | `ambient_K`        | K                        | `[298.151, 328.141]` |
| 5   | `h_w_m2k`          | W·m⁻²·K⁻¹                | `[500.042, 19991.8]` |
| 6   | `r_convec_k_per_w` | K·W⁻¹                    | `[0.00087, 0.208077]` |

Training ranges are the per-channel `[min, max]` of the S4 training split (10 800
samples), measured in S2.3 by `tools/export_stats.py` and stored in `contract/stats.json`
together with mean, std, p1 and p99. The per-channel mean/std recomputed from that split
agree with the normalizer pickled inside the authors' checkpoint to a relative 1.4e-5,
which is the evidence that this split is the one the model was trained on. The
normalization constants the ONNX/C++ path must use are the checkpoint's, copied into
`stats.json` under `checkpoint_normalizer`.

Facts about the channels that an implementer needs (all measured, see `stats.json`):

- Channels 0–2 carry no unit in the benchmark; the units above are inferred from the
  values. `chiplet_power` takes one value per chiplet (14 distinct values in training,
  0 outside chiplet footprints), so summing it over cells does **not** give chip power.
  `grid_x`/`grid_y` increase in uniform steps of a fraction of a millimetre whose size
  changes per sample (chip size varies by case).
- `grid_x` varies along the loader's axis `Y` (index 2) and `grid_y` along axis `X`
  (index 1): the MATLAB source is column-major and the names follow its convention. The
  axis names are labels of a fixed memory layout, not geometry.
- Channels 4–6 are constant within a sample (global boundary conditions encoded as maps).
- Output `temperature_K`: training range `[299.46, 506.27]` K,
  mean 356.30, std 29.17.

Assertions checked before inference (any failure is a hard reject, see §5):

- `I1` dtype is float32; no NaN or Inf.
- `I2` shape is exactly `(64, 64, 1, 7)` per sample.
- `I3` channel count equals the model's `input_channels` recorded in `models/metadata.json`.
- `I4` for every channel c: `min_c - margin_c <= value <= max_c + margin_c`, where
  `margin_c = oob_scale * (max_c - min_c)` and `oob_scale` comes from `stats.json`
  (`oob_scale = 0.0`, decided in S2b.3: 80 % of the unseen-layout set lies inside the
  training range and still fails the accuracy gate by 3–16×, so no margin is justified).
  Violation is an out-of-distribution (OOD) condition, not an error: it triggers
  fallback (§5).
- `I5` layout novelty: the configuration vector of the input (chip size x, chip size y,
  number of chiplets, total power, max power density, mean chiplet area), standardized
  with the training mean/std in `stats.json → layout_novelty`, must lie within distance
  `threshold` (0.091) of its nearest training sample. This is the guard the range check
  cannot provide: the model has learned the ten chip configurations of its training set
  (S2b.1, S2b.3), and inputs from other configurations are OOD even when every channel is
  in range. Violation is an OOD condition like `I4`; `oob_channels` then contains
  `"layout"`. Measured in S2b.3: rejects 100 % of the unseen-layout set and 1 % of the
  in-support test set. Known relaxation not applied: `ambient_K` extrapolates safely to
  1.5× its range (true error +0.2 K), so `I4` is over-conservative on that channel.

## 3. Output

| field           | type / shape        | meaning                                                   |
|-----------------|---------------------|-----------------------------------------------------------|
| `field`         | float32 `(64, 64)`  | steady-state temperature, K, same grid as the input       |
| `tmax`          | float32             | `max(field)`                                              |
| `tmax_index`    | `(ix, iy)`          | location of `tmax`                                        |
| `oob_flag`      | bool                | true if any input channel violated `I4`                   |
| `oob_channels`  | list of idx         | which channels violated `I4` (empty when `oob_flag` false)|
| `source`        | enum                | `surrogate` \| `reference_inprocess` \| `reference_external` |
| `latency_ms`    | float64             | wall-clock time of the call that produced `field`         |
| `model_version` | string              | from `models/metadata.json`; `""` when `source != surrogate` |
| `status`        | `Status{code, message}` | see §6                                                |

Error bounds (added S8.2): for a surrogate answer on an input that passed I1–I5,
`mae_bound_90_K` and `tmax_err_bound_90_K` are split-conformal 90 % bounds on the field
MAE and |Tmax error| of that answer, calibrated on the level4 validation split (1 200
samples) and conditioned on the predicted temperature rise (`contract/conformal.json`).
Measured coverage on the 50-sample test subset is asserted ≥ 0.85 by `Conformal.*`.
They are 0 for reference answers and for OOD inputs: the guarantee needs the input to be
exchangeable with the calibration set, which is what I4/I5 enforce.

Assertions on output: `O1` `field` contains no NaN/Inf; `O2` `tmax == max(field)`;
`O3` when `source == surrogate`, `model_version` equals the loaded metadata version.

## 4. Accuracy gate (surrogate path, S4 held-out test split)

| metric                              | gate                     | rule                                      |
|-------------------------------------|--------------------------|-------------------------------------------|
| field MAE over all test samples     | **2.012 K** (baseline 1.829 K) | baseline MAE × 1.1                   |
| worst-5 % per-sample MAE            | **3.792 K** (baseline 3.447 K) | 95th percentile of baseline per-sample MAE × 1.1 |
| `tmax` absolute error (median)      | **0.952 K** (baseline 0.865 K) | baseline × 1.1                       |

Baseline values were measured in S2.2 by running the unmodified IC-ThermBench
evaluation on the authors' level4 U-Net checkpoint (`docs/results/baseline_unet_level4.json`,
dataset-wide averages, within 5e-4 K of the authors' published numbers) and
`docs/results/baseline_unet_level4_per_sample.json` (per-sample distribution, from
`tools/baseline_per_sample.py`). Environment and timings: `docs/results/baseline_env.json`;
discussion: `docs/results/baseline_notes.md`.

Known limitation recorded at the same time: the *location* of the hotspot is not
reproduced reliably by the baseline (argmax of prediction equals argmax of label in
3.7 % of test samples, within 2 cells in 45 %). `tmax_index` in §3 is therefore
informative only and carries no gate; S2b revisits this.

The gate is enforced by a test (`S6.1`) on a fixed 50-sample subset; a failing gate fails
the build. The 1.1 factor leaves room for fp32 ONNX export drift and nothing else.
Numerical parity gates (purpose: catch wiring bugs such as a wrong layout, dtype or
normalization constant, which produce errors ≥ 0.1 K; not to measure floating-point noise):

| comparison | max abs diff | mean abs diff | basis |
|---|---|---|---|
| PyTorch fp32 (batch 1) ↔ ONNX Runtime (Python) | `< 5e-3 K` | `< 1e-4 K` | measured fp32 floors, S3.2 |
| ONNX Runtime (Python) ↔ ONNX Runtime (C++) | `< 1e-4 K` | — | same runtime, same file: ≈ 3 ulp |

History: the gate was first written as `1e-5` (S1.1, before the output magnitude was
known), revised to `1e-4 K` in S3.1 when the fixture showed exactly one float32 ulp
(3.05e-5 K at 350 K), and set to the values above in S3.2 from 100 test samples:
PyTorch fp32 itself differs from a float64 run of the same network by up to 3.4e-4 K,
ONNX Runtime by 8.5e-4 K, and the benchmark's own batched PyTorch evaluation path by
9.7e-3 K (its batched CPU convolution kernel accumulates in a different order). The
PyTorch↔ORT mean difference is 3.5e-6 K with 88 % of pixels bit-identical. Details:
`docs/results/parity_python.json`, `docs/results/parity_notes.md`.

## 5. Fallback rules

The dispatcher tries backends in order `surrogate → reference_inprocess → reference_external`.
A fallback is taken, and `source` set accordingly, when any of these holds:

- `F1` `oob_flag == true` (rule `I4`). The surrogate is **not** run.
- `F2` shape or channel mismatch (`I2`, `I3`). The surrogate is not run; the reference
  backend receives the original request and applies its own validation.
- `F3` model load failed at construction or `model_version` is incompatible (§7).
  `SurrogateBackend` reports `ModelNotLoaded` for every call.
- `F4` inference raised or returned a tensor violating `O1`/`O2`. The partial result is
  discarded.

If every backend fails, the call returns the last `Status` with `source` unset and an
empty `field`. Reference results are treated as ground truth for accuracy reporting
because the training data itself was generated by the same solver family.

## 6. Status codes

`Ok`, `InvalidInput` (`I1`), `ShapeMismatch` (`I2`), `ChannelMismatch` (`I3`),
`OutOfDistribution` (`I4`, informational when a fallback succeeded), `ModelNotLoaded`,
`InferenceFailed`, `ReferenceFailed`, `Timeout` (external backend only).
No exception crosses the library boundary; constructors are the only place that may throw.

## 7. Versioning

- `models/metadata.json` carries `model_version`, `schema_version`, `input_channels`,
  normalization parameters, opset and the SHA-256 of the source checkpoint.
- The library refuses to load a model whose `schema_version` differs from
  `contract/schema.json`'s `schema_version`. Refusal is `ModelNotLoaded`, not a crash.
- `Output.model_version` lets a caller audit which model produced a stored result.

## 9. Power what-if by physics (added S8.6)

Steady-state conduction is linear in the dissipated power (verified with HotSpot to
0.2 %, `docs/results/power_gain_verification.json`). The library therefore answers
"the same design with every chiplet power scaled by `f`" without asking the model:
`Dispatcher::solve_power_scaled(base, f, out)` validates and solves the *base* design
(rules I1–I5 and the fallback apply to it) and returns
`field' = ambient + f · (field − ambient)`, `tmax'` likewise, error bounds scaled by
`f`. The surrogate's own weak power response (gain 0.1–0.3, S2b.1/S8.1) never enters.
Not covered: scaling a subset of chiplets, which breaks the proportionality; such
queries go to the reference backends. Exposed as `ts_predict_power_scaled` (C ABI) and
`Engine.predict_power_scaled` (Python).

## 8. Latency

No gate in v1. Single-sample and batch latency are measured and reported (`S4.4`, `S4.5`);
the external reference backend has a configurable timeout (default `TBD(S5.3)` s).
