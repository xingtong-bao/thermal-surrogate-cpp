# Interface Contract — thermal surrogate v1

This document is the single source of truth for what the C++ library accepts, what it
returns, when it is allowed to answer from the surrogate, and when it must fall back to a
reference solver. Every rule is written as an assertion that the code checks. Placeholders
marked `TBD(S2.2)` / `TBD(S2.3)` are filled from measured data, never by hand.

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
| 0   | `chiplet_power`    | W per cell `TBD(S2.3)`    | `TBD(S2.3)`           |
| 1   | `grid_x`           | normalized coordinate `TBD(S2.3)` | `TBD(S2.3)`  |
| 2   | `grid_y`           | normalized coordinate `TBD(S2.3)` | `TBD(S2.3)`  |
| 3   | `local_thermal_k`  | W·m⁻¹·K⁻¹                | `TBD(S2.3)`           |
| 4   | `ambient_K`        | K                        | `TBD(S2.3)`           |
| 5   | `h_w_m2k`          | W·m⁻²·K⁻¹                | `TBD(S2.3)`           |
| 6   | `r_convec_k_per_w` | K·W⁻¹                    | `TBD(S2.3)`           |

Units for channels 0–2 are inferred from the data in S2.3 (value ranges and the upstream
ATPlace2.5D conventions) and confirmed there. Training ranges are the per-channel
`[min, max]` of the S4 training split, stored in `contract/stats.json` together with
mean, std, p1 and p99.

Assertions checked before inference (any failure is a hard reject, see §5):

- `I1` dtype is float32; no NaN or Inf.
- `I2` shape is exactly `(64, 64, 1, 7)` per sample.
- `I3` channel count equals the model's `input_channels` recorded in `models/metadata.json`.
- `I4` for every channel c: `min_c - margin_c <= value <= max_c + margin_c`, where
  `margin_c = oob_scale * (max_c - min_c)` and `oob_scale` comes from `stats.json`
  (default `TBD(S2b.3)`, derived from the extrapolation curve). Violation is an
  out-of-distribution (OOD) condition, not an error: it triggers fallback (§5).

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
Numerical parity gates: PyTorch ↔ ONNX Runtime (Python) and ONNX Runtime (Python) ↔ C++
max-abs-diff `< 1e-5` on fixed fixtures.

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

## 8. Latency

No gate in v1. Single-sample and batch latency are measured and reported (`S4.4`, `S4.5`);
the external reference backend has a configurable timeout (default `TBD(S5.3)` s).
