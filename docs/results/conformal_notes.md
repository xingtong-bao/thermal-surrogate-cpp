# S8.2 — Per-prediction error bounds (split conformal prediction)

Why: an optimiser needs to know how far *this* answer may be off, not the
dataset-wide MAE. Split conformal prediction gives a distribution-free
guarantee: calibrate a quantile of a nonconformity score on held-out data;
future scores stay under it with the chosen probability as long as the new
input is exchangeable with the calibration set. Rules I4/I5 are what make an
input exchangeable with the training distribution, so the bound is attached
only to surrogate answers on inputs that passed them.

Calibration (`tools/conformal.py`): level4 **validation** split, 1 200
samples the model never trained on and that the test split does not touch.
Scores: per-sample field MAE and |Tmax error|. Finite-sample quantile
ceil((n+1)(1−α))/n. Conditioned on the predicted rise (Tmax_pred − ambient,
known at inference), five equal-count bins, because S2b.2 showed the error
grows with the rise:

| predicted rise (K) | n | MAE 90 % bound | Tmax 90 % bound |
|---|---|---|---|
| < 34.8 | 240 | 1.62 | 1.39 |
| 34.8 – 48.3 | 240 | 2.35 | 2.09 |
| 48.3 – 63.1 | 240 | 2.64 | 2.47 |
| 63.1 – 88.0 | 240 | 2.90 | 2.72 |
| ≥ 88.0 | 240 | 3.69 | 3.65 |
| overall | 1 200 | 2.86 (95 %: 3.31) | 2.49 (95 %: 3.32) |

Held-out check on the committed 50-sample test subset (`Conformal.CoverageOnEval50`):
coverage 1.00 for the MAE bound and 1.00 for the Tmax bound (target ≥ 0.90; the
test asserts ≥ 0.85 to allow for 50-sample fluctuation). Mean MAE bound on
that subset 2.71 K against a realised MAE of 1.05 K: the bound is honest but
not tight, because the validation split contains the hot, dense cases that
the every-60th subset under-samples.

Library: `Output::mae_bound_90_K`, `tmax_err_bound_90_K` (C++), the same two
fields in `ts_result` (C ABI) and in the Python dict; 0 for reference
answers and OOD inputs. `contract/conformal.json` is versioned with the
model: a new model (S8.3) needs a new calibration.
