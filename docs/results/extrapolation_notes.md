# S2b.3 — Extrapolation curve, the OOB threshold, and a layout-novelty guard

Scripts: `tools/extrapolation.py` (experiments A, B, C) and
`tools/layout_novelty.py` (experiment D). Tables: `extrapolation.csv/.json`,
`level5_per_sample.csv`, `layout_novelty.csv/.json`; charts:
`extrapolation.png`, `layout_novelty.png`. All on CPU, seed 0, authors' level4
U-Net checkpoint.

## The question and why it is hard

Contract rule I4 lets the surrogate run only if every channel lies within the
training range widened by `oob_scale × range`. The backlog asked for the
threshold to be "the first scale whose error exceeds the gate". But a scaled
input has no ground-truth field, so an *error* vs. scale curve cannot be
measured for most channels. Four experiments were run instead, two with real
error and two without:

| exp. | data | has truth? | what it measures |
|---|---|---|---|
| A | level5, 5 000 samples, 5 unseen layouts | yes | true error vs. *OOB distance* d = smallest `oob_scale` that would accept the sample |
| B | level4 test, ambient_K stretched to k× range | yes (exact: field shifts by the ambient change) | true error of extrapolating one channel |
| C | level4 test, every channel stretched to k× range, k = 1.1 / 1.2 / 1.5 | no | how much the prediction *moves* (response), not error |
| D | level4 test vs level5 | yes | can a cheap "have we seen this layout?" score separate them |

## A — level5: the range check cannot see new layouts

| OOB distance bin | n | true MAE (K) | p95 (K) | median Tmax error (K) |
|---|---|---|---|---|
| [0, 0.05) — inside the training range | 4 000 | 14.09 | 53.1 | 16.8 |
| [0.2, 0.3) | 1 000 | 27.99 | 57.9 | 37.1 |

Per case: case 0 (72.7 mm chip, 23 % beyond the training size in grid_x and
grid_y) MAE 28.0 K; cases 1–4 are entirely **inside** the per-channel training
range and still carry MAE 11.8 / 32.0 / 7.5 / 5.1 K. The field-MAE gate is
2.01 K. So 80 % of the OOD set passes rule I4 untouched and fails by 3–16×.

Decision: **`oob_scale = 0.0`** (the first bin already fails, so any margin
would be a margin on top of a check that is necessary but not sufficient).
A per-channel range check catches only one of the five unseen cases.

## B — ambient_K: the one channel that extrapolates safely

| k | max d | true MAE (K) |
|---|---|---|
| 1.0 | 0 | 1.80 |
| 1.1 | 0.10 | 1.81 |
| 1.2 | 0.20 | 1.83 |
| 1.5 | 0.50 | 2.03 |

Stretching the ambient temperature 50 % beyond its training range (to ~343 K)
costs 0.2 K of error. Consistent with S2b.1 (gain 1.0): the network treats the
ambient as an additive offset. The range check is therefore over-conservative
on this channel; the contract keeps the uniform rule for simplicity and notes
the exemption as a possible relaxation.

## C — response curves: smooth, no cliff

Mean |T′ − T| in K for a single channel stretched to k× its training range:

| channel | 1.1 | 1.2 | 1.5 |
|---|---|---|---|
| chiplet_power | 0.64 | 1.91 | 9.08 |
| grid_x | 1.15 | 2.44 | 7.60 |
| grid_y | 1.28 | 3.03 | 9.72 |
| local_thermal_k | 0.69 | 2.80 | 8.18 |
| ambient_K | 1.42 | 2.81 | 6.93 |
| h_w_m2k | 0.56 | 1.07 | 2.45 |
| r_convec_k_per_w | 1.26 | 2.53 | 6.28 |
| all channels jointly | 4.52 | 10.30 | 29.83 |

The model keeps responding smoothly and monotonically; nothing saturates or
explodes up to 1.5×. That is reassuring for numerical robustness and useless
as a trust signal: a smooth wrong answer looks exactly like a smooth right
one. (Power at 1.5× finally moves Tmax by +6.4 K, about what physics expects
for +10 %, another view of the S2b.1 gain deficit.)

## D — layout novelty: what works and what does not

**Footprint nearest-neighbour (fails).** Score = minimum Hamming distance of
the chiplet footprint mask (16×16 or 32×32) to all 10 800 training masks.
level4 test and level5 have the same distribution (median 0.19 vs 0.19,
AUROC 0.53): placements vary so much within a case that every test sample is
"far" from the bank. Recorded as a negative result.

**Configuration nearest-neighbour (works on this benchmark).** Score =
Euclidean distance, in training-std units, from the 6-vector (chip size x,
chip size y, number of chiplets, total power, max power density, mean chiplet
area) to the nearest training sample.

| | level4 test | level5 |
|---|---|---|
| median score | 0.000 | 0.95 |
| p99 / min | p99 = 0.091 | min = 0.42 |
| rejected at the 99 %-pass threshold | 1 % | **100 %** (every case) |

AUROC 1.000; within level5 the score correlates with the true error
(Spearman 0.46). Cost: 6 floats per training sample (10 800 × 6) and one
nearest-neighbour search, trivially cheap in C++.

Why it works, and the caveat: in IC-ThermBench every case is a distinct chip
size with its own chiplet population, so the configuration vector is in effect
a whitelist of the ten chip configurations the model has seen. That is the
honest scope of the surrogate — it has learned ten chips, not thermal physics
(S2b.1) — and the rule says so explicitly. A chip whose configuration matches
training but whose placement is novel is still accepted, and S2b.2 shows those
are predicted at 1–2.5 K.

## What goes into the contract (rule I4 and new rule I5)

- `oob_scale = 0.0` in `stats.json`, `schema.json`, contract §2.
- New rule `I5`: configuration novelty score (features, training mean/std and
  threshold 0.091 stored in `stats.json` under `layout_novelty`) must not
  exceed the threshold; violation is an OOD condition and triggers fallback
  like I4. Status code stays `OutOfDistribution`; `oob_channels` lists
  `"layout"` for an I5 violation.
- Expected effect on the fallback rate (S2b.4 measures it): ~1 % of
  in-support inputs and 100 % of level5 inputs go to the reference solver.
