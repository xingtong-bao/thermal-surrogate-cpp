# Failure analysis of the level4 U-Net surrogate (S2b)

Everything below is measured in this repository; scripts and raw numbers are
in `tools/` and `docs/results/` (notes: `sensitivity_notes.md`,
`error_buckets_notes.md`, `extrapolation_notes.md`, `fallback_curve.json`).

## Three sentences

**Trusted region.** Inputs whose chip configuration (chip size, chiplet count,
total power, max power density, mean chiplet area) is within 0.091 training-std
units of a level4 training sample and whose channels are inside the training
range: field MAE 1.83 K on 3 000 held-out samples (3.4 % of the temperature
rise, 2.3 % on the hottest quintile), hotspot temperature error median 0.87 K,
no bucket of this region above the contract gates.

**Untrusted region.** (1) Any chip configuration the model has not seen: five
unseen layouts give MAE 5–32 K (mean 16.9 K, 9× the in-support value) even
though four of the five lie entirely inside the per-channel training range;
(2) any "what-if" on power magnitude: scaling every chiplet's power by ±10 %
moves the predicted hotspot by 0.5–0.8 K where physics requires 6.3 K (gain
≈ 0.1, same at ±2 %), and the inputs stay in range so no guard sees it;
(3) the hotspot *location*: argmax matches the truth in 3.7 % of samples.

**Physical reason.** The training set contains ten chips with 14 discrete
power values tied to their layouts, so the network learned a layout-to-field
lookup for those ten chips rather than the conduction operator; the only law
it generalises is the additive ambient offset (gain 1.00 even 50 % outside the
training range), because that offset is identical in every training sample.

## Where the error sits inside the trusted region (S2b.2)

Absolute error grows with thermal load and nothing else: temperature-rise
quintiles 1.22 → 2.51 K, max power density quintiles 1.02 → 2.40 K (Spearman
0.55); ambient, h and r have no effect (|ρ| ≤ 0.19). Relative error falls
with rise (4.6 % → 2.3 %). Per-case MAE spans 0.98 K (case 1) to 2.51 K
(case 8); the hard cases are the large, dense chips. Top-3 channels by
sensitivity: ambient_K ≫ r_convec_k_per_w > grid_y.

## What the guards do (S2b.3, S2b.4)

| guard | level4 test sent to reference | level5 sent to reference | blended MAE on a 50/50 stream |
|---|---|---|---|
| none | 0 % | 0 % | 9.35 K |
| I4 range check only (`oob_scale` = 0, its best setting) | 0 % | 20 % | 6.55 K |
| I5 configuration novelty (τ = 0.091) | 1 % | 100 % | 0.90 K |
| I4 + I5 (contract) | 1 % | 100 % | 0.90 K |

The range check can only catch the one unseen chip that is larger than any
training chip; the novelty check catches all five at the price of 1 % false
rejections. Cost per query with the contract guards on the 50/50 stream is
1 + 0.505·(R − 1) surrogate-call units, where R is the reference/surrogate
cost ratio (to be measured in S4.4 and S5.3); on an in-support stream it is
1 + 0.01·(R − 1).

## Consequences carried into the design

- `oob_scale = 0.0` and new contract rule I5 with its parameters in
  `contract/stats.json`; the C++ validator (S4.3) ships the 10 800 × 6
  configuration bank.
- `tmax_index` is informative only; no gate.
- Power what-if queries are documented as unsupported in the contract use
  case; the reference backend (S5) is the only path for them.
- Possible relaxation, not applied: exempt `ambient_K` from I4 (true error
  +0.2 K at 1.5× range).
