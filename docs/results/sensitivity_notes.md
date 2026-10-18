# S2b.1 — Channel sensitivity of the level4 U-Net

Produced by `tools/sensitivity.py` (200 level4 test samples, CPU, seed 0).
Raw numbers: `sensitivity.csv` / `sensitivity.json`; chart: `sensitivity.png`.

## Setup

Each input channel is scaled by 0.9 and 1.1 over the whole 64×64 map, one
channel at a time, and the model is re-run. Reported per (channel, factor),
averaged over the 200 samples:

- `dTmax`: change of the predicted hotspot temperature relative to the
  unperturbed prediction (signed, and absolute);
- `field MAD`: mean |T′ − T| over the field, i.e. the field MAE between the
  perturbed and unperturbed prediction;
- `ood_fraction`: share of samples whose perturbed channel leaves the training
  range in `contract/stats.json`.

For `chiplet_power` and `ambient_K` the steady-state heat equation gives an
exact expectation (the problem is linear in both: T = T_ambient + R·P), so a
`gain` = measured / expected is reported. A ±2 % probe was added for those two
channels to separate "model property" from "left the training range".

## Results (±10 %)

| channel | field MAD (K), ×0.9 / ×1.1 | dTmax (K), ×0.9 / ×1.1 | ood fraction ×1.1 | expected dTmax (K) | gain |
|---|---|---|---|---|---|
| ambient_K | 31.80 / 30.22 | −31.7 / +30.9 | 0.99 | ∓31.2 | 1.02 / 0.99 |
| r_convec_k_per_w | 1.31 / 1.32 | −1.38 / +1.29 | 0.00 | — | — |
| grid_y * | 1.16 / 1.28 | −0.63 / +1.89 | 0.20 | — | — |
| grid_x * | 0.91 / 1.16 | −0.86 / +1.48 | 0.30 | — | — |
| h_w_m2k | 0.78 / 0.72 | +0.92 / −0.86 | 0.03 | — | — |
| local_thermal_k | 0.70 / 0.69 | −0.52 / +0.77 | 1.00 | — | — |
| chiplet_power | 0.72 / 0.64 | −0.78 / +0.55 | 0.20 | ∓6.3 | 0.12 / 0.09 |

`*` geometric channel: scaling it means a 10 % larger or smaller chip, not a
changed parameter.

Ranking by mean field MAD: ambient_K ≫ r_convec_k_per_w > grid_y > grid_x >
h_w_m2k > local_thermal_k > chiplet_power.

±2 % probe: ambient_K gain 1.01 / 0.98; chiplet_power gain 0.09 / 0.06 (expected
∓1.26 K, measured −0.12 / +0.08 K).

## Reading

1. **Ambient temperature is handled exactly.** A shift of the ambient moves the
   whole predicted field by the same amount, gain 1.0 at both ±2 % and ±10 %,
   even though ±10 % puts almost every sample outside the training range of
   ambient_K. This is the one piece of physics the network reproduces
   as a law rather than as a fit.

2. **Power magnitude is almost ignored.** Scaling every chiplet's power by 10 %
   should move the hotspot by ~6.3 K; the model moves it by 0.5–0.8 K, a gain
   of about 0.1. The same gain appears at ±2 %, where 80 % of the samples stay
   inside the training range, so this is not an extrapolation artefact. The
   likely cause is in the data: `chiplet_power` takes only 14 distinct values
   in training and each value is tied to a case layout, so the network can
   predict the temperature rise from *which* chiplet pattern it sees rather
   than from the power number. Consequence for the product: the surrogate
   must not be used for "what if this chiplet draws 10 % more" questions, and
   the OOD check on power alone will not catch that misuse because the
   perturbed values are in range. This goes into `failure_analysis.md` (S2b.4)
   as the main untrusted region; S5.2's finite-difference solver can quantify
   the true ΔT for comparison.

3. **Boundary conditions matter more than materials.** r_convec and h move the
   field by ~1 K for a 10 % change with the physically right sign (higher h
   or lower r → cooler). local_thermal_k × 1.1 leaves the training range for
   every sample (the max is already 100), so its row measures extrapolation,
   not sensitivity.

4. **Geometry scaling is not a clean experiment.** Scaling grid_x/grid_y
   produces a plausible response (larger chip → hotter, +1.5–1.9 K) but the
   power map is not rescaled with it, so the physical meaning is muddled; the
   rows are kept for completeness only.

## Top-3 channels for S2b.4

ambient_K, r_convec_k_per_w, grid_y (by field MAD). The more important finding
for the failure analysis is the *missing* sensitivity to chiplet_power.

## Solver verification (S8.1)

The "expected" response above came from the linearity of the heat
equation. `tools/power_gain_verification.py` repeats the ±10 % power
experiment with HotSpot on the same inputs (10 seen + 10 unseen chips,
`power_gain_verification.json`). HotSpot itself is linear to 0.2 %
(dTmax / (0.1 × rise) = 1.000 ± 0.002), which validates the method. The
surrogate's hotspot response relative to HotSpot's:

| set | gain median | range over samples |
|---|---|---|
| level4 (seen chips) | 0.28 | −0.12 … 0.70 |
| level5 (unseen chips) | 0.08 | −0.41 … 0.44 |

The ratio is higher than the analytic 0.1 because HotSpot's package is
more conductive than the generator's (smaller rise in the denominator,
S5.3), but the conclusion stands and is now solver-backed: the surrogate
reproduces between a tenth and a third of the true power response, with the
wrong sign on some samples. Power what-if questions must go to the solver.
