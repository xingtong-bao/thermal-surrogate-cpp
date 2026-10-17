# S2b.2 — Where the level4 U-Net makes its errors

Produced by `tools/error_buckets.py` from the S2.2 per-sample errors (3 000
level4 test samples) joined with descriptors computed from the raw inputs.
Tables: `error_buckets.csv` (one row per bucket), `error_buckets_per_sample.csv`
(one row per sample, for S2b.4); chart: `error_buckets.png`.

Continuous descriptors are cut into 5 equal-count buckets (600 samples each);
`case` is one bucket per case (300 each). Spearman ρ is the rank correlation
between the descriptor and the per-sample MAE over all 3 000 samples; ρ_rel is
the same against the relative error MAE / (Tmax_true − ambient). Overall MAE is
1.83 K, overall relative error 3.4 %.

## One sentence per descriptor

| descriptor | ρ (MAE) | ρ_rel | MAE per bucket, low → high (K) | reading |
|---|---|---|---|---|
| true temperature rise (17–179 K) | +0.51 | −0.54 | 1.22, 1.54, 1.82, 2.07, 2.51 | Absolute error grows monotonically with how hot the chip gets, but the *relative* error falls from 4.6 % to 2.3 %: the network is better, not worse, on the hot cases in relative terms. |
| true Tmax (318–498 K) | +0.49 | −0.51 | 1.26, 1.56, 1.77, 2.03, 2.53 | Same picture as the rise, as expected since ambient has no effect (next row). |
| ambient temperature (298–328 K) | +0.02 | +0.03 | 1.85, 1.81, 1.81, 1.75, 1.93 | Flat; consistent with S2b.1 where ambient is handled as an exact offset. |
| max local power density (0.33–2.7 W/mm²) | +0.55 | +0.24 | 1.02, 1.72, 1.66, 2.27, 2.40 | The strongest monotonic driver of absolute error, and the only one that also raises the relative error: dense hotspots are where the field is hardest to predict. |
| total chip power (80–1548 W) | +0.52 | −0.12 | 1.38, 1.22, 2.22, 2.07, 2.22 | A step rather than a slope: below ~480 W the error is ~1.3 K, above it ~2.2 K, with no further growth; relative error is flat, so this is scale. |
| number of chiplets (2–61) | +0.36 | +0.31 | 1.35, 2.04, 1.97, 1.44, 2.35 | Non-monotonic; the count is confounded with the case (each case has a fixed layout family), so it is not a usable predictor on its own. |
| convection coefficient h (500–20 000 W/m²K) | −0.19 | +0.51 | 2.21, 1.95, 1.71, 1.65, 1.62 | Weak cooling (low h) gives hotter chips and a larger absolute error; relative error moves the other way because the rise shrinks faster than the error. |
| convective resistance r (0.0009–0.21 K/W) | +0.07 | −0.36 | 1.66, 1.72, 1.77, 2.03, 1.98 | Weak absolute trend, mirror image of h in relative terms. |
| chip area (10 distinct values) | +0.21 | −0.14 | 1.38, 2.35, 1.70, 1.51, 2.21 | No trend; chip size is a case property, see next row. |
| case (0–9) | +0.39 | — | 1.12, 0.98, 2.44, 1.91, 2.27, 2.04, 1.53, 1.23, 2.51, 2.27 | A 2.5× spread between the easiest (case 1, 0.98 K) and hardest (case 8, 2.51 K) layouts; cases 2, 4, 8, 9 sit above 2.2 K. Hard cases are the large, dense ones (case 8: 59×61 mm, case 2: 39×39 mm with the highest densities). |

## What this means for the product

- **Error is a function of thermal load, not of boundary conditions.** Power
  density and temperature rise explain the error; ambient, h and r barely do.
  A gate expressed in absolute Kelvin (contract §4) is therefore tightest on
  the cool, sparse designs and loosest on the hot, dense ones; the 2.3 %
  relative error on the hottest quintile is the number to quote for those.
- **The hardest 20 % of samples (rise > 88 K or density > 2.2 W/mm²) carry
  MAE ≈ 2.5 K**, still under the 3.79 K worst-5 % gate, so no bucket of the
  in-support set fails the contract. The failure mode is outside the data
  (S2b.1: power magnitude; S2b.3: range extrapolation), not inside it.
- **Case identity matters more than any single scalar**, which is consistent
  with S2b.1's finding that the network keys on layout patterns. S2b.3 should
  therefore extrapolate per channel *and* test an unseen layout (level5), not
  only scaled versions of seen ones.

## Caveats

- `n_chiplets` counts connected components of the power map per power value;
  two touching chiplets with identical power would merge. Counts range 2–61,
  so the effect is negligible for the bucket statistics.
- Quantile edges for descriptors with few distinct values (chip area, power
  with 14 levels) are uneven; the CSV records the exact edges.
