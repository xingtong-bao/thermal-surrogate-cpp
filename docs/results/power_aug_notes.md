# S8.5 — Fixing the power response with physics-informed augmentation

Scripts: `tools/finetune_power_aug.py` (training), `tools/lifecycle_poweraug.py`
(release gates). Results: `power_aug_lifecycle.json`,
`power_gain_verification_v1.2.json`, `power_aug_level4_eval50_v1.2.csv`,
`power_aug_level5_v1.2.csv`, chart `power_aug_gain.png`. Artefacts:
`models_v1.2_poweraug/` (metadata, bank, stats, FD params, recalibrated
conformal table; ONNX via the usual export).

## Idea

The published model reproduces only 0.1–0.3 of the true hotspot change when
chiplet power changes (S2b.1, S8.1), because its training set has 14
discrete power values tied to layouts. Steady-state conduction is linear in
the power (HotSpot: 1.000 ± 0.002), so every stored pair (x, T) yields exact
new pairs by scaling the power and the temperature rise together:
x′ = x with power × f, T′ = T_amb + f·(T − T_amb). No solver call needed.
Fine-tune the published weights on randomly scaled copies (f ~ U(0.7, 1.3),
one in four unscaled), all 10 800 training samples, 12 epochs, lr 1e-4,
29 min on the Apple GPU (mps).

## Results

| | v1.0 (published) | v1.2 (power-augmented) |
|---|---|---|
| power gain, analytic, 200 test chips (×0.9 / ×1.1) | 0.12 / 0.09 | **0.90 / 0.93** |
| power gain vs HotSpot, 10 seen / 10 unseen chips (median) | 0.28 / 0.08 | 1.60 / 0.28 |
| seen chips (eval50): MAE / gate 2.012 | 1.05 K | 1.46 K, pass |
| seen chips: worst-5 % / gate 3.792 | 1.66 K | 1.98 K, pass |
| seen chips: Tmax median error / gate 0.952 | ~0.6 K | **1.13 K, fail** |
| unseen chips (50): surrogate MAE | 14.8 K | 13.0 K (still OOD, 100 % fallback) |
| parity ONNX Runtime vs PyTorch | 0 K | 6.1e-5 K |
| conformal 90 % MAE bound (validation split) | 2.86 K | recalibrated, `models_v1.2_poweraug/conformal.json` |

A smaller run (2 000 samples, 8 epochs, 4 min) already moved the gain to
0.5–0.7 at a cost of 1.80 K MAE; the full run closes it.

## Reading

- **The flaw was the data, not the model.** Thirty minutes of augmentation
  built from a physical law the solver confirmed take the power response
  from "ignored" to 0.9 of the truth, on the same architecture. This is the
  concrete form of the requirement in the README: generate training data
  that varies each physical knob independently.
- **Why the HotSpot ratio reads 1.6 on seen chips:** HotSpot's package is
  more conductive than the dataset's generator (S5.3), so its temperature
  rise, the denominator, is smaller; against the dataset's own rise the gain
  is 0.9. On the unseen chips both numbers remain low: the model still does
  not know those layouts, which augmentation cannot fix (S8.3 can).
- **The release gate still says "candidate".** The field error stays well
  inside its gate (1.46 vs 2.01 K) but the hotspot median error rises from
  ~0.6 K to 1.13 K against a 0.95 K gate. The trade-off is explicit and
  measured: a model that answers what-ifs correctly, for 0.5 K of hotspot
  accuracy on the static designs. Whether that trade is worth taking is a
  product decision; a lower learning rate or a loss term on the hotspot are
  the obvious next experiments, runnable with the same scripts.
