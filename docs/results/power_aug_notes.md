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

## Second attempt: hotspot-weighted loss and a lower learning rate (v1.3)

Same augmentation, learning rate halved (5e-5), and the loss weight of the
64 hottest true cells of every sample raised 5× (`--hotspot-weight 4
--hotspot-topk 64`), 12 epochs, 32 min (`models_v1.2_poweraug_hot/`,
`power_aug_lifecycle_poweraug_hot.json`).

| | v1.2 (plain) | v1.3 (hotspot-weighted, lr/2) |
|---|---|---|
| power gain, analytic (×0.9 / ×1.1) | 0.90 / 0.93 | 0.85 / 0.90 |
| seen chips MAE / gate 2.012 | 1.46 K | 1.55 K |
| seen chips worst-5 % / gate 3.792 | 1.98 K | 1.99 K |
| seen chips Tmax median / gate 0.952 | 1.13 K | **1.36 K (worse)** |
| unseen chips MAE | 13.0 K | 12.2 K |

Weighting the hottest cells did not move the hotspot gate the right way;
it slightly hurt everything on the seen chips while the gain stayed. The
likely reason: the augmented targets scale the *whole* rise, so a loss
focused on the top cells of each scaled sample rewards fitting the scaling
law at the hotspot rather than the unscaled hotspot the gate measures.
Recorded as a negative result. Untried and cheap: a narrower scaling range
(0.85–1.15), fewer epochs with early stopping on the Tmax metric, or
freezing the encoder. Either way the trade-off remains explicit: what-if
correctness versus ~0.2–0.4 K of static hotspot accuracy, and the gate
decides, not a preference.

## Third round: narrower scaling, early stopping, frozen encoder (v1.4)

Two more variants, each with the contract gates evaluated on the 50-sample
subset every two epochs and the best epoch kept (`--eval-every 2`):

| variant | gain | eval50 MAE / p95 / Tmax median | 50-sample gates | full 3 000-sample gates |
|---|---|---|---|---|
| A: f ~ U(0.85, 1.15), all layers, 12 epochs | 0.85 | 1.415 / 1.860 / 0.725 K | **pass, pass, pass** | MAE 2.072 (gate 2.012) fail; p95 3.59 pass; Tmax 1.087 (gate 0.952) fail |
| B: f ~ U(0.7, 1.3), encoder + bottleneck frozen | 0.37 | 2.06 / 3.08 / 3.47 K | fail | — |
| reference v1.0 | 0.10 | 1.05 / 1.66 / 0.61 K | pass | MAE 1.829, p95 3.447, Tmax 0.865: pass |

Artefacts and C++ checks for A: `models_v1.2_poweraug_narrow/`,
`power_aug_lifecycle_poweraug_narrow.json` (parity 6e-5 K; HotSpot-verified
gain 1.66 seen / 0.20 unseen); full-split numbers in
`power_aug_narrow_fulltest.json`.

Three things learned:

1. **The narrower range is the right knob.** Scaling by ±15 % instead of
   ±30 % keeps the gain at 0.85 and costs the hotspot far less (0.73 K vs
   1.13 K on the subset). Early stopping did its job: epoch 10 of the same
   run had briefly failed the hotspot gate (1.37 K) and was skipped.
2. **Freezing the encoder does not work.** The power dependence lives in
   the encoder features, not only in the decoder: with the encoder frozen
   the gain stops at 0.37 and the hotspot degrades to 3.5 K.
3. **The 50-sample subset is a smoke gate, not a release gate.** Candidate
   A passes all three gates on the committed subset and fails two of them
   on the full 3 000-sample split (MAE 2.072 vs 2.012, Tmax median 1.087 vs
   0.952), because the every-60th subset under-samples the hot, dense
   cases. The lifecycle script now runs the full-split check as the release
   decision, and the contract says which is which. Candidate A stays a
   candidate: a model with a correct power response, 0.24 K worse on the
   field and 0.22 K worse on the hotspot than the published one. Whether a
   product takes that trade is its decision; the numbers are on the table.
