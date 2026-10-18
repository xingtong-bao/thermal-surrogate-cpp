# S8.3 — Solver-to-model lifecycle: can a solver answer improve the surrogate, safely?

Scripts: `tools/finetune_fewshot.py` (fine-tune, benchmark few-shot protocol,
weights saved), `tools/lifecycle_v11.py` (export → parity → I5 bank →
conformal recalibration → C++ evaluation). Results: `lifecycle_fewshot_k10.json`,
`lifecycle_fewshot_k50.json`, `lifecycle_fewshot_k50_replay250.json`, per-sample CSVs
`lifecycle_k*_level5_holdout_v1.*.csv`, `lifecycle_k*_level4_eval50_v1.1.csv`.
Artefacts per version in `models_v1.1_k10/`, `models_v1.1_k50/`, `models_v1.1_k50_replay250/`
(metadata, bank, conformal, stats; ONNX files gitignored like v1.0).

## The idea

In a design-exploration product every time the dispatcher falls back to the
solver, the (input, field) pair is a free training sample for the chips the
model has never seen. Here the "solver answers" are the benchmark's few-shot
samples: the first K samples of each of the five unseen chips. The rest of
the lifecycle is what a model update must survive in this library: the same
gates as v1.0.

## Fine-tuning (2 500 held-out samples of the unseen chips, PyTorch)

| model | shots per chip | fine-tune time | held-out MAE | Tmax error |
|---|---|---|---|---|
| v1.0 (no fine-tuning) | 0 | — | 17.09 K | 28.1 K |
| v1.1 k10 | 10 (50 samples) | 238 s | 5.58 K | 11.6 K |
| v1.1 k50 | 50 (250 samples) | 1 200 s | 2.78 K | 4.3 K |

## Through the gates (C++ dispatcher, surrogate → FD reference, no HotSpot)

| | v1.0 | v1.1 k10 | v1.1 k50 |
|---|---|---|---|
| parity ONNX Runtime vs PyTorch (shots) | 0 K / 8e-4 K | 3.1e-3 K | 1.1e-3 K (gate 5e-3) |
| I5 bank | 10 800 | 10 850 | 11 050 configurations |
| conformal 90 % MAE bound (recalibrated on the pool minus the shots) | — | 9.9 K | 4.5 K |
| unseen chips, held-out 50: surrogate MAE | 15.75 K | 4.88 K | 2.61 K |
| unseen chips: fallback rate / dispatcher MAE | 100 % / 19.7 K | 34 % / 6.4 K | 24 % / 3.5 K |
| **seen chips (eval50): surrogate MAE, gate 2.012 K** | **1.05 K, pass** | **3.66 K, FAIL** | **5.79 K, FAIL** |

## What the gates found

1. **Naive fine-tuning forgets the original chips.** 50 epochs on the new
   chips alone take the seen-chip error from 1.05 K to 3.7–5.8 K, past the
   contract gate. Without the gate a "better" v1.1 would have shipped and
   silently degraded every existing design. This is the single most useful
   thing the lifecycle produced.
2. **The I5 bank must grow with every solver answer, and the fallback must
   be good.** With only the shot configurations added, 24–34 % of the
   held-out samples of the *same* chips still read as novel (placements
   differ), and the FD fallback is worse than the fine-tuned surrogate, so
   the dispatcher scores worse than surrogate-only. In the product the
   fallback is the real solver and every answer extends the bank.
3. **Error bounds have to be recalibrated per version**, and they widen
   honestly: 9.9 K at 10 shots, 4.5 K at 50, against 2.9 K for v1.0 on its
   own distribution.

## Mitigation: rehearsal (part 2)

The fine-tuning set becomes the 250 new-chip samples plus 250 level4
*training* samples (never its test split), same 50 epochs (2 035 s).

| | v1.1 k50 (no replay) | v1.1 k50 + replay 250 |
|---|---|---|
| unseen chips, held-out 2 500 (PyTorch) | 2.78 K | 2.87 K |
| unseen chips, held-out 50 (C++): surrogate / dispatcher / fallback | 2.61 K / 3.47 K / 24 % | 2.83 K / 3.76 K / 24 % |
| seen chips (eval50): MAE (gate 2.012) | 5.79 K, fail | **1.45 K, pass** |
| seen chips: worst-5 % per-sample MAE (gate 3.792) | 11.6 K, fail | **2.10 K, pass** |
| seen chips: Tmax error median (gate 0.952) | — | **1.19 K, fail** |
| parity / conformal 90 % bound | 1.1e-3 K / 4.5 K | 2.8e-4 K / 4.7 K |

Rehearsal removes most of the forgetting (5.8 → 1.45 K on the seen chips)
at no cost on the new ones (2.78 → 2.87 K), but the hotspot gate still fails
by 25 %: the update is **not releasable as a replacement for v1.0** under
the contract. Two honest ways forward, both outside this story's budget:
more replay or a lower learning rate (cheap to try, measurable with the same
scripts), or keeping one model per chip family and routing by the I5 bank
(no forgetting by construction, more artefacts to version). What the story
delivers is the machinery and the measurement that make that decision a
number, not an opinion.
