# S8.7 / S8.8 — Swapping the model (U-FNO) and a two-model ensemble

Scripts: `tools/export_ufno_onnx.py`, `tools/export_ensemble_onnx.py`.
Results: `ufno_level4.csv`, `ufno_level5.csv`, `ensemble_level4.csv`,
`ensemble_spread.json`, `latency_ufno.json`, `latency_ensemble.json`.
Artefacts: `models_ufno/`, `models_ensemble/` (same layout as `models/`; the
C++ library loads them unchanged).

## Why U-FNO, and what it took to export it

The benchmark's own tables rank the neural operators above the U-Net on the
in-support scopes (S4 RMSE: Therm-FM 0.93, SAU-FNO 1.22, U-FNO ≈ 1.33,
U-Net 2.55). U-FNO's level4 checkpoint was pulled from the 46 GB archive
with 8 range requests (33.6 MB). Its six spectral layers call `torch.fft`,
which the PyTorch 2.0 exporter cannot translate. On a fixed 64×64 grid the
transform is a linear map, so each layer was rewritten as matrix products
with precomputed cosine/sine matrices over the 20×20 retained modes. A
detail that cost an hour: the model pads Z = 1 to 8 before the FFT and keeps
only kz = 0, so the Z transform is a sum forward and a 1/8-scaled broadcast
back. The rewrite reproduces the FFT layers to 3e-5 K (one float32 ulp) and
the exported graph matches PyTorch to 6e-5 K. Einsum ops were replaced by
MatMul; it made no difference to speed (see below).

## Accuracy and cost (C++ library, Apple M4, 4 threads)

| model | seen chips (eval50) MAE / p95 / Tmax median | unseen chips (50) MAE | latency p50 | size |
|---|---|---|---|---|
| U-Net (v1.0, shipped) | 1.05 / 1.66 / 0.60 K | 14.8 K | **30 ms** | 20 MB, 49 nodes |
| U-FNO | **0.60 / 0.83 / 0.27 K** | 24.6 K | 760 ms | 32 MB, 677 nodes |
| ensemble (mean of both) | 0.69 / 1.02 / 0.27 K | 18.4 K | 745 ms | 52 MB |

Authors' full-split numbers agree with the direction: U-FNO MAE 1.02 K vs
U-Net 1.83 K on the 3 000 test samples (`level4_UFNO/test_metrics.json`).

**Where the 25× goes.** Profiling the ONNX graph by operator: 61 % of the
time in `ConvTranspose` and 28 % in `Conv`, all inside the small 3-D U-Nets
that U-FNO runs on the Z-padded 64×64×8 volume; the spectral (DFT) part is
~1 %. PyTorch itself needs 857 ms for the same model on this CPU, so the
cost is the architecture, not the export. A 2-D variant of those blocks
(Z = 1 throughout) would remove most of it; that is a retraining job.

**Unseen chips.** U-FNO is *worse* out of distribution (24.6 vs 14.8 K),
consistent with the benchmark's finding that U-Net is the runner-up on S5
zero-shot. Accuracy in support and robustness out of support rank the
models differently; the dispatcher's guards matter more, not less, with the
more accurate model.

## Ensemble spread as a free uncertainty signal

The disagreement between the two models, mean |T_unet − T_ufno| per sample:

| | seen chips | unseen chips |
|---|---|---|
| spread | 1.04 K | **19.6 K** |
| Spearman(spread, ensemble error) | 0.70 | 0.43 |

The spread is 19× larger on unseen chips and tracks the error on seen ones:
a novelty and uncertainty indicator that needs no calibration set and no
training-set bank, at the price of running both models. It confirms rule I5
from an independent direction; it does not replace the conformal bounds,
which carry a guarantee.

## Verdict for the product

- For CPU-only deployment at ~30 ms, keep the U-Net (or retrain a 2-D U-FNO).
- Where 0.75 s per call is acceptable, or a GPU execution provider is
  available, U-FNO halves the in-support error and passes every contract
  gate with margin (Tmax median 0.27 K vs the 0.95 K gate).
- Model choice does not change the integration: both exports speak the same
  contract and run through the same validator, dispatcher and gates.
