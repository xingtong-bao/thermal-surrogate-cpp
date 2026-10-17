# S2.2 — Baseline reproduction notes (level4 U-Net)

Goal: prove that this machine loads the IC-ThermBench data and the authors'
checkpoint exactly as the authors did, so that every later accuracy comparison
(ONNX export in S3, C++ inference in S4) is anchored to a trusted number.

## How it was run

```
tools/run_baseline.sh --device cpu          # writes the three JSON files below
tools/baseline_per_sample.py --bench ../external/ThermalBench --device cpu \
    --output docs/results/baseline_unet_level4_per_sample.json
```

`run_baseline.sh` calls the benchmark's own `run.py --task test` unmodified.
Settings that must stay identical in S2.3 (`stats.json`) and in the C++ parity
tests: `--seed 0`, `--per_channel_norm 1` (mandatory for the P=7 levels, see the
benchmark's `run.py` docstring), evaluation batch 20, checkpoint
`checkpoints/level4_UNet/model.pt`. Test split = last 20 % of the 15 000 level4
samples (3 000), fixed by the benchmark loader, not by the seed.

Environment (`baseline_env.json`): Python 3.10.22, torch 2.0.1, numpy 1.26.4,
macOS arm64, Apple M4, device `cpu`.

## Results vs. the authors' published metrics

Authors' numbers come from `authors_level4_unet_test_metrics.json`, the file
shipped next to the checkpoint in the benchmark's archive.

| metric (level4 test, 3 000 samples) | ours (cpu) | authors | diff |
|---|---|---|---|
| MAE (K) | 1.82926 | 1.82907 | +1.9e-4 |
| RMSE (K) | 2.55099 | 2.55083 | +1.6e-4 |
| max abs error (K) | 11.16485 | 11.16458 | +2.7e-4 |
| Tmax abs error, mean (K) | 1.18551 | 1.18505 | +4.6e-4 |
| top-50 hottest-pixel MAE (K) | 1.46399 | 1.46364 | +3.5e-4 |
| R² | 0.990178 | 0.990179 | −1e-6 |

All differences are below 5e-4 K, i.e. two to three orders of magnitude under
the 0.01 K acceptance threshold. The residual is consistent with running the
same fp32 network on a different backend (authors: CUDA; here: CPU on arm64) —
convolution kernels accumulate in a different order and the differences
propagate through four U-Net stages. Nothing in the data path differs.

Cross-check on this machine: the same evaluation on the Apple GPU (`--device
mps`) agrees with the CPU run to 2e-6 on every metric, which confirms that the
remaining 1e-4-level gap to the authors is backend-related, not a loading bug.

Wall time for the 3 000-sample test pass: 122 s on CPU, 18 s on mps (includes
loading the 1.7 GB input file both times). These are PyTorch numbers and are
*not* the latency figures the project reports; those come from the C++ path.

## OOD zero-shot on level5 (recorded, not gated)

level5 is the benchmark's pure extrapolation set: five unseen cases, all 5 000
samples evaluated, no split, same level4 checkpoint.

| metric (level5, 5 000 samples) | value |
|---|---|
| MAE (K) | 16.87 |
| RMSE (K) | 19.10 |
| max abs error (K) | 41.00 |
| Tmax abs error, mean (K) | 27.90 |
| R² | 0.386 |

MAE is ~9× the in-support value and R² collapses from 0.99 to 0.39. This is
exactly why the contract (§2, §5) refuses out-of-distribution inputs and routes
them to the reference solver instead of trusting the surrogate: the model gives
no warning of its own, the field still looks plausible. S2b uses this set to
pick the `oob_scale` default and the thresholds for the fallback.

## Per-sample distribution and the contract gates

From `baseline_unet_level4_per_sample.json` (3 000 test samples):

| statistic | median | p95 | p99 | max |
|---|---|---|---|---|
| per-sample field MAE (K) | 1.664 | 3.447 | 4.842 | 9.168 |
| per-sample max abs error (K) | 9.96 | 24.01 | 32.49 | 66.23 |
| Tmax abs error (K) | 0.865 | 3.173 | 5.679 | 36.26 |

Gates written into `contract/contract.md` §4 (baseline × 1.1):

| gate | value |
|---|---|
| field MAE over all test samples | 2.012 K |
| worst-5 % per-sample MAE (p95) | 3.792 K |
| Tmax absolute error (median) | 0.952 K |

The 1.1 factor exists only to absorb fp32 export drift between PyTorch, ONNX
Runtime and the C++ build; it is not a tolerance for a worse model.

## Finding worth carrying into S2b

The hotspot *temperature* is reproduced well (median error 0.87 K) but the
hotspot *location* is not: argmax(prediction) equals argmax(label) in only
3.7 % of samples, is within 1 cell in 25 % and within 2 cells in 45 %; the
median Chebyshev distance is 3 cells, the worst 49. The likely explanation is
that the true field is nearly flat around the maximum, so the argmax is decided
by sub-Kelvin noise. The contract therefore marks `tmax_index` as informative
only. S2b should check how flat the top-k region is before deciding whether a
location metric belongs in the product at all.

## What would make this comparison invalid

- Running with `--per_channel_norm 0`: the three boundary-condition channels
  get crushed and MAE degrades by orders of magnitude.
- Running `run.py` from another working directory: the datasets path and
  checkpoint path are relative, the benchmark silently fails to find them.
- Changing the split logic in the loader: the test set would no longer be the
  one the checkpoint was *not* trained on.
