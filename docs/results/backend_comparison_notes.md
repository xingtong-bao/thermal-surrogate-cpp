# S5.3 / S5.5 — Three backends on the same 100 samples

Data: `backend_comparison.csv` (aggregate), `backend_comparison_level4.csv` and
`backend_comparison_level5.csv` (per sample), produced by
`ts_demo --compare` with HotSpot as the external reference. 50 level4 test
samples (committed subset) and 50 level5 samples (five unseen layouts).
Latencies are per call on the M4, surrogate with ONNX Runtime default threads.

| set | backend | MAE (K) | p95 MAE (K) | Tmax err (K) | median latency | dispatcher chose |
|---|---|---|---|---|---|---|
| level4 (seen chips) | surrogate | 1.05 | 1.86 | 0.75 | 22 ms | — |
| | FD reference (in-process) | 12.77 | 24.3 | 17.2 | 34 ms | — |
| | HotSpot (external) | 13.37 | 27.4 | 18.5 | 2.7 s | — |
| | **dispatcher** | **1.05** | 1.86 | 0.75 | 23 ms | surrogate 50/50, fallback 0 % |
| level5 (unseen chips) | surrogate | 14.82 | 36.9 | 23.5 | 22 ms | — |
| | FD reference | 17.79 | 35.8 | 15.1 | 41 ms | — |
| | HotSpot (external) | 3.36 | 7.0 | 3.9 | 2.0 s | — |
| | **dispatcher** | **3.36** | 7.0 | 3.9 | 2.0 s | external 50/50, fallback 100 % |

## Reading

- **The dispatcher does what the contract promised.** Seen chips go to the
  surrogate (1.05 K, 22 ms); every unseen chip is caught by rule I5 and
  answered by the first working reference (HotSpot, 3.4 K instead of the
  surrogate's 14.8 K). Blended error on the 100 samples: 2.2 K, versus
  7.9 K surrogate-only. Fallback rate: 0 % in-support, 100 % out-of-support,
  exactly the S2b.4 prediction.
- **Both references are reduced models of the generator.** The dataset was
  produced by HotSpot with a package configuration that is not published.
  Our HotSpot wrapper rebuilds the tool (ambient, r_convec, sink side from
  the h–r relation, per-block conductivity) and lands within 3.4 K on the
  level5 chips but 13 K low on the level4 chips, where the generator's
  package must have been more resistive. The in-process FD solver, with four
  scalars calibrated on 20 training samples, is 13–18 K off on both. Neither
  is an oracle; they are physics-consistent fallbacks, and the numbers above
  are what a user actually gets.
- **Cost of the fallback.** HotSpot costs ~100× the surrogate (2–2.7 s vs
  22 ms); the FD solver 1.5× (34–41 ms) but is not more accurate than the
  surrogate on unseen chips, so for this dataset the right chain is
  surrogate → HotSpot → FD (FD only when HotSpot is unavailable). With a
  cost ratio R ≈ 100 and the S2b.4 fallback rates, a 50/50 stream costs
  ~51 surrogate-call units per query; an in-support stream 1.0.
- **Where the surrogate error sits is unchanged** (S2b.2): the 50-sample
  level4 MAE of 1.05 K is below the 1.83 K of the full test split because the
  every-60th subset under-samples the hottest cases; the accuracy-gate test
  (S6.1) uses this fixed subset with the contract's gates.
