# S4.5 — Batch size, memory layout and output copy

Numbers: `throughput.json` (C++ bench `bench_throughput`, 4 intra-op threads,
median of 20 calls after warm-up, Apple M4). Cross-checked with the Python
ONNX Runtime in the same session (table at the end).

## Results

| path | layout | batch | ms per call | ms per sample | samples/s |
|---|---|---|---|---|---|
| predict_batch, model_dynamic | contract (B,X,Y,Z,P) | 1 | 20.5 | 20.5 | 48.8 |
| predict_batch, model_dynamic | contract | 8 | 236.6 | 29.6 | 33.8 |
| predict_batch, model_dynamic | contract | 32 | 932.9 | 29.2 | 34.3 |
| predict_batch, model_dynamic | contract | 64 | 1848.6 | 28.9 | 34.6 |
| raw session, model_nchw_dynamic | NCHW (B,P,X,Y) | 1 | 20.4 | 20.4 | 49.1 |
| raw session, model_nchw_dynamic | NCHW | 64 | 1848.4 | 28.9 | 34.6 |
| predict (allocates + copies) | contract | 1 | 31.3 | 31.3 | 31.9 |
| predict_into (zero-copy output) | contract | 1 | 30.8 | 30.8 | 32.5 |

Batch-64 results equal the single-sample result to 3.05e-5 K (one ulp) in
both layouts.

## What the differences mean

**Batching buys nothing on this CPU.** Per-sample cost is flat at about 29–30
ms from batch 8 upward and never beats batch 1. The U-Net is compute-bound
(four 58–348-channel convolution stages on a 64×64 grid) and ONNX Runtime
already parallelises a single sample across the intra-op threads, so a
larger batch only makes each call longer. This is the opposite of the GPU
intuition, and it means the integration should stay with single predictions
and get concurrency from *multiple sessions or threads*, not from batching.

**Layout does not matter.** The contract graph carries one Transpose node
(49 nodes vs 45 for NCHW) to turn (B,X,Y,Z,P) into the network's (B,P,X,Y).
It costs nothing measurable: 20.5 vs 20.4 ms at batch 1, 1848.6 vs 1848.4
ms at batch 64. Asking callers for NCHW buffers would move that transpose
into their code for a 0.3 % gain. Keep the contract layout.

**Zero-copy output is a 1.7 % effect.** The output is 16 KB; copying it is
~2 µs next to 30 ms of convolutions. `predict_into` exists for callers that
own their buffers (no allocation per call), not for speed. The input is
already zero-copy in both variants (the tensor wraps the caller's memory).

**Two steady states of the machine.** Across sessions the same batch-1 call
measured either ~20 ms or ~30 ms at 4 threads (also visible in the Python
cross-check: 20.5 vs 30.2 ms for the same model). The scheduler placing
threads on performance vs efficiency cores is the likely cause; it is not a
property of the graph. Fixed vs dynamic model, and all graph-optimisation
levels, are within noise of each other. With 8 threads (performance plus efficiency cores) the C++ bench gives p50 27 ms
with a wider spread (min 19, p95 34 ms); the Python cross-check reached 14 ms. See
`latency_single.json`.

## Python cross-check (ONNX Runtime 1.23.2, 4 threads unless stated, median ms)

| experiment | result |
|---|---|
| model.onnx vs model_dynamic.onnx, batch 1, either order | 28.8–30.5 vs 30.2–30.5 |
| dynamic batch 1 / 2 / 4 / 8 / 16, ms per sample | 30.2 / 40.8 / 29.9 / 29.6 / 29.3 |
| graph optimisation disable / basic / extended / all | 30.5 / 30.7 / 31.0 / 29.4 |
| dynamic batch 1, threads 1 / 2 / 4 / 8 / 10 | 79.8 / 39.3 / 20.5 / 14.2 / 14.0 |
