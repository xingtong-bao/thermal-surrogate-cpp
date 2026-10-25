# Red-black SOR thread scaling (S9.1)

Files: `sor_scaling.csv`, `sor_scaling.json` (machine block inside). Benchmark:
`bench/sor_scaling.cpp`. Solver: `src/fd_solver.cpp` (`sor_solve`), the same
code the dispatcher's last-resort backend runs at 64×64.

## Setup

- Apple M4 (4 performance + 6 efficiency cores), macOS, Apple clang 21,
  `-O2`, Homebrew libomp (OpenMP 5.1).
- Synthetic 16 mm die at n×n cells, n ∈ {64, 128, 256, 512}; uniform k,
  four 20 W blocks, lumped sink. Cell size shrinks with n, so the physics is
  fixed and only the grid gets finer.
- 200 sweeps timed (not "until converged"), so every thread count does the
  same arithmetic; median of 11 runs; nothing else running.
- speedup = t(1 thread) / t(N threads); efficiency = speedup / N.
- Separately, each configuration is solved to tolerance 1e-4 K and must give
  the same iteration count and a bit-identical field as the 1-thread run.

## Result (ms for 200 sweeps, static schedule)

| grid | 1 thread | 2 | 4 | 8 | speedup 4 / 8 | efficiency 4 / 8 | sweeps to converge |
|---|---|---|---|---|---|---|---|
| 64² | 1.00 | 2.56 | 3.63 | 9.25 | 0.28× / 0.11× | 0.07 / 0.01 | 4 548 |
| 128² | 4.51 | 4.94 | 4.92 | 10.14 | 0.92× / 0.44× | 0.23 / 0.06 | 15 429 |
| 256² | 21.25 | 14.19 | 9.42 | 12.37 | 2.26× / 1.72× | 0.56 / 0.21 | 50 178 |
| 512² | 81.66 | 44.18 | 30.10 | 34.37 | 2.71× / 2.38× | 0.68 / 0.30 | 153 834 |

Bit-identical to the 1-thread field and same iteration count: every row (28 of
28 configurations, static and dynamic). Run-to-run spread of the medians is
about ±10 % (a quick A/B on the same build gave 3.03× at 512²/4 threads; the
committed run is the full protocol).

## What the numbers say

1. **Parallel pays from 256² up, not at the production size.** A half-sweep at
   64² is ~2.5 µs of work; two barriers per sweep cost more than that, so 2–8
   threads are 2.5–9× *slower* than serial. `FdSolverParams::threads` therefore
   defaults to 1 and the dispatcher's 64×64 fallback stays serial. The parallel
   path is for larger grids (the dynamic-size path of the surrogate stops at
   512², the same place the solver starts to scale).
2. **4 threads is the sweet spot on this machine, 8 is worse.** The M4 has 4
   performance and 6 efficiency cores. With 8 threads the static split gives
   each thread the same rows, and the sweep ends when the slowest efficiency
   core finishes. Dynamic scheduling (chunks of 4 rows) was tried for exactly
   this reason and did not help at 512² (2.65× vs 2.71×, 8 threads 2.16× vs
   2.38×): the chunk hand-off costs as much as the imbalance it removes at this
   row size. A hypothesis, not a measurement: pinning the 4 threads to the
   performance cores would raise the 4-thread efficiency; macOS exposes no
   affinity API for that.
3. **Efficiency 0.68 at 4 threads: where the rest goes.** Per sweep the kernel
   touches ~10 MB at 512² (T read+write, q, kx, ky); 200 sweeps in 30 ms is
   ~65 GB/s of traffic across 4 cores, within the chip's bandwidth but no longer
   negligible against the ~7 cycles of arithmetic per cell. The two barriers
   per sweep are ~2 µs each against a 75 µs half-sweep. The remainder is the
   static split landing some rows on efficiency cores (see 2).
4. **The loop restructuring alone is worth 5–30 %.** The pre-S9.1 loop visited
   every cell twice and skipped the other colour (`legacy` column: 1.36 ms at
   64², 77 ms at 512²). The row loop with a parity start does the same
   arithmetic in the same order per cell (`FdSolver.RowSweepMatchesTheOriginalCellOrderSweepBitForBit`).
   At 512² legacy is slightly faster (77.0 vs 81.7 ms): the serial row loop has
   a worse inner-loop stride for the compiler; the difference is inside the
   run-to-run spread and the row loop is what parallelises.

## Three versions of the parallel region (why synchronisation count matters)

| version | per sweep | 256² / 4 thr | 512² / 4 thr | 512² / 8 thr |
|---|---|---|---|---|
| v1: `parallel for` per half-sweep | 2 fork/join | 9.5 ms | 30.5 ms | 40.3 ms |
| v2: one region, `single` resets a shared max, explicit barrier, `reduction(max)` | 4 sync points | 15.1 ms | 35.1 ms | 47.9 ms |
| v3: one region, per-thread padded slots, black loop `nowait` + one barrier | 2 barriers | 9.4 ms | 30.1 ms | 34.4 ms |

v2 was the textbook "amortise the fork" change and it was slower than v1 by
up to 60 %, because it added two synchronisation points per sweep. v3 keeps
one region and gets back to two barriers by removing the shared reduction
variable: each thread publishes its own max in a 128-byte-aligned slot and
reads all slots after the barrier. v1 and v3 tie at 4 threads; v3 wins at 8
(no team re-creation). The v1 numbers are the `region_per_halfsweep` column of
the csv, measured in the same run.

## Why the result does not depend on the thread count

Every neighbour of a red cell is black. Within a half-sweep a cell reads only
cells of the other colour, which were all written in the previous half-sweep,
so the operands of every cell update are the same whatever the row-to-thread
assignment, and the arithmetic per cell is identical. The only value that
crosses threads is the maximum update, and max is exact in any order. This is
asserted, not assumed: `FdSolver.ParallelSweepsAreBitIdenticalToSerial` (2, 4,
8 threads at 64² and 256², `memcmp` of the fields and equal iteration counts)
and `FdSolver.ThreadsParameterDoesNotChangeTheContractAnswer` (the production
entry point on the contract fixture).

## ThreadSanitizer

TSan cannot see the barriers inside the prebuilt OpenMP runtime, so a plain
TSan run reports 32 "races" on the shared max and on the temperature array
(false positives: every access pair is separated by a barrier). With LLVM's
Archer tool (`tools/build_archer.sh`, an OMPT plugin that forwards OpenMP
synchronisation to TSan's annotation API) the same tests run clean:

```
tools/build_archer.sh
OMP_TOOL_LIBRARIES=build-tsan/archer/libarcher.dylib TSAN_OPTIONS=ignore_noninstrumented_modules=1 \
  ./build-tsan/ts_tests --gtest_filter='FdSolver.*:Concurrency.*'
```

Result: `Archer detected OpenMP application with TSan`, all tests pass, 0
warnings (log: `sor_tsan.txt`).
