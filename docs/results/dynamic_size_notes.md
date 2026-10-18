# S4.6 — Larger grids and the "million-cell mesh" question

Data: `dynamic_size.csv` (bench `bench_dynamic_size`, models/model_dynamic.onnx,
4 threads, median of 5 calls, Apple M4 16 GB). Synthetic inputs = the 64×64
fixture tiled; the network is fully convolutional so cost depends on cell
count only.

| grid | cells | median ms | ms per 1k cells | peak RSS after (MB) |
|---|---|---|---|---|
| 64×64 | 4 096 | 20.7 | 5.06 | 75 |
| 128×128 | 16 384 | 117.2 | 7.15 | 100 |
| 256×256 | 65 536 | 468.0 | 7.14 | 196 |
| 512×512 | 262 144 | 1 871.7 | 7.14 | 620 |

Time is linear in the cell count from 128² upward (7.14 ms per thousand
cells, i.e. the 64² case still enjoys cache residency); memory grows ~1.6 MB
per thousand cells (activations of the widest stage, 348 channels × float32,
kept for the skip connections). Extrapolating: a 1024×1024 grid (1.05 M
cells) would take ~7.5 s and ~2.4 GB of RSS on this machine, single call.
The outputs stay finite and in the same range at every size, but note that
"finite" is all this says: the model was trained on 64×64 chips and S2b.3
shows it does not extrapolate physics, so a larger grid is a cost
measurement, not a usable prediction.

## Design note: how a million-cell mesh would be served

1. **Tile with a halo, not one call.** The U-Net's receptive field is bounded
   (three 2× poolings plus 3×3 convolutions: about 44 cells). Split the mesh
   into 256×256 tiles with a 32-cell halo on each side, run each 320×320
   tile independently (≈0.7 s, ≈300 MB), and keep only the interior. Peak
   memory becomes the tile's, not the mesh's; tiles can run on separate
   threads or machines.
2. **Stream the output.** Write each tile's interior straight into the
   caller's full-mesh buffer (`predict_into` already binds the output to
   caller memory); never materialise a second copy of the field.
3. **Switch execution provider past ~100 k cells.** On this CPU 7 ms per
   thousand cells means a million cells is seconds; a GPU execution provider
   (CUDA/CoreML) turns the same ONNX file into tens of milliseconds per
   tile. The switch point is a latency budget, not a correctness line: below
   ~100 k cells the CPU path is simpler and deterministic; above it, batch
   tiles on the GPU (where batching *does* pay off, unlike on this CPU).
4. **Validate per tile, fall back per tile.** Rules I4/I5 are evaluated on
   each tile's configuration; a tile outside the trusted region goes to the
   reference solver while its neighbours use the surrogate.
