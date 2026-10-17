#!/usr/bin/env python3
"""S2.3: per-channel statistics of the S4 (level4) *training* split -> contract/stats.json

Why this file exists
--------------------
The C++ library needs two things from the training data without ever reading the
1.7 GB dataset itself:

  1. the per-channel [min, max] of the training split, because contract.md
     assertion I4 (out-of-distribution check) is "value within training range
     +/- oob_scale * range";
  2. the per-channel mean/std the model was trained with, because the ONNX
     export (S3) must normalize inputs exactly like the PyTorch checkpoint did.

Both are small numbers that belong in a committed JSON file. This script derives
them from the data and, crucially, cross-checks (2) against the normalizer that
is pickled *inside* the authors' checkpoint. If our training split differed from
theirs by even one sample the mean/std would drift and the check would fail, so
this is also the proof that we reproduce their split.

How the split is obtained
-------------------------
We reuse the benchmark's own loader (`data_provider`) rather than re-implementing
the index arithmetic: the split is "test = last 20 %, then 9:1 train/val of the
leading 80 %", case-interleaved, never shuffled. Reusing the loader means a
future change in the benchmark shows up here as a failed checkpoint cross-check
instead of as a silent mismatch.

Outputs
-------
  contract/stats.json                    statistics + checkpoint normalizer
  tests/fixtures/sample_input.npz        one in-support test sample (+ truth)
  tests/fixtures/sample_input_level5.npz one OOD sample from level5 (+ truth)

Usage
-----
    python tools/export_stats.py --bench ../external/ThermalBench
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

# Channel names in C-axis order, copied from the benchmark's docs/DATASETS.md.
# The schema (contract/schema.json) is cross-checked against this list below.
CHANNELS = ["chiplet_power", "grid_x", "grid_y", "local_thermal_k",
            "ambient_K", "h_w_m2k", "r_convec_k_per_w"]

# Units. Channels 3-6 are named with their unit by the benchmark. Channels 0-2
# were inferred in S2.3 from the raw values (see docs/results/baseline_notes.md
# and the contract): power is one number per chiplet painted over its
# footprint, grid_x/grid_y are physical cell coordinates with a per-sample
# spacing of a fraction of a millimetre.
UNITS = {
    "chiplet_power": "W, total power of the chiplet covering the cell (0 outside chiplets); inferred",
    "grid_x": "mm, physical coordinate of the cell (0 at chip edge, uniform spacing per sample); inferred",
    "grid_y": "mm, physical coordinate of the cell (0 at chip edge, uniform spacing per sample); inferred",
    "local_thermal_k": "W/(m*K)",
    "ambient_K": "K",
    "h_w_m2k": "W/(m^2*K)",
    "r_convec_k_per_w": "K/W",
}


def parse_cli():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", required=True, help="IC-ThermBench checkout")
    p.add_argument("--repo", default=os.path.join(os.path.dirname(__file__), ".."),
                   help="this repository (default: parent of tools/)")
    p.add_argument("--checkpoint", default="checkpoints/level4_UNet/model.pt",
                   help="relative to --bench; used only for the cross-check")
    p.add_argument("--fixture-index", type=int, default=0,
                   help="index within the level4 TEST split used for the fixture")
    return p.parse_args()


def describe(a: np.ndarray) -> dict:
    """min/max/mean/std/p1/p99 of a flat float array, as plain Python floats."""
    a = a.astype(np.float64, copy=False)
    return {
        "min": float(a.min()), "max": float(a.max()),
        "mean": float(a.mean()),
        # ddof=1 to match torch.std(), which the checkpoint normalizer used.
        "std": float(a.std(ddof=1)),
        "p1": float(np.percentile(a, 1)), "p99": float(np.percentile(a, 99)),
    }


def load_split(bench: str, data: str):
    """Return (xtr, ytr, xva, yva, xte, yte) exactly as run.py --task test sees them."""
    from run import parse_args                    # benchmark CLI defaults
    from data_provider.data_factory import data_provider
    sys.argv = ["run.py", "--model", "UNet", "--data", data, "--task", "test", "--quiet"]
    return data_provider(parse_args())


def main():
    cli = parse_cli()
    bench, repo = os.path.abspath(cli.bench), os.path.abspath(cli.repo)

    # Same trick as run.py: the benchmark resolves `datasets/` relative to cwd
    # and imports its packages by bare name.
    os.chdir(bench)
    sys.path.insert(0, bench)
    os.environ["PER_CHANNEL_NORM"] = "1"
    from utils.compat import install
    install()
    from utils.tools import load_checkpoint

    # ---- 1. training split of level4 -------------------------------------------
    xtr, ytr, xva, yva, xte, yte = load_split(bench, "level4")
    assert xtr.shape[-1] == len(CHANNELS), f"P={xtr.shape[-1]} != {len(CHANNELS)}"
    x = xtr.numpy()                                # (B,X,Y,Z,P) float32, no copy
    y = ytr.numpy()                                # (B,X,Y,Z)
    B = x.shape[0]
    print(f"[split] train={B} val={xva.shape[0]} test={xte.shape[0]} sample={tuple(x.shape[1:])}")

    channels = []
    for c, name in enumerate(CHANNELS):
        a = x[..., c]                               # (B,X,Y,Z) view of one channel
        st = describe(a.reshape(-1))
        # Is the channel one scalar per sample (boundary conditions) or a map?
        per_sample_const = float(np.mean(a.reshape(B, -1).min(axis=1) == a.reshape(B, -1).max(axis=1)))
        n_unique = int(np.unique(a).size)
        # Which loader axis does the coordinate vary along? Needed because the
        # MATLAB file is column-major and the names grid_x/grid_y follow the
        # MATLAB convention, not the loader's (X, Y) index names.
        varies = {
            "X_axis1": bool(np.any(a[:, 1:, :, :] != a[:, :-1, :, :])),
            "Y_axis2": bool(np.any(a[:, :, 1:, :] != a[:, :, :-1, :])),
        }
        channels.append({
            "index": c, "name": name, "unit": UNITS[name], **st,
            "fraction_of_samples_constant": per_sample_const,
            "n_unique_values_in_train": n_unique,
            "varies_along": [k for k, v in varies.items() if v],
        })
        print(f"[ch{c}] {name:17s} min={st['min']:.6g} max={st['max']:.6g} "
              f"mean={st['mean']:.6g} std={st['std']:.6g} const_frac={per_sample_const:.2f} "
              f"n_unique={n_unique} varies={channels[-1]['varies_along']}")

    out_stats = describe(y.reshape(-1))
    print(f"[out] temperature_K min={out_stats['min']:.4f} max={out_stats['max']:.4f} "
          f"mean={out_stats['mean']:.4f} std={out_stats['std']:.4f}")

    # ---- 2. cross-check with the normalizer stored in the checkpoint ------------
    # The checkpoint pickles [x_norm, model, y_norm]; x_norm.mean/std were computed
    # by make_norm(xtr) during training. If our xtr is the authors' xtr, they match.
    x_norm, _, y_norm = load_checkpoint(cli.checkpoint)
    ck_mean = x_norm.mean.detach().cpu().double().numpy().reshape(-1)
    ck_std = x_norm.std.detach().cpu().double().numpy().reshape(-1)
    our_mean = np.array([ch["mean"] for ch in channels])
    our_std = np.array([ch["std"] for ch in channels])
    rel_mean = np.abs(our_mean - ck_mean) / np.maximum(np.abs(ck_mean), 1e-12)
    rel_std = np.abs(our_std - ck_std) / np.maximum(np.abs(ck_std), 1e-12)
    ck_ymean, ck_ystd = float(y_norm.mean), float(y_norm.std)
    rel_y = max(abs(out_stats["mean"] - ck_ymean) / ck_ymean,
                abs(out_stats["std"] - ck_ystd) / ck_ystd)
    worst = float(max(rel_mean.max(), rel_std.max(), rel_y))
    print(f"[check] max relative difference vs checkpoint normalizer: {worst:.2e}")
    # fp32 accumulation over 44 M values per channel explains ~1e-5; anything
    # above 1e-3 means a different split or a different dataset file.
    if worst > 1e-3:
        raise SystemExit("training-split statistics do not match the checkpoint normalizer")

    # ---- 3. write stats.json -----------------------------------------------------
    stats = {
        "schema_version": "1.0.0",
        "source": {
            "dataset": "IC-ThermBench level4_steady (S4), CC BY 4.0",
            "split": "train = first 90 % of the leading 80 % of samples (benchmark default, never shuffled)",
            "n_train": int(B), "n_val": int(xva.shape[0]), "n_test": int(xte.shape[0]),
            "sample_shape": [int(v) for v in x.shape[1:]],
            "axes": ["X", "Y", "Z", "P"],
            "generated_by": "tools/export_stats.py",
        },
        "channels": channels,
        "output": {"name": "temperature_K", "unit": "K", **out_stats},
        "checkpoint_normalizer": {
            "description": "per-channel mean/std pickled inside checkpoints/level4_UNet/model.pt; "
                           "the ONNX/C++ path must normalize with THESE values, not with a recomputation",
            "x_mean": ck_mean.tolist(), "x_std": ck_std.tolist(),
            "y_mean": ck_ymean, "y_std": ck_ystd,
            "max_relative_difference_vs_train_split": worst,
        },
        # Filled by S2b.3 from the extrapolation curve; null means "not decided".
        "oob_scale": None,
        "notes": [
            "grid_x varies along loader axis Y (index 2) and grid_y along loader axis X (index 1): "
            "the MATLAB source is column-major and names follow its convention. Treat the "
            "(X, Y) axis names as labels of a fixed layout, not as geometry.",
            "chiplet_power is one scalar per chiplet broadcast over its footprint, so the "
            "sum over cells is NOT the chip power.",
            "ambient_K, h_w_m2k and r_convec_k_per_w are constant within a sample "
            "(global boundary conditions encoded as maps).",
        ],
    }
    stats_path = os.path.join(repo, "contract", "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[stats] -> {os.path.relpath(stats_path, repo)}")

    # ---- 4. fixtures: one in-support sample, one OOD sample ---------------------
    # Stored as (X,Y,Z,P) float32 + the ground-truth field, so later stages can
    # (a) validate the schema, (b) run parity tests, (c) check the OOD path.
    # The loader hands out a transposed *view* of the HDF5 array, so the strides
    # are not C order; contract I2 promises a C-contiguous buffer, hence the copy.
    fix_dir = os.path.join(repo, "tests", "fixtures")
    os.makedirs(fix_dir, exist_ok=True)
    i = cli.fixture_index
    np.savez_compressed(os.path.join(fix_dir, "sample_input.npz"),
                        input=np.ascontiguousarray(xte[i].numpy(), dtype=np.float32),
                        expected_output=np.ascontiguousarray(yte[i].numpy(), dtype=np.float32),
                        dataset=np.array("level4_steady"), split=np.array("test"),
                        index_in_split=np.array(i),
                        global_index=np.array(B + xva.shape[0] + i))
    _, _, _, _, x5, y5 = load_split(bench, "level5")
    np.savez_compressed(os.path.join(fix_dir, "sample_input_level5.npz"),
                        input=np.ascontiguousarray(x5[0].numpy(), dtype=np.float32),
                        expected_output=np.ascontiguousarray(y5[0].numpy(), dtype=np.float32),
                        dataset=np.array("level5_steady"), split=np.array("all"),
                        index_in_split=np.array(0), global_index=np.array(0))
    for n in ("sample_input.npz", "sample_input_level5.npz"):
        print(f"[fixture] {n}: {os.path.getsize(os.path.join(fix_dir, n)) / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
