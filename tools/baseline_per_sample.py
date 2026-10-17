#!/usr/bin/env python3
"""S2.2 companion: per-sample error distribution of the level4 U-Net baseline.

Why this exists
---------------
The benchmark's `run.py --task test` reports dataset-wide *averages* (MAE, RMSE,
mean |Tmax error| ...). Two of the three gates in contract/contract.md section 4
are *distribution* statistics instead:

  * "worst-5 % per-sample MAE"  -> 95th percentile of the per-sample MAE
  * "tmax absolute error (median)" -> median of |max(pred) - max(label)|

Those cannot be derived from the averages, so this script re-runs the same
checkpoint through the same benchmark code path (`exp.exp_operator._predict`)
and keeps the per-sample numbers. It reuses the benchmark's own loader,
normalizer and prediction helpers on purpose: if we re-implemented any of them
here, a mismatch would be indistinguishable from a real accuracy regression.

The dataset-wide MAE printed by this script must equal the one in
baseline_unet_level4.json (same model, same split, same device); that equality
is the self-check that the two scripts looked at the same thing.

Usage
-----
    python tools/baseline_per_sample.py --bench ../external/ThermalBench \
        --device cpu --output docs/results/baseline_unet_level4_per_sample.json

The output JSON contains summary statistics, the three gate values
(baseline x 1.1, as the contract prescribes) and the raw per-sample arrays
rounded to 4 decimals so S2b (failure analysis) can read them without a GPU.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch


def parse_cli():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", required=True, help="IC-ThermBench checkout")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", required=True, help="where to write the JSON")
    p.add_argument("--checkpoint", default="checkpoints/level4_UNet/model.pt",
                   help="relative to --bench")
    p.add_argument("--data", default="level4")
    return p.parse_args()


def main():
    cli = parse_cli()
    bench = os.path.abspath(cli.bench)
    out_path = os.path.abspath(cli.output)

    # The benchmark resolves `datasets/` and the checkpoint relative to its own
    # directory and imports its packages by bare name, so mimic what run.py does:
    # chdir there and put it on sys.path before importing anything from it.
    os.chdir(bench)
    sys.path.insert(0, bench)
    os.environ["PER_CHANNEL_NORM"] = "1"      # mandatory for P=7 data (see run.py)

    from utils.compat import install          # legacy pickled checkpoints
    install()
    from run import parse_args                # reuse the benchmark's CLI defaults
    from data_provider.data_factory import data_provider
    from exp.exp_operator import _predict
    from utils.tools import load_checkpoint, set_seed

    # Build the exact argument set run.py would have used for `--task test`.
    sys.argv = ["run.py", "--model", "UNet", "--data", cli.data, "--task", "test",
                "--load", cli.checkpoint, "--device", cli.device,
                "--seed", "0", "--per_channel_norm", "1", "--quiet"]
    args = parse_args()
    set_seed(args.seed)

    # Loader returns (xtr, ytr, xva, yva, xte, yte); only the test split is used.
    _, _, _, _, xte, yte = data_provider(args)
    x_norm, model, y_norm = load_checkpoint(cli.checkpoint)
    model = model.to(cli.device)

    # pred/label: (B,X,Y,Z) in Kelvin, Z=1 for this dataset.
    pred = _predict(model, x_norm, y_norm, xte, "UNet", cli.device).numpy().astype(np.float64)
    label = yte.numpy().astype(np.float64)
    B = pred.shape[0]
    pred2d, label2d = pred.reshape(B, -1), label.reshape(B, -1)

    # ---- per-sample statistics -------------------------------------------------
    abs_err = np.abs(pred2d - label2d)                       # (B, X*Y)
    mae_per_sample = abs_err.mean(axis=1)                    # field MAE per sample
    maxae_per_sample = abs_err.max(axis=1)                   # worst pixel per sample
    tmax_err = np.abs(pred2d.max(axis=1) - label2d.max(axis=1))   # hotspot temperature
    # Hotspot *location*: grid distance between argmax(pred) and argmax(label).
    # The contract's `tmax_index` field is only useful if this is usually 0.
    X, Y = pred.shape[1], pred.shape[2]
    ip, il = pred2d.argmax(axis=1), label2d.argmax(axis=1)
    dx = np.abs(ip // Y - il // Y)
    dy = np.abs(ip % Y - il % Y)
    loc_dist = np.maximum(dx, dy)                            # Chebyshev distance in cells

    def pct(a):
        return {"mean": float(a.mean()), "median": float(np.median(a)),
                "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
                "max": float(a.max())}

    summary = {
        "n_test": int(B),
        "field_shape": [int(X), int(Y)],
        "per_sample_mae_K": pct(mae_per_sample),
        "per_sample_max_abs_err_K": pct(maxae_per_sample),
        "tmax_abs_err_K": pct(tmax_err),
        "tmax_location_chebyshev_cells": {
            "exact_match_fraction": float((loc_dist == 0).mean()),
            "within_1_cell_fraction": float((loc_dist <= 1).mean()),
            "within_2_cells_fraction": float((loc_dist <= 2).mean()),
            "median": float(np.median(loc_dist)), "max": int(loc_dist.max()),
        },
    }
    # Gates: contract section 4 says baseline x 1.1; the factor only covers fp32
    # export drift. Values are stored unrounded; the contract table rounds them.
    gates = {
        "field_mae_K": float(mae_per_sample.mean() * 1.1),
        "worst5pct_per_sample_mae_K": float(np.percentile(mae_per_sample, 95) * 1.1),
        "tmax_abs_err_median_K": float(np.median(tmax_err) * 1.1),
        "factor": 1.1,
    }
    result = {
        "model": "UNet", "data": cli.data, "device": cli.device,
        "checkpoint": cli.checkpoint, "seed": 0, "per_channel_norm": 1,
        "summary": summary, "gates": gates,
        "per_sample": {
            "mae_K": [round(float(v), 4) for v in mae_per_sample],
            "max_abs_err_K": [round(float(v), 4) for v in maxae_per_sample],
            "tmax_abs_err_K": [round(float(v), 4) for v in tmax_err],
            "tmax_location_chebyshev_cells": [int(v) for v in loc_dist],
        },
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Self-check line: must match baseline_unet_level4.json's unet/mean_absolute_error.
    print(f"dataset MAE = {abs_err.mean():.6f} K  (compare with run.py output)")
    print(json.dumps({"summary": summary, "gates": gates}, indent=2))
    print(f"[per-sample] -> {out_path}")


if __name__ == "__main__":
    main()
