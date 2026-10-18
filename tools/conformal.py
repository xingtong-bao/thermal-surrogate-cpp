#!/usr/bin/env python3
"""S8.2: per-prediction error bounds by split conformal prediction.

Why
---
An optimiser (or an engineer) needs to know how far a surrogate answer may
be off *for this input*, not just the dataset-wide MAE. Split conformal
prediction gives a distribution-free guarantee: calibrate the quantile of a
nonconformity score on held-out data, and future scores fall below that
quantile with the chosen probability, for inputs exchangeable with the
calibration set. "Exchangeable" is exactly what rules I4/I5 enforce, so the
bound applies to in-support inputs and is meaningless for OOD ones (which the
library routes elsewhere anyway).

Calibration set: the level4 VALIDATION split (1 200 samples the model did
not train on; the test split stays untouched for evaluation).
Scores: per-sample field MAE and |Tmax error| (both in K).
Conditioning: the predicted temperature rise (tmax_pred - ambient) is known
at inference time and S2b.2 showed the error grows with it, so quantiles are
also given per rise quintile (Mondrian / group-conditional conformal), which
tightens the bound for cool designs.

Writes contract/conformal.json; the C++ Dispatcher fills Output.mae_bound_90_K
and tmax_err_bound_90_K from it. tests/test_conformal.cpp checks coverage
on the eval50 subset.

Usage
-----
    python tools/conformal.py --bench ../external/ThermalBench
"""
import argparse
import json
import os
import sys

import numpy as np

LEVELS = (0.90, 0.95)
N_BINS = 5


def conformal_quantile(scores, alpha):
    """Finite-sample corrected quantile: ceil((n+1)(1-alpha))/n-th order statistic."""
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(np.sort(scores)[k - 1])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", required=True)
    p.add_argument("--repo", default=os.path.join(os.path.dirname(__file__), ".."))
    p.add_argument("--checkpoint", default="checkpoints/level4_UNet/model.pt", help="relative to --bench, or absolute")
    p.add_argument("--version", default="level4-unet-1.0.0")
    p.add_argument("--out", default=None, help="default: <repo>/contract/conformal.json")
    a = p.parse_args()
    bench, repo = os.path.abspath(a.bench), os.path.abspath(a.repo)
    os.chdir(bench)
    sys.path.insert(0, bench)
    os.environ["PER_CHANNEL_NORM"] = "1"
    from utils.compat import install
    install()
    from run import parse_args
    from data_provider.data_factory import data_provider
    from exp.exp_operator import _predict
    from utils.tools import load_checkpoint, set_seed

    sys.argv = ["run.py", "--model", "UNet", "--data", "level4", "--task", "test", "--quiet"]
    set_seed(0)
    _, _, xva, yva, _, _ = data_provider(parse_args())
    try:
        x_norm, model, y_norm = load_checkpoint(a.checkpoint)
    except ValueError:
        from exp.exp_basic import build_model
        x_norm, model, y_norm = load_checkpoint(a.checkpoint, build_model("UNet", 7, 1, 64))
    model.eval()
    pred = _predict(model, x_norm, y_norm, xva.contiguous(), "UNet", "cpu").numpy().astype(np.float64)   # (n,X,Y,Z)
    truth = yva.numpy().astype(np.float64)
    n = pred.shape[0]
    mae = np.abs(pred - truth).reshape(n, -1).mean(axis=1)
    tmax_err = np.abs(pred.reshape(n, -1).max(axis=1) - truth.reshape(n, -1).max(axis=1))
    amb = xva[:, 0, 0, 0, 4].numpy().astype(np.float64)
    rise_pred = pred.reshape(n, -1).max(axis=1) - amb                  # available at inference
    print(f"[calib] n={n} MAE mean {mae.mean():.3f} K, |Tmax err| mean {tmax_err.mean():.3f} K")

    edges = np.quantile(rise_pred, np.linspace(0, 1, N_BINS + 1))
    edges[0], edges[-1] = -np.inf, np.inf                                 # open-ended outer bins
    bins = []
    for b in range(N_BINS):
        m = (rise_pred >= edges[b]) & (rise_pred < edges[b + 1])
        row = {"rise_lo_K": None if b == 0 else float(edges[b]), "rise_hi_K": None if b == N_BINS - 1 else float(edges[b + 1]), "n": int(m.sum())}
        for lvl in LEVELS:
            row[f"mae_q{int(lvl * 100)}_K"] = conformal_quantile(mae[m], 1 - lvl)
            row[f"tmax_err_q{int(lvl * 100)}_K"] = conformal_quantile(tmax_err[m], 1 - lvl)
        bins.append(row)
        print(f"[bin {b}] rise [{row['rise_lo_K']}, {row['rise_hi_K']}) n={row['n']}  MAE q90 {row['mae_q90_K']:.3f}  Tmax q90 {row['tmax_err_q90_K']:.3f}")
    overall = {}
    for lvl in LEVELS:
        overall[f"mae_q{int(lvl * 100)}_K"] = conformal_quantile(mae, 1 - lvl)
        overall[f"tmax_err_q{int(lvl * 100)}_K"] = conformal_quantile(tmax_err, 1 - lvl)
    print("[overall]", {k: round(v, 3) for k, v in overall.items()})
    doc = {"method": "split conformal prediction; scores = per-sample field MAE and |Tmax error| (K); "
                     "finite-sample quantile ceil((n+1)(1-alpha))/n; group-conditional on predicted rise = tmax_pred - ambient",
           "calibration_set": "level4 validation split (1200 samples; never the test split)", "model_version": a.version,
           "levels": list(LEVELS), "n_calibration": int(n), "overall": overall, "rise_bins": bins,
           "applies_to": "inputs that pass rules I1-I5 (exchangeable with the calibration set); undefined for OOD inputs",
           "validation_mae_mean_K": float(mae.mean())}
    out = a.out or os.path.join(repo, "contract", "conformal.json")
    with open(out, "w") as f:
        json.dump(doc, f, indent=2)
    print("->", os.path.relpath(out, repo))


if __name__ == "__main__":
    main()
