#!/usr/bin/env python3
"""S2b.3: extrapolation curve and the out-of-distribution (OOD) threshold.

What `oob_scale` is
-------------------
contract.md rule I4 accepts a sample for the surrogate only if every channel
lies within [train_min - m, train_max + m] with m = oob_scale * (train_max -
train_min). This script decides the value of oob_scale from measurements
instead of taste.

The honest difficulty
---------------------
A scaled input has no ground-truth temperature field, so "error vs. scale"
cannot be measured directly for most channels. Three things CAN be measured:

  A. level5 (5 000 samples, five unseen layouts, ground truth available).
     For every sample we compute its *OOB distance*
         d = max over channels c and cells of
             max(0, (train_min_c - v) / range_c, (v - train_max_c) / range_c)
     i.e. the smallest oob_scale that would let the sample through. Binning
     the true per-sample error by d gives a real "error vs. how far outside"
     curve. This is the primary evidence for the threshold.

  B. ambient_K on level4 test samples. The heat equation is linear in the
     ambient temperature, so the true field for a shifted ambient is the
     original field shifted by the same amount. Extrapolating this channel
     therefore has an exact error, and it is the only channel for which a
     scaled-input error is honest.

  C. every channel on level4 test samples, scaled to k x training range
     (k = 1.0, 1.1, 1.2, 1.5, stretched from train_min). No truth exists, so
     we only record how much the prediction moves (mean |T' - T| and dTmax).
     This is a *response* curve, not an error curve; it tells whether the
     model keeps reacting smoothly or saturates/breaks, nothing more.

Threshold rule (from the backlog): the first OOB distance bin whose true error
exceeds the contract's field-MAE gate decides oob_scale. The level5 curve is
the only one with truth across channels, so it is the one that decides.

Outputs
-------
  docs/results/extrapolation.csv        all curves, long format
  docs/results/extrapolation.json       same + the decided threshold + settings
  docs/results/extrapolation.png        two panels: A (with gates) and B+C
  docs/results/level5_per_sample.csv    per-sample OOB distance and true error on level5

Usage
-----
    python tools/extrapolation.py --bench ../external/ThermalBench --device cpu
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch

CHANNELS = ["chiplet_power", "grid_x", "grid_y", "local_thermal_k",
            "ambient_K", "h_w_m2k", "r_convec_k_per_w"]
SCALES = (1.0, 1.1, 1.2, 1.5)
# Bin edges for the level5 OOB distance (fraction of the training range).
DIST_EDGES = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, np.inf]


def parse_cli():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", required=True)
    p.add_argument("--repo", default=os.path.join(os.path.dirname(__file__), ".."))
    p.add_argument("--device", default="cpu")
    p.add_argument("--n", type=int, default=200, help="level4 test samples for B and C")
    p.add_argument("--checkpoint", default="checkpoints/level4_UNet/model.pt")
    return p.parse_args()


def main():
    cli = parse_cli()
    bench, repo = os.path.abspath(cli.bench), os.path.abspath(cli.repo)
    out_dir = os.path.join(repo, "docs", "results")

    os.chdir(bench)                                     # benchmark wants to be cwd
    sys.path.insert(0, bench)
    os.environ["PER_CHANNEL_NORM"] = "1"
    from utils.compat import install
    install()
    from run import parse_args
    from data_provider.data_factory import data_provider
    from exp.exp_operator import _predict
    from utils.tools import load_checkpoint, set_seed

    def load(level):
        sys.argv = ["run.py", "--model", "UNet", "--data", level, "--task", "test", "--quiet"]
        return data_provider(parse_args())

    set_seed(0)
    x_norm, model, y_norm = load_checkpoint(cli.checkpoint)
    model = model.to(cli.device)
    with open(os.path.join(repo, "contract", "stats.json"), encoding="utf-8") as f:
        stats = json.load(f)
    with open(os.path.join(repo, "docs", "results", "baseline_unet_level4_per_sample.json"), encoding="utf-8") as f:
        gates = json.load(f)["gates"]
    lo = np.array([c["min"] for c in stats["channels"]]); hi = np.array([c["max"] for c in stats["channels"]])
    rng = hi - lo

    def predict(xb):
        return _predict(model, x_norm, y_norm, xb, "UNet", cli.device).numpy().astype(np.float64).reshape(xb.shape[0], -1)

    def oob_distance(xb):
        """Per sample: smallest oob_scale that accepts it (0 = inside the training range)."""
        v = xb.numpy().reshape(xb.shape[0], -1, len(CHANNELS)).astype(np.float64)
        below = (lo - v.min(axis=1)) / rng
        above = (v.max(axis=1) - hi) / rng
        per_channel = np.maximum(0.0, np.maximum(below, above))      # (B, P)
        return per_channel.max(axis=1), per_channel

    rows = []

    # ---- A. level5: true error vs OOB distance ------------------------------------
    _, _, _, _, x5, y5 = load("level5")
    pred5 = predict(x5)
    truth5 = y5.numpy().astype(np.float64).reshape(x5.shape[0], -1)
    mae5 = np.abs(pred5 - truth5).mean(axis=1)
    tmax_err5 = np.abs(pred5.max(axis=1) - truth5.max(axis=1))
    d5, d5_per_channel = oob_distance(x5)
    which = {CHANNELS[c]: float(np.mean(d5_per_channel[:, c] > 0)) for c in range(len(CHANNELS))}
    print(f"[A] level5 n={len(d5)} MAE={mae5.mean():.3f} K; fraction of samples outside the range per channel: "
          + ", ".join(f"{k}={v:.2f}" for k, v in which.items() if v > 0))
    for b in range(len(DIST_EDGES) - 1):
        m = (d5 >= DIST_EDGES[b]) & (d5 < DIST_EDGES[b + 1])
        if m.sum() == 0:
            continue
        rows.append({"experiment": "A_level5_vs_oob_distance", "channel": "all", "x": DIST_EDGES[b],
                     "x_hi": DIST_EDGES[b + 1], "n": int(m.sum()),
                     "mae_mean_K": float(mae5[m].mean()), "mae_p95_K": float(np.percentile(mae5[m], 95)),
                     "tmax_err_median_K": float(np.median(tmax_err5[m])),
                     "response_mean_K": None, "dTmax_mean_K": None})
        print(f"    d in [{DIST_EDGES[b]:.2f},{DIST_EDGES[b+1]:.2f}) n={m.sum():5d} MAE={mae5[m].mean():.2f} "
              f"p95={np.percentile(mae5[m],95):.2f} Tmax_err_med={np.median(tmax_err5[m]):.2f}")
    # Same curve per level5 case (index // 1000), to show that error is a layout property.
    case5 = np.arange(len(d5)) // 1000
    # Per-sample level5 record for S2b.4 (fallback curve) and tools/layout_novelty.py.
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "level5_per_sample.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["index", "case", "oob_distance", "mae_K", "tmax_abs_err_K"]
                   + [f"oob_{c}" for c in CHANNELS])
        for i in range(len(d5)):
            w.writerow([i, int(case5[i]), round(float(d5[i]), 5), round(float(mae5[i]), 4),
                        round(float(tmax_err5[i]), 4)] + [round(float(v), 5) for v in d5_per_channel[i]])
    for c in range(5):
        m = case5 == c
        rows.append({"experiment": "A_level5_per_case", "channel": f"case{c}", "x": float(np.median(d5[m])),
                     "x_hi": float(d5[m].max()), "n": int(m.sum()), "mae_mean_K": float(mae5[m].mean()),
                     "mae_p95_K": float(np.percentile(mae5[m], 95)),
                     "tmax_err_median_K": float(np.median(tmax_err5[m])), "response_mean_K": None, "dTmax_mean_K": None})
        print(f"    level5 case{c}: median d={np.median(d5[m]):.2f} max d={d5[m].max():.2f} MAE={mae5[m].mean():.2f}")

    # ---- B + C. level4 test samples, channels stretched to k x training range -----
    _, _, _, _, xte, yte = load("level4")
    x = xte[:cli.n].contiguous()
    base = predict(x)
    base_tmax = base.max(axis=1)
    amb_idx = CHANNELS.index("ambient_K")

    def stretched(xb, chans, k):
        """v' = train_min + (v - train_min) * k on the given channels."""
        xp = xb.clone()
        for c in chans:
            xp[..., c] = float(lo[c]) + (xp[..., c] - float(lo[c])) * k
        return xp

    for name in CHANNELS + ["all_channels"]:
        chans = list(range(len(CHANNELS))) if name == "all_channels" else [CHANNELS.index(name)]
        for k in SCALES:
            xp = stretched(x, chans, k)
            pred = predict(xp)
            d, _ = oob_distance(xp)
            resp = np.abs(pred - base).mean(axis=1)
            d_tmax = pred.max(axis=1) - base_tmax
            row = {"experiment": "C_response_vs_scale", "channel": name, "x": k, "x_hi": float(d.max()),
                   "n": int(x.shape[0]), "mae_mean_K": None, "mae_p95_K": None, "tmax_err_median_K": None,
                   "response_mean_K": float(resp.mean()), "dTmax_mean_K": float(d_tmax.mean())}
            if name == "ambient_K":
                # B: exact truth = base prediction's *truth* shifted by the ambient change.
                shift = (xp[:, 0, 0, 0, amb_idx] - x[:, 0, 0, 0, amb_idx]).numpy().astype(np.float64)
                truth = yte[:cli.n].numpy().astype(np.float64).reshape(cli.n, -1) + shift[:, None]
                err = np.abs(pred - truth).mean(axis=1)
                row.update({"experiment": "B_ambient_true_error_vs_scale",
                            "mae_mean_K": float(err.mean()), "mae_p95_K": float(np.percentile(err, 95)),
                            "tmax_err_median_K": float(np.median(np.abs(pred.max(axis=1) - truth.max(axis=1))))})
            rows.append(row)
            extra = f" true MAE={row['mae_mean_K']:.2f}" if row["mae_mean_K"] is not None else ""
            print(f"[{'B' if name == 'ambient_K' else 'C'}] {name:17s} k={k:.1f} max d={d.max():.2f} "
                  f"response={resp.mean():.2f} K dTmax={d_tmax.mean():+.2f}{extra}")

    # ---- threshold decision -----------------------------------------------------
    gate = gates["field_mae_K"]
    a_rows = [r for r in rows if r["experiment"] == "A_level5_vs_oob_distance"]
    failing = [r for r in a_rows if r["mae_mean_K"] > gate]
    first_fail = min(failing, key=lambda r: r["x"]) if failing else None
    # Rule: oob_scale = lower edge of the first failing bin. If the very first
    # bin (samples inside the range) already fails, the honest answer is 0.
    oob_scale = float(first_fail["x"]) if first_fail else float(DIST_EDGES[-2])
    decision = {
        "oob_scale": oob_scale, "gate_field_mae_K": gate,
        "first_failing_bin": None if first_fail is None else {"d_lo": first_fail["x"], "d_hi": first_fail["x_hi"],
                                                              "n": first_fail["n"], "mae_mean_K": first_fail["mae_mean_K"]},
        "in_range_level5_samples": int((d5 == 0).sum()),
        "in_range_level5_mae_K": float(mae5[d5 == 0].mean()) if (d5 == 0).any() else None,
        "level5_out_of_range_fraction_per_channel": which,
        "ambient_true_mae_at_1p5x_K": next(r["mae_mean_K"] for r in rows if r["channel"] == "ambient_K" and r["x"] == 1.5),
    }
    print(f"[decision] oob_scale={oob_scale} (gate {gate:.3f} K); in-range level5 samples: "
          f"{decision['in_range_level5_samples']} with MAE {decision['in_range_level5_mae_K']}")

    # ---- outputs ----------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "extrapolation.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(out_dir, "extrapolation.json"), "w", encoding="utf-8") as f:
        json.dump({"settings": {"n_level4": cli.n, "scales": list(SCALES), "dist_edges": [e if np.isfinite(e) else "inf" for e in DIST_EDGES],
                                "device": cli.device, "checkpoint": cli.checkpoint},
                   "decision": decision, "rows": rows}, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    def hi_label(v):
        return "inf" if not np.isfinite(v) else f"{v:.2f}"
    labels = [f"[{r['x']:.2f},{hi_label(r['x_hi'])})" for r in a_rows]
    ax1.bar(range(len(a_rows)), [r["mae_mean_K"] for r in a_rows], color="tab:orange", label="level5 true MAE (mean)")
    ax1.plot(range(len(a_rows)), [r["mae_p95_K"] for r in a_rows], "k^", label="p95")
    for i, r in enumerate(a_rows):
        ax1.text(i, r["mae_mean_K"], f"n={r['n']}", ha="center", va="bottom", fontsize=7)
    ax1.axhline(gate, color="tab:red", ls="--", label=f"field-MAE gate {gate:.2f} K")
    ax1.axhline(gates["worst5pct_per_sample_mae_K"], color="tab:red", ls=":", label="worst-5 % gate")
    ax1.set_xticks(range(len(a_rows))); ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax1.set_xlabel("OOB distance d (fraction of training range)"); ax1.set_ylabel("per-sample MAE (K)")
    ax1.set_title("A: level5 true error vs. how far outside the training range"); ax1.legend(fontsize=7)
    # k = 1.0 is the identity (response exactly 0) and cannot be drawn on a log axis.
    for name in CHANNELS + ["all_channels"]:
        rs = [r for r in rows if r["channel"] == name and r["experiment"].startswith(("B", "C")) and r["x"] > 1.0]
        ax2.plot([r["x"] for r in rs], [r["response_mean_K"] for r in rs], marker="o",
                 lw=2.5 if name == "all_channels" else 1, label=name)
    rb = [r for r in rows if r["experiment"] == "B_ambient_true_error_vs_scale" and r["x"] > 1.0]
    ax2.plot([r["x"] for r in rb], [r["mae_mean_K"] for r in rb], "k--", marker="s", label="ambient_K TRUE error")
    ax2.set_yscale("log"); ax2.set_xlabel("k (channel stretched to k x training range)")
    ax2.set_ylabel("mean |T' - T| (K, log)"); ax2.set_title("B+C: response of the prediction to stretching one channel")
    ax2.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "extrapolation.png"), dpi=120)
    print("[out] extrapolation.csv / .json / .png")


if __name__ == "__main__":
    main()
