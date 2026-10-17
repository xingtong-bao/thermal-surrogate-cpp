#!/usr/bin/env python3
"""S2b.3 experiment D: can a cheap "layout novelty" score detect unseen layouts?

Why
---
Experiment A in tools/extrapolation.py showed that 80 % of the level5 samples
lie *inside* the per-channel training range yet carry a 14 K error: the model
fails on new chiplet layouts, and a range check cannot see that. The product
needs a second guard that looks at the layout itself.

Score
-----
For a sample, take the chiplet footprint mask (chiplet_power > 0), downsample
it to g x g cells by majority (g = 16 and 32), and compute the Hamming distance
(fraction of differing cells) to every training footprint; the score is the
minimum, i.e. "how different is this layout from the closest one the model
was trained on". Binary masks make the score independent of the power values
(which S2b.1 showed the model barely uses anyway) and tiny: 10 800 masks of
32x32 bits is 1.4 MB, cheap enough to ship inside a C++ library.

Second score: configuration distance
------------------------------------
The footprint score turned out not to separate seen from unseen layouts
(placements vary a lot within a case, so every test sample is "far" from the
bank). A coarser, physically meaningful descriptor is tried as well: the
6-vector (chip size x, chip size y, number of chiplets, total power, max power
density, mean chiplet area), standardized by the training std, with the score
= Euclidean distance to the nearest training sample. This asks "have we seen
a chip of this size with this kind of chiplet population?", which is exactly
the question the per-channel range check cannot ask.

What is measured
----------------
* distribution of the score on level4 test (seen layout families) vs level5
  (unseen): if the two separate, a threshold on the score is a usable gate;
* the threshold at which 99 % of level4 test samples pass, and the fraction of
  level5 samples it rejects;
* within level5, Spearman correlation between the score and the true error.

Outputs
-------
  docs/results/layout_novelty.csv   per-sample scores (level4 test + level5)
  docs/results/layout_novelty.json  summary and the proposed threshold
  docs/results/layout_novelty.png   histograms

Usage
-----
    python tools/layout_novelty.py --data ../external/thermbench_data
"""
import argparse
import csv
import json
import os

import h5py
import numpy as np
from scipy import stats

from scipy import ndimage

N_TEST = 3000
GRIDS = (16, 32)
PASS_FRACTION = 0.99          # threshold = 99th percentile of the in-support score


def parse_cli():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="folder containing level4_steady/ and level5_steady/")
    p.add_argument("--repo", default=os.path.join(os.path.dirname(__file__), ".."))
    return p.parse_args()


def config_features(path, sl):
    """(n, 6) configuration vector per sample: chip x/y size (mm), n chiplets,
    total power (W), max power density (W/mm^2), mean chiplet area (mm^2).
    Same chiplet logic as tools/error_buckets.py (components per power value)."""
    with h5py.File(path, "r") as f:
        d = f["data"]
        pw, gx, gy = d[sl, 0, 0], d[sl, 1, 0], d[sl, 2, 0]
    out = np.zeros((pw.shape[0], 6))
    for i in range(pw.shape[0]):
        dx = float(np.max(np.diff(gx[i], axis=0))) or float(np.max(np.diff(gx[i], axis=1)))
        dy = float(np.max(np.diff(gy[i], axis=1))) or float(np.max(np.diff(gy[i], axis=0)))
        cell = dx * dy
        total, n, dens, areas = 0.0, 0, 0.0, []
        for val in np.unique(pw[i][pw[i] > 0]):
            labels, k = ndimage.label(pw[i] == val)
            n += k; total += float(val) * k
            for comp in range(1, k + 1):
                a = float((labels == comp).sum()) * cell
                areas.append(a); dens = max(dens, float(val) / a)
        out[i] = [gx[i].max(), gy[i].max(), n, total, dens, np.mean(areas) if areas else 0.0]
    return out


def min_euclid(q, bank, chunk=1000):
    """Min Euclidean distance from each query row to the bank (standardized features)."""
    out = np.empty(q.shape[0])
    bb = (bank ** 2).sum(axis=1)
    for s in range(0, q.shape[0], chunk):
        qq = q[s:s + chunk]
        d2 = (qq ** 2).sum(axis=1)[:, None] + bb[None, :] - 2.0 * qq @ bank.T
        out[s:s + chunk] = np.sqrt(np.maximum(d2.min(axis=1), 0.0))
    return out


def footprints(path, sl):
    """Binary chiplet masks (n, 64, 64) for a slice of input.mat (channel 0 = power)."""
    with h5py.File(path, "r") as f:
        pw = f["data"][sl, 0, 0]                     # (n, 64, 64) raw (Y, X) order
    return pw > 0


def downsample(mask, g):
    """Majority-vote downsample of (n,64,64) bool masks to (n, g*g) float32 in {0,1}."""
    n = mask.shape[0]
    f = 64 // g
    m = mask.reshape(n, g, f, g, f).mean(axis=(2, 4))
    return (m >= 0.5).astype(np.float32).reshape(n, g * g)


def min_hamming(q, bank, chunk=1000):
    """Min over the bank of the Hamming distance (fraction of cells) for each query row.

    For binary vectors a, b: hamming = (|a| + |b| - 2 a.b) / dim, so one matmul
    per chunk gives all pairwise distances."""
    dim = q.shape[1]
    bank_sum = bank.sum(axis=1)
    out = np.empty(q.shape[0])
    for s in range(0, q.shape[0], chunk):
        qq = q[s:s + chunk]
        d = (qq.sum(axis=1)[:, None] + bank_sum[None, :] - 2.0 * qq @ bank.T) / dim
        out[s:s + chunk] = d.min(axis=1)
    return out


def main():
    cli = parse_cli()
    repo = os.path.abspath(cli.repo)
    out_dir = os.path.join(repo, "docs", "results")
    p4 = os.path.join(cli.data, "level4_steady", "input.mat")
    p5 = os.path.join(cli.data, "level5_steady", "input.mat")
    with h5py.File(p4, "r") as f:
        B = f["data"].shape[0]
    n_train = int(int(B * 0.8) * 0.9)                # benchmark split: 10 800
    train = footprints(p4, slice(0, n_train))
    test4 = footprints(p4, slice(B - N_TEST, B))
    test5 = footprints(p5, slice(0, None))
    print(f"[masks] train={train.shape[0]} level4-test={test4.shape[0]} level5={test5.shape[0]}")

    # level5 true error (from tools/extrapolation.py) for the correlation.
    err5 = None
    l5 = os.path.join(out_dir, "level5_per_sample.csv")
    if os.path.exists(l5):
        with open(l5, encoding="utf-8") as f:
            err5 = np.array([float(r["mae_K"]) for r in csv.DictReader(f)])
    # level4 test per-sample error from S2.2.
    with open(os.path.join(out_dir, "baseline_unet_level4_per_sample.json"), encoding="utf-8") as f:
        err4 = np.array(json.load(f)["per_sample"]["mae_K"])

    summary, per_sample = {}, {"level4_test": {}, "level5": {}}
    for g in GRIDS:
        bank = downsample(train, g)
        s4 = min_hamming(downsample(test4, g), bank)
        s5 = min_hamming(downsample(test5, g), bank)
        thr = float(np.quantile(s4, PASS_FRACTION))
        rej5 = float(np.mean(s5 > thr))
        rho5 = float(stats.spearmanr(s5, err5)[0]) if err5 is not None else None
        rho4 = float(stats.spearmanr(s4, err4)[0])
        # AUROC: probability that a random level5 score exceeds a random level4 score.
        auroc = float(stats.mannwhitneyu(s5, s4, alternative="greater").statistic / (len(s5) * len(s4)))
        summary[f"grid{g}"] = {
            "level4_test_score": {"median": float(np.median(s4)), "p99": thr, "max": float(s4.max())},
            "level5_score": {"median": float(np.median(s5)), "min": float(s5.min()), "max": float(s5.max())},
            "threshold_99pct_level4": thr, "level5_rejected_fraction": rej5, "auroc": auroc,
            "spearman_score_vs_error_level5": rho5, "spearman_score_vs_error_level4": rho4,
            "level5_per_case_median": [float(np.median(s5[c * 1000:(c + 1) * 1000])) for c in range(5)],
            "bank_size_bytes": int(n_train * g * g / 8),
        }
        per_sample["level4_test"][g], per_sample["level5"][g] = s4, s5
        print(f"[grid {g}x{g}] level4-test median={np.median(s4):.3f} p99={thr:.3f} max={s4.max():.3f} | "
              f"level5 median={np.median(s5):.3f} min={s5.min():.3f} | rejects {rej5:.1%} of level5 | "
              f"AUROC={auroc:.3f} | rho(score,err) level5={rho5} level4={rho4:.2f}")
        print(f"    level5 per-case median: {['%.3f' % v for v in summary[f'grid{g}']['level5_per_case_median']]}")

    # ---- configuration-distance score ---------------------------------------
    ftr = config_features(p4, slice(0, n_train))
    f4 = config_features(p4, slice(B - N_TEST, B))
    f5 = config_features(p5, slice(0, None))
    mu, sd = ftr.mean(axis=0), ftr.std(axis=0)
    sd[sd == 0] = 1.0
    z = lambda a: (a - mu) / sd
    c4, c5 = min_euclid(z(f4), z(ftr)), min_euclid(z(f5), z(ftr))
    thr_c = float(np.quantile(c4, PASS_FRACTION))
    auroc_c = float(stats.mannwhitneyu(c5, c4, alternative="greater").statistic / (len(c5) * len(c4)))
    summary["config"] = {
        "features": ["chip_x_mm", "chip_y_mm", "n_chiplets", "total_power_W", "max_power_density_W_mm2", "mean_chiplet_area_mm2"],
        "train_mean": mu.tolist(), "train_std": sd.tolist(),
        "level4_test_score": {"median": float(np.median(c4)), "p99": thr_c, "max": float(c4.max())},
        "level5_score": {"median": float(np.median(c5)), "min": float(c5.min()), "max": float(c5.max())},
        "threshold_99pct_level4": thr_c, "level5_rejected_fraction": float(np.mean(c5 > thr_c)), "auroc": auroc_c,
        "spearman_score_vs_error_level5": float(stats.spearmanr(c5, err5)[0]) if err5 is not None else None,
        "spearman_score_vs_error_level4": float(stats.spearmanr(c4, err4)[0]),
        "level5_per_case_median": [float(np.median(c5[c * 1000:(c + 1) * 1000])) for c in range(5)],
        "level5_per_case_rejected": [float(np.mean(c5[c * 1000:(c + 1) * 1000] > thr_c)) for c in range(5)],
    }
    s = summary["config"]
    print(f"[config] level4-test median={s['level4_test_score']['median']:.3f} p99={thr_c:.3f} max={c4.max():.3f} | "
          f"level5 median={s['level5_score']['median']:.3f} min={c5.min():.3f} | rejects {s['level5_rejected_fraction']:.1%} "
          f"of level5 | AUROC={auroc_c:.3f} | rho(score,err) level5={s['spearman_score_vs_error_level5']}")
    print(f"    level5 per-case median: {['%.2f' % v for v in s['level5_per_case_median']]} rejected: {['%.2f' % v for v in s['level5_per_case_rejected']]}")

    # ---- outputs ----------------------------------------------------------------
    with open(os.path.join(out_dir, "layout_novelty.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["set", "index"] + [f"score_grid{g}" for g in GRIDS] + ["score_config", "mae_K"])
        for i in range(N_TEST):
            w.writerow(["level4_test", i] + [round(float(per_sample["level4_test"][g][i]), 5) for g in GRIDS]
                       + [round(float(c4[i]), 4), round(float(err4[i]), 4)])
        for i in range(test5.shape[0]):
            w.writerow(["level5", i] + [round(float(per_sample["level5"][g][i]), 5) for g in GRIDS]
                       + [round(float(c5[i]), 4), round(float(err5[i]), 4) if err5 is not None else ""])
    with open(os.path.join(out_dir, "layout_novelty.json"), "w", encoding="utf-8") as f:
        json.dump({"settings": {"grids": list(GRIDS), "pass_fraction": PASS_FRACTION, "bank": "level4 train footprints"},
                   "summary": summary}, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(GRIDS) + 1, figsize=(6 * (len(GRIDS) + 1), 4))
    axc = axes[-1]
    bins = np.linspace(0, max(c4.max(), c5.max()), 40)
    axc.hist(c4, bins, alpha=0.6, label="level4 test (seen)"); axc.hist(c5, bins, alpha=0.6, label="level5 (unseen)")
    axc.axvline(thr_c, color="k", ls="--", label="99 % of level4 pass")
    axc.set_xlabel("distance to nearest training configuration (std units)"); axc.set_ylabel("samples")
    axc.set_title(f"configuration novelty score; AUROC={auroc_c:.3f}"); axc.legend(fontsize=8)
    for ax, g in zip(axes[:-1], GRIDS):
        s4, s5 = per_sample["level4_test"][g], per_sample["level5"][g]
        bins = np.linspace(0, max(s4.max(), s5.max()), 40)
        ax.hist(s4, bins, alpha=0.6, label="level4 test (seen layouts)")
        ax.hist(s5, bins, alpha=0.6, label="level5 (unseen layouts)")
        ax.axvline(summary[f"grid{g}"]["threshold_99pct_level4"], color="k", ls="--", label="99 % of level4 pass")
        ax.set_xlabel(f"min Hamming distance to training footprints ({g}x{g})"); ax.set_ylabel("samples")
        ax.set_title(f"layout novelty score, grid {g}x{g}; AUROC={summary[f'grid{g}']['auroc']:.3f}")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "layout_novelty.png"), dpi=120)
    print("[out] layout_novelty.csv / .json / .png")


if __name__ == "__main__":
    main()
