#!/usr/bin/env python3
"""S2b.2: where does the level4 U-Net make its errors?

Question answered
-----------------
The dataset-wide MAE (1.83 K) hides structure. This script attaches physical
descriptors to every test sample and reports the error per bucket of each
descriptor, so S2b.4 can say "trusted here, untrusted there" with numbers.

Inputs
------
* docs/results/baseline_unet_level4_per_sample.json  (S2.2): per-sample MAE,
  max abs error and |Tmax error| for the 3 000 test samples, in test-split order.
* datasets/level4_steady/{input,output}.mat: the test split is the last 3 000
  samples (benchmark rule "test = last 20 %", index-based, verified in S2.3),
  read directly with h5py here to avoid loading the 1.7 GB training part.

Descriptors per sample
----------------------
  case              index % 10 -- the archive is case-interleaved (10 cases,
                    300 test samples each); confirmed by the chip extent being
                    identical within each residue class
  total_power_W     sum over chiplets of the chiplet power (NOT the sum over
                    cells: the map stores one value per chiplet broadcast over
                    its footprint, see contract section 2)
  n_chiplets        connected components of the non-zero power map, where two
                    adjacent cells belong to the same component only if they
                    carry the same power value (adjacent chiplets with equal
                    power would merge; rare and accepted)
  max_power_density chiplet power / chiplet area, max over chiplets, W/mm^2;
                    cell area = (grid_x spacing) * (grid_y spacing)
  chip_area_mm2     (max grid_x) * (max grid_y)
  tmax_true_K       max of the ground-truth field
  rise_true_K       tmax_true - ambient (temperature rise above ambient)
  ambient_K, h_w_m2k, r_convec_k_per_w   boundary conditions (constant per sample)

Bucketing
---------
Continuous descriptors (n_chiplets included: it spans 2..61): 5 quantile bins
(equal count), so every bucket has ~600 samples and the per-bucket MAE is
comparable. `case` is discrete: one bucket per case. Per bucket: n, mean and
median per-sample MAE, mean max-abs error, median |Tmax error|, and the mean
*relative* error MAE / (Tmax_true - ambient), which tells whether an increase
of MAE with temperature is just scale. Spearman rank correlation between
descriptor and per-sample MAE summarises whether a trend is monotonic.

Outputs
-------
  docs/results/error_buckets.csv      one row per (descriptor, bucket)
  docs/results/error_buckets.json     same + Spearman table
  docs/results/error_buckets_per_sample.csv   descriptors + errors per test sample
  docs/results/error_buckets.png      MAE per bucket, one panel per descriptor

Usage
-----
    python tools/error_buckets.py --data ../external/thermbench_data/level4_steady
"""
import argparse
import csv
import json
import os

import h5py
import numpy as np
from scipy import ndimage, stats

P_IDX = {"chiplet_power": 0, "grid_x": 1, "grid_y": 2, "local_thermal_k": 3,
         "ambient_K": 4, "h_w_m2k": 5, "r_convec_k_per_w": 6}
N_TEST = 3000
N_CASES = 10
N_QBINS = 5


def parse_cli():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="folder with level4 input.mat / output.mat")
    p.add_argument("--repo", default=os.path.join(os.path.dirname(__file__), ".."))
    return p.parse_args()


def chiplet_descriptors(power: np.ndarray, gx: np.ndarray, gy: np.ndarray):
    """total power, chiplet count, max power density (W/mm^2) for one sample.

    power, gx, gy: (Y, X) maps as stored in the HDF5 file (axis order does not
    matter for these descriptors as long as the three maps agree)."""
    # Cell size from the coordinate spacing. gx varies along one axis, gy along
    # the other (MATLAB column-major naming, see contract); take the positive
    # step of each along the axis it varies on.
    dx = float(np.max(np.diff(gx, axis=0))) or float(np.max(np.diff(gx, axis=1)))
    dy = float(np.max(np.diff(gy, axis=1))) or float(np.max(np.diff(gy, axis=0)))
    cell_mm2 = dx * dy
    total, n, max_density = 0.0, 0, 0.0
    # Label per distinct power value so that adjacent chiplets with different
    # power stay separate components.
    for val in np.unique(power[power > 0]):
        labels, k = ndimage.label(power == val)       # 4-connectivity
        n += k
        total += float(val) * k                        # one 'val' per chiplet
        for comp in range(1, k + 1):
            area = float((labels == comp).sum()) * cell_mm2
            max_density = max(max_density, float(val) / area)
    return total, n, max_density, cell_mm2


def main():
    cli = parse_cli()
    repo = os.path.abspath(cli.repo)
    out_dir = os.path.join(repo, "docs", "results")

    # ---- per-sample errors from S2.2 -------------------------------------------
    with open(os.path.join(out_dir, "baseline_unet_level4_per_sample.json"), encoding="utf-8") as f:
        ps = json.load(f)["per_sample"]
    mae = np.array(ps["mae_K"]); maxae = np.array(ps["max_abs_err_K"]); tmax_err = np.array(ps["tmax_abs_err_K"])
    assert mae.size == N_TEST

    # ---- descriptors from the raw test inputs ----------------------------------
    with h5py.File(os.path.join(cli.data, "input.mat"), "r") as f:
        x = f["data"]                                   # (B,P,Z,Y,X)
        B = x.shape[0]
        x_te = x[B - N_TEST:]                           # (3000,7,1,64,64)
    with h5py.File(os.path.join(cli.data, "output.mat"), "r") as f:
        y_te = f["data"][B - N_TEST:]                   # (3000,1,64,64)
    print(f"[data] test slice {B - N_TEST}..{B - 1}: input {x_te.shape} output {y_te.shape}")

    rows_ps = []
    for i in range(N_TEST):
        pw, gx, gy = x_te[i, P_IDX["chiplet_power"], 0], x_te[i, P_IDX["grid_x"], 0], x_te[i, P_IDX["grid_y"], 0]
        total, n, dens, cell = chiplet_descriptors(pw, gx, gy)
        amb = float(x_te[i, P_IDX["ambient_K"], 0, 0, 0])
        tmax = float(y_te[i].max())
        rows_ps.append({
            "index_in_split": i, "global_index": B - N_TEST + i, "case": (B - N_TEST + i) % N_CASES,
            "total_power_W": total, "n_chiplets": n, "max_power_density_W_mm2": dens,
            "chip_area_mm2": float(gx.max()) * float(gy.max()),
            "tmax_true_K": tmax, "rise_true_K": tmax - amb, "ambient_K": amb,
            "h_w_m2k": float(x_te[i, P_IDX["h_w_m2k"], 0, 0, 0]),
            "r_convec_k_per_w": float(x_te[i, P_IDX["r_convec_k_per_w"], 0, 0, 0]),
            "mae_K": float(mae[i]), "max_abs_err_K": float(maxae[i]), "tmax_abs_err_K": float(tmax_err[i]),
            # relative error: field MAE as a fraction of the true temperature rise
            "rel_mae": float(mae[i]) / (tmax - amb),
        })
    col = lambda k: np.array([r[k] for r in rows_ps], dtype=np.float64)

    # ---- bucketing -------------------------------------------------------------
    continuous = ["total_power_W", "n_chiplets", "max_power_density_W_mm2", "tmax_true_K", "rise_true_K",
                  "ambient_K", "h_w_m2k", "r_convec_k_per_w", "chip_area_mm2"]
    discrete = ["case"]
    rel = col("rel_mae")
    buckets, spearman = [], {}

    def add_bucket(desc, label, lo, hi, mask):
        buckets.append({
            "descriptor": desc, "bucket": label, "lo": lo, "hi": hi, "n": int(mask.sum()),
            "mae_mean_K": float(mae[mask].mean()), "mae_median_K": float(np.median(mae[mask])),
            "max_abs_err_mean_K": float(maxae[mask].mean()),
            "tmax_abs_err_median_K": float(np.median(tmax_err[mask])),
            "rel_mae_mean": float(rel[mask].mean()),
        })

    for d in continuous:
        v = col(d)
        edges = np.quantile(v, np.linspace(0, 1, N_QBINS + 1))
        for b in range(N_QBINS):
            lo, hi = edges[b], edges[b + 1]
            mask = (v >= lo) & ((v < hi) if b < N_QBINS - 1 else (v <= hi))
            add_bucket(d, f"Q{b + 1}", float(lo), float(hi), mask)
        rho, pval = stats.spearmanr(v, mae)
        rho_r, _ = stats.spearmanr(v, rel)
        spearman[d] = {"rho_vs_mae": float(rho), "p": float(pval), "rho_vs_rel_mae": float(rho_r)}
    for d in discrete:
        v = col(d)
        for val in np.unique(v):
            mask = v == val
            add_bucket(d, str(int(val)), float(val), float(val), mask)
        rho, pval = stats.spearmanr(v, mae)
        spearman[d] = {"rho_vs_mae": float(rho), "p": float(pval)}

    for d in continuous + discrete:
        bs = [b for b in buckets if b["descriptor"] == d]
        print(f"[{d}] rho={spearman[d]['rho_vs_mae']:+.2f} rho_rel={spearman[d].get('rho_vs_rel_mae', float('nan')):+.2f}  "
              "MAE | rel per bucket: " + "  ".join(f"{b['bucket']}={b['mae_mean_K']:.2f}|{b['rel_mae_mean']*100:.1f}%" for b in bs))

    # ---- outputs ---------------------------------------------------------------
    with open(os.path.join(out_dir, "error_buckets.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(buckets[0].keys())); w.writeheader(); w.writerows(buckets)
    with open(os.path.join(out_dir, "error_buckets.json"), "w", encoding="utf-8") as f:
        json.dump({"settings": {"n_test": N_TEST, "quantile_bins": N_QBINS, "source": "S2.2 per-sample errors"},
                   "spearman_vs_mae": spearman, "buckets": buckets}, f, indent=2)
    # Per-sample descriptors go to their own CSV (S2b.4 joins them with the
    # fallback decision); rounded so the file stays small enough to commit.
    with open(os.path.join(out_dir, "error_buckets_per_sample.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_ps[0].keys())); w.writeheader()
        w.writerows([{k: (round(v, 5) if isinstance(v, float) else v) for k, v in r.items()} for r in rows_ps])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = ["total_power_W", "n_chiplets", "max_power_density_W_mm2", "tmax_true_K", "rise_true_K",
              "case", "ambient_K", "h_w_m2k", "r_convec_k_per_w"]
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    for ax, d in zip(axes.flat, panels):
        bs = [b for b in buckets if b["descriptor"] == d]
        labels = [b["bucket"] if d in discrete else f"{b['lo']:.3g}–{b['hi']:.3g}" for b in bs]
        ax.bar(range(len(bs)), [b["mae_mean_K"] for b in bs], color="tab:blue")
        ax.plot(range(len(bs)), [b["mae_median_K"] for b in bs], "k.", label="median")
        ax.set_xticks(range(len(bs))); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
        ax.set_title(f"{d}  (Spearman ρ={spearman[d]['rho_vs_mae']:+.2f})", fontsize=9)
        ax.set_ylabel("per-sample MAE (K)", fontsize=8)
        ax.axhline(mae.mean(), color="tab:red", lw=0.8, ls="--", label="overall")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle("level4 U-Net test error per bucket (3000 samples; bars = mean, dots = median)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "error_buckets.png"), dpi=120)
    print("[out] error_buckets.csv / .json / .png / _per_sample.csv")


if __name__ == "__main__":
    main()
