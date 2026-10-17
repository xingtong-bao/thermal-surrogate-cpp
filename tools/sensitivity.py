#!/usr/bin/env python3
"""S2b.1: channel sensitivity of the level4 U-Net surrogate.

Question answered
-----------------
"If one input channel changes by +/-10 %, how much does the predicted field and
the predicted hotspot temperature move?"  This ranks the channels by influence,
which later tells us (a) which channels the OOD check must guard most tightly
(S2b.3) and (b) whether the surrogate reacts to a physical change the way the
heat equation says it should.

Method
------
* 200 samples from the level4 TEST split (first 200; the split is
  case-interleaved, so this covers all 10 cases). Test rather than train so the
  model has not seen them.
* For every channel c and factor f in {0.9, 1.1}: x' = x with channel c scaled
  by f everywhere, then predict. Scaling the whole map keeps the spatial
  pattern and changes only the magnitude, which is what "+/-10 %" means for a
  boundary condition or a material property.
* Recorded per (channel, factor), averaged over the 200 samples:
    dTmax_mean_K        signed change of the predicted hotspot temperature
    dTmax_abs_mean_K    |change|, so +/- do not cancel
    field_mad_mean_K    mean |T'(x,y) - T(x,y)| over the field = "field MAE"
                        between the perturbed and the unperturbed prediction
    ood_fraction        share of samples whose perturbed channel leaves the
                        training range in stats.json (a 10 % change in the
                        ambient temperature is ~31 K, which exits the range
                        for many samples -- that is itself a finding)
* Physics expectation, where the steady-state heat equation gives one
  (T = T_ambient + R * P is linear in P and in T_ambient):
    chiplet_power x f : Tmax - T_amb scales by f, so expected dTmax = (f-1)*(Tmax-T_amb)
    ambient_K     x f : whole field shifts by (f-1)*T_amb
  The ratio measured/expected ("gain") should be ~1 if the surrogate learned
  the physics rather than a lookup table. Other channels enter non-linearly, so
  no expectation is recorded for them.

Outputs
-------
  docs/results/sensitivity.csv   one row per (channel, factor)
  docs/results/sensitivity.json  same numbers + ranking + settings
  docs/results/sensitivity.png   bar chart of field_mad_mean_K per channel

Usage
-----
    python tools/sensitivity.py --bench ../external/ThermalBench --device cpu
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
# grid_x / grid_y are coordinates, not physical parameters; scaling them means
# "a 10 % larger chip". They are included for completeness but flagged.
GEOMETRIC = {"grid_x", "grid_y"}
FACTORS = (0.9, 1.1)
# Small-step probe, only for the two channels with an exact physics expectation.
# If the measured/expected gain is the same at +/-2 % as at +/-10 %, the deficit
# is a property of the model, not an artefact of leaving the training range.
PROBE_FACTORS = (0.98, 1.02)
PROBE_CHANNELS = {"chiplet_power", "ambient_K"}


def parse_cli():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", required=True)
    p.add_argument("--repo", default=os.path.join(os.path.dirname(__file__), ".."))
    p.add_argument("--device", default="cpu")
    p.add_argument("--n", type=int, default=200, help="number of test samples")
    p.add_argument("--checkpoint", default="checkpoints/level4_UNet/model.pt")
    return p.parse_args()


def main():
    cli = parse_cli()
    bench, repo = os.path.abspath(cli.bench), os.path.abspath(cli.repo)
    out_dir = os.path.join(repo, "docs", "results")

    # Same bootstrap as the other tools: the benchmark wants to be cwd and on sys.path.
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
    args = parse_args()
    set_seed(0)
    _, _, _, _, xte, yte = data_provider(args)
    x = xte[:cli.n].contiguous()                       # (n,X,Y,Z,P)
    y_true = yte[:cli.n].numpy().astype(np.float64)    # (n,X,Y,Z)
    x_norm, model, y_norm = load_checkpoint(cli.checkpoint)
    model = model.to(cli.device)

    with open(os.path.join(repo, "contract", "stats.json"), encoding="utf-8") as f:
        stats = json.load(f)
    train_rng = {c["name"]: (c["min"], c["max"]) for c in stats["channels"]}

    def predict(xb):
        """(n,X,Y,Z,P) -> (n, X*Y) field in Kelvin, float64."""
        return _predict(model, x_norm, y_norm, xb, "UNet", cli.device).numpy().astype(np.float64).reshape(xb.shape[0], -1)

    base = predict(x)                                  # unperturbed prediction
    base_tmax = base.max(axis=1)
    t_amb = x[:, 0, 0, 0, CHANNELS.index("ambient_K")].numpy().astype(np.float64)  # constant per sample
    print(f"[base] n={cli.n} MAE vs truth = {np.abs(base - y_true.reshape(cli.n, -1)).mean():.4f} K "
          f"(sanity: should be ~1.8 K like S2.2)")

    rows = []
    for c, name in enumerate(CHANNELS):
        factors = FACTORS + (PROBE_FACTORS if name in PROBE_CHANNELS else ())
        for f in factors:
            xp = x.clone()
            xp[..., c] *= f                             # scale the whole channel map
            pred = predict(xp)
            d_tmax = pred.max(axis=1) - base_tmax
            field_mad = np.abs(pred - base).mean(axis=1)
            lo, hi = train_rng[name]
            ch = xp[..., c].reshape(cli.n, -1).numpy()
            ood = float(np.mean((ch.min(axis=1) < lo) | (ch.max(axis=1) > hi)))

            # Physics expectation (see docstring); None where the equation gives none.
            expected = None
            if name == "chiplet_power":
                expected = float(((f - 1.0) * (base_tmax - t_amb)).mean())
            elif name == "ambient_K":
                expected = float(((f - 1.0) * t_amb).mean())
            gain = None if expected is None or abs(expected) < 1e-9 else float(d_tmax.mean() / expected)

            row = {
                "channel": name, "index": c, "geometric": name in GEOMETRIC, "factor": f,
                "probe": f in PROBE_FACTORS,            # True = +/-2 % probe row
                "dTmax_mean_K": float(d_tmax.mean()),
                "dTmax_abs_mean_K": float(np.abs(d_tmax).mean()),
                "dTmax_abs_p95_K": float(np.percentile(np.abs(d_tmax), 95)),
                "field_mad_mean_K": float(field_mad.mean()),
                "field_mad_p95_K": float(np.percentile(field_mad, 95)),
                "ood_fraction": ood,
                "expected_dTmax_K": expected, "gain": gain,
            }
            rows.append(row)
            exp_s = "" if expected is None else f" expected={expected:+.3f} gain={gain:.2f}"
            print(f"[{name:17s} x{f}] dTmax={row['dTmax_mean_K']:+.3f} K  |dTmax|={row['dTmax_abs_mean_K']:.3f}  "
                  f"field MAD={row['field_mad_mean_K']:.3f} K  ood={ood:.2f}{exp_s}")

    # Ranking: mean field MAD over the two factors. Field MAD rather than dTmax
    # because it is less noisy (averages 4096 cells instead of one argmax).
    score = {}
    for r in rows:
        if not r["probe"]:
            score.setdefault(r["channel"], []).append(r["field_mad_mean_K"])
    ranking = sorted(((np.mean(v), k) for k, v in score.items()), reverse=True)
    ranking = [{"rank": i + 1, "channel": k, "field_mad_mean_K": float(s),
                "geometric": k in GEOMETRIC} for i, (s, k) in enumerate(ranking)]

    # ---- outputs ----------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "sensitivity.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "sensitivity.json"), "w", encoding="utf-8") as f:
        json.dump({"settings": {"n_samples": cli.n, "split": "level4 test, first n",
                                "factors": list(FACTORS), "probe_factors": list(PROBE_FACTORS), "device": cli.device,
                                "checkpoint": cli.checkpoint},
                   "ranking": ranking, "rows": rows}, f, indent=2)

    # Bar chart: one bar per channel, the two factors side by side.
    import matplotlib
    matplotlib.use("Agg")                               # no display needed
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    idx = np.arange(len(CHANNELS))
    for j, f in enumerate(FACTORS):
        vals = [next(r["field_mad_mean_K"] for r in rows if r["channel"] == n and r["factor"] == f)
                for n in CHANNELS]
        ax.bar(idx + (j - 0.5) * 0.38, vals, width=0.38, label=f"x{f}")
    ax.set_xticks(idx)
    ax.set_xticklabels([n + ("*" if n in GEOMETRIC else "") for n in CHANNELS], rotation=30, ha="right")
    # Log scale: ambient_K moves the field by ~30 K and would flatten every
    # other bar (sub-Kelvin) on a linear axis.
    ax.set_yscale("log")
    ax.set_ylabel("mean |T' - T| over field (K, log scale)")
    ax.set_title(f"level4 U-Net: field change for a +/-10 % change of one channel (n={cli.n})\n"
                 "* geometric channel (scaling = larger/smaller chip)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "sensitivity.png"), dpi=120)
    print("[out] sensitivity.csv / .json / .png ->", os.path.relpath(out_dir, repo))
    print("[ranking]", " > ".join(r["channel"] for r in ranking))


if __name__ == "__main__":
    main()
