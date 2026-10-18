#!/usr/bin/env python3
"""S2b.4: fallback-rate curve -- what the two OOD guards cost and buy.

Question answered
-----------------
contract rules I4 (per-channel range, margin oob_scale) and I5 (configuration
novelty, threshold tau) send an input to the reference solver instead of the
surrogate. For every candidate threshold this script reports, on three input
streams, the fraction of inputs that fall back, the error the user actually
sees (surrogate error where accepted, reference truth where fallen back), and
the mean cost per query. The curve is the evidence behind the thresholds that
S2b.3 wrote into the contract, and it shows why I4 alone can never work.

Inputs (all produced by earlier stories, no model is run here)
--------------------------------------------------------------
  docs/results/error_buckets_per_sample.csv   level4 test: true per-sample MAE
  docs/results/level5_per_sample.csv          level5: OOB distance + true MAE
  docs/results/layout_novelty.csv             configuration score, both sets
  contract/stats.json                         training ranges (for level4 OOB distance)
  datasets/level4_steady/input.mat            to compute level4-test OOB distances

Streams
-------
  level4   3 000 in-support test samples (seen chip configurations)
  level5   5 000 unseen-layout samples
  mixed    50/50 weighting of the two per-sample sets. The real mix in a
           design flow is unknown; 50/50 is the neutral choice and the
           per-stream numbers are reported so any other mix can be derived.

Blended error assumes the reference solver reproduces the ground truth
exactly. That is literally true here (HotSpot generated the dataset) and is
the premise of a fallback: the slow path is trusted.

Cost model: reference solve = R x surrogate call; mean cost per query in
surrogate units = 1 + fallback_rate * (R - 1). R is unknown until S4.4/S5.3
measure both paths, so curves are given for R = 100 and R = 1000.

Outputs
-------
  docs/results/fallback_curve.csv   long table: rule, threshold, stream, metrics
  docs/results/fallback_curve.json  operating points at the contract thresholds
  docs/results/fallback_curve.png   blended MAE vs fallback rate; rates vs threshold

Usage
-----
    python tools/fallback_curve.py --data ../external/thermbench_data/level4_steady
"""
import argparse
import csv
import json
import os

import h5py
import numpy as np

N_TEST = 3000
GATE_KEY = "field_mae_K"
OOB_SCALES = [round(v, 3) for v in np.arange(0.0, 0.51, 0.025)]       # I4 sweep
COST_RATIOS = (100, 1000)


def parse_cli():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="folder with level4 input.mat")
    p.add_argument("--repo", default=os.path.join(os.path.dirname(__file__), ".."))
    return p.parse_args()


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    cli = parse_cli()
    repo = os.path.abspath(cli.repo)
    res = os.path.join(repo, "docs", "results")
    with open(os.path.join(repo, "contract", "stats.json"), encoding="utf-8") as f:
        stats = json.load(f)
    with open(os.path.join(res, "baseline_unet_level4_per_sample.json"), encoding="utf-8") as f:
        gate = json.load(f)["gates"][GATE_KEY]
    tau_contract = stats["layout_novelty"]["threshold"]
    oob_contract = stats["oob_scale"]

    # ---- level4 test: MAE, novelty score, OOB distance ------------------------
    l4 = read_csv(os.path.join(res, "error_buckets_per_sample.csv"))
    mae4 = np.array([float(r["mae_K"]) for r in l4])
    nov = read_csv(os.path.join(res, "layout_novelty.csv"))
    s4 = np.array([float(r["score_config"]) for r in nov if r["set"] == "level4_test"])
    s5 = np.array([float(r["score_config"]) for r in nov if r["set"] == "level5"])
    lo = np.array([c["min"] for c in stats["channels"]]); hi = np.array([c["max"] for c in stats["channels"]])
    with h5py.File(os.path.join(cli.data, "input.mat"), "r") as f:
        d = f["data"]; B = d.shape[0]
        x = d[B - N_TEST:]                                   # (3000,7,1,64,64)
    v = x.reshape(N_TEST, len(lo), -1)
    per_ch = np.maximum(0, np.maximum((lo[None, :] - v.min(axis=2)) / (hi - lo), (v.max(axis=2) - hi) / (hi - lo)))
    d4 = per_ch.max(axis=1)
    # ---- level5 ----
    l5 = read_csv(os.path.join(res, "level5_per_sample.csv"))
    mae5 = np.array([float(r["mae_K"]) for r in l5]); d5 = np.array([float(r["oob_distance"]) for r in l5])
    assert len(mae4) == len(s4) == len(d4) == N_TEST and len(mae5) == len(s5) == len(d5)
    print(f"[data] level4 test: {np.mean(d4 > 0):.1%} outside the training range (max d={d4.max():.3f}); "
          f"level5: {np.mean(d5 > 0):.1%} outside")

    streams = {"level4": (mae4, d4, s4), "level5": (mae5, d5, s5)}

    def metrics(accept, mae):
        """accept: bool mask (True = surrogate used). Reference assumed exact."""
        r = 1.0 - accept.mean()
        blended = float(np.where(accept, mae, 0.0).mean())
        acc_mae = float(mae[accept].mean()) if accept.any() else float("nan")
        out = {"fallback_rate": float(r), "blended_mae_K": blended, "accepted_mae_K": acc_mae,
               "accepted_passes_gate": bool(accept.any() and acc_mae <= gate)}
        for R in COST_RATIOS:
            out[f"cost_R{R}"] = float(1.0 + r * (R - 1))
        return out

    def mixed(m4, m5):
        out = {}
        for k in m4:
            if k == "accepted_passes_gate":
                continue
            if k == "accepted_mae_K":
                # weight by accepted counts so the mixed accepted-MAE is a true mean
                w4, w5 = 0.5 * (1 - m4["fallback_rate"]), 0.5 * (1 - m5["fallback_rate"])
                out[k] = (m4[k] * w4 + (m5[k] if not np.isnan(m5[k]) else 0) * w5) / (w4 + w5) if (w4 + w5) > 0 else float("nan")
            else:
                out[k] = 0.5 * m4[k] + 0.5 * m5[k]
        out["accepted_passes_gate"] = bool(out["accepted_mae_K"] <= gate)
        return out

    rows = []

    def sweep(rule, thr_list, accept_fn):
        for thr in thr_list:
            per = {name: metrics(accept_fn(dd, ss, thr), mm) for name, (mm, dd, ss) in streams.items()}
            per["mixed"] = mixed(per["level4"], per["level5"])
            for name, m in per.items():
                rows.append({"rule": rule, "threshold": thr, "stream": name, **m})

    # I4 only: accept if OOB distance <= oob_scale
    sweep("I4_range", OOB_SCALES, lambda d, s, t: d <= t)
    # I5 only: accept if novelty score <= tau; tau grid from level4 quantiles + absolute values
    taus = sorted(set([round(float(np.quantile(s4, q)), 4) for q in (0.5, 0.9, 0.95, 0.99, 0.999, 1.0)]
                      + [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.0, 4.0]))
    sweep("I5_novelty", taus, lambda d, s, t: s <= t)
    # I4 + I5 at the contract's oob_scale, sweeping tau
    sweep("I4+I5", taus, lambda d, s, t: (d <= oob_contract) & (s <= t))

    # ---- operating points at the contract thresholds --------------------------
    def at(rule, thr):
        return {r["stream"]: {k: r[k] for k in r if k not in ("rule", "threshold", "stream")}
                for r in rows if r["rule"] == rule and abs(r["threshold"] - thr) < 1e-9}
    ops = {"gate_field_mae_K": gate, "oob_scale": oob_contract, "novelty_threshold": tau_contract,
           "I4_only_at_contract": at("I4_range", oob_contract),
           "I5_only_at_contract": at("I5_novelty", round(tau_contract, 4)) or at("I5_novelty", tau_contract),
           "I4+I5_at_contract": at("I4+I5", round(tau_contract, 4)) or at("I4+I5", tau_contract),
           "I4_best_possible": None}
    # Best I4 can do on the mixed stream: the smallest blended MAE over the sweep.
    best = min((r for r in rows if r["rule"] == "I4_range" and r["stream"] == "mixed"), key=lambda r: r["blended_mae_K"])
    ops["I4_best_possible"] = {"threshold": best["threshold"], **{k: best[k] for k in best if k not in ("rule", "threshold", "stream")}}
    for name in ("I4_only_at_contract", "I5_only_at_contract", "I4+I5_at_contract"):
        m = ops[name]["mixed"]
        print(f"[{name}] mixed: fallback {m['fallback_rate']:.1%}  blended MAE {m['blended_mae_K']:.2f} K  "
              f"accepted MAE {m['accepted_mae_K']:.2f} K  cost(R=100) {m['cost_R100']:.1f}x  | "
              f"level4 fallback {ops[name]['level4']['fallback_rate']:.1%}  level5 fallback {ops[name]['level5']['fallback_rate']:.1%}")
    print(f"[I4 best possible on mixed] oob_scale={best['threshold']}: blended MAE {best['blended_mae_K']:.2f} K at fallback {best['fallback_rate']:.1%}")

    # ---- outputs ----------------------------------------------------------------
    with open(os.path.join(res, "fallback_curve.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(os.path.join(res, "fallback_curve.json"), "w", encoding="utf-8") as f:
        json.dump({"settings": {"streams": "level4 test (3000), level5 (5000), mixed = 50/50",
                                "reference_assumed_exact": True, "cost_ratios": list(COST_RATIOS)},
                   "operating_points": ops}, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    for rule, style in (("I4_range", "o-"), ("I5_novelty", "s-"), ("I4+I5", "^--")):
        rs = sorted((r for r in rows if r["rule"] == rule and r["stream"] == "mixed"), key=lambda r: r["fallback_rate"])
        ax1.plot([r["fallback_rate"] for r in rs], [r["blended_mae_K"] for r in rs], style, label=rule, ms=4)
    ax1.axhline(gate, color="tab:red", ls="--", label=f"gate {gate:.2f} K")
    m = ops["I4+I5_at_contract"]["mixed"]
    ax1.plot([m["fallback_rate"]], [m["blended_mae_K"]], "k*", ms=14, label="contract thresholds")
    ax1.set_xlabel("fallback rate on the 50/50 mixed stream"); ax1.set_ylabel("blended MAE seen by the user (K)")
    ax1.set_title("What a guard buys: error vs. how often the slow solver runs"); ax1.legend(fontsize=8)
    ax1.set_xlim(-0.02, 1.02)
    for stream, c in (("level4", "tab:blue"), ("level5", "tab:orange")):
        rs = sorted((r for r in rows if r["rule"] == "I5_novelty" and r["stream"] == stream), key=lambda r: r["threshold"])
        ax2.plot([r["threshold"] for r in rs], [r["fallback_rate"] for r in rs], "s-", color=c, ms=4, label=f"I5 fallback rate, {stream}")
        rs = sorted((r for r in rows if r["rule"] == "I4_range" and r["stream"] == stream), key=lambda r: r["threshold"])
        ax2.plot([r["threshold"] for r in rs], [r["fallback_rate"] for r in rs], "o:", color=c, ms=3, alpha=0.6, label=f"I4 fallback rate, {stream}")
    ax2.axvline(tau_contract, color="k", ls="--", lw=0.8, label=f"contract tau={tau_contract:.3f}")
    ax2.set_xscale("symlog", linthresh=0.1); ax2.set_xlabel("threshold (I5: novelty score; I4: oob_scale)")
    ax2.set_ylabel("fallback rate"); ax2.set_title("Which inputs each guard sends to the reference"); ax2.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(res, "fallback_curve.png"), dpi=120)
    print("[out] fallback_curve.csv / .json / .png")


if __name__ == "__main__":
    main()
