#!/usr/bin/env python3
"""S8.4: a small design-exploration loop -- surrogate vs solver decisions.

Scenario: for one seen chip (level4 test sample #12000), how weak can the
cooling be while the hotspot stays under a limit? "Cooling" is the data's
own knob: the convective resistance r (K/W), with h = 1/(r * 0.0282 m^2)
kept consistent with every sample in the dataset. 12 values of r spaced
logarithmically across the training range are evaluated by the surrogate
(ONNX Runtime) and by HotSpot; the chosen design is the largest r with
Tmax <= limit.

What this shows (and what it cannot): the surrogate answers the sweep in
~0.3 s instead of ~30 s; whether its Tmax-vs-r *slope* matches the solver
tells us if "cooling what-ifs" are in the trusted region (S2b.1 found that
power what-ifs are not). HotSpot's absolute temperatures carry a package
offset against the dataset (S5.3), so the two curves are also compared
after removing each one's value at the sample's own r (the design point the
dataset calibrates); the decision is reported both raw and offset-corrected.

Usage
-----
    python tools/exploration_demo.py --limit 360
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
SINK_AREA = 0.0282                       # m^2, from h = 1/(r * A) in every sample (S5.2)


def hotspot(x):
    with tempfile.TemporaryDirectory() as d:
        np.save(os.path.join(d, "in.npy"), x)
        r = subprocess.run([sys.executable, os.path.join(HERE, "reference_cli.py"), os.path.join(d, "in.npy"), os.path.join(d, "out.npy")], capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-400:].decode())
        return np.load(os.path.join(d, "out.npy"))[..., 0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=float, default=360.0, help="Tmax limit in K")
    p.add_argument("--points", type=int, default=12)
    a = p.parse_args()
    stats = json.load(open(os.path.join(REPO, "contract", "stats.json")))
    r_lo, r_hi = [c for c in stats["channels"] if c["name"] == "r_convec_k_per_w"][0]["min"], [c for c in stats["channels"] if c["name"] == "r_convec_k_per_w"][0]["max"]
    x0 = np.load(os.path.join(REPO, "tests", "fixtures", "sample_input.npy"))
    r0, amb = float(x0[0, 0, 0, 6]), float(x0[0, 0, 0, 4])
    rs = np.geomspace(r_lo, r_hi, a.points)
    sess = ort.InferenceSession(os.path.join(REPO, "models", "model.onnx"), providers=["CPUExecutionProvider"])

    def design(r):
        x = x0.copy()
        x[..., 6] = r
        x[..., 5] = 1.0 / (r * SINK_AREA)
        return x

    t0 = time.time()
    tmax_s = [float(sess.run(["field"], {"input": design(r)[None]})[0].max()) for r in rs]
    t_s = time.time() - t0
    t0 = time.time()
    tmax_h = [float(hotspot(design(r)).max()) for r in rs]
    t_h = time.time() - t0
    # Offset correction at the dataset's own design point r0 (interpolated on log r).
    s0 = float(np.interp(np.log(r0), np.log(rs), tmax_s)); h0 = float(np.interp(np.log(r0), np.log(rs), tmax_h))
    tmax_h_corr = [v - h0 + s0 for v in tmax_h]

    def choose(curve):
        ok = [r for r, t in zip(rs, curve) if t <= a.limit]
        return float(max(ok)) if ok else None

    slope_s = np.polyfit(np.log(rs), tmax_s, 1)[0]; slope_h = np.polyfit(np.log(rs), tmax_h, 1)[0]
    out = {"chip": "level4 test sample 12000 (tests/fixtures/sample_input.npy)", "ambient_K": amb, "limit_K": a.limit,
           "r_values": rs.tolist(), "tmax_surrogate_K": tmax_s, "tmax_hotspot_K": tmax_h, "tmax_hotspot_offset_corrected_K": tmax_h_corr,
           "wall_time_s": {"surrogate_12_evals": t_s, "hotspot_12_evals": t_h},
           "slope_dTmax_dlnr_K": {"surrogate": float(slope_s), "hotspot": float(slope_h), "ratio": float(slope_s / slope_h)},
           "decision_max_r_meeting_limit": {"surrogate": choose(tmax_s), "hotspot_raw": choose(tmax_h), "hotspot_offset_corrected": choose(tmax_h_corr)},
           "dataset_design_point_r": r0}
    with open(os.path.join(REPO, "docs", "results", "exploration_demo.json"), "w") as f:
        json.dump(out, f, indent=2)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.semilogx(rs, tmax_s, "o-", label=f"surrogate ({t_s:.2f} s for {a.points} evals)")
    ax.semilogx(rs, tmax_h, "s--", label=f"HotSpot raw ({t_h:.0f} s)")
    ax.semilogx(rs, tmax_h_corr, "s:", color="gray", label="HotSpot, offset-corrected at the dataset's r")
    ax.axhline(a.limit, color="tab:red", ls="--", label=f"limit {a.limit:.0f} K"); ax.axvline(r0, color="k", lw=0.8, ls=":", label="dataset design point")
    ax.set_xlabel("convective resistance r (K/W), h = 1/(r*0.0282)"); ax.set_ylabel("Tmax (K)")
    ax.set_title("Cooling sweep on one seen chip: surrogate vs solver"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(REPO, "docs", "results", "exploration_demo.png"), dpi=120)
    print(json.dumps({k: out[k] for k in ("wall_time_s", "slope_dTmax_dlnr_K", "decision_max_r_meeting_limit")}, indent=1))


if __name__ == "__main__":
    main()
