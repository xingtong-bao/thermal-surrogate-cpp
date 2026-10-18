#!/usr/bin/env python3
"""S8.1: verify the surrogate's power sensitivity against a physics solver.

S2b.1 compared the surrogate's response to a +/-10 % chiplet-power change with
an *analytic* expectation (linear heat equation: dT = +/-10 % of the rise).
A reviewer may rightly ask: did a solver confirm that? This script runs
HotSpot (tools/reference_cli.py) on the same perturbed inputs and reports

    gain = dTmax_surrogate / dTmax_hotspot

HotSpot's absolute temperatures differ from the dataset's generator (S5.3),
but the *relative* response to power is a property of the linear conduction
problem and does not depend on the package parameters, so the ratio is a fair
test. Also reported: HotSpot's own linearity (dT / (0.1 * rise)), which must
be 1.00 if the solver is linear, as a sanity check on the method.

Usage
-----
    python tools/power_gain_verification.py --n 10
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
PY = sys.executable


def hotspot(x):
    with tempfile.TemporaryDirectory() as d:
        np.save(os.path.join(d, "in.npy"), x)
        r = subprocess.run([PY, os.path.join(HERE, "reference_cli.py"), os.path.join(d, "in.npy"), os.path.join(d, "out.npy")], capture_output=True)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-500:].decode())
        return np.load(os.path.join(d, "out.npy"))[..., 0]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=10, help="samples per set")
    p.add_argument("--factors", default="0.9,1.1")
    a = p.parse_args()
    factors = [float(f) for f in a.factors.split(",")]
    sess = ort.InferenceSession(os.path.join(REPO, "models", "model.onnx"), providers=["CPUExecutionProvider"])
    sets = {"level4": os.path.join(REPO, "tests", "fixtures", "eval50", "level4_inputs.npy"),
            "level5": os.path.join(REPO, "..", "external", "eval_sets", "level5_inputs.npy")}
    rows, summary = [], {}
    for name, path in sets.items():
        X = np.load(path)[:a.n]
        for i, x in enumerate(X):
            amb = float(x[0, 0, 0, 4])
            base_s = sess.run(["field"], {"input": x[None]})[0][0, ..., 0]
            base_h = hotspot(x)
            for f in factors:
                xp = x.copy()
                xp[..., 0] *= f
                ds = float(sess.run(["field"], {"input": xp[None]})[0][0, ..., 0].max() - base_s.max())
                dh = float(hotspot(xp).max() - base_h.max())
                lin = dh / ((f - 1.0) * (base_h.max() - amb))         # HotSpot's own linearity: 1.0 expected
                rows.append({"set": name, "i": i, "factor": f, "rise_hotspot_K": float(base_h.max() - amb),
                             "dTmax_surrogate_K": ds, "dTmax_hotspot_K": dh, "gain_vs_hotspot": ds / dh if dh else None,
                             "hotspot_linearity": lin})
                print(f"{name} {i} x{f}: surrogate dTmax {ds:+.2f} K, HotSpot {dh:+.2f} K, gain {ds / dh:.2f}, HotSpot linearity {lin:.3f}")
        g = [r["gain_vs_hotspot"] for r in rows if r["set"] == name and r["gain_vs_hotspot"] is not None]
        l = [r["hotspot_linearity"] for r in rows if r["set"] == name]
        summary[name] = {"n": len(X), "gain_mean": float(np.mean(g)), "gain_median": float(np.median(g)),
                         "gain_min": float(np.min(g)), "gain_max": float(np.max(g)),
                         "hotspot_linearity_mean": float(np.mean(l)), "hotspot_linearity_std": float(np.std(l))}
    out = {"method": "HotSpot grid model (tools/reference_cli.py) on the same inputs with chiplet power scaled; "
                     "gain = surrogate dTmax / HotSpot dTmax; HotSpot linearity = dTmax / (df * rise) should be 1",
           "factors": factors, "summary": summary, "rows": rows}
    with open(os.path.join(REPO, "docs", "results", "power_gain_verification.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
