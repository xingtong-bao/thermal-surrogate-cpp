#!/usr/bin/env python3
"""Export fixed evaluation sets as plain .npy for the C++ side.

Three sets, all in contract layout (n, 64, 64, 1, 7) float32 + truth (n, 64, 64, 1):

  tests/fixtures/eval50/level4_inputs.npy, level4_truth.npy   COMMITTED (6.5 MB)
      50 level4 TEST samples (every 60th), used by the S6.1 accuracy-gate test,
      the S5.2 solver validation and the S5.5 backend comparison.
  <out>/level5_inputs.npy, level5_truth.npy                    not committed
      50 level5 samples (every 100th), S5.5 only.
  <out>/train20_inputs.npy, train20_truth.npy                  not committed
      20 level4 TRAINING samples, used only to calibrate the finite-difference
      reference solver (S5.2): calibration must never see the test split.

Usage
-----
    python tools/export_eval_sets.py --data ../external/thermbench_data --out ../external/eval_sets
"""
import argparse
import json
import os

import h5py
import numpy as np


def load(path_dir, idx):
    with h5py.File(os.path.join(path_dir, "input.mat"), "r") as f:
        x = f["data"][sorted(idx)]                       # (n,P,Z,Y,X)
    with h5py.File(os.path.join(path_dir, "output.mat"), "r") as f:
        y = f["data"][sorted(idx)]                       # (n,Z,Y,X)
    # Same transpose as the benchmark loader: (B,P,Z,Y,X) -> (B,X,Y,Z,P), (B,Z,Y,X) -> (B,X,Y,Z)
    x = np.ascontiguousarray(np.transpose(x, (0, 4, 3, 2, 1)), dtype=np.float32)
    y = np.ascontiguousarray(np.transpose(y, (0, 3, 2, 1)), dtype=np.float32)
    return x, y


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True, help="folder for the non-committed sets")
    p.add_argument("--repo", default=os.path.join(os.path.dirname(__file__), ".."))
    a = p.parse_args()
    repo = os.path.abspath(a.repo)
    os.makedirs(a.out, exist_ok=True)
    fix = os.path.join(repo, "tests", "fixtures", "eval50")
    os.makedirs(fix, exist_ok=True)

    l4 = os.path.join(a.data, "level4_steady")
    l5 = os.path.join(a.data, "level5_steady")
    with h5py.File(os.path.join(l4, "input.mat"), "r") as f:
        B = f["data"].shape[0]
    n_train = int(int(B * 0.8) * 0.9)
    test_start = int(B * 0.8)

    sets = {
        "level4": (l4, list(range(test_start, B, 60))[:50], fix),
        "level5": (l5, list(range(0, 5000, 100))[:50], a.out),
        "train20": (l4, list(range(0, n_train, n_train // 20))[:20], a.out),
    }
    manifest = {}
    for name, (d, idx, where) in sets.items():
        x, y = load(d, idx)
        np.save(os.path.join(where, f"{name}_inputs.npy"), x)
        np.save(os.path.join(where, f"{name}_truth.npy"), y)
        manifest[name] = {"dir": os.path.relpath(where, repo) if where.startswith(repo) else where,
                          "global_indices": idx, "n": len(idx), "input_shape": list(x.shape), "truth_shape": list(y.shape)}
        print(f"[{name}] n={len(idx)} -> {where}  ({x.nbytes/1e6:.1f} MB + {y.nbytes/1e6:.1f} MB)")
    with open(os.path.join(fix, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"description": "Fixed evaluation subsets; level4 is the committed 50-sample test subset "
                                  "(contract section 4 gate test), level5 and train20 are regenerated locally.",
                   "sets": manifest}, f, indent=2)


if __name__ == "__main__":
    main()
