#!/usr/bin/env python3
"""S4.3 support: export the I5 configuration bank and validator fixtures.

The C++ validator implements contract rule I5 (layout novelty): distance from
the input's 6-feature configuration vector to the nearest level4 training
sample, in training-std units. It needs the 10 800 training vectors. This
script writes them once as models/config_bank.npy (float32, (10800, 6),
260 KB, committed) using exactly the feature code of tools/layout_novelty.py,
and writes the expected scores for the two validator fixtures so the C++
re-implementation of the feature extraction can be checked bit for bit.

Also converts the two .npz fixtures to plain .npy (the C++ reader supports
only .npy) and records their per-channel OOB distances for the I4 tests.

Usage
-----
    python tools/export_config_bank.py --data ../external/thermbench_data
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from layout_novelty import config_features, min_euclid   # noqa: E402  (same code as S2b.3)

N_TEST = 3000


def features_from_contract_tensor(x):
    """config_features() reads the raw (Y,X) HDF5 layout; the fixtures are in
    contract layout (X,Y,Z,P). Both orderings give the same descriptors (the
    features are orientation-invariant), so we only need to present the maps
    in a (n, P, Z, Y, X)-like order to reuse the function via a temp file."""
    import tempfile
    raw = np.transpose(x[None], (0, 4, 3, 2, 1)).astype(np.float32)      # (1,P,Z,Y,X)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "input.mat")
        with h5py.File(p, "w") as f:
            f.create_dataset("data", data=raw)
        return config_features(p, slice(0, None))[0]


def main():
    cli = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    cli.add_argument("--data", required=True)
    cli.add_argument("--repo", default=os.path.join(os.path.dirname(__file__), ".."))
    a = cli.parse_args()
    repo = os.path.abspath(a.repo)
    p4 = os.path.join(a.data, "level4_steady", "input.mat")
    with h5py.File(p4, "r") as f:
        B = f["data"].shape[0]
    n_train = int(int(B * 0.8) * 0.9)
    bank = config_features(p4, slice(0, n_train)).astype(np.float32)
    np.save(os.path.join(repo, "models", "config_bank.npy"), np.ascontiguousarray(bank))
    print(f"[bank] {bank.shape} -> models/config_bank.npy ({bank.nbytes/1024:.0f} KiB)")

    with open(os.path.join(repo, "contract", "stats.json"), encoding="utf-8") as f:
        stats = json.load(f)
    ln = stats["layout_novelty"]
    mu, sd = np.array(ln["train_mean"]), np.array(ln["train_std"])
    assert np.allclose(bank.mean(axis=0), mu, rtol=1e-4) and np.allclose(bank.std(axis=0), sd, rtol=1e-4), "stats.json drifted"
    lo = np.array([c["min"] for c in stats["channels"]]); hi = np.array([c["max"] for c in stats["channels"]])

    fix = os.path.join(repo, "tests", "fixtures")
    expected = {"threshold": ln["threshold"], "oob_scale": stats["oob_scale"], "fixtures": {}}
    for name in ("sample_input", "sample_input_level5"):
        z = np.load(os.path.join(fix, name + ".npz"))
        x = np.ascontiguousarray(z["input"], dtype=np.float32)
        np.save(os.path.join(fix, name + ".npy"), x)
        np.save(os.path.join(fix, name.replace("input", "output") + ".npy"), np.ascontiguousarray(z["expected_output"], dtype=np.float32))
        feat = features_from_contract_tensor(x)
        score = float(min_euclid(((feat - mu) / sd)[None], (bank - mu) / sd)[0])
        v = x.reshape(-1, 7)
        d = np.maximum(0, np.maximum((lo - v.min(axis=0)) / (hi - lo), (v.max(axis=0) - hi) / (hi - lo)))
        expected["fixtures"][name] = {"features": feat.tolist(), "novelty_score": score,
                                      "oob_distance_per_channel": d.tolist(), "oob_channels": [stats["channels"][i]["name"] for i in np.where(d > 0)[0]],
                                      "expected_status": "Ok" if (score <= ln["threshold"] and not (d > 0).any()) else "OutOfDistribution"}
        print(f"[{name}] features={np.round(feat, 4).tolist()} score={score:.4f} oob={expected['fixtures'][name]['oob_channels']} -> {expected['fixtures'][name]['expected_status']}")
    with open(os.path.join(fix, "validator_expected.json"), "w", encoding="utf-8") as f:
        json.dump(expected, f, indent=2)
    print("[out] tests/fixtures/validator_expected.json")


if __name__ == "__main__":
    main()
