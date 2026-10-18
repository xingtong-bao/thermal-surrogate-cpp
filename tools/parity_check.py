#!/usr/bin/env python3
"""S3.2: PyTorch vs ONNX Runtime parity, and reference pairs for the C++ tests.

Two questions
-------------
1. Does the exported graph (models/model.onnx) compute the same thing as the
   PyTorch model, over enough samples to trust it? 100 level4 test samples,
   compared in physical Kelvin (the graph has normalization baked in, see
   tools/export_onnx.py). Gate: max |diff| < 1e-4 K (contract §4, about 3
   float32 ulp at 350 K) was the first idea; see GATE_* below for what the
   measurements turned it into.

   Three references are computed, because the first run of this script
   taught us that "PyTorch" is not one number:
     torch_b1   PyTorch fp32, one sample at a time   -> the parity reference
     torch_b20  PyTorch fp32 in batches of 20 (what the benchmark's evaluate()
                does) -> differs from torch_b1 by up to ~1e-2 K on a few pixels,
                because the batched CPU convolution kernel accumulates in a
                different order
     fp64       PyTorch float64, one sample at a time -> the "true" value of
                the network; both fp32 paths are judged against it
   ONNX Runtime agrees with torch_b1 to 1 ulp and is as close to fp64 as
   torch_b1 is; the batched path is the least accurate of the three. The
   gate is applied to ORT vs torch_b1 and to ORT vs fp64.
2. What does the C++ predictor (S4.2) compare itself against? Ten of those
   samples are written as plain .npy files with the ONNX Runtime (Python)
   output as the expected field. C++ runs the same ONNX file through the
   same runtime, so its result must match to well under the 1e-4 K gate; a
   larger difference points at a buffer-layout or dtype bug, not at numerics.

Outputs
-------
  docs/results/parity_python.json       summary over 100 samples
  tests/fixtures/parity/input_XX.npy    (64,64,1,7) float32, C-contiguous
  tests/fixtures/parity/field_XX.npy    (64,64,1)   float32, ORT output in K
  tests/fixtures/parity/torch_XX.npy    (64,64,1)   float32, PyTorch output
  tests/fixtures/parity/manifest.json   indices, shapes, SHA-256 of each file

Usage
-----
    python tools/parity_check.py --bench ../external/ThermalBench
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

N_SAMPLES = 100
N_FIXTURES = 10
# Gates (contract section 4, revised in S3.2 from the measured fp32 floors):
#   PyTorch <-> ONNX Runtime: two fp32 implementations with different kernels.
#   Measured floors over 100 samples: torch fp32 (batch 1) vs fp64 3.4e-4 K,
#   ORT vs fp64 8.5e-4 K, torch fp32 batched vs fp64 9.7e-3 K. A wiring bug
#   (wrong layout, dtype or constant) produces >= 0.1 K, so 5e-3 K max and
#   1e-4 K mean separate "noise" from "bug" with margin on both sides.
#   ORT(Python) <-> ORT(C++): same runtime, same file -> 1e-4 K (about 3 ulp).
GATE_MAX_K = 5e-3
GATE_MEAN_K = 1e-4
GATE_CPP_K = 1e-4


def parse_cli():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", required=True)
    p.add_argument("--repo", default=os.path.join(os.path.dirname(__file__), ".."))
    p.add_argument("--model", default="models/model.onnx", help="relative to --repo")
    p.add_argument("--checkpoint", default="checkpoints/level4_UNet/model.pt", help="relative to --bench")
    return p.parse_args()


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    cli = parse_cli()
    bench, repo = os.path.abspath(cli.bench), os.path.abspath(cli.repo)
    model_path = os.path.join(repo, cli.model)
    fix_dir = os.path.join(repo, "tests", "fixtures", "parity")

    os.chdir(bench)                                  # benchmark imports by bare name
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
    _, _, _, _, xte, yte = data_provider(parse_args())
    # Spread the samples over the whole test split (every 30th) so all ten
    # cases and the whole temperature range are represented.
    idx = np.arange(0, xte.shape[0], xte.shape[0] // N_SAMPLES)[:N_SAMPLES]
    x = xte[idx].contiguous()                                           # (100,64,64,1,7)
    truth = yte[idx].numpy().astype(np.float64)

    # ---- PyTorch references --------------------------------------------------
    x_norm, net, y_norm = load_checkpoint(cli.checkpoint)
    net.eval()

    def torch_run(xb, dtype, batch):
        """Same wiring as the ONNX graph: normalize -> net -> de-normalize, in `dtype`."""
        m = net.to(dtype)
        xm, xs = x_norm.mean.detach().to(dtype), x_norm.std.detach().to(dtype)
        ym, ys = y_norm.mean.detach().to(dtype), y_norm.std.detach().to(dtype)
        outs = []
        with torch.no_grad():
            for s in range(0, xb.shape[0], batch):
                xt = xb[s:s + batch].to(dtype)
                xn = ((xt - xm) / xs).squeeze(-2).permute(0, 3, 1, 2)
                outs.append((m(xn) * ys + ym).permute(0, 2, 3, 1))
        return torch.cat(outs).numpy()

    y_torch = torch_run(x, torch.float32, 1).astype(np.float32)              # torch_b1 (reference)
    y_b20 = torch_run(x, torch.float32, 20).astype(np.float32)               # benchmark's batched path
    y_64 = torch_run(x.double(), torch.float64, 1)                           # high-precision truth of the net
    net.to(torch.float32)
    # Cross-check: the batched path must equal the benchmark's own evaluate() output.
    y_bench = _predict(net, x_norm, y_norm, x, "UNet", "cpu").numpy().astype(np.float32)
    assert np.abs(y_bench - y_b20).max() < 1e-6, "torch_b20 does not reproduce the benchmark path"

    # ---- ONNX Runtime, one sample at a time (the fixed-shape graph has B=1) --
    import onnxruntime as ort
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    xin = x.numpy().astype(np.float32)
    y_ort = np.stack([sess.run(["field"], {"input": xin[i:i + 1]})[0][0] for i in range(len(idx))])

    diff = np.abs(y_ort.astype(np.float64) - y_torch.astype(np.float64))
    per_sample_max = diff.reshape(len(idx), -1).max(axis=1)
    # Express the difference in float32 ulps of the output magnitude, so the
    # number is meaningful regardless of the temperature level.
    ulp = np.spacing(np.abs(y_torch).astype(np.float32)).astype(np.float64)
    diff_ulp = diff / ulp
    def mx(a, b):
        return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
    cross = {"ort_vs_torch_b1_K": mx(y_ort, y_torch), "ort_vs_fp64_K": mx(y_ort, y_64),
             "torch_b1_vs_fp64_K": mx(y_torch, y_64), "torch_b20_vs_fp64_K": mx(y_b20, y_64),
             "torch_b20_vs_torch_b1_K": mx(y_b20, y_torch), "ort_vs_torch_b20_K": mx(y_ort, y_b20)}
    print("[cross] " + "  ".join(f"{k}={v:.2e}" for k, v in cross.items()))
    summary = {
        "n_samples": int(len(idx)), "sample_indices_in_test_split": idx.tolist(),
        "reference": "torch_b1 = PyTorch fp32, batch 1, same normalize/net/de-normalize wiring as the graph",
        "max_abs_diff_cross_table_K": cross,
        "model": cli.model, "model_sha256": sha256(model_path),
        "onnxruntime": ort.__version__, "torch": torch.__version__, "device": "cpu",
        "max_abs_diff_K": float(diff.max()), "mean_abs_diff_K": float(diff.mean()),
        "p99_abs_diff_K": float(np.percentile(diff, 99)),
        "max_diff_ulp": float(diff_ulp.max()), "fraction_pixels_exact": float((diff == 0).mean()),
        "per_sample_max_abs_diff_K": {"min": float(per_sample_max.min()), "median": float(np.median(per_sample_max)),
                                      "max": float(per_sample_max.max())},
        "gates": {"max_abs_diff_K": GATE_MAX_K, "mean_abs_diff_K": GATE_MEAN_K, "cpp_vs_python_ort_K": GATE_CPP_K},
        "passes_gate": bool(diff.max() < GATE_MAX_K and diff.mean() < GATE_MEAN_K
                            and cross["ort_vs_fp64_K"] < GATE_MAX_K),
        # Both paths against the dataset truth: must agree to the same digits.
        "mae_vs_truth_K": {"torch_b1": float(np.abs(y_torch - truth).mean()), "torch_b20": float(np.abs(y_b20 - truth).mean()),
                           "onnxruntime": float(np.abs(y_ort - truth).mean())},
    }
    print(f"[parity] n={len(idx)}  max|diff|={summary['max_abs_diff_K']:.3e} K ({summary['max_diff_ulp']:.0f} ulp)  "
          f"mean={summary['mean_abs_diff_K']:.2e}  exact pixels={summary['fraction_pixels_exact']:.1%}  "
          f"gates max<{GATE_MAX_K} mean<{GATE_MEAN_K} -> {'PASS' if summary['passes_gate'] else 'FAIL'}")
    print(f"[parity] MAE vs truth: torch_b1 {summary['mae_vs_truth_K']['torch_b1']:.4f}  ORT {summary['mae_vs_truth_K']['onnxruntime']:.4f}")

    # ---- fixtures for the C++ test ------------------------------------------
    os.makedirs(fix_dir, exist_ok=True)
    manifest = {"description": "Reference pairs for the C++ parity test (S4.2). field_XX is the ONNX Runtime "
                               "(Python) output for input_XX; torch_XX is the PyTorch fp32 batch-1 output. All float32, C order.",
                "model_sha256": summary["model_sha256"], "gate_cpp_vs_python_ort_K": GATE_CPP_K, "pairs": []}
    pick = np.linspace(0, len(idx) - 1, N_FIXTURES).round().astype(int)
    total = 0
    for k, j in enumerate(pick):
        files = {}
        for tag, arr in (("input", xin[j]), ("field", y_ort[j]), ("torch", y_torch[j])):
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            path = os.path.join(fix_dir, f"{tag}_{k:02d}.npy")
            np.save(path, arr)
            files[tag] = {"file": os.path.basename(path), "shape": list(arr.shape), "sha256": sha256(path)}
            total += os.path.getsize(path)
        manifest["pairs"].append({"pair": k, "test_split_index": int(idx[j]), "global_index": int(12000 + idx[j]),
                                  "tmax_K": float(y_ort[j].max()), "ort_vs_torch_max_abs_diff_K": float(per_sample_max[j]), **files})
    with open(os.path.join(fix_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    summary["fixtures"] = {"dir": "tests/fixtures/parity", "pairs": N_FIXTURES, "total_bytes": int(total)}
    print(f"[fixtures] {N_FIXTURES} pairs, {total / 1e6:.2f} MB -> tests/fixtures/parity/")

    os.makedirs(os.path.join(repo, "docs", "results"), exist_ok=True)
    with open(os.path.join(repo, "docs", "results", "parity_python.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("[out] docs/results/parity_python.json")
    sys.exit(0 if summary["passes_gate"] else 1)


if __name__ == "__main__":
    main()
