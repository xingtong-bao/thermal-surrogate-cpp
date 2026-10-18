#!/usr/bin/env python3
"""S3.1: export the level4 U-Net checkpoint to ONNX + write models/metadata.json

What is exported
----------------
Not the bare network. The exported graph takes the *contract* tensor
(B, X, Y, Z, P) = (1, 64, 64, 1, 7) float32 in physical units and returns the
temperature field (B, X, Y, Z) in Kelvin. The per-channel input
normalization, the layout permutation to the network's (B, P, X, Y) and the
output de-normalization are baked into the graph as constant ops:

    raw (B,X,Y,Z,P) -> (x - mean_c) / std_c -> squeeze Z, permute -> U-Net
                    -> (B,1,X,Y) -> * y_std + y_mean -> permute -> (B,X,Y,Z)

Why bake it in: every number the C++ side would otherwise have to re-implement
(7 means, 7 stds, 2 output constants, one permutation) is a place where parity
can silently break. With it inside the graph, the C++ predictor is "copy the
input buffer in, read the output buffer out", and the parity test (S3.2/S4.2)
compares physical Kelvin values end to end. The constants are still written to
metadata.json for inspection and so a client can validate them against
contract/stats.json.

Two files
---------
  models/model.onnx          fixed shape (1,64,64,1,7); the production artefact
  models/model_dynamic.onnx  batch, X and Y dynamic (multiples of 8, because of
                             the three 2x pools); for the S4.6 size experiment
Both are gitignored (20 MB each); models/metadata.json and this script are
committed, so anyone with the benchmark checkpoint can regenerate them
byte-for-byte: see models/README.md.

Usage
-----
    python tools/export_onnx.py --bench ../external/ThermalBench
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sys

import numpy as np
import torch

OPSET = 17                     # highest opset torch 2.0.1 exports; ORT 1.23 reads it
MODEL_VERSION = "level4-unet-1.0.0"   # bumped whenever the checkpoint or the graph changes
CHANNELS = ["chiplet_power", "grid_x", "grid_y", "local_thermal_k",
            "ambient_K", "h_w_m2k", "r_convec_k_per_w"]


def parse_cli():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", required=True)
    p.add_argument("--repo", default=os.path.join(os.path.dirname(__file__), ".."))
    p.add_argument("--checkpoint", default="checkpoints/level4_UNet/model.pt", help="relative to --bench")
    return p.parse_args()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ContractWrapper(torch.nn.Module):
    """U-Net plus normalization and layout, so the graph speaks the contract."""

    def __init__(self, net, x_mean, x_std, y_mean, y_std):
        super().__init__()
        self.net = net
        # Registered as buffers so they become graph constants (not inputs).
        self.register_buffer("x_mean", x_mean.reshape(1, 1, 1, 1, -1).float())
        self.register_buffer("x_std", x_std.reshape(1, 1, 1, 1, -1).float())
        self.register_buffer("y_mean", y_mean.reshape(()).float())
        self.register_buffer("y_std", y_std.reshape(()).float())

    def forward(self, x):                       # x: (B,X,Y,Z,P) physical units
        xn = (x - self.x_mean) / self.x_std     # per-channel normalization
        xn = xn.squeeze(-2).permute(0, 3, 1, 2) # (B,P,X,Y) as the net expects
        out = self.net(xn)                      # (B,1,X,Y) normalized temperature
        out = out * self.y_std + self.y_mean    # back to Kelvin
        return out.permute(0, 2, 3, 1)          # (B,X,Y,Z=1)


def main():
    cli = parse_cli()
    bench, repo = os.path.abspath(cli.bench), os.path.abspath(cli.repo)
    models_dir = os.path.join(repo, "models")
    os.makedirs(models_dir, exist_ok=True)

    os.chdir(bench)                              # benchmark imports by bare name
    sys.path.insert(0, bench)
    os.environ["PER_CHANNEL_NORM"] = "1"
    from utils.compat import install
    install()
    from utils.tools import load_checkpoint
    x_norm, net, y_norm = load_checkpoint(cli.checkpoint)
    net.eval()
    wrapper = ContractWrapper(net, x_norm.mean.detach(), x_norm.std.detach(),
                              y_norm.mean.detach(), y_norm.std.detach()).eval()

    # Cross-check the baked constants against contract/stats.json (S2.3 copied
    # the same checkpoint normalizer there). Any mismatch means a stale file.
    with open(os.path.join(repo, "contract", "stats.json"), encoding="utf-8") as f:
        stats = json.load(f)
    ck = stats["checkpoint_normalizer"]
    assert np.allclose(ck["x_mean"], x_norm.mean.detach().numpy(), rtol=1e-6), "stats.json x_mean stale"
    assert np.allclose(ck["x_std"], x_norm.std.detach().numpy(), rtol=1e-6), "stats.json x_std stale"

    # Sanity before export: the wrapper must reproduce the benchmark's own
    # prediction path on the fixture (same ops, different wiring).
    z = np.load(os.path.join(repo, "tests", "fixtures", "sample_input.npz"))
    x = torch.from_numpy(z["input"][None])      # (1,64,64,1,7)
    with torch.no_grad():
        ref = y_norm.inverse(net(x_norm.forward(x).squeeze(-2).permute(0, 3, 1, 2))).permute(0, 2, 3, 1)
        got = wrapper(x)
    wrap_diff = float((ref - got).abs().max())
    assert wrap_diff < 1e-5, f"wrapper differs from benchmark path by {wrap_diff}"
    fixture_mae = float((got[0, ..., 0].numpy() - z["expected_output"][..., 0]).__abs__().mean())
    print(f"[wrapper] max|diff| vs benchmark path = {wrap_diff:.2e}; fixture MAE vs truth = {fixture_mae:.3f} K")

    # ---- export: fixed shape, then dynamic ----------------------------------------
    outputs = {}
    for name, dynamic in (("model.onnx", False), ("model_dynamic.onnx", True)):
        path = os.path.join(models_dir, name)
        kw = dict(dynamic_axes={"input": {0: "batch", 1: "X", 2: "Y"}, "field": {0: "batch", 1: "X", 2: "Y"}}) if dynamic else {}
        torch.onnx.export(wrapper, x, path, opset_version=OPSET, input_names=["input"], output_names=["field"],
                          do_constant_folding=True, **kw)
        import onnx
        m = onnx.load(path)
        onnx.checker.check_model(m, full_check=True)
        # Run it once through ONNX Runtime and compare with the PyTorch wrapper.
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        y_ort = sess.run(["field"], {"input": z["input"][None]})[0]
        diff = float(np.abs(y_ort - got.numpy()).max())
        outputs[name] = {"size_bytes": os.path.getsize(path), "sha256": sha256(path), "max_abs_diff_vs_torch_K": diff,
                         "dynamic_axes": bool(dynamic), "ir_version": m.ir_version,
                         "opset": [o.version for o in m.opset_import if o.domain in ("", "ai.onnx")][0],
                         "node_count": len(m.graph.node)}
        print(f"[{name}] {outputs[name]['size_bytes']/1e6:.1f} MB, {outputs[name]['node_count']} nodes, opset {outputs[name]['opset']}, "
              f"checker ok, ORT vs torch max|diff| = {diff:.2e}")

    # ---- metadata.json --------------------------------------------------------------
    meta = {
        "model_version": MODEL_VERSION,
        "schema_version": stats["schema_version"],
        "source": {"benchmark": "IC-ThermBench (CC BY 4.0)", "checkpoint": cli.checkpoint,
                   "checkpoint_sha256": sha256(cli.checkpoint), "training_data": "level4_steady (S4), split per contract/stats.json",
                   "architecture": "UNet in=7 out=1 channels (58,116,232,348), 5,062,125 parameters"},
        "export": {"tool": "tools/export_onnx.py", "torch": torch.__version__, "onnx": onnx.__version__,
                   "onnxruntime": ort.__version__, "opset": OPSET, "date": dt.date.today().isoformat()},
        "io": {"input": {"name": "input", "shape": [1, 64, 64, 1, 7], "axes": ["B", "X", "Y", "Z", "P"], "dtype": "float32",
                         "layout": "C-contiguous, physical units as in contract section 2"},
               "output": {"name": "field", "shape": [1, 64, 64, 1], "axes": ["B", "X", "Y", "Z"], "dtype": "float32", "unit": "K"}},
        "normalization": {"baked_into_graph": True,
                          "note": "The graph normalizes its input and de-normalizes its output; the client passes physical values. "
                                  "Constants are listed for inspection and must equal contract/stats.json checkpoint_normalizer.",
                          "channels": CHANNELS, "x_mean": ck["x_mean"], "x_std": ck["x_std"], "y_mean": ck["y_mean"], "y_std": ck["y_std"]},
        "files": outputs,
        "dynamic_constraints": "model_dynamic.onnx: X and Y must be multiples of 8 (three 2x max-pools); Z must be 1; P must be 7",
    }
    with open(os.path.join(models_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("[meta] -> models/metadata.json")


if __name__ == "__main__":
    main()
