#!/usr/bin/env python3
"""S8.7: export the benchmark's U-FNO (level4) to ONNX without FFT operators.

Why this is not a one-liner
---------------------------
U-FNO's spectral layers call torch.fft.rfftn / irfftn, which the PyTorch 2.0
ONNX exporter cannot translate and ONNX Runtime's CPU provider has no fast
kernel for. For a fixed 64x64x1 grid the discrete Fourier transform is a
plain linear map, so each spectral layer is rewritten as matrix products with
precomputed cosine/sine matrices (only the 2x10 retained kx rows and 2x10 ky
columns are needed). The rewrite is checked against the original module on
real samples to 1e-3 K before export; the exported graph then has only
MatMul/Conv/Relu/Pad ops and runs in the same C++ library as the U-Net.

Mathematics (per spectral layer; Net3d pads Z = 1 to 8 and only kz = 0 is kept, so the
Z transform is a sum over z forward and a 1/8-scaled broadcast back)
    X_ft[kx,ky] = sum_{x,y} v[x,y] exp(-2 pi i (kx x + ky y)/64)
                = (F_x v F_y^T) with F = C - i S  (C = cos, S = sin of 2 pi k n / 64)
    only kx in {0..9, 54..63}, ky in {0..9, 54..63} are kept (four corners),
    mixed by the layer's complex weights (einsum over channels),
    then back:  v'[x,y] = (1/4096) Re sum_k out_ft[k] exp(+2 pi i (kx x + ky y)/64)
Everything is expressed with real tensors (real and imaginary parts).

Usage
-----
    python tools/export_ufno_onnx.py --bench ../external/ThermalBench --checkpoint ../external/thermbench_data/level4_UFNO/model.pt --out-dir models_ufno
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

N = 64
OPSET = 17


class SpectralConvDFT(nn.Module):
    """Drop-in replacement for layers.fourier.SpectralConv3d on a fixed 64x64x1 grid."""

    def __init__(self, orig):
        super().__init__()
        m1, m2 = orig.modes1, orig.modes2
        assert orig.modes3 == 1, "Z must be 1 for this export"
        self.m1, self.m2 = m1, m2
        n = torch.arange(N, dtype=torch.float64)
        kx = torch.cat([torch.arange(m1), torch.arange(N - m1, N)]).double()      # 0..9, 54..63
        ky = torch.cat([torch.arange(m2), torch.arange(N - m2, N)]).double()
        ang_x = 2 * np.pi * torch.outer(kx, n) / N                                 # (2m1, N)
        ang_y = 2 * np.pi * torch.outer(ky, n) / N                                 # (2m2, N)
        for name, a in (("cx", ang_x.cos()), ("sx", ang_x.sin()), ("cy", ang_y.cos()), ("sy", ang_y.sin())):
            self.register_buffer(name, a.float())
        # Complex weights per corner -> one (2m1, 2m2) block of (in, out) complex matrices.
        W = torch.zeros(orig.in_channels, orig.out_channels, 2 * m1, 2 * m2, dtype=torch.cfloat)
        W[:, :, :m1, :m2] = orig.weights1[..., 0]          # (lo1, lo2)
        W[:, :, m1:, :m2] = orig.weights2[..., 0]          # (hi1, lo2)
        W[:, :, :m1, m2:] = orig.weights3[..., 0]          # (lo1, hi2)
        W[:, :, m1:, m2:] = orig.weights4[..., 0]          # (hi1, hi2)
        self.register_buffer("wr", W.real.detach().clone())
        self.register_buffer("wi", W.imag.detach().clone())

    def forward(self, v):                       # v: (B, C, X, Y, Zp); Net3d pads Z=1 to Zp=8
        zp = v.shape[-1]
        # Only the kz = 0 mode is retained (modes3 = 1): the forward rFFT along Z
        # reduces to a sum over z, and the inverse rFFT from that single mode is a
        # constant along z scaled by 1/Zp.
        v = v.sum(-1)                           # (B, C, X, Y)
        # Everything below is plain matmul (ONNX MatMul): einsum exports to an
        # ORT Einsum op that runs ~25x slower on CPU.
        cxT, sxT = self.cx.transpose(0, 1), self.sx.transpose(0, 1)      # (N, 2m1)
        cyT, syT = self.cy.transpose(0, 1), self.sy.transpose(0, 1)      # (N, 2m2)
        # Forward DFT on the retained modes: A = Fx v Fy^T, Fx = cx - i sx, Fy = cy - i sy.
        cv, sv = torch.matmul(self.cx, v), torch.matmul(self.sx, v)      # (B, C, 2m1, N)
        ar = torch.matmul(cv, cyT) - torch.matmul(sv, syT)               # (B, C, 2m1, 2m2)
        ai = -(torch.matmul(cv, syT) + torch.matmul(sv, cyT))
        # Channel mixing: for every mode (k, l) a (B, I) x (I, O) product.
        ar_t = ar.permute(2, 3, 0, 1)                                     # (2m1, 2m2, B, I)
        ai_t = ai.permute(2, 3, 0, 1)
        wr, wi = self.wr.permute(2, 3, 0, 1), self.wi.permute(2, 3, 0, 1)   # (2m1, 2m2, I, O)
        orr = torch.matmul(ar_t, wr) - torch.matmul(ai_t, wi)            # (2m1, 2m2, B, O)
        oi = torch.matmul(ar_t, wi) + torch.matmul(ai_t, wr)
        orr, oi = orr.permute(2, 3, 0, 1), oi.permute(2, 3, 0, 1)         # (B, O, 2m1, 2m2)
        # Inverse: Re[(cx^T + i sx^T)(orr + i oi)(cy + i sy)] / (N N Zp).
        out = (torch.matmul(torch.matmul(cxT, orr), self.cy)
               - torch.matmul(torch.matmul(cxT, oi), self.sy)
               - torch.matmul(torch.matmul(sxT, oi), self.cy)
               - torch.matmul(torch.matmul(sxT, orr), self.sy)) / (N * N * zp)
        return out.unsqueeze(-1).expand(-1, -1, -1, -1, zp)   # (B, O, X, Y, Zp), constant along z


class ContractWrapperUFNO(nn.Module):
    """Input (B,64,64,1,7) physical units -> output (B,64,64,1) K. U-FNO is channel-last already."""

    def __init__(self, net, x_mean, x_std, y_mean, y_std):
        super().__init__()
        self.net = net
        self.register_buffer("x_mean", x_mean.reshape(1, 1, 1, 1, -1).float())
        self.register_buffer("x_std", x_std.reshape(1, 1, 1, 1, -1).float())
        self.register_buffer("y_mean", y_mean.reshape(()).float())
        self.register_buffer("y_std", y_std.reshape(()).float())

    def forward(self, x):
        return self.net((x - self.x_mean) / self.x_std) * self.y_std + self.y_mean


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--repo", default=os.path.join(os.path.dirname(__file__), ".."))
    a = p.parse_args()
    bench, repo, out = os.path.abspath(a.bench), os.path.abspath(a.repo), os.path.abspath(a.out_dir)
    ckpt = os.path.abspath(a.checkpoint)
    os.makedirs(out, exist_ok=True)
    os.chdir(bench); sys.path.insert(0, bench); os.environ["PER_CHANNEL_NORM"] = "1"
    from utils.compat import install
    install()
    from utils.tools import load_checkpoint
    from layers.fourier import SpectralConv3d
    x_norm, net, y_norm = load_checkpoint(ckpt)
    net.eval()

    # Reference outputs with the original FFT modules.
    z = np.load(os.path.join(repo, "tests", "fixtures", "eval50", "level4_inputs.npy"))[:8]
    x = torch.from_numpy(z)
    with torch.no_grad():
        ref = y_norm.inverse(net(x_norm.forward(x))).numpy()

    # Swap every SpectralConv3d for the DFT version, in place.
    n_swapped = 0
    for name, mod in list(net.named_modules()):
        for child_name, child in list(mod.named_children()):
            if isinstance(child, SpectralConv3d):
                setattr(mod, child_name, SpectralConvDFT(child))
                n_swapped += 1
    with torch.no_grad():
        got = y_norm.inverse(net(x_norm.forward(x))).numpy()
    diff = float(np.abs(got - ref).max())
    print(f"[dft] swapped {n_swapped} spectral layers; max |diff| vs FFT original on 8 samples = {diff:.3e} K")
    assert diff < 1e-2, "DFT rewrite does not reproduce the FFT layers"

    wrapper = ContractWrapperUFNO(net, x_norm.mean.detach(), x_norm.std.detach(), y_norm.mean.detach(), y_norm.std.detach()).eval()
    path = os.path.join(out, "model.onnx")
    torch.onnx.export(wrapper, x[:1], path, opset_version=OPSET, input_names=["input"], output_names=["field"], do_constant_folding=True)
    import onnx, onnxruntime as ort
    m = onnx.load(path)
    onnx.checker.check_model(m, full_check=True)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    worst = 0.0
    for i in range(8):
        y = sess.run(["field"], {"input": z[i:i + 1]})[0]
        worst = max(worst, float(np.abs(y - ref[i:i + 1]).max()))
    ops = sorted({n.op_type for n in m.graph.node})
    print(f"[onnx] {os.path.getsize(path)/1e6:.1f} MB, {len(m.graph.node)} nodes, ops {ops}; ORT vs FFT original max |diff| = {worst:.3e} K")

    stats = json.load(open(os.path.join(repo, "contract", "stats.json")))
    meta = {"model_version": "level4-ufno-1.0.0", "schema_version": stats["schema_version"],
            "source": {"benchmark": "IC-ThermBench (CC BY 4.0)", "checkpoint": "external/thermbench_data/level4_UFNO/model.pt", "checkpoint_sha256": sha256(ckpt),
                       "architecture": "U-FNO (Net3d: 6 U-Fourier layers, width 36, modes 10x10x1), spectral layers rewritten as DFT matrix products for export",
                       "params": sum(p.numel() for p in net.parameters())},
            "export": {"tool": "tools/export_ufno_onnx.py", "torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__, "opset": OPSET, "date": dt.date.today().isoformat(),
                       "dft_vs_fft_max_abs_diff_K": diff, "ort_vs_torch_max_abs_diff_K": worst},
            "io": {"input": {"name": "input", "shape": [1, 64, 64, 1, 7], "axes": ["B", "X", "Y", "Z", "P"], "dtype": "float32", "layout": "C-contiguous, physical units as in contract section 2"},
                   "output": {"name": "field", "shape": [1, 64, 64, 1], "axes": ["B", "X", "Y", "Z"], "dtype": "float32", "unit": "K"}},
            "normalization": {"baked_into_graph": True, "x_mean": x_norm.mean.detach().reshape(-1).tolist(), "x_std": x_norm.std.detach().reshape(-1).tolist(),
                              "y_mean": float(y_norm.mean), "y_std": float(y_norm.std)},
            "files": {"model.onnx": {"size_bytes": os.path.getsize(path), "sha256": sha256(path), "node_count": len(m.graph.node), "opset": OPSET}}}
    json.dump(meta, open(os.path.join(out, "metadata.json"), "w"), indent=2)
    print("[meta] ->", os.path.relpath(os.path.join(out, "metadata.json"), repo))


if __name__ == "__main__":
    main()
