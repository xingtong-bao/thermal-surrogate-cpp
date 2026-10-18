#!/usr/bin/env python3
"""S8.8: a two-model ensemble (U-Net + U-FNO) as one ONNX graph, plus a spread study.

The ensemble graph outputs the mean of the two contract-layout models (same
input, same output, so the C++ library loads it unchanged). The study part
measures, on the 50 seen and 50 unseen chips, the error of each model and of
the mean, and whether the *disagreement* |T_unet - T_ufno| predicts the true
error (a free uncertainty signal that needs no calibration set).

Usage
-----
    python tools/export_ensemble_onnx.py --bench ../external/ThermalBench
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

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)


class Ensemble(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.a, self.b = a, b

    def forward(self, x):
        return 0.5 * (self.a(x) + self.b(x))


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", required=True)
    p.add_argument("--out-dir", default=os.path.join(REPO, "models_ensemble"))
    a = p.parse_args()
    bench, out = os.path.abspath(a.bench), os.path.abspath(a.out_dir)
    os.makedirs(out, exist_ok=True)
    os.chdir(bench); sys.path.insert(0, bench); os.environ["PER_CHANNEL_NORM"] = "1"
    from utils.compat import install
    install()
    from utils.tools import load_checkpoint
    from layers.fourier import SpectralConv3d
    from export_onnx import ContractWrapper
    from export_ufno_onnx import SpectralConvDFT, ContractWrapperUFNO

    xn, unet, yn = load_checkpoint("checkpoints/level4_UNet/model.pt"); unet.eval()
    w_unet = ContractWrapper(unet, xn.mean.detach(), xn.std.detach(), yn.mean.detach(), yn.std.detach()).eval()
    xf, ufno, yf = load_checkpoint(os.path.join(REPO, "..", "external", "thermbench_data", "level4_UFNO", "model.pt")); ufno.eval()
    for name, mod in list(ufno.named_modules()):
        for cn, child in list(mod.named_children()):
            if isinstance(child, SpectralConv3d):
                setattr(mod, cn, SpectralConvDFT(child))
    w_ufno = ContractWrapperUFNO(ufno, xf.mean.detach(), xf.std.detach(), yf.mean.detach(), yf.std.detach()).eval()
    ens = Ensemble(w_unet, w_ufno).eval()

    x0 = torch.from_numpy(np.load(os.path.join(REPO, "tests", "fixtures", "sample_input.npy"))[None])
    path = os.path.join(out, "model.onnx")
    torch.onnx.export(ens, x0, path, opset_version=17, input_names=["input"], output_names=["field"], do_constant_folding=True)
    import onnx, onnxruntime as ort
    onnx.checker.check_model(onnx.load(path), full_check=True)
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = ens(x0).numpy()
    print(f"[onnx] ensemble {os.path.getsize(path)/1e6:.1f} MB; ORT vs torch {float(np.abs(sess.run(['field'], {'input': x0.numpy()})[0] - ref).max()):.2e} K")

    # ---- spread study ------------------------------------------------------------
    s_unet = ort.InferenceSession(os.path.join(REPO, "models", "model.onnx"), providers=["CPUExecutionProvider"])
    s_ufno = ort.InferenceSession(os.path.join(REPO, "models_ufno", "model.onnx"), providers=["CPUExecutionProvider"])
    sets = {"level4": (os.path.join(REPO, "tests", "fixtures", "eval50", "level4_inputs.npy"), os.path.join(REPO, "tests", "fixtures", "eval50", "level4_truth.npy")),
            "level5": (os.path.join(REPO, "..", "external", "eval_sets", "level5_inputs.npy"), os.path.join(REPO, "..", "external", "eval_sets", "level5_truth.npy"))}
    from scipy import stats as sps
    study = {}
    for name, (xi, yi) in sets.items():
        X, Y = np.load(xi), np.load(yi)
        e_u, e_f, e_m, spread = [], [], [], []
        for i in range(len(X)):
            pu = s_unet.run(["field"], {"input": X[i:i + 1]})[0][0]; pf = s_ufno.run(["field"], {"input": X[i:i + 1]})[0][0]
            pm = 0.5 * (pu + pf); t = Y[i]
            e_u.append(np.abs(pu - t).mean()); e_f.append(np.abs(pf - t).mean()); e_m.append(np.abs(pm - t).mean()); spread.append(np.abs(pu - pf).mean())
        rho = float(sps.spearmanr(spread, e_m)[0])
        study[name] = {"n": len(X), "mae_unet_K": float(np.mean(e_u)), "mae_ufno_K": float(np.mean(e_f)), "mae_ensemble_K": float(np.mean(e_m)),
                       "spread_mean_K": float(np.mean(spread)), "spearman_spread_vs_ensemble_error": rho}
        print(f"[{name}] MAE unet {np.mean(e_u):.2f}  ufno {np.mean(e_f):.2f}  ensemble {np.mean(e_m):.2f} K; spread {np.mean(spread):.2f} K, rho(spread, err) {rho:.2f}")
    # spread as an OOD signal: does level5 spread exceed level4 spread?
    meta = {"model_version": "level4-ensemble-unet-ufno-1.0.0", "schema_version": "1.0.0",
            "source": {"members": ["models/model.onnx (level4-unet-1.0.0)", "models_ufno/model.onnx (level4-ufno-1.0.0)"], "combination": "arithmetic mean of the two fields"},
            "export": {"tool": "tools/export_ensemble_onnx.py", "torch": torch.__version__, "onnxruntime": ort.__version__, "opset": 17, "date": dt.date.today().isoformat()},
            "io": {"input": {"name": "input", "shape": [1, 64, 64, 1, 7], "axes": ["B", "X", "Y", "Z", "P"], "dtype": "float32"},
                   "output": {"name": "field", "shape": [1, 64, 64, 1], "axes": ["B", "X", "Y", "Z"], "dtype": "float32", "unit": "K"}},
            "normalization": {"baked_into_graph": True}, "files": {"model.onnx": {"size_bytes": os.path.getsize(path), "sha256": sha256(path)}}, "spread_study": study}
    json.dump(meta, open(os.path.join(out, "metadata.json"), "w"), indent=2)
    json.dump(study, open(os.path.join(REPO, "docs", "results", "ensemble_spread.json"), "w"), indent=2)
    print("[meta] -> models_ensemble/metadata.json; docs/results/ensemble_spread.json")


if __name__ == "__main__":
    main()
