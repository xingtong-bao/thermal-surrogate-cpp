#!/usr/bin/env python3
"""S8.3 step 1: solver-to-model feedback, few-shot fine-tuning on unseen chips.

Mirrors the benchmark's operator few-shot protocol (exp/exp_fewshot.py):
  * level5 manifest grouped by case; first K samples per case = fine-tuning
    set (K shots x 5 cases), last 500 per case = held-out evaluation (2 500);
  * start from the level4 checkpoint, keep its normalizers, Adam lr 1e-4,
    weight decay 1e-4, batch 20, MSE on normalized targets, `epochs` epochs;
  * evaluate the held-out set before and after.
Differences from the benchmark: the fine-tuned weights are SAVED (the
benchmark only writes a curve), in the same checkpoint format, so
tools/export_onnx.py can export them as model v1.1; and the fine-tuning
inputs are exported so the I5 configuration bank can be extended.

In the product this is the "data flywheel": every time the dispatcher runs
the reference solver, the (input, field) pair is a free training sample.

Usage
-----
    python tools/finetune_fewshot.py --bench ../external/ThermalBench --shots 10 --epochs 50 \
        --out ../external/thermbench_data/level5_UNet_k10
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", required=True)
    p.add_argument("--shots", type=int, default=10)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out", required=True, help="directory for model.pt, metrics.json, shots_inputs.npy")
    p.add_argument("--device", default="cpu")
    p.add_argument("--replay", type=int, default=0,
                   help="rehearsal: mix this many level4 TRAINING samples into the fine-tuning set so the original chips are not forgotten")
    a = p.parse_args()
    bench = os.path.abspath(a.bench)
    out = os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    os.chdir(bench)
    sys.path.insert(0, bench)
    os.environ["PER_CHANNEL_NORM"] = "1"
    from utils.compat import install
    install()
    from data_provider.data_loader import load_mat_pair
    from exp.exp_operator import _predict
    from utils.tools import load_checkpoint, set_seed
    from exp.exp_basic import build_model

    set_seed(0)
    x_all, y_all, _, _ = load_mat_pair("datasets/level5_steady")       # (5000,X,Y,Z,P), (5000,X,Y,Z)
    manifest = json.load(open("datasets/level5_steady/manifest.json"))
    groups = {}
    for i, rec in enumerate(manifest):
        groups.setdefault(rec["case"], []).append(i)
    ft_idx, ho_idx = [], []
    for case, idx in groups.items():
        ft_idx += idx[:a.shots]
        ho_idx += idx[-500:]
    print(f"[split] cases={list(groups)} finetune={len(ft_idx)} holdout={len(ho_idx)}")

    x_norm, model, y_norm = load_checkpoint("checkpoints/level4_UNet/model.pt")
    model = model.to(a.device)

    def evaluate():
        pred = _predict(model, x_norm, y_norm, x_all[ho_idx].contiguous(), "UNet", a.device).numpy().astype(np.float64)
        truth = y_all[ho_idx].numpy().astype(np.float64)
        mae = np.abs(pred - truth).reshape(len(ho_idx), -1).mean(axis=1)
        tmax = np.abs(pred.reshape(len(ho_idx), -1).max(axis=1) - truth.reshape(len(ho_idx), -1).max(axis=1))
        per_case = {}
        for c, idx in groups.items():
            m = np.isin(ho_idx, idx[-500:])
            per_case[c] = float(mae[m].mean())
        return {"mae_K": float(mae.mean()), "mae_p95_K": float(np.percentile(mae, 95)), "tmax_abs_err_mean_K": float(tmax.mean()), "per_case_mae_K": per_case}

    before = evaluate()
    print(f"[before] holdout MAE {before['mae_K']:.3f} K, Tmax err {before['tmax_abs_err_mean_K']:.3f} K, per case {dict((k, round(v, 2)) for k, v in before['per_case_mae_K'].items())}")

    x_ft, y_ft = x_all[ft_idx], y_all[ft_idx]
    if a.replay > 0:
        # Rehearsal against catastrophic forgetting: evenly spaced samples from the
        # level4 TRAINING split (never its test split) join the fine-tuning set.
        x4, y4, _, _ = load_mat_pair("datasets/level4_steady")
        n_train4 = int(int(x4.shape[0] * 0.8) * 0.9)
        rep = list(range(0, n_train4, max(1, n_train4 // a.replay)))[:a.replay]
        x_ft = torch.cat([x_ft, x4[rep]]); y_ft = torch.cat([y_ft, y4[rep]])
        print(f"[replay] + {len(rep)} level4 training samples -> fine-tuning set {x_ft.shape[0]}")
    xn = x_norm.forward(x_ft).squeeze(-2).permute(0, 3, 1, 2).contiguous()      # (n,P,X,Y)
    yn = y_norm.forward(y_ft).squeeze(-1)                                      # (n,X,Y)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xn, yn), batch_size=20, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4, foreach=False)
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb.to(a.device)).squeeze(1)
            loss = F.mse_loss(pred, yb.to(a.device))
            loss.backward()
            opt.step()
            tot += loss.item(); nb += 1
        if ep % 10 == 0 or ep == a.epochs:
            print(f"[ep {ep}] train mse (normalized) {tot / max(nb, 1):.5f}", flush=True)
    ft_time = time.time() - t0
    model.eval()
    after = evaluate()
    print(f"[after ] holdout MAE {after['mae_K']:.3f} K, Tmax err {after['tmax_abs_err_mean_K']:.3f} K, per case {dict((k, round(v, 2)) for k, v in after['per_case_mae_K'].items())}  ({ft_time:.0f} s)")

    # The benchmark's state_dict format (utils.tools.save_checkpoint): tensors only,
    # so it loads regardless of the compat shim's class identities.
    from utils.tools import save_checkpoint
    save_checkpoint(os.path.join(out, "model.pt"), model.cpu(), x_norm, y_norm,
                    {"finetuned_from": "level4_UNet", "shots": a.shots, "epochs": a.epochs, "lr": a.lr})
    np.save(os.path.join(out, "shots_inputs.npy"), np.ascontiguousarray(x_all[ft_idx].numpy(), dtype=np.float32))
    json.dump({"shots": a.shots, "epochs": a.epochs, "lr": a.lr, "replay": a.replay, "n_finetune": int(xn.shape[0]), "n_holdout": len(ho_idx),
               "finetune_time_s": ft_time, "before": before, "after": after, "holdout_indices": ho_idx},
              open(os.path.join(out, "metrics.json"), "w"), indent=2)
    print("->", out)


if __name__ == "__main__":
    main()
