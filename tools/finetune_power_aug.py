#!/usr/bin/env python3
"""S8.5: fix the power insensitivity with physics-informed data augmentation.

The flaw (S2b.1, S8.1): the surrogate moves the hotspot by only 0.1-0.3 of
the true amount when chiplet power changes, because the training set has 14
discrete power values tied to layouts; the model never saw "same chip,
different power".

The fix needs no solver call. Steady-state conduction is linear in the power
(HotSpot confirmed it to 0.2 %, S8.1): if (x, T) is a training pair with
ambient T_amb, then scaling every chiplet power by f gives the exact pair
    x' = x with chiplet_power * f,   T' = T_amb + f * (T - T_amb).
So every stored sample yields a continuum of correct what-if samples.
Fine-tuning on randomly scaled copies (f ~ U(0.7, 1.3), one in four
unscaled) teaches the network the dependence it was missing, while the
unscaled copies keep the original accuracy.

Outputs: checkpoint (state_dict format) + metrics.json with the power gain on
200 test samples before/after (analytic expectation, same as S2b.1) and the
test MAE before/after on the 50-sample eval subset indices.

Usage
-----
    python tools/finetune_power_aug.py --bench ../external/ThermalBench --out ../external/thermbench_data/level4_UNet_poweraug \
        --n-train 2000 --epochs 8 --device mps
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
    p.add_argument("--out", required=True)
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--f-range", default="0.7,1.3")
    p.add_argument("--hotspot-weight", type=float, default=0.0,
                   help="extra loss weight on the hottest cells: w = 1 + W * [cell in the true top-k]; 0 = plain MSE")
    p.add_argument("--hotspot-topk", type=int, default=64, help="how many of the hottest true cells get the extra weight")
    a = p.parse_args()
    f_lo, f_hi = [float(v) for v in a.f_range.split(",")]
    bench, out = os.path.abspath(a.bench), os.path.abspath(a.out)
    os.makedirs(out, exist_ok=True)
    os.chdir(bench); sys.path.insert(0, bench); os.environ["PER_CHANNEL_NORM"] = "1"
    from utils.compat import install
    install()
    from run import parse_args
    from data_provider.data_factory import data_provider
    from exp.exp_operator import _predict
    from utils.tools import load_checkpoint, save_checkpoint, set_seed

    sys.argv = ["run.py", "--model", "UNet", "--data", "level4", "--task", "test", "--quiet"]
    set_seed(0)
    xtr, ytr, _, _, xte, yte = data_provider(parse_args())
    x_norm, model, y_norm = load_checkpoint("checkpoints/level4_UNet/model.pt")
    dev = a.device
    try:
        model = model.to(dev)
        torch.zeros(1).to(dev)
    except Exception as e:                       # mps not usable -> cpu
        print(f"[device] {dev} unavailable ({e}); using cpu"); dev = "cpu"; model = model.to(dev)

    # ---- measurements -------------------------------------------------------
    sub = list(range(0, 3000, 60))[:50]            # same indices as tests/fixtures/eval50
    probe = xte[:200].contiguous()                # power-gain probe, as in S2b.1

    def eval_all():
        model.eval()
        pred = _predict(model, x_norm, y_norm, xte[sub].contiguous(), "UNet", dev).numpy().astype(np.float64)
        truth = yte[sub].numpy().astype(np.float64)
        mae = float(np.abs(pred - truth).mean())
        base = _predict(model, x_norm, y_norm, probe, "UNet", dev).numpy().astype(np.float64).reshape(200, -1)
        amb = probe[:, 0, 0, 0, 4].numpy().astype(np.float64)
        gains = {}
        for f in (0.9, 1.1):
            xp = probe.clone(); xp[..., 0] *= f
            pr = _predict(model, x_norm, y_norm, xp, "UNet", dev).numpy().astype(np.float64).reshape(200, -1)
            d = pr.max(axis=1) - base.max(axis=1)
            expected = (f - 1.0) * (base.max(axis=1) - amb)
            gains[f"x{f}"] = float(d.mean() / expected.mean())
        return {"eval50_mae_K": mae, "power_gain": gains}

    before = eval_all()
    print(f"[before] eval50 MAE {before['eval50_mae_K']:.3f} K, power gain {before['power_gain']}")

    # ---- augmented fine-tuning ----------------------------------------------
    n = min(a.n_train, xtr.shape[0])
    idx = torch.arange(0, xtr.shape[0], xtr.shape[0] // n)[:n]
    X, Y = xtr[idx].contiguous(), ytr[idx].contiguous()        # (n,X,Y,Z,P), (n,X,Y,Z)
    amb_all = X[:, 0, 0, 0, 4].view(-1, 1, 1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-4, foreach=False)
    gen = torch.Generator().manual_seed(0)
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        tot, nb = 0.0, 0
        for s in range(0, n, 20):
            b = perm[s:s + 20]
            xb, yb, ab = X[b].clone(), Y[b].clone(), amb_all[b]
            f = f_lo + (f_hi - f_lo) * torch.rand(len(b), generator=gen)
            f[torch.rand(len(b), generator=gen) < 0.25] = 1.0            # keep some originals
            f4 = f.view(-1, 1, 1, 1)
            xb[..., 0] *= f4                                             # scale chiplet power
            yb = ab + f4 * (yb - ab)                                     # exact linear response
            xn = x_norm.forward(xb).squeeze(-2).permute(0, 3, 1, 2).to(dev)
            yn = y_norm.forward(yb).squeeze(-1).to(dev)
            opt.zero_grad()
            pred = model(xn).squeeze(1)
            if a.hotspot_weight > 0:
                # Hotspot-weighted MSE: the contract's tightest gate is the Tmax error,
                # so the hottest true cells of every sample count (1 + W) times more.
                flat = yn.reshape(yn.shape[0], -1)
                top = flat.topk(a.hotspot_topk, dim=1).indices
                w = torch.ones_like(flat)
                w.scatter_(1, top, 1.0 + a.hotspot_weight)
                loss = (w * (pred.reshape(flat.shape) - flat) ** 2).sum() / w.sum()
            else:
                loss = F.mse_loss(pred, yn)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"[ep {ep}/{a.epochs}] mse {tot / nb:.5f}  ({time.time() - t0:.0f} s)", flush=True)
    ft_time = time.time() - t0
    after = eval_all()
    print(f"[after ] eval50 MAE {after['eval50_mae_K']:.3f} K, power gain {after['power_gain']}  ({ft_time:.0f} s)")
    model.cpu()
    save_checkpoint(os.path.join(out, "model.pt"), model, x_norm, y_norm,
                    {"finetuned_from": "level4_UNet", "augmentation": f"power scale U({f_lo},{f_hi}), 25% unscaled", "n_train": n, "epochs": a.epochs, "lr": a.lr, "hotspot_weight": a.hotspot_weight, "hotspot_topk": a.hotspot_topk})
    json.dump({"n_train": n, "epochs": a.epochs, "lr": a.lr, "f_range": [f_lo, f_hi], "hotspot_weight": a.hotspot_weight, "hotspot_topk": a.hotspot_topk, "device": dev, "finetune_time_s": ft_time,
               "before": before, "after": after}, open(os.path.join(out, "metrics.json"), "w"), indent=2)
    print("->", out)


if __name__ == "__main__":
    main()
