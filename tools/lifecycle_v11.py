#!/usr/bin/env python3
"""S8.3 step 2: turn a fine-tuned checkpoint into a versioned, validated model release.

What a model update must go through in this library (the same gates as v1.0):
  1. export       tools/export_onnx.py  -> <out>/model.onnx + metadata.json (new model_version)
  2. parity       ONNX Runtime vs PyTorch on the fine-tuning samples (gate 5e-3 K)
  3. I5 bank      models/config_bank.npy + the configurations the solver answered
                  (the shots) -> <out>/config_bank.npy: those chips become in-support
  4. error bounds conformal recalibration on data the new model did not train on
                  (level5 pool minus the shots) -> <out>/conformal.json
  5. evaluation   C++ dispatcher on a 50-sample level5 held-out subset with v1.0
                  and v1.1 artefacts: MAE, fallback rate, bounds coverage
Nothing is overwritten: v1.0 stays in models/, v1.1 lives in models_v1.1/
(gitignored .onnx, committed metadata/bank/conformal), one directory per shot count.

Usage
-----
    python tools/lifecycle_v11.py --bench ../external/ThermalBench --finetuned ../external/thermbench_data/level5_UNet_k10 --shots 10
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)


def conformal_quantile(scores, alpha):
    n = len(scores)
    k = min(max(int(np.ceil((n + 1) * (1 - alpha))), 1), n)
    return float(np.sort(scores)[k - 1])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", required=True)
    p.add_argument("--finetuned", required=True, help="directory written by tools/finetune_fewshot.py")
    p.add_argument("--shots", type=int, required=True)
    p.add_argument("--out", default=None, help="default: <repo>/models_v1.1_k<shots>")
    p.add_argument("--tag", default=None, help="label for result files (default: k<shots>)")
    p.add_argument("--demo", default=os.path.join(REPO, "build", "ts_demo"))
    a = p.parse_args()
    bench, ft = os.path.abspath(a.bench), os.path.abspath(a.finetuned)
    tag = a.tag or f"k{a.shots}"
    out = os.path.abspath(a.out) if a.out else os.path.join(REPO, f"models_v1.1_{tag}")
    os.makedirs(out, exist_ok=True)
    version = f"level4-unet-ft-level5-{tag}-1.1.0"

    # 1. export
    subprocess.run([sys.executable, os.path.join(HERE, "export_onnx.py"), "--bench", bench, "--checkpoint", os.path.join(ft, "model.pt"),
                    "--version", version, "--out-dir", out, "--contract-only"], check=True)
    import onnxruntime as ort
    sess = ort.InferenceSession(os.path.join(out, "model.onnx"), providers=["CPUExecutionProvider"])

    # 2. parity on the shots (PyTorch reference = benchmark path, batch 1)
    os.chdir(bench); sys.path.insert(0, bench); os.environ["PER_CHANNEL_NORM"] = "1"
    from utils.compat import install
    install()
    from utils.tools import load_checkpoint
    import torch
    try:
        x_norm, model, y_norm = load_checkpoint(os.path.join(ft, "model.pt"))
    except ValueError:
        from exp.exp_basic import build_model
        x_norm, model, y_norm = load_checkpoint(os.path.join(ft, "model.pt"), build_model("UNet", 7, 1, 64))
    model.eval()
    shots = np.load(os.path.join(ft, "shots_inputs.npy"))
    worst = 0.0
    with torch.no_grad():
        for x in shots:
            xt = torch.from_numpy(x[None])
            ref = (y_norm.inverse(model(x_norm.forward(xt).squeeze(-2).permute(0, 3, 1, 2))).permute(0, 2, 3, 1)).numpy()
            got = sess.run(["field"], {"input": x[None]})[0]
            worst = max(worst, float(np.abs(got - ref).max()))
    print(f"[parity] ORT vs torch on {len(shots)} shots: max |diff| {worst:.2e} K")
    assert worst < 5e-3

    # 3. bank v1.1
    from export_config_bank import features_from_contract_tensor
    bank = np.load(os.path.join(REPO, "models", "config_bank.npy"))
    new = np.array([features_from_contract_tensor(x) for x in shots], dtype=np.float32)
    np.save(os.path.join(out, "config_bank.npy"), np.ascontiguousarray(np.concatenate([bank, new])))
    print(f"[bank] {bank.shape[0]} + {new.shape[0]} configurations")
    # stats.json: same ranges, same standardisation (so thresholds keep their meaning); copy with a note
    stats = json.load(open(os.path.join(REPO, "contract", "stats.json")))
    stats["layout_novelty"]["bank_note"] = f"models_v1.1/config_bank.npy = level4 training bank + {new.shape[0]} level5 shots ({version})"
    json.dump(stats, open(os.path.join(out, "stats.json"), "w"), indent=2)
    shutil.copy(os.path.join(REPO, "models", "fd_solver_params.json"), os.path.join(out, "fd_solver_params.json"))

    # 4. conformal on the level5 pool minus the shots (never trained on, never evaluated on)
    from data_provider.data_loader import load_mat_pair
    x_all, y_all, _, _ = load_mat_pair("datasets/level5_steady")
    manifest = json.load(open("datasets/level5_steady/manifest.json"))
    groups = {}
    for i, rec in enumerate(manifest): groups.setdefault(rec["case"], []).append(i)
    calib = [i for idx in groups.values() for i in idx[a.shots:500]]
    ho = [i for idx in groups.values() for i in idx[-500:]]
    xs, ys = x_all.numpy(), y_all.numpy()
    mae, terr, rise = [], [], []
    for i in calib:
        pr = sess.run(["field"], {"input": xs[i:i + 1]})[0][0, ..., 0]
        mae.append(np.abs(pr - ys[i][..., 0]).mean()); terr.append(abs(pr.max() - ys[i].max())); rise.append(pr.max() - xs[i][0, 0, 0, 4])
    mae, terr, rise = map(np.array, (mae, terr, rise))
    edges = np.quantile(rise, np.linspace(0, 1, 6)); edges[0], edges[-1] = -np.inf, np.inf
    bins = []
    for b in range(5):
        m = (rise >= edges[b]) & (rise < edges[b + 1])
        bins.append({"rise_lo_K": None if b == 0 else float(edges[b]), "rise_hi_K": None if b == 4 else float(edges[b + 1]), "n": int(m.sum()),
                     "mae_q90_K": conformal_quantile(mae[m], 0.1), "tmax_err_q90_K": conformal_quantile(terr[m], 0.1),
                     "mae_q95_K": conformal_quantile(mae[m], 0.05), "tmax_err_q95_K": conformal_quantile(terr[m], 0.05)})
    conf = {"method": "split conformal, group-conditional on predicted rise", "calibration_set": f"level5 pool minus the {a.shots} shots per case ({len(calib)} samples)",
            "model_version": version, "levels": [0.9, 0.95], "n_calibration": len(calib),
            "overall": {"mae_q90_K": conformal_quantile(mae, 0.1), "tmax_err_q90_K": conformal_quantile(terr, 0.1), "mae_q95_K": conformal_quantile(mae, 0.05), "tmax_err_q95_K": conformal_quantile(terr, 0.05)},
            "rise_bins": bins, "calibration_mae_mean_K": float(mae.mean())}
    json.dump(conf, open(os.path.join(out, "conformal.json"), "w"), indent=2)
    print(f"[conformal] n={len(calib)} MAE mean {mae.mean():.3f} K, q90 {conf['overall']['mae_q90_K']:.3f} K")

    # 5. held-out 50-sample subset as .npy (not committed) and C++ comparison with v1.0 and v1.1
    sel = ho[::50][:50]
    ev = os.path.join(REPO, "..", "external", "eval_sets")
    np.save(os.path.join(ev, "level5_holdout50_inputs.npy"), np.ascontiguousarray(xs[sel], dtype=np.float32))
    np.save(os.path.join(ev, "level5_holdout50_truth.npy"), np.ascontiguousarray(ys[sel], dtype=np.float32))
    results = {}
    for vtag, mdir, cdir in (("v1.0", os.path.join(REPO, "models"), os.path.join(REPO, "contract")), ("v1.1", out, out)):
        csv_path = os.path.join(REPO, "docs", "results", f"lifecycle_{tag}_level5_holdout_{vtag}.csv")
        env = dict(os.environ, TS_MODELS_DIR_OVERRIDE=mdir, TS_CONTRACT_DIR_OVERRIDE=cdir)
        subprocess.run([a.demo, "--compare", f"level5_holdout_{tag}_{vtag}", os.path.join(ev, "level5_holdout50_inputs.npy"), os.path.join(ev, "level5_holdout50_truth.npy"), "50", csv_path],
                       env=env, check=True, capture_output=True)
        import csv
        rows = list(csv.DictReader(open(csv_path)))
        disp = [r for r in rows if r["backend"] == "dispatcher"]; sur = [r for r in rows if r["backend"] == "surrogate"]
        results[vtag] = {"surrogate_mae_K": float(np.mean([float(r["mae_K"]) for r in sur])),
                        "dispatcher_mae_K": float(np.mean([float(r["mae_K"]) for r in disp if r["status"] == "Ok"])),
                        "fallback_rate": 1 - sum(r["source"] == "surrogate" for r in disp) / len(disp),
                        "dispatcher_sources": {s: sum(r["source"] == s for r in disp) for s in set(r["source"] for r in disp)}}
        print(f"[{vtag}] surrogate MAE {results[vtag]['surrogate_mae_K']:.2f} K; dispatcher MAE {results[vtag]['dispatcher_mae_K']:.2f} K, fallback {results[vtag]['fallback_rate']:.0%}")
    ftm = json.load(open(os.path.join(ft, "metrics.json")))
    doc = {"version": version, "tag": tag, "shots_per_case": a.shots, "replay": ftm.get("replay", 0), "finetune": {k: ftm[k] for k in ("n_finetune", "n_holdout", "epochs", "lr", "finetune_time_s", "before", "after")},
           "parity_max_abs_diff_K": worst, "bank": {"v1.0": int(bank.shape[0]), "v1.1": int(bank.shape[0] + new.shape[0])},
           "conformal_v1.1": {"n_calibration": len(calib), "overall": conf["overall"]}, "holdout50_cpp": results,
           "note": "dispatcher chain in --compare is surrogate -> FD reference (no HotSpot) so the v1.0 fallback answers come from the FD solver"}
    json.dump(doc, open(os.path.join(REPO, "docs", "results", f"lifecycle_fewshot_{tag}.json"), "w"), indent=2)
    print(f"-> docs/results/lifecycle_fewshot_{tag}.json")


if __name__ == "__main__":
    main()
