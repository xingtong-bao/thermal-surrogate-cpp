#!/usr/bin/env python3
"""S8.5 step 2: release gates for the power-augmented candidate (model v1.2).

Same gates as every other version, plus the one this candidate exists for:
  1. export        models_v1.2_poweraug/model.onnx + metadata.json
  2. parity        ONNX Runtime vs PyTorch on the 50-sample eval subset
  3. artefacts     bank (unchanged: same chips), stats (unchanged ranges), FD params; conformal
                   recalibrated on the validation split with the new model
  4. C++ gates     ts_demo --compare on eval50 (seen chips): the three contract gates
  5. unseen check  ts_demo --compare on the 50 level5 samples: must not get worse
  6. power gain    tools/power_gain_verification.py with the new model (HotSpot truth)
Writes docs/results/power_aug_lifecycle.json.

Usage
-----
    python tools/lifecycle_poweraug.py --bench ../external/ThermalBench --finetuned ../external/thermbench_data/level4_UNet_poweraug
"""
import argparse
import csv
import json
import os
import shutil
import statistics as st
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
PY = sys.executable


def compare(demo, label, inputs, truth, mdir, cdir, out_csv):
    env = dict(os.environ, TS_MODELS_DIR_OVERRIDE=mdir, TS_CONTRACT_DIR_OVERRIDE=cdir)
    subprocess.run([demo, "--compare", label, inputs, truth, "50", out_csv], env=env, check=True, capture_output=True)
    rows = list(csv.DictReader(open(out_csv)))
    sur = [r for r in rows if r["backend"] == "surrogate"]
    mae = [float(r["mae_K"]) for r in sur]; tm = [float(r["tmax_abs_err_K"]) for r in sur]
    disp = [r for r in rows if r["backend"] == "dispatcher"]
    return {"surrogate_mae_K": st.mean(mae), "surrogate_mae_p95_K": sorted(mae)[int(0.95 * len(mae)) - 1],
            "tmax_abs_err_median_K": sorted(tm)[len(tm) // 2], "fallback_rate": 1 - sum(r["source"] == "surrogate" for r in disp) / len(disp)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bench", required=True)
    p.add_argument("--finetuned", required=True)
    p.add_argument("--demo", default=os.path.join(REPO, "build", "ts_demo"))
    p.add_argument("--tag", default="poweraug")
    p.add_argument("--version", default="level4-unet-poweraug-1.2.0")
    a = p.parse_args()
    bench, ft = os.path.abspath(a.bench), os.path.abspath(a.finetuned)
    out = os.path.join(REPO, f"models_v1.2_{a.tag}")
    os.makedirs(out, exist_ok=True)
    version = a.version
    ftm = json.load(open(os.path.join(ft, "metrics.json")))

    subprocess.run([PY, os.path.join(HERE, "export_onnx.py"), "--bench", bench, "--checkpoint", os.path.join(ft, "model.pt"),
                    "--version", version, "--out-dir", out, "--contract-only"], check=True)
    m = json.load(open(os.path.join(out, "metadata.json")))
    m["source"]["checkpoint"] = f"external/thermbench_data/{os.path.basename(ft)}/model.pt (tools/finetune_power_aug.py)"
    m["source"]["augmentation"] = ftm
    json.dump(m, open(os.path.join(out, "metadata.json"), "w"), indent=2)

    # parity on eval50 (ORT vs PyTorch batch 1)
    os.chdir(bench); sys.path.insert(0, bench); os.environ["PER_CHANNEL_NORM"] = "1"
    from utils.compat import install
    install()
    from utils.tools import load_checkpoint
    from exp.exp_basic import build_model
    import torch, onnxruntime as ort
    x_norm, model, y_norm = load_checkpoint(os.path.join(ft, "model.pt"), build_model("UNet", 7, 1, 64)); model.eval()
    sess = ort.InferenceSession(os.path.join(out, "model.onnx"), providers=["CPUExecutionProvider"])
    X = np.load(os.path.join(REPO, "tests", "fixtures", "eval50", "level4_inputs.npy"))
    worst = 0.0
    with torch.no_grad():
        for x in X:
            ref = (y_norm.inverse(model(x_norm.forward(torch.from_numpy(x[None])).squeeze(-2).permute(0, 3, 1, 2))).permute(0, 2, 3, 1)).numpy()
            worst = max(worst, float(np.abs(sess.run(["field"], {"input": x[None]})[0] - ref).max()))
    print(f"[parity] max |diff| {worst:.2e} K"); assert worst < 5e-3

    for f in ("config_bank.npy", "fd_solver_params.json"):
        shutil.copy(os.path.join(REPO, "models", f), os.path.join(out, f))
    shutil.copy(os.path.join(REPO, "contract", "stats.json"), os.path.join(out, "stats.json"))
    subprocess.run([PY, os.path.join(HERE, "conformal.py"), "--bench", bench, "--checkpoint", os.path.join(ft, "model.pt"), "--version", version,
                    "--out", os.path.join(out, "conformal.json")], check=True, capture_output=True)
    conf = json.load(open(os.path.join(out, "conformal.json")))

    gates = json.load(open(os.path.join(REPO, "docs", "results", "baseline_unet_level4_per_sample.json")))["gates"]
    res = os.path.join(REPO, "docs", "results")
    seen = compare(a.demo, f"level4_eval50_{a.tag}", os.path.join(REPO, "tests", "fixtures", "eval50", "level4_inputs.npy"), os.path.join(REPO, "tests", "fixtures", "eval50", "level4_truth.npy"),
                   out, out, os.path.join(res, f"power_aug_{a.tag}_level4_eval50.csv"))
    seen["passes"] = {"field_mae": seen["surrogate_mae_K"] < gates["field_mae_K"], "worst5pct": seen["surrogate_mae_p95_K"] < gates["worst5pct_per_sample_mae_K"],
                      "tmax_median": seen["tmax_abs_err_median_K"] < gates["tmax_abs_err_median_K"]}
    seen["passes_all"] = all(seen["passes"].values())
    ev = os.path.join(REPO, "..", "external", "eval_sets")
    unseen = compare(a.demo, f"level5_{a.tag}", os.path.join(ev, "level5_inputs.npy"), os.path.join(ev, "level5_truth.npy"), out, out, os.path.join(res, f"power_aug_{a.tag}_level5.csv"))
    print(f"[seen] MAE {seen['surrogate_mae_K']:.3f} p95 {seen['surrogate_mae_p95_K']:.3f} Tmax med {seen['tmax_abs_err_median_K']:.3f} -> {seen['passes']}")
    print(f"[unseen] surrogate MAE {unseen['surrogate_mae_K']:.2f} K, fallback {unseen['fallback_rate']:.0%}")

    subprocess.run([PY, os.path.join(HERE, "power_gain_verification.py"), "--n", "10", "--model", os.path.join(out, "model.onnx"),
                    "--out", os.path.join(res, f"power_gain_verification_{a.tag}.json")], check=True, capture_output=True)
    gain = json.load(open(os.path.join(res, f"power_gain_verification_{a.tag}.json")))["summary"]
    print(f"[hotspot gain] level4 median {gain['level4']['gain_median']:.2f}, level5 median {gain['level5']['gain_median']:.2f}")

    json.dump({"version": version, "finetune": ftm, "parity_max_abs_diff_K": worst, "conformal_overall": conf["overall"], "gates": gates,
               "seen_eval50_cpp": seen, "unseen_level5_cpp": unseen, "hotspot_gain_v1.2": gain,
               "hotspot_gain_v1.0": json.load(open(os.path.join(res, "power_gain_verification.json")))["summary"]},
              open(os.path.join(res, f"power_aug_lifecycle_{a.tag}.json"), "w"), indent=2)
    print("-> docs/results/power_aug_lifecycle.json")


if __name__ == "__main__":
    main()
