#!/usr/bin/env bash
# =============================================================================
# tools/run_baseline.sh -- S2.2: reproduce the published level4 U-Net metrics
#
# What this script does
#   1. Runs the *unmodified* IC-ThermBench evaluation (`run.py --task test`) on
#      the level4 test split with the authors' checkpoint. This is the "did we
#      load the data and the weights the same way the authors did?" check. Every
#      later accuracy comparison (ONNX export, C++ inference) is anchored to the
#      numbers produced here, so this must match the authors' test_metrics.json.
#   2. Runs the same checkpoint on level5. level5 is a pure extrapolation set
#      (five unseen cases, never trained on), so these are zero-shot numbers.
#      They are recorded for the OOD story only and are NOT used as a gate:
#      the contract's accuracy gate is about reproducing in-support behaviour.
#   3. Writes a small environment/timing record next to the metrics so that the
#      notes in docs/results/baseline_notes.md can cite exact versions.
#
# Why it cds into the benchmark directory
#   run.py resolves `--root_path datasets` and the checkpoint path relative to
#   the current working directory and also inserts its own directory on
#   sys.path. Running it from anywhere else breaks both. Absolute paths are
#   passed only for the *outputs*, which live in this repository.
#
# Usage
#   tools/run_baseline.sh [--device cpu|mps|cuda] [--python <interpreter>]
#
#   THERMBENCH_DIR   override the benchmark checkout (default: ../external/ThermalBench
#                    relative to this repository; a sibling checkout layout)
#
# Exit code is non-zero if either evaluation fails, so CI can use it directly.
# =============================================================================
set -euo pipefail

# ---- locate directories ----------------------------------------------------
# REPO is this repository root (the script lives in <repo>/tools/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCH="${THERMBENCH_DIR:-$REPO/../external/ThermalBench}"
OUT_DIR="$REPO/docs/results"

# ---- arguments -------------------------------------------------------------
DEVICE="cpu"          # Apple Silicon can use "mps"; CPU is the portable default.
PYTHON="${PYTHON:-python}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --device) DEVICE="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done

# ---- sanity checks before spending minutes on inference --------------------
[[ -f "$BENCH/run.py" ]] || { echo "benchmark not found at $BENCH (set THERMBENCH_DIR)" >&2; exit 1; }
CKPT="checkpoints/level4_UNet/model.pt"   # relative to $BENCH, as run.py expects
[[ -f "$BENCH/$CKPT" ]] || { echo "checkpoint missing: $BENCH/$CKPT (run tools/fetch_assets.py)" >&2; exit 1; }
for lvl in level4_steady level5_steady; do
  [[ -f "$BENCH/datasets/$lvl/input.mat" ]] || { echo "dataset missing: $lvl (run tools/fetch_assets.py)" >&2; exit 1; }
done
mkdir -p "$OUT_DIR"

# ---- fixed evaluation settings ---------------------------------------------
# These must stay identical in S2.3 (stats.json) and in the C++ parity tests:
#   --seed 0              the benchmark's default; the test split is a fixed
#                         slice (last 20 %) so the seed only affects RNG state.
#   --per_channel_norm 1  the default and the only valid choice for P=7 data
#                         (run.py's docstring explains why: the three boundary
#                         condition channels differ from power by orders of
#                         magnitude, so a scalar normalizer would crush them).
SEED=0
PCN=1

# run_eval <level> <output-json>
#   Executes one evaluation and prints wall-clock seconds on stdout's last line.
run_eval() {
  local level="$1" out="$2"
  echo "=== [$level] device=$DEVICE seed=$SEED per_channel_norm=$PCN ==="
  local t0 t1
  t0=$(date +%s)
  ( cd "$BENCH" && "$PYTHON" run.py \
        --model UNet --data "$level" --task test \
        --load "$CKPT" --device "$DEVICE" \
        --seed "$SEED" --per_channel_norm "$PCN" \
        --output "$out" )
  t1=$(date +%s)
  echo "=== [$level] wall time: $((t1 - t0)) s ==="
  echo $((t1 - t0))
}

# ---- 1. in-support reproduction (the accuracy anchor) ----------------------
T4=$(run_eval level4 "$OUT_DIR/baseline_unet_level4.json" | tee /dev/stderr | tail -n1)

# ---- 2. OOD zero-shot (recorded only, no gate) -----------------------------
T5=$(run_eval level5 "$OUT_DIR/baseline_unet_level5_zeroshot.json" | tee /dev/stderr | tail -n1)

# ---- 3. environment + timing record ----------------------------------------
# No absolute paths go in here: the file is committed to a public repository.
"$PYTHON" - "$OUT_DIR/baseline_env.json" "$DEVICE" "$SEED" "$PCN" "$T4" "$T5" <<'PY'
import json, platform, subprocess, sys
import torch, numpy
out, device, seed, pcn, t4, t5 = sys.argv[1:]
# `sysctl` gives the marketing CPU name on macOS; fall back to platform on others.
try:
    cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                         capture_output=True, text=True, check=True).stdout.strip()
except Exception:
    cpu = platform.processor() or platform.machine()
json.dump({
    "python": platform.python_version(),
    "torch": torch.__version__,
    "numpy": numpy.__version__,
    "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
    "cpu": cpu,
    "device": device,
    "seed": int(seed),
    "per_channel_norm": int(pcn),
    "checkpoint": "checkpoints/level4_UNet/model.pt",
    "eval_batch": 20,
    "wall_time_s": {"level4_test": int(t4), "level5_zeroshot": int(t5)},
}, open(out, "w"), indent=2)
print(f"[env] -> {out}")
PY

echo "done: metrics in $OUT_DIR"
