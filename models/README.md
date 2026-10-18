# models/

The ONNX files are **not committed** (20 MB each, `*.onnx` is gitignored).
Only `metadata.json` is, and it pins everything needed to regenerate them
byte-for-byte and to verify a regenerated file (SHA-256 of the checkpoint and
of each export).

## Regenerate

```bash
# 1. fetch the IC-ThermBench checkpoint (19 MB pulled out of the 46 GB archive)
python tools/fetch_assets.py            # see tools/fetch_assets.py --help

# 2. export (needs the benchmark checkout next to this repo, or THERMBENCH_DIR)
python tools/export_onnx.py --bench ../external/ThermalBench
```

Produces:

| file | shape in → out | use |
|---|---|---|
| `model.onnx` | `(1,64,64,1,7)` → `(1,64,64,1)` | production; what the C++ library loads |
| `model_dynamic.onnx` | `(B,X,Y,1,7)` → `(B,X,Y,1)`, X and Y multiples of 8 | S4.6 size experiment only |

## What is inside the graph

The graph speaks the interface contract directly: input in physical units,
output in Kelvin. Per-channel normalization, the layout change to the
network's `(B,P,X,Y)`, and the output de-normalization are constant ops inside
the graph (see `tools/export_onnx.py`, class `ContractWrapper`). The C++
predictor therefore does no arithmetic on the tensor at all. The constants are
repeated in `metadata.json → normalization` and must equal
`contract/stats.json → checkpoint_normalizer`; the exporter asserts this.

## Verify a regenerated file

```bash
python - <<'PY'
import hashlib, json
m = json.load(open("models/metadata.json"))
for name, info in m["files"].items():
    h = hashlib.sha256(open(f"models/{name}", "rb").read()).hexdigest()
    print(name, "OK" if h == info["sha256"] else f"MISMATCH (got {h[:12]}…)")
PY
```

A mismatch is expected if torch/onnx versions differ from `metadata.json →
export`; in that case run `tools/parity_check.py` (S3.2), which is the
functional test.

## Precision note

Outputs are float32 values around 300–500 K, where one float32 ulp is
3.05e-5 K. PyTorch and ONNX Runtime accumulate convolutions in different
orders, so a 1-ulp difference on some pixels is the floor, not a bug. The
contract's parity gate is therefore 1e-4 K (about 3 ulp), see contract §4.
