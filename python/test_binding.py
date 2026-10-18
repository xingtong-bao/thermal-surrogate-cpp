"""S7.2 check: the Python binding returns exactly what the C++ parity fixtures say.

    PYTHONPATH=build/python python python/test_binding.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.environ.get("TS_PY_BUILD", "build/python"))
import thermal_surrogate as ts  # noqa: E402

root = os.path.join(os.path.dirname(__file__), "..")
eng = ts.Engine(os.path.join(root, "models"), os.path.join(root, "contract"))
print("thermal_surrogate", ts.__version__, "ORT", ts.onnxruntime_version, "model", eng.model_version)

man = json.load(open(os.path.join(root, "tests", "fixtures", "parity", "manifest.json")))
worst = 0.0
for p in man["pairs"]:
    x = np.load(os.path.join(root, "tests", "fixtures", "parity", p["input"]["file"]))
    ref = np.load(os.path.join(root, "tests", "fixtures", "parity", p["field"]["file"]))
    out = eng.predict(x)
    assert out["status"] == "Ok" and out["source"] == "surrogate", out
    worst = max(worst, float(np.abs(out["field"] - ref).max()))
print(f"10 parity pairs: max |python binding - C++/ORT reference| = {worst:.3e} K")
assert worst < 1e-4

x5 = np.load(os.path.join(root, "tests", "fixtures", "sample_input_level5.npy"))
out = eng.predict(x5)
print("level5 fixture:", out["status"], out["source"], out["oob_channels"], f"tmax {out['tmax']:.2f} K")
assert out["oob_flag"] and out["source"] == "reference_inprocess"
bad = eng.predict(np.zeros((64, 64, 1, 6), np.float32))
print("6 channels:", bad["status"])
assert bad["status"] == "ChannelMismatch"
print("binding OK")
