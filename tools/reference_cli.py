#!/usr/bin/env python3
"""External reference solver: HotSpot's grid model behind the ExternalReferenceBackend protocol.

    python tools/reference_cli.py <input.npy> <output.npy>

Reads one contract tensor (64,64,1,7), writes the steady-state temperature
field (64,64,1) in Kelvin. Steps:
  1. chiplets = connected components of equal non-zero power -> one floorplan
     block each (bounding box; chiplets are rectangles in this dataset);
     the background between chiplets is one more block with zero power, so the
     floorplan tiles the whole chip (HotSpot requires full coverage);
  2. chip size from the coordinate channels (mm -> m);
  3. HotSpot config from the channels: ambient, r_convec; the material map
     (local_thermal_k) enters as per-block resistivity in the floorplan;
     sink side 0.168 m (from the h-r relation in the data); other package
     parameters are HotSpot's defaults;
  4. run `hotspot -model_type grid -grid_rows 64 -grid_cols 64` on a
     one-row power trace and read the grid steady-state file of layer 0.

The dataset was produced by HotSpot too, but with the generator's own package
parameters (spreader/sink sizes, interface, per-block materials) that are not
part of the public data. This wrapper therefore reproduces the *tool*, not the
*configuration*; its accuracy against the dataset truth is measured in S5.3 and
reported as is. Set HOTSPOT_BIN to the binary (default: ../external/hotspot/src/hotspot).
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
from scipy import ndimage

HOTSPOT = os.environ.get("HOTSPOT_BIN", os.path.join(os.path.dirname(__file__), "..", "..", "external", "hotspot", "src", "hotspot"))


def main(inp, outp):
    x = np.load(inp)
    assert x.shape == (64, 64, 1, 7), x.shape
    pw, gx, gy, k = x[..., 0, 0], x[..., 0, 1], x[..., 0, 2], x[..., 0, 3]
    amb, r = float(x[0, 0, 0, 4]), float(x[0, 0, 0, 6])
    # Cell size (m): the coordinate channels increase along one axis each.
    dx = max(np.diff(gx, axis=0).max(), np.diff(gx, axis=1).max()) * 1e-3
    dy = max(np.diff(gy, axis=0).max(), np.diff(gy, axis=1).max()) * 1e-3
    W, H = 64 * dx, 64 * dy                       # chip width (x) and height (y)

    blocks, powers = [], []
    covered = np.zeros_like(pw, dtype=bool)
    for v in np.unique(pw[pw > 0]):
        lab, n = ndimage.label(pw == v)
        for c in range(1, n + 1):
            m = lab == c
            xs, ys = np.where(m)
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            blocks.append((f"c{len(blocks)}", (x1 - x0) * dx, (y1 - y0) * dy, x0 * dx, y0 * dy, float(k[m].mean())))
            powers.append(float(v))
            covered |= m
    # Background: HotSpot needs the floorplan to tile the chip without overlaps.
    # One zero-power block per horizontal run of uncovered cells (simple, never overlaps).
    k_chip = float(k[pw > 0].mean()) if (pw > 0).any() else 100.0
    for i in range(64):
        j = 0
        while j < 64:
            if covered[i, j]:
                j += 1
                continue
            j1 = j
            while j1 < 64 and not covered[i, j1]:
                j1 += 1
            blocks.append((f"bg{len(blocks)}", dx, (j1 - j) * dy, i * dx, j * dy, float(k[i, j:j1].mean())))
            powers.append(0.0)
            j = j1
    # Package: HotSpot's defaults assume a 30 mm spreader and 60 mm sink; the
    # chips here are up to 73 mm, so scale both with the chip (spreader 1.5x, sink 3x).
    side = max(W, H)
    # Sink side from the data itself: in every sample h = 1 / (r * 0.0282 m^2), i.e.
    # the generator's sink area was 0.0282 m^2 (0.168 m side). Spreader and
    # thicknesses are HotSpot defaults scaled to the chip unless overridden by
    # TS_HOTSPOT_SPREADER_FACTOR / TS_HOTSPOT_T_SPREADER (used by the S5.3 sweep).
    s_sink = float(os.environ.get("TS_HOTSPOT_S_SINK", 0.168))
    s_spreader = min(s_sink, max(side * float(os.environ.get("TS_HOTSPOT_SPREADER_FACTOR", 1.5)), side))
    t_spreader = os.environ.get("TS_HOTSPOT_T_SPREADER", "0.001")

    with tempfile.TemporaryDirectory() as d:
        flp = os.path.join(d, "chip.flp")
        with open(flp, "w") as f:
            # Optional 6th/7th columns: specific heat (J/m^3/K) and resistivity (m K / W).
            # Per-block resistivity = 1 / local_thermal_k carries the dataset's material map
            # (chiplets ~100 W/mK, background 0.2-5 W/mK) into HotSpot's die layer.
            for name, w, h, left, bottom, kk in blocks:
                f.write(f"{name}\t{w:.6e}\t{h:.6e}\t{left:.6e}\t{bottom:.6e}\t1.75e6\t{1.0 / max(kk, 1e-3):.6e}\n")
        ptrace = os.path.join(d, "chip.ptrace")
        with open(ptrace, "w") as f:
            f.write("\t".join(b[0] for b in blocks) + "\n")
            f.write("\t".join(f"{p:.6g}" for p in powers) + "\n")
            f.write("\t".join(f"{p:.6g}" for p in powers) + "\n")
        grid_file = os.path.join(d, "grid.steady")
        cmd = [HOTSPOT, "-f", flp, "-p", ptrace, "-model_type", "grid", "-grid_rows", "64", "-grid_cols", "64",
               "-ambient", f"{amb:.6f}", "-r_convec", f"{r:.6g}",
               "-s_spreader", f"{s_spreader:.6g}", "-s_sink", f"{s_sink:.6g}", "-t_spreader", t_spreader,
               "-steady_file", os.path.join(d, "block.steady"), "-grid_steady_file", grid_file, "-o", os.path.join(d, "t.ttrace")]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not os.path.exists(grid_file):
            sys.stderr.write(res.stdout[-2000:] + res.stderr[-2000:])
            return 2
        # Grid steady file: per layer, "LAYER n" header? Format: lines "<index>\t<temperature>" for each cell
        # of each layer. We take the first 64*64 entries (layer 0 = the die).
        vals = []
        with open(grid_file) as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2 and parts[0].isdigit():
                    vals.append(float(parts[1]))
                if len(vals) >= 64 * 64:
                    break
        if len(vals) < 64 * 64:
            sys.stderr.write(f"grid steady file has {len(vals)} values\n")
            return 3
        T = np.array(vals[:64 * 64], dtype=np.float32).reshape(64, 64)
        # HotSpot's grid rows run from the top of the floorplan downward and our
        # floorplan's 'left' is the contract X axis: flipping the rows aligns the
        # grid with the (X, Y) field (checked against the dataset truth: 0.95
        # correlation, vs 0.81 unflipped, S5.3).
        T = T[::-1, :]
        np.save(outp, np.ascontiguousarray(T[..., None], dtype=np.float32))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
