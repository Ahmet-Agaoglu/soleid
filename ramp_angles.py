"""Measure the six Camargo ramp inclinations from the force plates.

On an inclined plate the centre of pressure travels up or down the plate surface during a step, so
the slope of its height against its fore-aft position is the plate inclination. For every loaded
step on every plate the slope is fitted over the frames carrying at least half of that step's peak
load; plates are pooled per inclination code and the median over the inclined steps is the angle. Axes: y vertical, z fore-aft (the Camargo convention used elsewhere).

Output: results/ramp_angles.json
"""
import glob
import json
import os

import numpy as np
import pandas as pd

import camargo_adapter as ca
import soleid_planar as sp_

MIN_N = 100.0            # N, a plate counts as loaded
MIN_STEP_S = 0.25


def step_slopes(fp, plate, fs):
    fy = fp[f"{plate}_vy"].values
    on = fy > MIN_N
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    out = []
    for a, b in zip(edges[::2], edges[1::2]):
        if (b - a) / fs < MIN_STEP_S:
            continue
        seg = slice(a, b)
        w = fy[seg] >= 0.5 * fy[seg].max()
        y, z = fp[f"{plate}_py"].values[seg][w], fp[f"{plate}_pz"].values[seg][w]
        if np.ptp(z) < 0.05 * (1000 if np.nanmax(np.abs(z)) > 50 else 1):   # CoP must travel >= 5 cm
            continue
        k = np.polyfit(z, y, 1)[0]
        out.append(float(np.degrees(np.arctan(abs(k)))))
    return out


def main():
    rows = []
    for subj in sorted(os.listdir(ca.CSV)):
        for f in sorted(glob.glob(os.path.join(ca.CSV, subj, "ramp", "fp", "ramp_*.csv"))):
            code = int(os.path.basename(f).split("_")[1])
            fp = pd.read_csv(f)
            fs = 1.0 / np.median(np.diff(fp.Header.values))
            for plate in sorted({c.rsplit("_vy", 1)[0] for c in fp.columns if c.endswith("_vy")}):
                for ang in step_slopes(fp, plate, fs):
                    rows.append(dict(subject=subj, code=code, plate=plate, angle=ang))
    df = pd.DataFrame(rows)
    inclined = df[df.angle > 2.0]
    per_plate = inclined.groupby(["code", "plate"]).angle.agg(["median", "count"])
    out = {int(c): float(g.angle.median()) for c, g in inclined.groupby("code")}
    json.dump(dict(angles_deg=out, per_plate=per_plate.reset_index().to_dict("records"),
                   n_steps=int(len(inclined))), open(os.path.join(sp_.OUT, "ramp_angles.json"), "w"),
              indent=1)
    print(per_plate.round(2).to_string())
    print({c: round(a, 1) for c, a in out.items()})


if __name__ == "__main__":
    main()
