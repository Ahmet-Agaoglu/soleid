"""Where does the hip moment bias live: which subjects, which limb, which phase, which speed?

The paper reports one number for the systematic hip offset. This script breaks it down by
subject, limb, phase and speed, with consistent definitions, to establish its consequences rather
than to remove it. Bias is estimate minus reference throughout (N m/kg); phases are fractions of that
limb's own stance (early, mid, late thirds) and its swing.

The by-data-set figure comes from reference_agreement.py, which computes SoleID minus the planar
reference on all three layers with the same sign convention.

Output: results/hip_bias_breakdown.csv, results/hip_bias_breakdown.json
"""
import json
import os
import time

import numpy as np
import pandas as pd

import experiments as ex
import soleid_planar as sp_
from protocol import CAL_SUBJECTS, available_trials

CAL = json.load(open(os.path.join(sp_.OUT, "calibration.json")))["chosen"]
PHASES = ("early", "mid", "late", "swing")


def phase_masks(Fy, trim):
    """Each loaded episode split in thirds by its own duration; the rest of the trimmed record is
    swing."""
    on = Fy > sp_.F_CONTACT
    lab = np.full(len(Fy), "swing", dtype=object)
    d = np.diff(on.astype(int))
    starts, ends = list(np.where(d == 1)[0] + 1), list(np.where(d == -1)[0] + 1)
    if on[0]:
        starts = [0] + starts
    if on[-1]:
        ends = ends + [len(on)]
    for a, b in zip(starts, ends):
        k = b - a
        lab[a:a + k // 3] = "early"
        lab[a + k // 3:a + 2 * k // 3] = "mid"
        lab[a + 2 * k // 3:b] = "late"
    return {p: (lab == p) & trim for p in PHASES}


def run_trial(subj, trial, speed, age):
    d = ex.prepare(subj, trial)
    sp_.configure(subj, trial)
    S = ex.newton_sum(d, 1.0)
    pr = ex.vpp_prior(d, CAL["h_vpp"])
    fR, fL, _ = ex.solve(d, S, fc_smooth=CAL["fc_smooth"], prior=pr, w_prior=CAL["w_prior"],
                         w_torque=0.0)
    row = dict(subject=subj, trial=trial, speed=speed, age=age)
    for side, f in (("R", fR), ("L", fL)):
        g = d["grf"][side]
        ref = sp_.inverse_dynamics(d["segs"], side, g["Fx"], g["Fy"], g["copx"])["hip"]
        est = sp_.inverse_dynamics(d["segs"], side, f, g["Fy"], g["copx"])["hip"]
        e = (est - ref) / sp_.MASS
        masks = phase_masks(g["Fy"], d["trim"])
        row[f"{side}_cycle"] = float(np.mean(e[d["trim"]]))
        for p, m in masks.items():
            row[f"{side}_{p}"] = float(np.mean(e[m])) if m.any() else np.nan
        stance = masks["early"] | masks["mid"] | masks["late"]
        row[f"{side}_stance"] = float(np.mean(e[stance]))
    return row


def main():
    trials = [t for t in available_trials() if t[0] not in CAL_SUBJECTS]
    rows, t0 = [], time.time()
    for i, (s, t, spd, age) in enumerate(trials):
        try:
            rows.append(run_trial(s, t, spd, age))
        except Exception as e:
            print(f"  skip {s} {t}: {e!r}", flush=True)
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(trials)} [{(time.time()-t0)/60:.1f} min]", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(sp_.OUT, "hip_bias_breakdown.csv"), index=False)

    out = {}
    out["limb"] = {s: dict(stance=float(df[f"{s}_stance"].mean()), cycle=float(df[f"{s}_cycle"].mean()))
                   for s in ("R", "L")}
    out["phase"] = {p: dict(R=float(df[f"R_{p}"].mean()), L=float(df[f"L_{p}"].mean())) for p in PHASES}
    bins = pd.cut(df.speed, [0, 0.8, 1.2, 1.6, 9], labels=["<0.8", "0.8-1.2", "1.2-1.6", ">1.6"])
    out["speed"] = {str(k): float(v) for k, v in df.groupby(bins, observed=True).R_stance.mean().items()}
    out["speed_slope"] = float(np.polyfit(df.speed, df.R_stance, 1)[0])
    subj = df.groupby("subject").R_stance.mean()
    out["subject"] = dict(n=int(len(subj)), positive=int((subj > 0).sum()), min=float(subj.min()),
                          max=float(subj.max()), sd=float(subj.std()))
    out["age"] = {k: float(v) for k, v in df.groupby("age").R_stance.mean().items()}
    ra = os.path.join(sp_.OUT, "reference_agreement.json")
    if os.path.exists(ra):
        agr = json.load(open(ra))
        out["dataset_cycle"] = {ds: agr[ds]["soleid_vs_planar"]["hip"]["bias"] for ds in agr}
    json.dump(out, open(os.path.join(sp_.OUT, "hip_bias_breakdown.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
