"""Stride-normalised curves for the statistical analysis.

For every trial of the Fukuchi test set the right-leg shear force and the three sagittal joint
moments are produced for the reference and for each arm, averaged over strides and resampled to
101 points of the gait cycle. The result feeds the SPM analysis in statistics.py.

Output: results/curves_fukuchi.npz
"""
import os

import numpy as np

import experiments as ex
import soleid_planar as sp_
from protocol import CAL_SUBJECTS, available_trials
import json

CAL = json.load(open(os.path.join(sp_.OUT, "calibration.json")))["chosen"]
NPTS = 101


def trial_curves(subj, trial):
    d = ex.prepare(subj, trial)
    sp_.configure(subj, trial)
    n = d["n"]
    grf = d["grf"]
    S = ex.newton_sum(d)
    prior = ex.vpp_prior(d, CAL["h_vpp"])
    sR, sL, _ = ex.solve(d, S, fc_smooth=CAL["fc_smooth"], prior=prior, w_prior=CAL["w_prior"], w_torque=0.0)
    vR, vL, _ = ex.solve(d, S, fc_smooth=CAL["fc_smooth"], prior=None, w_prior=0.0, w_torque=0.0)
    bR, bL = sp_.newton_baseline(S, grf["R"]["Fy"], grf["L"]["Fy"])
    kin, _ = sp_.kinematics_only_baseline(d["P"], d["segs"], n)

    BW = sp_.MASS * 9.81
    hs = sp_.heel_strikes(grf["R"]["Fy"])
    hs = hs[(hs > sp_.FS) & (hs < n - sp_.FS)]
    if len(hs) < 4:
        raise ValueError("too few strides")

    ref = sp_.inverse_dynamics(d["segs"], "R", grf["R"]["Fx"], grf["R"]["Fy"], grf["R"]["copx"])
    arms = {"SoleID": (sR, grf["R"]["Fy"], grf["R"]["copx"]),
            "SoleID_noVPP": (vR, grf["R"]["Fy"], grf["R"]["copx"]),
            "Newton_prop": (bR, grf["R"]["Fy"], grf["R"]["copx"]),
            "NoShear": (np.zeros(n), grf["R"]["Fy"], grf["R"]["copx"]),
            "Kin_only": (kin["R"]["Fx"], kin["R"]["Fy"], kin["R"]["copx"])}

    out = {"reference_shear": sp_.stride_normalize(100 * grf["R"]["Fx"] / BW, hs, NPTS).mean(0)}
    for j in ("ankle", "knee", "hip"):
        out[f"reference_{j}"] = sp_.stride_normalize(ref[j] / sp_.MASS, hs, NPTS).mean(0)
    for name, (est, Fy, cop) in arms.items():
        out[f"{name}_shear"] = sp_.stride_normalize(100 * est / BW, hs, NPTS).mean(0)
        M = sp_.inverse_dynamics(d["segs"], "R", est, Fy, cop)
        for j in ("ankle", "knee", "hip"):
            out[f"{name}_{j}"] = sp_.stride_normalize(M[j] / sp_.MASS, hs, NPTS).mean(0)
    out["stance_frac"] = np.mean(grf["R"]["Fy"] > sp_.F_CONTACT)
    return out


def main():
    trials = [t for t in available_trials() if t[0] not in CAL_SUBJECTS]
    store, meta = {}, []
    for i, (subj, trial, speed, age) in enumerate(trials):
        try:
            c = trial_curves(subj, trial)
        except Exception as e:
            print(f"  {subj} {trial}: skip ({e!r})", flush=True)
            continue
        for k, v in c.items():
            store.setdefault(k, []).append(v)
        meta.append((subj, trial, speed, age))
        if (i + 1) % 40 == 0:
            print(f"  {i + 1}/{len(trials)}", flush=True)
    arrays = {k: np.array(v) for k, v in store.items()}
    np.savez_compressed(os.path.join(sp_.OUT, "curves_fukuchi.npz"),
                        subjects=np.array([m[0] for m in meta]),
                        trials=np.array([m[1] for m in meta]),
                        speeds=np.array([m[2] for m in meta], float),
                        ages=np.array([m[3] for m in meta]), **arrays)
    print(f"saved {len(meta)} trials, {len(arrays)} curve sets")


if __name__ == "__main__":
    main()
