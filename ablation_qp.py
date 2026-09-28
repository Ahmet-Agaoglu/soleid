"""What does each part of the program contribute, and what does the program add over the prior
used directly?

The paper's no-prior ablation shows the prior helps the program. It does not show the program helps
the prior, and it does not separate the program's other two ingredients -- temporal smoothness and
the contact constraints -- from each other. This removes them one at a time, and adds the two simple
alternatives a skeptical reader would try first, all with the same measured vertical force,
center of pressure and contact timing:

  full           the estimator as published
  no smoothing   the second-difference penalty removed
  no cone        the friction bounds removed (swing is still forced to zero: that is contact
                 timing, which is measured)
  blend          both removed: a per-frame weighted least-squares blend of the whole-body balance
                 and the prior, with the published weights -- the simplest dynamics/prior blend
  prior          the direction prior itself, used directly as the estimate
  prior, smooth  the same, low-passed at the estimator's own cut-off

With equal weights on the balance and the prior, the blend in single support is the plain average
of the two, which is worth knowing when reading the half-slope result in the paper.

Output: results/ablation_qp.csv, results/ablation_qp.json
"""
import json
import os
import time

import numpy as np
import osqp
import pandas as pd
import scipy.sparse as sp
from scipy.signal import butter, filtfilt

import experiments as ex
import soleid_planar as sp_
from protocol import CAL_SUBJECTS, available_trials

CAL = json.load(open(os.path.join(sp_.OUT, "calibration.json")))["chosen"]
BIG = 1e30


def solve(d, S, prior, smooth=True, cone=True, fc=CAL["fc_smooth"], w_prior=CAL["w_prior"],
          w_dyn=1.0, w_mag=1e-4, mu=sp_.MU):
    """ex.solve with the smoothness term and the friction bounds individually switchable."""
    n = d["n"]
    dt = 1.0 / sp_.FS
    FyR, FyL = d["grf"]["R"]["Fy"], d["grf"]["L"]["Fy"]
    I = sp.identity(n, format="csc")
    A = sp.hstack([I, I], format="csc")
    P = w_dyn * A.T @ A + w_mag * sp.identity(2 * n)
    if smooth:
        w_s = w_dyn / (2 * np.pi * fc * dt) ** 4
        D2 = sp.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(n - 2, n), format="csc")
        D2b = sp.block_diag([D2, D2], format="csc")
        P = P + w_s * D2b.T @ D2b
    q = -w_dyn * (A.T @ S)
    onR, onL = FyR > sp_.F_CONTACT, FyL > sp_.F_CONTACT
    Wp = sp.diags(np.concatenate([onR, onL]).astype(float))
    P = P + w_prior * Wp
    q = q - w_prior * (Wp @ np.concatenate([prior["R"], prior["L"]]))
    if cone:
        lb = np.concatenate([np.where(onR, -mu * FyR, 0.0), np.where(onL, -mu * FyL, 0.0)])
        ub = -lb
    else:
        lb = np.concatenate([np.where(onR, -BIG, 0.0), np.where(onL, -BIG, 0.0)])
        ub = -lb
    prob = osqp.OSQP()
    prob.setup(P=sp.csc_matrix(2 * P), q=2 * q, A=sp.identity(2 * n, format="csc"), l=lb, u=ub,
               verbose=False, eps_abs=1e-6, eps_rel=1e-6, max_iter=50000, polish=True)
    r = prob.solve()
    return r.x[:n], r.x[n:]


def metrics(d, fR, g):
    tr = d["trim"]
    FyR, FyL = d["grf"]["R"]["Fy"], d["grf"]["L"]["Fy"]
    onR, onL = FyR > sp_.F_CONTACT, FyL > sp_.F_CONTACT
    BW = sp_.MASS * 9.81
    rm = lambda m: 100 * float(np.sqrt(np.mean((fR[m] - g["Fx"][m]) ** 2))) / BW
    out = dict(stance=rm(onR & tr), ds=rm(onR & onL & tr), ss=rm(onR & ~onL & tr))
    ref = sp_.inverse_dynamics(d["segs"], "R", g["Fx"], g["Fy"], g["copx"])
    est = sp_.inverse_dynamics(d["segs"], "R", fR, g["Fy"], g["copx"])
    for j in ("ankle", "knee", "hip"):
        out[j] = float(np.sqrt(np.mean(((est[j] - ref[j]) / sp_.MASS)[tr] ** 2)))
    return out


ARMS = ("full", "no_smooth", "no_cone", "blend", "prior", "prior_smooth")


def run_trial(subj, trial):
    d = ex.prepare(subj, trial)
    sp_.configure(subj, trial)
    g = d["grf"]["R"]
    S = ex.newton_sum(d, 1.0)
    pr = ex.vpp_prior(d, CAL["h_vpp"])
    onR = g["Fy"] > sp_.F_CONTACT
    b, a = butter(4, CAL["fc_smooth"] / (sp_.FS / 2), btype="low")
    est = {
        "full": solve(d, S, pr)[0],
        "no_smooth": solve(d, S, pr, smooth=False)[0],
        "no_cone": solve(d, S, pr, cone=False)[0],
        "blend": solve(d, S, pr, smooth=False, cone=False)[0],
        "prior": np.where(onR, pr["R"], 0.0),
        "prior_smooth": np.where(onR, filtfilt(b, a, pr["R"]), 0.0),
    }
    row = dict(subject=subj, trial=trial)
    for arm, fR in est.items():
        for k, v in metrics(d, fR, g).items():
            row[f"{arm}_{k}"] = v
    return row


def main():
    trials = [(s, t) for s, t, *_ in available_trials() if s not in CAL_SUBJECTS]
    rows, t0 = [], time.time()
    for i, (s, t) in enumerate(trials):
        try:
            rows.append(run_trial(s, t))
        except Exception as e:
            print(f"  skip {s} {t}: {e!r}", flush=True)
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(trials)} [{(time.time()-t0)/60:.1f} min]", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(sp_.OUT, "ablation_qp.csv"), index=False)
    keys = ("stance", "ds", "ss", "ankle", "knee", "hip")
    summary = {a: {k: float(df[f"{a}_{k}"].mean()) for k in keys} for a in ARMS}
    json.dump(summary, open(os.path.join(sp_.OUT, "ablation_qp.json"), "w"), indent=1)
    print(f"\n{len(df)} test trials\n")
    print(f"{'arm':>14s}" + "".join(f"{k:>9s}" for k in keys))
    for a in ARMS:
        print(f"{a:>14s}" + "".join(f"{summary[a][k]:9.3f}" for k in keys))
    return df


if __name__ == "__main__":
    main()
