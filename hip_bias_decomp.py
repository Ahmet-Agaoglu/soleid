"""How much of the hip moment error is systematic?

The manuscript says "more than half", and that figure came from comparing a bias measured over
stance against an RMSE measured over the whole gait cycle. The windows have to match: swing, where
the estimate and the reference agree because there is no force to get wrong, dilutes the RMSE but
not the offset, so mixing the two inflates the systematic share.

There is also more than one thing "systematic" can mean, and they give different answers:

  population  one offset shared by every trial      bias(pooled)^2 / MSE
  per trial   each trial has its own constant offset  mean(bias_i^2) / MSE

The second is the larger and is the fairer reading of "systematic and traceable", since the claim
is that the prior's geometry biases each estimate, not that every subject is biased identically.
Both are reported, in both windows, so the sentence in the paper can be written from whichever is
actually meant.

Output: results/hip_bias_decomp.csv, results/hip_bias_decomp.json
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


def metrics(subj, trial):
    d = ex.prepare(subj, trial)
    sp_.configure(subj, trial)
    S = ex.newton_sum(d, 1.0)
    pr = ex.vpp_prior(d, CAL["h_vpp"])
    fR, _, _ = ex.solve(d, S, fc_smooth=CAL["fc_smooth"], prior=pr,
                        w_prior=CAL["w_prior"], w_torque=0.0)
    g = d["grf"]["R"]
    FxR, FyR, cop = g["Fx"], g["Fy"], g["copx"]
    tr = d["trim"]
    st = (FyR > sp_.F_CONTACT) & tr
    ref = sp_.inverse_dynamics(d["segs"], "R", FxR, FyR, cop)["hip"] / sp_.MASS
    est = sp_.inverse_dynamics(d["segs"], "R", fR, FyR, cop)["hip"] / sp_.MASS
    e = est - ref
    return dict(
        subject=subj, trial=trial,
        hip_rmse_cycle=float(np.sqrt(np.mean(e[tr] ** 2))),
        hip_rmse_stance=float(np.sqrt(np.mean(e[st] ** 2))),
        hip_bias_cycle=float(np.mean(e[tr])),
        hip_bias_stance=float(np.mean(e[st])),
        n_cycle=int(tr.sum()), n_stance=int(st.sum()),
    )


def share(bias, rmse):
    """Systematic share of the mean squared error, two ways."""
    mse = float(np.mean(rmse ** 2))
    return dict(mse=mse,
                population=100 * float(np.mean(bias)) ** 2 / mse,
                per_trial=100 * float(np.mean(bias ** 2)) / mse,
                bias_mean=float(np.mean(bias)), bias_sd=float(np.std(bias)),
                rmse_mean=float(np.mean(rmse)),
                same_sign=int(np.sum(np.sign(bias) == np.sign(np.mean(bias)))))


def main():
    trials = [t for t in available_trials() if t[0] not in CAL_SUBJECTS]
    rows, t0 = [], time.time()
    for i, (subj, trial, *_) in enumerate(trials):
        try:
            rows.append(metrics(subj, trial))
        except Exception as e:
            print(f"  skip {subj} {trial}: {e!r}", flush=True)
        if (i + 1) % 40 == 0:
            print(f"  {i+1}/{len(trials)} [{(time.time()-t0)/60:.1f} min]", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(sp_.OUT, "hip_bias_decomp.csv"), index=False)

    out = {}
    print(f"\n{len(df)} test trials\n")
    print(f"{'window':>10s} {'RMSE':>8s} {'bias':>8s} {'sd':>7s} {'same sign':>10s} "
          f"{'population':>11s} {'per trial':>10s}")
    for w in ("cycle", "stance"):
        s = share(df[f"hip_bias_{w}"].values, df[f"hip_rmse_{w}"].values)
        out[w] = s
        print(f"{w:>10s} {s['rmse_mean']:8.4f} {s['bias_mean']:+8.4f} {s['bias_sd']:7.4f} "
              f"{s['same_sign']:>7d}/{len(df)} {s['population']:10.1f}% {s['per_trial']:9.1f}%")
    # the mismatch the manuscript made: stance bias against whole-cycle RMSE
    mixed = 100 * df.hip_bias_stance.mean() ** 2 / float(np.mean(df.hip_rmse_cycle ** 2))
    out["mixed_stance_bias_vs_cycle_rmse"] = mixed
    print(f"\nmixing the windows (stance bias vs whole-cycle MSE): {mixed:.1f} %"
          f"  <- how the manuscript got its figure")
    with open(os.path.join(sp_.OUT, "hip_bias_decomp.json"), "w") as f:
        json.dump(out, f, indent=1)
    return df


if __name__ == "__main__":
    main()
