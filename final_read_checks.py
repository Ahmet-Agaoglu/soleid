"""Single values of the paper that no other analysis produces, computed from saved results or, where
needed, from the trials themselves. Writes results/final_read_checks.json for verify_manuscript.py.
The slow part (--lever) prepares every test trial again.

  smallest_lever_m        II-D   smallest vertical lever of the prior over loaded test frames
  mass_rescaled_residual  V-D    dynamics residual with the body mass rescaled by the force ratio
  vertical_error_of_range IV-A   pivot-point split vertical-force error as a share of its stride range
  combined_increment_ci   IV-C   subject bootstrap of the combined wearable-grade increment
  wang_without_outlier    III-D  Wang shear and hip error without the one subject that fails QC
  spm_max_at_pct          IV-A   where in the cycle the mean shear difference is largest
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import experiments as ex
import soleid_planar as sp_

R = sp_.OUT
OUT = os.path.join(R, "final_read_checks.json")
t = pd.read_csv(os.path.join(R, "test_all_trials.csv"))
q = pd.read_csv(os.path.join(R, "fukuchi_quality.csv"))
res = json.load(open(OUT)) if os.path.exists(OUT) else {}


def save():
    json.dump(res, open(OUT, "w"), indent=1)


def strides(on):
    hs = np.where(np.diff(on.astype(int)) == 1)[0] + 1
    return list(zip(hs[:-1], hs[1:]))


# ---- quick ones, from saved results --------------------------------------------------------
# vertical force per stride, from the saved time series of the test trials
rng_pct, resid, resid_k = [], [], []
ratio = q.set_index(["subject", "trial"]).bw_ratio
for s, tr, m in zip(t.subject, t.trial, t.mass):
    p = os.path.join(R, f"{s}{tr}_timeseries.csv")
    if not os.path.exists(p):
        continue
    ts = pd.read_csv(p)
    BW = m * 9.81
    on = ts.FyR.values > sp_.F_CONTACT
    r = [100 * (ts.FyR.values[a:b].max() - ts.FyR.values[a:b].min()) / BW for a, b in strides(on)]
    rng_pct += r
    # dynamics residual as soleid_planar computes it (1 s trimmed at each end), and with the mass
    # rescaled by the trial's force-to-weight ratio: every segment mass, so S and BW, scale by k
    n = len(ts)
    tr_ = slice(int(1.0 * sp_.FS), n - int(1.0 * sp_.FS))
    S, F = ts.S_newton.values[tr_], (ts.FxR_meas + ts.FxL_meas).values[tr_]
    k = float(ratio.loc[(s, tr)])
    resid.append(100 * np.sqrt(np.mean((S - F) ** 2)) / BW)
    resid_k.append(100 * np.sqrt(np.mean((k * S - F) ** 2)) / (k * BW))
vs = pd.read_csv(os.path.join(R, "vpp_split_test.csv"))
res["vertical_force_stride_range_pctBW"] = float(np.mean(rng_pct))
res["vertical_error_of_range_pct"] = float(100 * vs.vert_stance.mean() / np.mean(rng_pct))
res["residual_from_timeseries"] = float(np.mean(resid))
res["residual_newton_check"] = float(t.newton_check.mean())
res["mass_rescaled_residual"] = float(np.mean(resid_k))
res["n_timeseries"] = len(resid)

# combined wearable-grade increment, subject bootstrap of the trial-weighted mean
dg = pd.read_csv(os.path.join(R, "final_degradation_trials.csv"))
dg = dg[dg.arm == "SoleID"]
c = dg[dg.condition == "clean"].set_index(["subject", "trial"]).stance
w = dg[dg.condition == "wearable"].set_index(["subject", "trial"]).stance
inc = (w - c).dropna().reset_index()
g = inc.groupby("subject").stance
sums, cnts = g.sum().values, g.count().values
boot = np.random.default_rng(0).integers(0, len(sums), size=(5000, len(sums)))
bm = sums[boot].sum(1) / cnts[boot].sum(1)
res["combined_increment"] = float(inc.stance.mean())
res["combined_increment_ci"] = [float(np.percentile(bm, 2.5)), float(np.percentile(bm, 97.5))]

# Wang without the subject that fails the quality check: worst force quality and alignment
wa = pd.read_csv(os.path.join(R, "wang_application.csv"))
bysub = wa.groupby("subject")[["fy_rmse_pctBW", "r_align", "SoleID_stance"]].mean()
worst = bysub.fy_rmse_pctBW.idxmax()
res["wang_outlier"] = dict(subject=str(worst), worst_alignment=bool(bysub.r_align.idxmin() == worst),
                           shear_over_median=float(bysub.SoleID_stance[worst] / bysub.SoleID_stance.median()))
keep = wa[wa.subject != worst]
res["wang_without_outlier"] = dict(stance=float(keep.SoleID_stance.mean()), hip=float(keep.SoleID_hip.mean()))

# where the mean shear difference from the reference is largest (subject means, as for SPM)
z = np.load(os.path.join(R, "curves_fukuchi.npz"), allow_pickle=True)
subj = np.asarray(z["subjects"])
d = np.asarray(z["SoleID_shear"], float) - np.asarray(z["reference_shear"], float)
dm = np.array([d[subj == s_].mean(0) for s_ in np.unique(subj)]).mean(0)
res["spm_max_at_pct"] = int(np.argmax(np.abs(dm)))
save()
print("quick checks written")

# ---- the slow one: every test trial prepared again --------------------------------------------
if "--lever" in sys.argv:
    low = []
    for i, (s, tr) in enumerate(zip(t.subject, t.trial)):
        try:
            dd = ex.prepare(s, tr)
        except Exception as e:                      # noqa: BLE001 -- report and move on
            print("  skip", s, tr, repr(e))
            continue
        sp_.configure(s, tr)
        py = dd["pelvis"][:, 1] + 0.2
        for side in ("R", "L"):
            Fy = dd["grf"][side]["Fy"]
            cy = dd["grf"][side].get("copy")
            cy = np.zeros_like(Fy) if cy is None else cy
            on = Fy > sp_.F_CONTACT
            if on.any():
                low.append(float(np.min(py[on] - cy[on])))
        if i % 25 == 0:
            print(f"  {i}/{len(t)}", flush=True)
    res["smallest_lever_m"] = float(np.min(low))
    res["lever_trials"] = len(low) // 2
    save()
    print("smallest lever", res["smallest_lever_m"])
