"""External validation of SoleID (frozen v1.0) on the Camargo et al. 2021 dataset.

Different lab, different marker protocol, instrumented treadmill whose belt speed ramps within a
trial. Nothing is tuned here: the weights come from results/calibration.json (Fukuchi calibration
subjects). Arms are the same as in the main validation.

Per stride metrics are recorded so that the continuously varying belt speed can be used as a
covariate instead of a fixed condition.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

import camargo_adapter as ca
import experiments as ex
import soleid_planar as sp_
import config

CAL = json.load(open(os.path.join(sp_.OUT, "calibration.json")))["chosen"]
INFO = pd.read_csv(config.CAMARGO_INFO).set_index("Subject")
MIN_SPEED = 0.4       # ignore standing / very slow parts of the ramp-up protocol


def run_trial(subj, trial, mode="treadmill"):
    d = ca.load_trial(subj, trial, mode)
    n = d["n"]
    mass, height = float(INFO.loc[subj, "Weight"]), float(INFO.loc[subj, "Height"])
    sp_.MASS, sp_.HEIGHT, sp_.FS = mass, height, ca.FS
    segs, pel = ca.build_segments(d["P"], n, mass, height)
    grf = d["grf"]
    BW = mass * 9.81

    onR = grf["R"]["Fy"] > sp_.F_CONTACT
    onL = grf["L"]["Fy"] > sp_.F_CONTACT
    ds = onR & onL
    trim = np.zeros(n, bool)
    trim[int(ca.FS):n - int(ca.FS)] = True
    trim &= d["speed"] > MIN_SPEED

    dd = dict(n=n, segs=segs, pelvis=pel, grf=grf, trim=trim)
    S = ex.newton_sum(dd)
    prior = ex.vpp_prior(dd, CAL["h_vpp"])
    sR, sL, _ = ex.solve(dd, S, fc_smooth=CAL["fc_smooth"], prior=prior, w_prior=CAL["w_prior"], w_torque=0.0)
    vR, vL, _ = ex.solve(dd, S, fc_smooth=CAL["fc_smooth"], prior=None, w_prior=0.0, w_torque=0.0)
    bR, bL = sp_.newton_baseline(S, grf["R"]["Fy"], grf["L"]["Fy"])
    kin, _ = sp_.kinematics_only_baseline(d["P"], segs, n)

    ref = sp_.inverse_dynamics(segs, "R", grf["R"]["Fx"], grf["R"]["Fy"], grf["R"]["copx"])
    arms = {"SoleID": (sR, grf["R"]["Fy"], grf["R"]["copx"]),
            "SoleID_noVPP": (vR, grf["R"]["Fy"], grf["R"]["copx"]),
            "Newton_prop": (bR, grf["R"]["Fy"], grf["R"]["copx"]),
            "Kin_only": (kin["R"]["Fx"], kin["R"]["Fy"], kin["R"]["copx"])}

    # ---- per-stride rows (speed as covariate)
    hs = sp_.heel_strikes(grf["R"]["Fy"])
    rows = []
    for a, b in zip(hs[:-1], hs[1:]):
        if b - a < 0.5 * ca.FS or b - a > 2.5 * ca.FS:
            continue
        sl = slice(a, b)
        if not trim[sl].all():
            continue
        spd = float(np.mean(d["speed"][sl]))
        m_on = onR[sl]
        m_ds = ds[sl]
        if m_on.sum() < 20:
            continue
        row = dict(subject=subj, trial=trial, mode=mode, speed=spd, mass=mass, stride_s=(b - a) / ca.FS,
                   newton_check=100 * np.sqrt(np.mean((S[sl] - (np.where(onR, grf["R"]["Fx"], 0) +
                                                                np.where(onL, grf["L"]["Fx"], 0))[sl]) ** 2)) / BW)
        for name, (est, Fy_u, cop_u) in arms.items():
            e, meas = est[sl], grf["R"]["Fx"][sl]
            row[f"{name}_stance"] = 100 * np.sqrt(np.mean((e[m_on] - meas[m_on]) ** 2)) / BW
            row[f"{name}_ds"] = (100 * np.sqrt(np.mean((e[m_ds] - meas[m_ds]) ** 2)) / BW) if m_ds.sum() > 5 else np.nan
            M = sp_.inverse_dynamics(segs, "R", est, Fy_u, cop_u)
            for j in ("ankle", "knee", "hip"):
                row[f"{name}_{j}"] = float(np.sqrt(np.mean((M[j][sl] / mass - ref[j][sl] / mass) ** 2)))
        rows.append(row)
    return rows


def main(limit_subjects=None, mode="treadmill"):
    subs = ca.subjects()
    if limit_subjects:
        subs = subs[:limit_subjects]
    allrows = []
    for s in subs:
        for t in ca.trials(s, mode):
            try:
                r = run_trial(s, t, mode)
                allrows += r
                print(f"  {s} {t}: {len(r)} strides", flush=True)
            except Exception as e:
                print(f"  {s} {t}: ERROR {e!r}", flush=True)
    df = pd.DataFrame(allrows)
    df.to_csv(os.path.join(sp_.OUT, f"camargo_{mode}_strides.csv"), index=False)
    summarize(df)
    return df


def summarize(df):
    pd.set_option("display.width", 220)
    arms = ["Kin_only", "Newton_prop", "SoleID_noVPP", "SoleID"]
    print("")
    print(f"CAMARGO ({df['mode'].iloc[0]}): {df.subject.nunique()} subjects, "
          f"{df.groupby(['subject', 'trial']).ngroups} subject-trials, {len(df)} strides")
    df = df.copy()
    df["speed_bin"] = pd.cut(df.speed, [0.4, 0.8, 1.2, 1.6, 3.0], labels=["0.4-0.8", "0.8-1.2", "1.2-1.6", ">1.6"])
    for key in ("stance", "ds", "knee", "hip"):
        cols = [f"{a}_{key}" for a in arms]
        print(f"\n{key}: " + "  ".join(f"{a}={df[f'{a}_{key}'].mean():.3f}" for a in arms))
        print(df.groupby("speed_bin", observed=True)[cols].mean().round(3).to_string())
    print("\nwhole-body Newton check: %.2f %%BW" % df.newton_check.mean())
    print("SoleID better than Newton_prop in %.0f%% of strides (stance)" % (100 * (df.SoleID_stance < df.Newton_prop_stance).mean()))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        summarize(pd.read_csv(os.path.join(sp_.OUT, "camargo_treadmill_strides.csv")))
    else:
        main(limit_subjects=int(sys.argv[1]) if len(sys.argv) > 1 else None)
