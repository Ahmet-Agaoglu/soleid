"""Honest evaluation protocol for SoleID (planar).

1. Calibration subjects (fixed list): grid-search SoleID weights (VPP height, prior weight, torque
   smoothness weight, smoothing cutoff) minimising mean stance-phase Fx RMSE over ALL their trials.
2. Freeze the chosen parameters; run every arm on every trial of the remaining (test) subjects.
3. Summaries by speed bin and age group.

Usage:  python protocol.py calibrate      -> results/calibration.json
        python protocol.py test           -> results/test_all_trials.csv + summary
"""
import itertools
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import experiments as ex
import soleid_planar as sp_

CAL_SUBJECTS = ["WBDS01", "WBDS05", "WBDS10", "WBDS20", "WBDS30"]   # 5 of 51; young + older mix
CAL_JSON = os.path.join(sp_.OUT, "calibration.json")


QUALITY_CSV = os.path.join(sp_.OUT, "fukuchi_quality.csv")


def trial_quality(subj, trial):
    """Body-weight consistency and handrail use for one trial (see sp_.bw_ratio)."""
    sp_.configure(subj, trial)
    gr = pd.read_csv(os.path.join(sp_.DATA, f"{subj}{trial}grf.txt"), sep="\t")
    grf = {s: dict(Fy=gr[f"Fy{sp_.PLATE[s]}"].values) for s in ("R", "L")}
    ratio = sp_.bw_ratio(grf, sp_.MASS)
    dead = [s for s in ("R", "L") if np.max(grf[s]["Fy"]) < sp_.F_CONTACT]
    info = pd.read_csv(os.path.join(sp_.DATA, "WBDSinfo.csv"))
    row = info[info.FileName == f"{subj}{trial}grf.txt"]
    hands = str(row.TreadHands.iat[0]) if len(row) else "?"
    scale_bad = not (sp_.BW_RATIO_LO <= ratio <= sp_.BW_RATIO_HI)
    ok = not dead and not scale_bad and hands.lower() != "yes"
    reason = ("dead belt" if dead else
              "handrail" if hands.lower() == "yes" else
              f"force/mass scale (bw ratio {ratio:.2f})" if scale_bad else "")
    return dict(subject=subj, trial=trial, bw_ratio=ratio, hands=hands, dead_belt=bool(dead),
                ok=bool(ok), reason=reason)


def quality_table(refresh=False):
    if os.path.exists(QUALITY_CSV) and not refresh:
        return pd.read_csv(QUALITY_CSV)
    rows = []
    for subj, trial, speed, age in _all_trials():
        try:
            rows.append(trial_quality(subj, trial))
        except Exception as e:
            rows.append(dict(subject=subj, trial=trial, bw_ratio=np.nan, hands="?", dead_belt=True,
                             ok=False, reason=repr(e)[:60]))
    df = pd.DataFrame(rows)
    df.to_csv(QUALITY_CSV, index=False)
    return df


def _all_trials():
    info = pd.read_csv(os.path.join(sp_.DATA, "WBDSinfo.csv"))
    tr = info[info.FileName.str.contains("walkT") & info.FileName.str.endswith("grf.txt")].copy()
    tr["subj"] = tr.FileName.str[:6]
    tr["trial"] = tr.FileName.str[6:13]
    ok = []
    for _, r in tr.iterrows():
        base = os.path.join(sp_.DATA, f"{r.subj}{r.trial}")
        if all(os.path.exists(base + k + ".txt") for k in ("mkr", "grf", "knt")):
            ok.append((r.subj, r.trial, float(r["GaitSpeed(m/s)"]), r.AgeGroup))
    return ok


def available_trials(apply_quality=True):
    """Trials whose files are present and, unless disabled, that pass the data-quality gate."""
    trials = _all_trials()
    if not apply_quality:
        return trials
    q = quality_table()
    good = {(r.subject, r.trial) for r in q.itertuples() if r.ok}
    return [t for t in trials if (t[0], t[1]) in good]


def calibrate():
    trials = [t for t in available_trials() if t[0] in CAL_SUBJECTS]
    print(f"calibration trials: {len(trials)}")
    data = {}
    for subj, trial, speed, age in trials:
        try:
            data[(subj, trial)] = ex.prepare(subj, trial)
        except sp_.MarkerGapError as e:
            print("skip", subj, trial, e)
    grid = list(itertools.product([0.1, 0.2, 0.3, 0.4],      # h_vpp
                                  [0.5, 1.0, 2.0],           # w_prior
                                  [0.0, 1.0, 2.0],           # w_torque
                                  [5.0, 8.0, 12.0]))         # fc_smooth
    rows = []
    t0 = time.time()
    for h, wp, wt, fc in grid:
        st, ds, hip, prop = [], [], [], []
        for (subj, trial), d in data.items():
            sp_.configure(subj, trial)
            S = ex.newton_sum(d, 1.0)
            pr = ex.vpp_prior(d, h)
            fR, fL, status = ex.solve(d, S, fc_smooth=fc, prior=pr, w_prior=wp, w_torque=wt)
            m = ex.metrics(d, fR)
            st.append(m["stance"]); ds.append(m["ds"]); hip.append(m["hip"])
        rows.append(dict(h_vpp=h, w_prior=wp, w_torque=wt, fc_smooth=fc,
                         stance=np.mean(st), ds=np.mean(ds), hip=np.mean(hip)))
    df = pd.DataFrame(rows).sort_values("stance")
    df.to_csv(os.path.join(sp_.OUT, "calibration_grid.csv"), index=False)
    best = df.iloc[0]
    chosen = dict(w_prior=float(best.w_prior), h_vpp=float(best.h_vpp), w_torque=float(best.w_torque), fc_smooth=float(best.fc_smooth))
    with open(CAL_JSON, "w") as f:
        json.dump(dict(subjects=CAL_SUBJECTS, n_trials=len(data), chosen=chosen,
                       cal_stance=float(best.stance), cal_ds=float(best.ds), cal_hip=float(best.hip),
                       grid_size=len(grid), seconds=time.time() - t0), f, indent=2)
    print(df.head(10).round(3).to_string(index=False))
    print("chosen:", chosen)
    return chosen


def test():
    chosen = json.load(open(CAL_JSON))["chosen"]
    sp_.V2_PARAMS.update(chosen)
    trials = [t for t in available_trials() if t[0] not in CAL_SUBJECTS]
    print(f"test trials: {len(trials)} with params {chosen}")
    rows, skipped = [], []
    for i, (subj, trial, speed, age) in enumerate(trials):
        try:
            r = sp_.main(subj, trial)
        except sp_.MarkerGapError as e:
            skipped.append((subj, trial, str(e)))
            continue
        except Exception as e:  # keep going, record
            skipped.append((subj, trial, repr(e)))
            continue
        row = dict(subject=subj, trial=trial, speed=speed, age=age, mass=r["mass"],
                   newton_check=r["newton_check"]["rmse_pctBW"])
        for arm in ("SoleID", "SoleID_v1", "Newton_prop", "Kin_only"):
            f = r["Fx_right_foot"][arm]
            j = r["joint_moments_right_Nm_per_kg"][arm]
            row[f"{arm}_stance"] = f["stance_R"]["rmse_pctBW"]
            row[f"{arm}_ds"] = f["double_support"]["rmse_pctBW"]
            row[f"{arm}_ss"] = f["single_support_R"]["rmse_pctBW"]
            for jn in ("ankle", "knee", "hip"):
                row[f"{arm}_{jn}"] = j[jn]["rmse_all"]
            row[f"{arm}_prop_peak_err"] = r["Fx_right_peaks"][arm]["propulsion_peak_err_N_mean"]
            row[f"{arm}_brake_peak_err"] = r["Fx_right_peaks"][arm]["braking_peak_err_N_mean"]
        row["ref_v3d_hip_r"] = abs(r["reference_vs_dataset_moments"]["hip"]["r"])
        row["hip_source"] = r["hip_source"]
        row["pelvis_markers_used"] = r["pelvis_markers_used"]
        rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(trials)}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(sp_.OUT, "test_all_trials.csv"), index=False)
    pd.DataFrame(skipped, columns=["subject", "trial", "reason"]).to_csv(os.path.join(sp_.OUT, "test_skipped.csv"), index=False)
    summarize(df)
    return df


def summarize(df):
    df = df.copy()
    df["speed_bin"] = pd.cut(df.speed, [0, 0.8, 1.2, 1.6, 3.0], labels=["<0.8", "0.8-1.2", "1.2-1.6", ">1.6"])
    arms = ["Kin_only", "Newton_prop", "SoleID_v1", "SoleID"]
    pd.set_option("display.width", 220)
    print(f"\nTEST SET: {df.subject.nunique()} subjects, {len(df)} trials")
    for key, lab in (("stance", "Fx RMSE stance (%BW)"), ("ds", "Fx RMSE double support (%BW)"),
                     ("hip", "hip moment RMSE (Nm/kg)"), ("knee", "knee moment RMSE (Nm/kg)")):
        print(f"\n{lab}: mean [sd] by arm")
        print("  overall: " + "  ".join(f"{a}={df[f'{a}_{key}'].mean():.2f}[{df[f'{a}_{key}'].std():.2f}]" for a in arms))
        g = df.groupby("speed_bin", observed=True)[[f"{a}_{key}" for a in arms]].mean().round(2)
        print(g.to_string())
        g = df.groupby("age", observed=True)[[f"{a}_{key}" for a in arms]].mean().round(2)
        print(g.to_string())
    print("\nSoleID propulsion peak error (N): mean %.1f sd %.1f ; braking %.1f sd %.1f" % (
        df.SoleID_prop_peak_err.mean(), df.SoleID_prop_peak_err.std(), df.SoleID_brake_peak_err.mean(), df.SoleID_brake_peak_err.std()))
    print("paired improvement SoleID vs Newton_prop, stance: %.1f%% of trials better" % (100 * (df.SoleID_stance < df.Newton_prop_stance).mean()))
    if "hip_source" in df:
        print("\nhip source counts:", df.hip_source.value_counts().to_dict())
        g = df.groupby("hip_source")[["SoleID_stance", "SoleID_ds", "SoleID_hip", "Newton_prop_hip"]].mean().round(3)
        print(g.to_string())


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "calibrate"
    if cmd == "calibrate":
        calibrate()
    elif cmd == "test":
        test()
    elif cmd == "summary":
        summarize(pd.read_csv(os.path.join(sp_.OUT, "test_all_trials.csv")))
