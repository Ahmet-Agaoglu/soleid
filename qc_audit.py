"""Trial flow of the controlled layer, where the exclusions fall, and whether they could have
flattered the result.

Exclusions are defensible individually -- a dead belt has no reference, a handrail adds an
unmeasured external force -- but together they could remove difficult subjects or speeds. This
script accounts for every trial, shows where the exclusions land by subject and by speed, tests
whether the acceptance band matters, runs the estimator on the excluded trials that still have a
usable reference, and looks at the population's force-to-weight ratio of 0.93.

Handrail use is taken from the data set's own annotation (the TreadHands field recorded by the
original investigators), not inferred from the force, so it is independent of the criterion it is
reported next to.

Output: results/qc_audit.json, results/qc_excluded_runs.csv
"""
import json
import os

import numpy as np
import pandas as pd

import experiments as ex
import soleid_planar as sp_
from protocol import CAL_SUBJECTS, _all_trials, quality_table

CAL = json.load(open(os.path.join(sp_.OUT, "calibration.json")))["chosen"]


def soleid_errors(subj, trial, scale=1.0):
    """Stance shear RMSE and hip moment RMSE, with every force channel divided by `scale`."""
    d = ex.prepare(subj, trial)
    sp_.configure(subj, trial)
    if scale != 1.0:
        for s in ("R", "L"):
            for k in ("Fx", "Fy"):
                d["grf"][s][k] = d["grf"][s][k] / scale
    g = d["grf"]["R"]
    S = ex.newton_sum(d, 1.0)
    pr = ex.vpp_prior(d, CAL["h_vpp"])
    fR, _, _ = ex.solve(d, S, fc_smooth=CAL["fc_smooth"], prior=pr, w_prior=CAL["w_prior"],
                        w_torque=0.0)
    on = (g["Fy"] > sp_.F_CONTACT) & d["trim"]
    BW = sp_.MASS * 9.81
    ref = sp_.inverse_dynamics(d["segs"], "R", g["Fx"], g["Fy"], g["copx"])["hip"]
    est = sp_.inverse_dynamics(d["segs"], "R", fR, g["Fy"], g["copx"])["hip"]
    return dict(stance=100 * float(np.sqrt(np.mean((fR[on] - g["Fx"][on]) ** 2))) / BW,
                hip=float(np.sqrt(np.mean(((est - ref) / sp_.MASS)[d["trim"]] ** 2))))


def main():
    q = quality_table()
    info = pd.DataFrame(_all_trials(), columns=["subject", "trial", "speed", "age"])
    q = q.merge(info, on=["subject", "trial"], how="left")
    q["group"] = np.where(q.ok, "retained",
                 np.where(q.dead_belt, "dead belt",
                 np.where(q.hands.str.lower() == "yes", "handrail", "force scale")))
    gaps = pd.read_csv(os.path.join(sp_.OUT, "test_skipped.csv"))
    test = pd.read_csv(os.path.join(sp_.OUT, "test_all_trials.csv"))
    out = {}

    # ---- 1. flow
    flow = q.group.value_counts().to_dict()
    ret = q[q.ok]
    n_cal = int(ret.subject.isin(CAL_SUBJECTS).sum())
    out["flow"] = dict(total=len(q), **{k: int(v) for k, v in flow.items()},
                       retained_calibration=n_cal, retained_test=int(len(ret) - n_cal),
                       marker_gap_test=int(len(gaps)), final_test=int(len(test)),
                       final_calibration=int(json.load(open(os.path.join(sp_.OUT, "calibration.json")))["n_trials"]))
    print("flow:", out["flow"])

    # ---- 2. where they fall
    by = {}
    for grp in ("dead belt", "handrail", "force scale"):
        g = q[q.group == grp]
        subj_counts = g.subject.value_counts()
        whole = [s for s in subj_counts.index if subj_counts[s] == (q.subject == s).sum()]
        by[grp] = dict(trials=int(len(g)), subjects=int(g.subject.nunique()),
                       subjects_losing_every_trial=whole,
                       calibration_subjects_affected=sorted(set(g.subject) & set(CAL_SUBJECTS)),
                       speed_mean=float(g.speed.mean()), speed_min=float(g.speed.min()),
                       speed_max=float(g.speed.max()),
                       per_subject=subj_counts.to_dict())
        print(f"\n{grp}: {len(g)} trials in {g.subject.nunique()} subjects; whole subjects lost: "
              f"{whole}; speed {g.speed.min():.2f}-{g.speed.max():.2f} m/s (mean {g.speed.mean():.2f})")
    out["by_group"] = by
    out["speed_retained_mean"] = float(ret.speed.mean())
    subj_all = q.subject.nunique()
    subj_kept = test.subject.nunique() + len(set(ret.subject) & set(CAL_SUBJECTS))
    out["subjects_total"], out["subjects_kept"] = int(subj_all), int(subj_kept)
    print(f"\nsubjects: {subj_all} in the data set, {subj_kept} contribute retained trials")

    # ---- 3. the acceptance band
    r = ret.bw_ratio
    out["retained_ratio"] = dict(min=float(r.min()), max=float(r.max()),
                                 p5=float(r.quantile(0.05)), p95=float(r.quantile(0.95)),
                                 median=float(r.median()))
    nonhand = q[q.hands.str.lower() != "yes"]
    bands = [(0.60, 1.80), (0.70, 1.30), (0.80, 1.20), (0.85, 1.10), (0.88, 1.00)]
    out["band_sensitivity"] = {f"{a:.2f}-{b:.2f}": int(((nonhand.bw_ratio >= a) &
                                                         (nonhand.bw_ratio <= b)).sum())
                               for a, b in bands}
    print("retained ratio:", {k: round(v, 3) for k, v in out["retained_ratio"].items()})
    print("trials inside each band (handrail trials excluded separately):",
          out["band_sensitivity"])

    # ---- 4. the 0.93: instrument or records?
    per = ret.groupby("subject").bw_ratio
    within = float(np.sqrt(per.var().mean()))
    between = float(per.mean().std())
    rs = float(np.corrcoef(ret.speed, ret.bw_ratio)[0, 1])
    out["ratio_structure"] = dict(within_subject_sd=within, between_subject_sd=between,
                                  corr_with_speed=rs,
                                  subject_means_min=float(per.mean().min()),
                                  subject_means_max=float(per.mean().max()))
    print(f"ratio sd within subjects {within:.4f}, between subjects {between:.4f}, "
          f"r with speed {rs:+.3f}; subject means {per.mean().min():.3f}-{per.mean().max():.3f}")
    cam = pd.read_csv(os.path.join(sp_.OUT, "camargo_treadmill_strides.csv"))
    if "fy_ratio" in cam.columns:
        out["camargo_ratio_median"] = float(cam.fy_ratio.median())

    # ---- 5. the excluded trials that still have a reference
    rows = []
    for grp in ("handrail", "force scale"):
        for t in q[(q.group == grp) & ~q.subject.isin(CAL_SUBJECTS)].itertuples():
            try:
                sc = t.bw_ratio if grp == "force scale" else 1.0
                e = soleid_errors(t.subject, t.trial, sc)
                rows.append(dict(group=grp, subject=t.subject, trial=t.trial, speed=t.speed,
                                 ratio=t.bw_ratio, **e))
            except Exception as ex_:
                print(f"  skip {t.subject} {t.trial}: {ex_!r}")
    ex_df = pd.DataFrame(rows)
    ex_df.to_csv(os.path.join(sp_.OUT, "qc_excluded_runs.csv"), index=False)
    out["excluded_runs"] = {g: dict(n=int(len(v)), stance=float(v.stance.mean()),
                                    hip=float(v.hip.mean())) for g, v in ex_df.groupby("group")}
    out["test_mean"] = dict(stance=float(test.SoleID_stance.mean()), hip=float(test.SoleID_hip.mean()))
    allin = pd.concat([test[["SoleID_stance", "SoleID_hip"]].rename(
        columns={"SoleID_stance": "stance", "SoleID_hip": "hip"}), ex_df[["stance", "hip"]]])
    out["test_plus_excluded"] = dict(n=int(len(allin)), stance=float(allin.stance.mean()),
                                     hip=float(allin.hip.mean()))
    print("\nexcluded trials with a usable reference:", out["excluded_runs"])
    print("test set:", out["test_mean"], " test + those:", out["test_plus_excluded"])
    json.dump(out, open(os.path.join(sp_.OUT, "qc_audit.json"), "w"), indent=1, default=str)
    return out


if __name__ == "__main__":
    main()
